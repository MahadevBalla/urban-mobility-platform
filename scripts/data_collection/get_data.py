"""
Mumbai OD routes real-time sampler (Google Routes API, traffic-aware).

- Loads zones from CSV (zone_id, lat, lng, area_km2)
- Generates random sub-points per zone
- Samples OD pairs between zones
- Uses traffic-aware routing with alternative routes
- Supports optional road snapping (Roads API)
- Imputes intra-zonal trips using Wardrop's formula
- Runs continuously at fixed intervals

Output columns:
    origin_id, dest_id, origin_zone, dest_zone,
    origin_lat, origin_lng, dest_lat, dest_lng,
    capture_time_utc, capture_time_ist,
    route_rank, route_label, distance_m, duration_s, is_imputed
"""


import os
import csv
import time
import json
from datetime import datetime, timedelta, timezone
import random
from dotenv import load_dotenv
import requests
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if not (ENV_PATH).exists():
    print("[WARN] .env file not found; ensure GOOGLE_MAPS_API_KEY is set in environment variables.")

load_dotenv(ENV_PATH)

# ---------------- CONFIGURATION ----------------
# Google Maps API key 
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Zones file: CSV with columns: zone_id,lat,lng
ZONES_CSV = BASE_DIR / "mumbai_zones.csv"

# Output file
OUTPUT_CSV = BASE_DIR / "google_routes_realtime.csv"

# Interval b/w batches
STEP_MINUTES = 5

# OD sampling and routes
MAX_OD_PAIRS = 50        # cap no. of OD pairs to keep total calls under free tier
MAX_ROUTES_PER_OD = 0    # 0 or None -> take ALL routes returned by API

# Simple rate limiting to avoid QPS spikes
SLEEP_BETWEEN_CALLS_SEC = 0.2    # tune up/down based on observed rate

# Retry config for transient errors (e.g. network issues). HTTP errors like 400/403/429 are not retried since they indicate a problem with the request or quota.
MAX_HTTP_RETRIES = 3

# Random seed for reproducibility (if shuffling OD pairs or departure times is desired)
RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)

# Sub-points per zone
POINTS_PER_ZONE = 3            # sub-points to generate per zone centroid
DEFAULT_RADIUS_KM = 0.4        # fallback scatter radius if area_km2 is missing

# Intra-zonal imputation
INTRA_ZONAL_SPEED_KMH = 20     # conservative urban speed for Wardrop estimate
INTRA_ZONAL_DEFAULT_S = 300    # fallback if area_km2 also missing (5 min)

# Road snapping (requires Roads API enabled in your project)
# NOTE: Disabled for now due to API cost and unnecessary for current use-case
SNAP_TO_ROADS = False

# ---------------- HELPERS ----------------
def load_zones(path):
    """
    Load zones from a CSV file.

    Expects columns: zone_id, lat, lng

    Returns:
        List of dicts like:
        {
            "id": str,
            "lat": float,
            "lng": float
        }
    """
    zones = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("zone_id"):
                continue
            area = row.get("area_km2", "").strip()
            zones.append({
                "id": row["zone_id"].strip(),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "area_km2": float(area) if area else None,
            })
    return zones

def _radius_deg(zone):
    """
    Compute scatter radius in degrees for a zone.
    Derived from zone area: assume circle → r = sqrt(A/π).
    Use half that radius so sub-points stay within the zone boundary.
    """
    if zone["area_km2"] is not None and zone["area_km2"] > 0:
        r_km = math.sqrt(zone["area_km2"] / math.pi) * 0.5
    else:
        r_km = DEFAULT_RADIUS_KM

    lat_deg = r_km / 111.0

    cos_lat = math.cos(math.radians(zone["lat"]))
    cos_lat = max(cos_lat, 1e-6)
    lng_deg = r_km / (111.0 * cos_lat)

    return lat_deg, lng_deg

def _sample_point(zone, lat_r, lng_r, rng):
    """
    Sample a random point inside a circular region around zone centroid.

    Uses rejection sampling to ensure uniform distribution inside circle.

    Args:
        zone (dict): zone with lat/lng
        lat_r (float): latitude radius in degrees
        lng_r (float): longitude radius in degrees
        rng (random.Random): random generator

    Returns:
        tuple: (lat, lng) of sampled point
    """
    if lat_r < 1e-9 or lng_r < 1e-9:
        return zone["lat"], zone["lng"]

    while True:
        dlat = rng.uniform(-lat_r, lat_r)
        dlng = rng.uniform(-lng_r, lng_r)
        if (dlat / lat_r) ** 2 + (dlng / lng_r) ** 2 <= 1:
            return zone["lat"] + dlat, zone["lng"] + dlng

def generate_sub_points(zones, n, rng):
    """
    Generate random sub-points for each zone.

    Each zone centroid is expanded into `n` points scattered within
    a radius derived from zone area.

    Args:
        zones (list): list of zone dicts
        n (int): number of sub-points per zone
        rng (random.Random): random generator

    Returns:
        list: list of sub-point dicts
    """
    sub_points = []
    for zone in zones:
        lat_r, lng_r = _radius_deg(zone)

        for i in range(n):
            lat, lng = _sample_point(zone, lat_r, lng_r, rng)

            sub_points.append({
                "id": f"{zone['id']}_{i}",
                "zone_id": zone["id"],
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "area_km2": zone["area_km2"],
            })
    return sub_points

def snap_points_to_roads(points):
    """
    Snap coordinates to nearest road using Google Roads API.

    Points are batched (max 100 per request). If snapping fails,
    original coordinates are retained.

    Args:
        points (list): list of sub-point dicts

    Returns:
        dict: mapping {(lat, lng): (snapped_lat, snapped_lng)}
    """
    snap_cache = {}
    url = "https://roads.googleapis.com/v1/nearestRoads"
    batch_size = 100

    coords = list({(p["lat"], p["lng"]) for p in points})  # deduplicate

    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        points_str = "|".join(f"{lat},{lng}" for lat, lng in batch)

        try:
            resp = requests.get(url, params={"points": points_str, "key": GOOGLE_MAPS_API_KEY}, timeout=10)
            resp.raise_for_status()

            try:
                data = resp.json()
            except Exception:
                print("[WARN] Invalid JSON from Roads API")
                continue

            for sp in data.get("snappedPoints", []):
                idx = sp.get("originalIndex")
                if idx is not None and idx < len(batch):
                    loc = sp["location"]
                    snap_cache[batch[idx]] = (loc["latitude"], loc["longitude"])
        except Exception as e:
            print(f"[WARN] Road snapping failed for batch {i//batch_size}: {e}")

    # For any point that failed to snap, fall back to original coords
    for lat, lng in coords:
        if (lat, lng) not in snap_cache:
            snap_cache[(lat, lng)] = (lat, lng)

    return snap_cache

def apply_snapping(sub_points, snap_cache):
    """
    Replace sub-point coordinates with snapped road coordinates.

    Falls back to original coordinates if snapping is missing.

    Args:
        sub_points (list): list of sub-point dicts
        snap_cache (dict): mapping of original → snapped coords

    Returns:
        list: updated sub_points (in-place)
    """
    for sp in sub_points:
        key = (sp["lat"], sp["lng"])
        snapped = snap_cache.get(key, key)
        sp["lat"] = snapped[0]
        sp["lng"] = snapped[1]
    return sub_points

def wardrop_intra_zonal(area_km2):
    """
    Estimate intra-zonal distance and travel time.

    Uses Wardrop-based approximation:
        avg distance ≈ 0.52 * sqrt(area_km2) [km]

    Travel time is computed using a fixed average speed.
    If area is missing or invalid, returns default duration.

    Args:
        area_km2 (float or None): zone area in square kilometers

    Returns:
        tuple:
            distance_m (int): estimated trip distance in meters
            duration_s (int): estimated travel time in seconds
    """
    if area_km2 and area_km2 > 0:
        dist_km = 0.52 * math.sqrt(area_km2)
        duration_s = (dist_km / INTRA_ZONAL_SPEED_KMH) * 3600
        return round(dist_km * 1000), round(duration_s)
    return 0, INTRA_ZONAL_DEFAULT_S


def is_intra_zonal(origin, dest):
    """
    Check if an OD pair should be treated as intra-zonal.

    Conditions:
        - Same zone_id
        - OR coordinates are identical (after snapping)

    Args:
        origin (dict): origin sub-point
        dest (dict): destination sub-point

    Returns:
        bool: True if intra-zonal, else False
    """
    if origin["zone_id"] == dest["zone_id"]:
        return True
    if abs(origin["lat"] - dest["lat"]) < 1e-7 and abs(origin["lng"] - dest["lng"]) < 1e-7:
        return True
    return False

def build_request(origin, destination):
    """
    Build request body for Google Routes API.
    Uses DRIVE mode with traffic-aware routing and allows alternative routes.

    Args:
        origin (dict)     : {"id", "lat", "lng"}
        destination (dict): {"id", "lat", "lng"}

    Returns:
        dict: JSON request body for computeRoutes API
    """
    return {
        "origin": {"location": {"latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}}},
        "destination": {"location": {"latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "computeAlternativeRoutes": True,
    }


def call_compute_routes(request_body):
    """
    Call Google Routes API (computeRoutes).
    Retries on temporary errors like timeouts.
    Does NOT retry on hard errors like:
        - 400 (bad request)
        - 403 (auth/quota)
        - 429 (rate limit)

    Args:
        request_body (dict): API request payload

    Returns:
        dict: Parsed JSON response

    Raises:
        HTTPError   : for hard API errors
        RuntimeError: if all retries fail
    """
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        # Field mask: only ask for what we need
        # distanceMeters, duration, routeLabels are enough for our use case.
        # We can add routes.polyline.encodedPolyline if we want geometry.
        "X-Goog-FieldMask": (
            "routes.distanceMeters,"
            "routes.duration,"
            "routes.routeLabels"
        ),
    }

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(request_body), timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            try:
                msg = e.response.json().get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            print(f"    HTTP {status} (attempt {attempt}): {msg}")
            if status in (400, 403, 429):
                raise
        except requests.exceptions.Timeout:
            print(f"    Timeout on attempt {attempt}")
        except Exception as e:
            print(f"    Non-HTTP error on attempt {attempt}: {e}")
        time.sleep(1.0 * attempt)  # basic backoff
    raise RuntimeError("computeRoutes failed after retries")


def parse_routes(routes, max_routes=None):
    """
    Process API routes and extract useful fields.
    - Sorts routes by travel time (fastest first)
    - Optionally limits number of routes

    Args:
        routes (list)           : Raw routes from API response
        max_routes (int or None): max routes to keep

    Returns:
        List of dicts:
        {
            "rank": int,
            "distance_m": float,
            "duration_s": float,
            "labels": str
        }
    """
    def duration_sec(r):
        dur_str = r.get("duration", "0s")
        try:
            return float(dur_str.rstrip("s"))
        except ValueError:
            return 0.0

    routes = sorted(routes or [], key=duration_sec)

    if max_routes and max_routes > 0:
        routes = routes[:max_routes]

    return [
        {
            "rank": idx,
            "distance_m": r.get("distanceMeters", 0),
            "duration_s": duration_sec(r),
            "labels": ",".join(r.get("routeLabels", [])),
        }
        for idx, r in enumerate(routes, start=1)
    ]

# ---------------- MAIN SAMPLING LOGIC ----------------
FIELDNAMES = [
    "origin_id", "dest_id",
    "origin_zone", "dest_zone",
    "origin_lat", "origin_lng",
    "dest_lat", "dest_lng",
    "capture_time_utc", "capture_time_ist",
    "route_rank", "route_label",
    "distance_m", "duration_s",
    "is_imputed",
]

IST = timezone(timedelta(hours=5, minutes=30))

def write_rows(writer, origin, dest, capture_utc_iso, capture_ist_iso, parsed_routes, is_imputed):
    """Write one or more route rows for an OD pair to the CSV."""
    rows = []
    for r in parsed_routes:
        rows.append({
            "origin_id":        origin["id"],
            "dest_id":          dest["id"],
            "origin_zone":      origin["zone_id"],
            "dest_zone":        dest["zone_id"],
            "origin_lat":       origin["lat"],
            "origin_lng":       origin["lng"],
            "dest_lat":         dest["lat"],
            "dest_lng":         dest["lng"],
            "capture_time_utc": capture_utc_iso,
            "capture_time_ist": capture_ist_iso,
            "route_rank":       r["rank"],
            "route_label":      r["labels"],
            "distance_m":       r["distance_m"],
            "duration_s":       r["duration_s"],
            "is_imputed":       is_imputed,
        })
    writer.writerows(rows)

def run_batch(od_pairs, capture_utc, writer):
    """
    Run one batch of API calls for all OD pairs.
    For each OD pair:
        - Call Routes API
        - Parse routes
        - Write results to CSV

    Args:
        od_pairs (list)        : list of (origin, destination) tuples
        capture_utc (datetime) : batch start time (UTC)
        writer (csv.DictWriter): CSV writer

    Returns:
        bool:
            True  -> batch completed successfully
            False -> stop execution (quota/auth error)
    """
    capture_ist = capture_utc.astimezone(IST)
    capture_utc_iso = capture_utc.isoformat().replace("+00:00", "Z")
    capture_ist_iso = capture_ist.isoformat()

    print(f"\nBatch at {capture_ist_iso} IST ({capture_utc_iso})")
    print(f"OD pairs this batch: {len(od_pairs)}")

    for origin, dest in od_pairs:

        if is_intra_zonal(origin, dest):
            area = origin["area_km2"]
            dist_m, dur_s = wardrop_intra_zonal(area)
            imputed_route = [{"rank": 1, "distance_m": dist_m, "duration_s": dur_s, "labels": "INTRA_ZONAL"}]
            write_rows(writer, origin, dest, capture_utc_iso, capture_ist_iso, imputed_route, is_imputed=True)
            continue

        body = build_request(origin, dest)
        try:
            data = call_compute_routes(body)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (403, 429):
                print("Quota/auth error; stopping scraper to protect key...")
                return False
            # 400 or other: log and skip this pair only
            print(f"  Skipping OD pair: {origin['id']}→{dest['id']} (HTTP {status}).")
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            continue
        except RuntimeError as e:
            print(f"  Skipping OD pair: {origin['id']}→{dest['id']} after retries: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            continue

        parsed = parse_routes(
            data.get("routes", []),
            max_routes=MAX_ROUTES_PER_OD if MAX_ROUTES_PER_OD else None,
        )

        if not parsed:
            continue
        write_rows(writer, origin, dest, capture_utc_iso, capture_ist_iso, parsed, is_imputed=False)
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    return True

def initialize():
    """
    Load zones, validate config, and prepare lookup structures.
    """
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set GOOGLE_MAPS_API_KEY env var.")

    zones = load_zones(ZONES_CSV)
    if not zones:
        raise RuntimeError(f"No zones loaded from {ZONES_CSV}.")

    zone_ids = [z["id"] for z in zones]

    max_possible_pairs = len(zone_ids) ** 2
    if MAX_OD_PAIRS > max_possible_pairs:
        raise ValueError(
            f"MAX_OD_PAIRS={MAX_OD_PAIRS} exceeds possible unique OD pairs={max_possible_pairs}"
        )

    return zones, zone_ids

def prepare_sub_points(zones, rng):
    """
    Generate sub-points and optionally snap to roads.
    """
    sub_points = generate_sub_points(zones, POINTS_PER_ZONE, rng)

    if SNAP_TO_ROADS:
        print("Snapping sub-points to nearest roads (Roads API)...")
        snap_cache = snap_points_to_roads(sub_points)
        sub_points = apply_snapping(sub_points, snap_cache)
        print("Snapping complete.")

    zone_to_points = {}
    for sp in sub_points:
        zone_to_points.setdefault(sp["zone_id"], []).append(sp)

    return sub_points, zone_to_points

def sample_od_pairs(zone_ids, zone_to_points, rng):
    """
    Sample OD pairs by shuffling all possible zone pairs and selecting a subset.
    Ensures no duplicates and avoids infinite loop issues.
    """
    all_pairs = [(a, b) for a in zone_ids for b in zone_ids]
    rng.shuffle(all_pairs)

    od_pairs = []
    for oz_id, dz_id in all_pairs:
        if oz_id == dz_id:
            pts = zone_to_points[oz_id]
            if len(pts) < 2:
                continue  # can't make an intra-zonal pair with only 1 sub-point
            o_sp, d_sp = rng.sample(pts, 2)  # guaranteed different sub-points
        else:
            o_sp = rng.choice(zone_to_points[oz_id])
            d_sp = rng.choice(zone_to_points[dz_id])
        od_pairs.append((o_sp, d_sp))
        if len(od_pairs) == MAX_OD_PAIRS:
            break

    return od_pairs

def run_loop(zone_ids, zone_to_points, write_header):
    """
    Continuous batch execution loop.
    """
    while True:
        od_pairs = sample_od_pairs(zone_ids, zone_to_points, rng)

        batch_start = datetime.now(timezone.utc)

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

            if write_header:
                writer.writeheader()
                write_header = False

            ok = run_batch(od_pairs, batch_start, writer)
            f.flush()

        if not ok:
            print("Stopping due to hard API error...")
            return
        sleep_until_next_batch(batch_start)

def sleep_until_next_batch(batch_start):
    """
    Sleep until next scheduled batch time.
    """
    now = datetime.now(timezone.utc)
    elapsed = (now - batch_start).total_seconds()

    next_run = batch_start + timedelta(minutes=STEP_MINUTES)
    sleep_sec = max(0, (next_run - now).total_seconds())

    print(f"Batch done in {elapsed:.1f}s. Sleeping {sleep_sec:.1f}s...")
    time.sleep(sleep_sec)

def main():
    """
    Main execution loop for OD sampling.

    Workflow:
        - Load zones
        - Generate sub-points
        - (Optional) snap to roads
        - Sample OD pairs every batch
        - Call Routes API
        - Write results to CSV

    Runs continuously until:
        - quota/auth error
        - user interruption (Ctrl+C)
    """
    zones, zone_ids = initialize()

    sub_points, zone_to_points = prepare_sub_points(zones, rng)

    print(f"Zones: {len(zones)}")
    print(f"Sub-points: {len(sub_points)} ({POINTS_PER_ZONE} per zone)")
    print(f"Sampling {MAX_OD_PAIRS} OD pairs per batch")
    print(f"Interval: every {STEP_MINUTES} min")
    print(f"Output → {OUTPUT_CSV}")
    print("Press Ctrl+C to stop.\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    try:
        run_loop(zone_ids, zone_to_points, write_header)
    except KeyboardInterrupt:
        print("\nStopped by user; exiting...")

if __name__ == "__main__":
    main()
