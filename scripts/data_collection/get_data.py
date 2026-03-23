"""
Mumbai OD routes real-time sampler (Routes API, traffic-aware, alt routes).

- Reads zones from mumbai_zones.csv (zone_id, lat, lng).
- Every STEP_MINUTES, queries live traffic-aware travel times for a sampled set of MAX_OD_PAIRS OD pairs.
- Appends results to OUTPUT_CSV continuously until stopped (Ctrl+C) or a quota/auth error occurs.

Columns written:
    origin_id, dest_id, origin_lat, origin_lng,
    dest_lat, dest_lng, capture_time_utc, capture_time_ist,
    route_rank, route_label, distance_m, duration_s
"""


import os
import csv
import time
import json
from itertools import product
from datetime import datetime, timedelta, timezone
import random
from dotenv import load_dotenv
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------- CONFIGURATION ----------------
# 1. Google Maps API key 
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# 2. Zones file: CSV with columns: zone_id,lat,lng
ZONES_CSV = BASE_DIR / "mumbai_zones.csv"

# 3. Output file
OUTPUT_CSV = BASE_DIR / "google_routes_5min_sample.csv"

# 4. Interval b/w batches
STEP_MINUTES = 5

# 5. OD sampling and routes
MAX_OD_PAIRS = 50        # cap no. of OD pairs to keep total calls under free tier
MAX_ROUTES_PER_OD = 0    # 0 or None -> take ALL routes returned by API

# 6. Simple rate limiting to avoid QPS spikes
SLEEP_BETWEEN_CALLS_SEC = 0.2    # tune up/down based on observed rate

# 7. Retry config for transient errors (e.g. network issues). HTTP errors like 400/403/429 are not retried since they indicate a problem with the request or quota.
MAX_HTTP_RETRIES = 3

# 8. Random seed for reproducibility (if shuffling OD pairs or departure times is desired)
RANDOM_SEED = 42

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
            zones.append({
                "id": row["zone_id"],
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
            })
    return zones

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
        "origin": {"location": {"latLng": {"latitude": origin["lat"],      "longitude": origin["lng"]}}},
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
                body = e.response.json()
            except Exception:
                body = {}
            print(f"    HTTP {status} (attempt {attempt}): {body.get('error', {}).get('message', e)}")
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

    routes = routes or []
    routes_sorted = sorted(routes, key=duration_sec)

    if max_routes and max_routes > 0:
        routes_sorted = routes_sorted[:max_routes]

    parsed = []
    for idx, r in enumerate(routes_sorted, start=1):
        parsed.append({
            "rank":       idx,
            "distance_m": r.get("distanceMeters", 0),
            "duration_s": duration_sec(r),
            "labels":     ",".join(r.get("routeLabels", [])),
        })
    return parsed

# ---------------- MAIN SAMPLING LOGIC ----------------
FIELDNAMES = [
    "origin_id", "dest_id",
    "origin_lat", "origin_lng",
    "dest_lat",   "dest_lng",
    "capture_time_utc",
    "capture_time_ist",
    "route_rank", "route_label",
    "distance_m", "duration_s",
]

IST = timezone(timedelta(hours=5, minutes=30))

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

    for origin, dest in od_pairs:
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
        for r in parsed:
            writer.writerow({
                "origin_id": origin["id"],
                "dest_id": dest["id"],
                "origin_lat": origin["lat"],
                "origin_lng": origin["lng"],
                "dest_lat": dest["lat"],
                "dest_lng": dest["lng"],
                "capture_time_utc": capture_utc_iso,
                "capture_time_ist": capture_ist_iso,
                "route_rank": r["rank"],
                "route_label": r["labels"],
                "distance_m": r["distance_m"],
                "duration_s": r["duration_s"],
            })

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
    return True


def main():
    """
    Entry point for the script.
    - Loads zones
    - Samples OD pairs
    - Runs batches every STEP_MINUTES
    - Writes results to CSV continuously

    Stops if:
        - API quota/auth error occurs
        - User interrupts (Ctrl+C)
    """
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError("Set GOOGLE_MAPS_API_KEY env var (Routes API enabled).")

    zones = load_zones(ZONES_CSV)
    if not zones:
        raise RuntimeError(f"No zones loaded from {ZONES_CSV}.")

    all_pairs = [(o, d) for o, d in product(zones, zones) if o["id"] != d["id"]]

    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    od_pairs = all_pairs[:MAX_OD_PAIRS]

    print(f"Loaded {len(zones)} zones & {len(od_pairs)} OD pairs.")
    print(f"Collecting every {STEP_MINUTES} min. Output will be stored in: {OUTPUT_CSV}")
    print("Press Ctrl+C to stop.\n")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)

    while True:
        batch_start = datetime.now(timezone.utc)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_batch:
            w = csv.DictWriter(f_batch, fieldnames=FIELDNAMES)
            if write_header:
                w.writeheader()
                write_header = False
            ok = run_batch(od_pairs, batch_start, w)
            f_batch.flush()

            if not ok:
                print("Stopping due to hard API error...")
                break

            # Sleep until next batch time
            elapsed = (datetime.now(timezone.utc) - batch_start).total_seconds()
            sleep_sec = max(0, STEP_MINUTES * 60 - elapsed)
            print(f"Batch done in {elapsed:.1f}s. Sleeping {sleep_sec:.1f}s...")
            time.sleep(sleep_sec)

if __name__ == "__main__":
    main()
