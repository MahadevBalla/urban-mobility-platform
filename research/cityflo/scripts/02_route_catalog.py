"""
02_route_catalog.py

Extracts unique route templates from trips_clean.csv and builds the route
catalog used by all downstream scripts.

trips_clean.csv uses str(list) route format produced by the reference notebook:
    "[(142, '19:30:00'), (152, '19:30:00'), ...]"
Parse with ast.literal_eval, not regex or json.loads.

Route template definition:
    A template is the unique ordered stop_id sequence regardless of run times.
    Two trips sharing the same stop sequence belong to the same template.
    This matches the report's definition (Section 5, Step 2): templates are
    structural route variants, not timetable variants.

Outputs:
    route_catalog.parquet  — one row per unique stop sequence
    route_catalog.json     — human-readable debug copy
"""

import ast
import json
import sys
from pathlib import Path
from statistics import median

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROUTE_CATALOG, ROUTE_CATALOG_JSON, TRIPS_FILE


# Parsing
def parse_route(route_str: str) -> list[tuple[int, str]]:
    """
    Parse a trip_route string into an ordered list of (stop_id, time) tuples.

    Expected format: "[(142, '19:30:00'), (152, '19:30:00'), ...]"
    Uses ast.literal_eval for safe parsing of Python-literal tuple lists.

    Returns empty list on failure; caller is responsible for counting failures.
    Does NOT use bare `except Exception` — only catches the specific errors
    ast.literal_eval can raise, so genuine programming errors still propagate.
    """
    try:
        parsed = ast.literal_eval(route_str)
    except (ValueError, SyntaxError):
        return []

    result = []
    for entry in parsed:
        # Guard against malformed tuples (wrong arity, wrong types)
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return []  # reject the whole route — partial routes are unusable
        sid, t = entry
        try:
            result.append((int(sid), str(t)))
        except (ValueError, TypeError):
            return []
    return result


def time_to_minutes(t: str) -> float:
    """Convert HH:MM:SS to minutes since midnight."""
    h, m, s = map(int, t.split(":"))
    return h * 60.0 + m + s / 60.0


# Catalog builder
def build_route_catalog(
    trips_path: Path,
    out_parquet: Path,
    out_json: Path,
) -> pd.DataFrame:
    """
    Build route template catalog from trips_clean.csv.

    A route template = unique ordered stop_id sequence.
    Multiple trip_ids share one template (same stops, different scheduled times).

    Scheduled reference times use statistics.median() across all runs of a
    template. Median is preferred over mean for transit schedules because
    anomalous early/late runs (deadheads, disruptions) are common and would
    bias the mean, but the median remains stable.
    """
    trips = pd.read_csv(trips_path)
    print(f"Trips loaded      : {len(trips):,}  columns: {list(trips.columns)}")

    # Parse all route strings
    trips["parsed"] = trips["trip_route"].apply(parse_route)

    n_empty = (trips["parsed"].apply(len) == 0).sum()
    n_null_routes = trips["trip_route"].isna().sum()
    print(f"Parse failures    : {n_empty:,} empty routes  |  {n_null_routes:,} null route strings")
    if n_empty > 0:
        print(f"  → {n_empty} trips dropped (unparseable or null route string)")

    trips = trips[trips["parsed"].apply(len) > 0].copy()
    trips["stop_seq"] = trips["parsed"].apply(lambda p: tuple(sid for sid, _ in p))
    trips["n_stops"] = trips["parsed"].apply(len)

    # Date column — needed for first/last seen
    trips["trip_date"] = pd.to_datetime(trips["trip_date"], errors="coerce")

    n_templates = trips["stop_seq"].nunique()
    print(f"Unique templates  : {n_templates:,}")
    print(f"Trips retained    : {len(trips):,}\n")

    # Build catalog rows
    catalog_rows = []

    for template_id, (seq_tuple, group) in enumerate(
        trips.groupby("stop_seq", sort=False)
    ):
        stop_ids = list(seq_tuple)
        n_stops = len(stop_ids)

        time_accum: dict[int, list[float]] = {}

        for route in group["parsed"]:
            for sid, t in route:
                time_accum.setdefault(sid, []).append(time_to_minutes(t))

        median_sched = {
            sid: round(median(v), 2)
            for sid, v in time_accum.items()
        }

        # Typical start time (median across runs, not mean)
        first_times = [
            time_to_minutes(route[0][1]) / 60.0 for route in group["parsed"] if route
        ]
        typical_start_h = round(median(first_times), 2) if first_times else 0.0

        # Date coverage (first/last seen)
        first_seen = (
            str(group["trip_date"].min().date())
            if not group["trip_date"].isna().all()
            else None
        )
        last_seen = (
            str(group["trip_date"].max().date())
            if not group["trip_date"].isna().all()
            else None
        )
        n_active_dates = int(group["trip_date"].dt.date.nunique())

        # Example trip IDs (sorted for reproducibility)
        example_ids = sorted(group["trip_id"].tolist())[:3]

        catalog_rows.append(
            {
                "template_id": template_id,
                "stop_sequence": json.dumps(stop_ids),
                "n_stops": n_stops,
                "first_stop_id": stop_ids[0],
                "last_stop_id": stop_ids[-1],
                "typical_start_hour": typical_start_h,
                "n_trips_in_catalog": len(group),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "n_active_dates": n_active_dates,
                "median_schedule_json": json.dumps(median_sched),
                "example_trip_ids": json.dumps(example_ids),
            }
        )

    catalog = pd.DataFrame(catalog_rows)

    # Summary stats
    total_trips = catalog["n_trips_in_catalog"].sum()
    print("Template statistics:")
    print(f"  Total templates         : {len(catalog):,}")
    print(f"  Total trips covered     : {total_trips:,}")
    print(f"  Coverage                : {100 * total_trips / len(trips):.1f}% of retained trips")
    print(f"  Stops/template  median  : {catalog['n_stops'].median():.0f}")
    print(f"  Stops/template  range   : {catalog['n_stops'].min()}–{catalog['n_stops'].max()}")
    print(f"  Trips/template  median  : {catalog['n_trips_in_catalog'].median():.0f}")
    print(f"  Trips/template  range   : {catalog['n_trips_in_catalog'].min()}–{catalog['n_trips_in_catalog'].max()}")

    print("\nTop 10 templates by trip count:")
    display_cols = [
        "template_id",
        "n_stops",
        "typical_start_hour",
        "n_trips_in_catalog",
        "first_stop_id",
        "last_stop_id",
        "first_seen",
        "last_seen",
    ]
    print(
        catalog.sort_values("n_trips_in_catalog", ascending=False)
        .head(10)[display_cols]
        .to_string(index=False)
    )

    # Write outputs
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_parquet, index=False, compression="zstd")

    debug = [
        {
            "template_id": int(r["template_id"]),
            "n_stops": int(r["n_stops"]),
            "n_trips": int(r["n_trips_in_catalog"]),
            "typical_start_hour": float(r["typical_start_hour"]),
            "first_stop": int(r["first_stop_id"]),
            "last_stop": int(r["last_stop_id"]),
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "n_active_dates": int(r["n_active_dates"]),
            "stops": json.loads(r["stop_sequence"]),
        }
        for _, r in catalog.iterrows()
    ]
    with open(out_json, "w") as fh:
        json.dump(debug, fh, indent=2)

    print(f"\nRoute catalog  → {out_parquet}")
    print(f"Debug JSON     → {out_json}")
    return catalog


if __name__ == "__main__":
    build_route_catalog(TRIPS_FILE, ROUTE_CATALOG, ROUTE_CATALOG_JSON)
