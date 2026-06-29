"""
15_policy_outputs.py

Computes three TDM policy output tables from the pipeline's processed data.

1.  Mode-shift scores  (per OD corridor, per month)
    MSS = MODE_SHIFT_WEIGHTS["load"] × load_factor_proxy
        + MODE_SHIFT_WEIGHTS["reliability"] × headway_reliability
        + MODE_SHIFT_WEIGHTS["regularity"] × service_regularity

    load_factor_proxy :
        service_supply.avg_runs_per_active_bin
        (avg vehicle-trips per active 30-min bin on the corridor).
        No passenger-occupancy data exists in the GPS-only pipeline, so a
        true load factor cannot be computed.  This proxy captures corridor
        throughput intensity.  Normalised to [0,1] via 99th-percentile clip.

    service_regularity :
        active_time_bins / MAX_BINS_PER_MONTH
        Fraction of possible 30-min bins where the corridor was observed.
        This is the proxy for captive-rider ratio as documented in the report.
        MAX_BINS_PER_MONTH = 1464  (30.5 d × 48 bins/d).

    headway_reliability :
        From headway_stats.parquet, averaged over all day_type × period strata
        per (origin_stop, month_num).  Already in [0,1]; clipped to guard
        against pathological negative values at heavily-bunched stops.

    Tier classification >=0.70 → Tier1_invest
                        >=0.50 → Tier2_monitor
                        < 0.50 → Tier3_review

2.  CO2 savings  (per OD corridor, per month)
    CO2_saved_kg = total_runs × trip_distance_km × (CAR_EF − BUS_EF)
    cars_replaced = total_runs / AVG_CAR_OCCUPANCY

    IMPORTANT: total_runs is vehicle-trips, not passenger-trips.
    Passenger-level calculations require occupancy data unavailable from GPS.
    Emission factors: IPCC AR6 road-transport averages.
        CAR_EF  = 0.171 kg CO₂/km  (private car, India fleet average)
        BUS_EF  = 0.030 kg CO₂/km  (bus per passenger, assuming ~40% seat fill)

3.  Service gaps  (OD pairs not directly served or 1-transfer reachable)
    Reads route_catalog.parquet (stop_sequence stored as JSON string;
    parse with json.loads, NOT ast.literal_eval — catalog writer uses
    json.dumps since the 02_route_catalog.py refactor).

    Classification:
        direct      — both stops on the same route template
        transfer_1  — reachable via exactly one shared intermediate stop
        unserved    — neither; DRT / new-route candidate

Input artefacts
    od_agg.parquet
    service_supply.parquet
    headway_stats.parquet
    route_catalog.parquet

Output artefacts
    mode_shift_scores.parquet
    co2_savings.parquet
    service_gaps.parquet
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AVG_CAR_OCCUPANCY,
    BUS_EMISSION_KG_PER_KM,
    CAR_EMISSION_KG_PER_KM,
    CO2_SAVINGS,
    HEADWAY_STATS,
    MAX_BINS_PER_MONTH,
    MODE_SHIFT,
    MODE_SHIFT_WEIGHTS,
    OD_AGG,
    ROUTE_CATALOG,
    SERVICE_GAPS,
    SERVICE_SUPPLY,
    TIER1_THRESHOLD,
    TIER2_THRESHOLD,
)


# 1. Mode-shift scores
def compute_mode_shift_scores(
    supply_path: Path,
    headway_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """
    Compute mode-shift score per (origin_stop_id, dest_stop_id, month).

    Reads
        service_supply.parquet : grain = (origin_stop_id, dest_stop_id, month)
                                 columns: total_runs, active_time_bins,
                                          avg_runs_per_active_bin
        headway_stats.parquet  : grain = (stop_id, day_type, period, month_num,
                                          is_monsoon)
                                 columns: headway_reliability  [0 .. 1]

    Month merging
        service_supply stores a full month timestamp while headway_stats currently
        stores only month_num. A YYYY-MM period string is materialised for the output,
        but the join currently uses month_num because that is the only key available
        in headway_stats. If headway_stats is extended to include a year-month field,
        the merge should use the full period instead.
    """
    # Load service_supply and normalise month to YYYY-MM string
    supply = pd.read_parquet(supply_path)
    supply["month"] = (
        pd.to_datetime(supply["month"], utc=True).dt.to_period("M").astype(str)
    )
    supply["month_num"] = pd.to_datetime(supply["month"]).dt.month

    supply = supply.rename(columns={"avg_runs_per_active_bin": "load_factor_proxy"})

    supply["service_regularity"] = (
        supply["active_time_bins"] / MAX_BINS_PER_MONTH
    ).clip(0, 1)

    # Aggregate monthly headway reliability
    hw = pd.read_parquet(headway_path)
    hw_monthly = (
        hw.groupby(["stop_id", "month_num"])["headway_reliability"]
        .mean()
        .reset_index()
        .rename(columns={"stop_id": "origin_stop_id"})
    )

    # Join supply and reliability
    merged = supply.merge(
        hw_monthly,
        on=["origin_stop_id", "month_num"],
        how="left",
        validate="many_to_one",
    )

    merged["headway_reliability"] = merged["headway_reliability"].clip(0, 1).fillna(0.0)

    # Normalise load_factor_proxy to [0,1] via 99th-percentile cap
    lf_cap = merged["load_factor_proxy"].quantile(0.99)
    if lf_cap > 0:
        merged["load_factor_norm"] = (merged["load_factor_proxy"] / lf_cap).clip(0, 1)
    else:
        merged["load_factor_norm"] = 0.0

    merged["mode_shift_score"] = (
        MODE_SHIFT_WEIGHTS["load"] * merged["load_factor_norm"]
        + MODE_SHIFT_WEIGHTS["regularity"] * merged["service_regularity"]
        + MODE_SHIFT_WEIGHTS["reliability"] * merged["headway_reliability"]
    ).clip(0, 1)

    merged["priority_tier"] = pd.cut(
        merged["mode_shift_score"],
        bins=[-np.inf, TIER2_THRESHOLD, TIER1_THRESHOLD, np.inf],
        labels=["Tier3_review", "Tier2_monitor", "Tier1_invest"],
    )

    out_cols = [
        "origin_stop_id",
        "dest_stop_id",
        "month",
        "total_runs",
        "active_time_bins",
        "load_factor_proxy",
        "load_factor_norm",
        "service_regularity",
        "headway_reliability",
        "mode_shift_score",
        "priority_tier",
    ]
    result = merged[[c for c in out_cols if c in merged.columns]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False, compression="zstd")

    tier_counts = result["priority_tier"].value_counts().sort_index()
    print(f"Mode-shift scores : {len(result):,} OD-month records")
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count:,}")
    return result

# 2. CO2 savings
def compute_co2_savings(
    supply_path: Path,
    od_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """
    Estimate CO2 savings (kg/month) and cars replaced per OD corridor.

    Uses service_supply for trip counts (consistent with mode-shift function)
    and od_agg for corridor metadata (stop names, distance).

    CO2_saved_kg = total_runs × trip_distance_km × (CAR_EF − BUS_EF)
    cars_replaced = total_runs / AVG_CAR_OCCUPANCY

    Caveat: total_runs counts vehicle-trips.  Passenger-level savings require
    occupancy data unavailable from GPS pings alone.
    """
    supply = pd.read_parquet(supply_path)
    supply["month"] = (
        pd.to_datetime(supply["month"], utc=True).dt.to_period("M").astype(str)
    )

    # Corridor metadata: one representative row per (origin, dest) pair
    od = pd.read_parquet(
        od_path,
        columns=[
            "origin_stop_id",
            "dest_stop_id",
            "origin_name",
            "dest_name",
            "trip_distance_km",
        ],
    )
    corridor_meta = (
        od.groupby(["origin_stop_id", "dest_stop_id"])
        .agg(
            origin_name=("origin_name", "first"),
            dest_name=("dest_name", "first"),
            trip_distance_km=("trip_distance_km", "median"),
        )
        .reset_index()
    )

    monthly = supply.merge(
        corridor_meta,
        on=["origin_stop_id", "dest_stop_id"],
        how="left",
        validate="many_to_one",
    )

    emission_delta = CAR_EMISSION_KG_PER_KM - BUS_EMISSION_KG_PER_KM  # 0.141
    monthly["co2_saved_kg"] = (
        monthly["total_runs"] * monthly["trip_distance_km"].fillna(0) * emission_delta
    ).clip(lower=0)

    monthly["cars_replaced"] = (monthly["total_runs"] / AVG_CAR_OCCUPANCY).round(1)

    out_cols = [
        "origin_stop_id",
        "dest_stop_id",
        "month",
        "origin_name",
        "dest_name",
        "trip_distance_km",
        "total_runs",
        "co2_saved_kg",
        "cars_replaced",
    ]
    result = monthly[[c for c in out_cols if c in monthly.columns]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False, compression="zstd")

    total_co2 = result["co2_saved_kg"].sum()
    total_cars = result["cars_replaced"].sum()
    print(
        f"CO2 savings : {total_co2 / 1_000:.1f} t CO₂  |  "
        f"Cars replaced : {total_cars:,.0f}"
    )
    return result

# 3. Service gaps
def _build_reachability(catalog_path: Path) -> tuple[set, set]:
    """
    Build directly-served and 1-transfer-reachable stop pairs.

    Returns
        directly_served : set of (min_id, max_id) tuples
        transfer_1      : set of (min_id, max_id) tuples (superset of direct)
    """
    cat = pd.read_parquet(catalog_path, columns=["template_id", "stop_sequence"])

    template_stops: dict[int, list[int]] = {}
    for _, row in cat.iterrows():
        template_stops[int(row["template_id"])] = json.loads(row["stop_sequence"])

    # Inverted index: stop_id → set of template_ids it appears in
    stop_to_templates: dict[int, set[int]] = {}
    for tid, stops in template_stops.items():
        for sid in stops:
            stop_to_templates.setdefault(sid, set()).add(tid)

    # Directly served: any two stops appearing on the same template
    directly_served: set[tuple[int, int]] = set()
    for stops in template_stops.values():
        for i in range(len(stops)):
            for j in range(i + 1, len(stops)):
                directly_served.add((min(stops[i], stops[j]), max(stops[i], stops[j])))

    # 1-transfer: stops on template A reachable from template B via a
    # shared intermediate stop.
    template_stop_sets = {
        tid: frozenset(stops) for tid, stops in template_stops.items()
    }
    tids = list(template_stops.keys())
    transfer_1: set[tuple[int, int]] = set(directly_served)

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            tid_a, tid_b = tids[i], tids[j]
            if not (template_stop_sets[tid_a] & template_stop_sets[tid_b]):
                continue
            for sa in template_stops[tid_a]:
                for sb in template_stops[tid_b]:
                    transfer_1.add((min(sa, sb), max(sa, sb)))

    return directly_served, transfer_1


def compute_service_gaps(
    od_path: Path,
    catalog_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """
    Classify each observed OD pair as direct / transfer_1 / unserved.
    """
    directly_served, transfer_1 = _build_reachability(catalog_path)

    od = pd.read_parquet(
        od_path,
        columns=[
            "origin_stop_id",
            "dest_stop_id",
            "trip_count",
            "time_bin_30min",
            "origin_name",
            "dest_name",
            "trip_distance_km",
        ],
    )

    if od.empty:
        print("Warning: od_agg is empty — service gap classification skipped.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(out_path, index=False)
        return pd.DataFrame()

    # Vectorised pair-key construction
    oid = od["origin_stop_id"].astype(int).values
    did = od["dest_stop_id"].astype(int).values
    od["pair_key"] = list(
        zip(np.minimum(oid, did).tolist(), np.maximum(oid, did).tolist())
    )

    od["is_direct"] = od["pair_key"].isin(directly_served)
    od["is_transfer_1"] = od["pair_key"].isin(transfer_1)
    od["service_status"] = np.where(
        od["is_direct"],
        "direct",
        np.where(od["is_transfer_1"], "transfer_1", "unserved"),
    )

    status_agg = (
        od.groupby(
            [
                "origin_stop_id",
                "dest_stop_id",
                "service_status",
                "origin_name",
                "dest_name",
                "trip_distance_km",
            ],
            dropna=False,
        )
        .agg(
            total_trips=("trip_count", "sum"),
            n_time_bins=("time_bin_30min", "nunique"),
        )
        .reset_index()
        .sort_values("total_trips", ascending=False)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_agg.to_parquet(out_path, index=False, compression="zstd")

    print("\nService classification (OD time-bin rows):")
    print(od["service_status"].value_counts().to_string())

    gaps = status_agg[status_agg["service_status"] == "unserved"]
    print(f"\nUnserved OD corridors (DRT candidates) : {len(gaps):,}")
    if not gaps.empty:
        print(
            gaps[["origin_name", "dest_name", "total_trips", "trip_distance_km"]]
            .head(10)
            .to_string(index=False)
        )
    return status_agg

# Entry point
def main() -> None:
    print("1. Mode-shift scores —")
    compute_mode_shift_scores(
        supply_path=SERVICE_SUPPLY,
        headway_path=HEADWAY_STATS,
        out_path=MODE_SHIFT,
    )

    print("\n2. CO2 savings —")
    compute_co2_savings(
        supply_path=SERVICE_SUPPLY,
        od_path=OD_AGG,
        out_path=CO2_SAVINGS,
    )

    print("\n3. Service gaps —")
    compute_service_gaps(
        od_path=OD_AGG,
        catalog_path=ROUTE_CATALOG,
        out_path=SERVICE_GAPS,
    )

    print("\nPolicy outputs written:")
    for p in [MODE_SHIFT, CO2_SAVINGS, SERVICE_GAPS]:
        if p.exists():
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")
        else:
            print(f"  {p.name}  [NOT FOUND — upstream step may have failed]")


if __name__ == "__main__":
    main()
