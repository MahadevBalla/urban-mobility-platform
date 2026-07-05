#!/usr/bin/env python3
"""
dry_run.py — end-to-end smoke test for the Cityflo pipeline on a small
subset of data, without touching the real data/processed, data/interim,
or outputs/ directories.

WHAT THIS DOES
--------------
1. Creates a sandbox folder:  research/cityflo/.dryrun_sandbox/
   (a throwaway copy of just enough data to run the pipeline).
2. Subsamples the three big raw GPS files (systematic row sampling --
   every Nth line, so the sample still spans the full date range and
   many vehicles, not just whatever happens to be first in the file).
3. Copies the small reference files (stops_clean.csv, trips_clean.csv,
   the weather grid, mumbai_wards.kml) as-is -- they're already small,
   no need to subsample them.
4. Runs scripts 01_1 through 15 in dependency order INSIDE the sandbox,
   by setting the CITYFLO_ROOT environment variable for each subprocess
   (see the config.py patch -- this makes every path in config.py
   resolve under the sandbox instead of your real data/outputs).
5. After each stage: checks the process exited cleanly, the expected
   output file(s) exist and are non-empty, expected columns are
   present, and a handful of stage-specific sanity checks (e.g. did
   stop-snapping actually snap anything, are model metrics finite).
6. Prints a PASS/FAIL/SKIP/WARN summary table at the end. Full
   stdout/stderr for every stage is saved under .dryrun_sandbox/logs/
   so a failure can be root-caused without re-running anything.

USAGE
-----
    cd research/cityflo
    python dry_run.py                  # first run: builds sandbox + runs everything
    python dry_run.py --resample       # rebuild the GPS subsample (e.g. different size)
    python dry_run.py --sample-every 500   # denser sample (more rows kept)
    python dry_run.py --purge          # delete the sandbox and exit
    python dry_run.py --skip-heavy     # skip 10, 12, 13, 14 (geopandas/torch/xgboost heavy)
    python dry_run.py --only 02,03     # runs the specified stages only (useful for debugging a single stage)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Locations
PROJECT_ROOT = Path(__file__).resolve().parent  # research/cityflo
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
REAL_DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REAL_DATA_RAW = PROJECT_ROOT / "data" / "raw"

SANDBOX = PROJECT_ROOT / ".dryrun_sandbox"
SANDBOX_RAW = SANDBOX / "data" / "raw"
SANDBOX_PROCESSED = SANDBOX / "data" / "processed"
SANDBOX_INTERIM = SANDBOX / "data" / "interim"
SANDBOX_OUTPUTS = SANDBOX / "outputs"
LOG_DIR = SANDBOX / "logs"

GPS_FILES = [
    "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756",
    "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756_part2",
    "before_2022-10-22_698096e5f4994518a37a0b9c59bb9756_part3",
]
WEATHER_SRC = REAL_DATA_RAW / "WeatherData" / "mumbai_openmeteo_10km_grid_data"
WARDS_KML = "mumbai_wards.kml"

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


# Preflight
def preflight():
    problems = []
    if not (SCRIPTS_DIR / "config.py").exists():
        problems.append(
            f"Can't find scripts/config.py under {PROJECT_ROOT}. "
            "Run this from research/cityflo, and make sure dry_run.py "
            "sits next to scripts/, data/, outputs/."
        )
    for f in ["stops_clean.csv", "trips_clean.csv"]:
        p = REAL_DATA_PROCESSED / f
        if not p.exists() or p.stat().st_size == 0:
            problems.append(
                f"Missing or empty: data/processed/{f}. "
                "These come from the two EDA notebooks (01_reference_data_audit.ipynb, "
                "02_gps_data_audit.ipynb) per the README -- run those first."
            )
    for f in GPS_FILES:
        p = REAL_DATA_RAW / f
        if not p.exists():
            problems.append(f"Missing raw GPS file: data/raw/{f}")
    if not WEATHER_SRC.exists():
        problems.append(
            f"Missing weather folder: {WEATHER_SRC}. "
            "Check the config.py patch matches your actual folder name."
        )
    if not (REAL_DATA_RAW / WARDS_KML).exists():
        problems.append(f"Missing data/raw/{WARDS_KML} (needed by stage 10).")
    with open(SCRIPTS_DIR / "config.py") as fh:
        cfg_src = fh.read()
    if "CITYFLO_ROOT" not in cfg_src:
        problems.append(
            "scripts/config.py does not contain the CITYFLO_ROOT override yet -- "
            "apply the config.py patch before running this."
        )
    if problems:
        print("Preflight checks failed:\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Preflight checks passed.\n")


# Sandbox construction
def build_sandbox(sample_every: int, resample: bool):
    for d in [
        SANDBOX_RAW,
        SANDBOX_PROCESSED,
        SANDBOX_INTERIM,
        LOG_DIR,
        SANDBOX_OUTPUTS / "figures",
        SANDBOX_OUTPUTS / "models",
        SANDBOX_OUTPUTS / "tables",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # subsample the raw GPS files
    for fname in GPS_FILES:
        src = REAL_DATA_RAW / fname
        dst = SANDBOX_RAW / fname
        if dst.exists() and not resample:
            print(f"  [skip] {fname} already sampled (use --resample to redo)")
            continue
        print(
            f"  Sampling every {sample_every}th line of {fname} "
            f"({src.stat().st_size / 1e9:.2f} GB source) ..."
        )
        t0 = time.time()
        with open(dst, "w") as out:
            subprocess.run(
                ["awk", f"NR % {sample_every} == 0"],
                stdin=open(src, "r", errors="ignore"),
                stdout=out,
                check=True,
            )
        n_lines = sum(1 for _ in open(dst))
        print(f"    -> {n_lines:,} rows kept in {time.time() - t0:.0f}s")
        if n_lines == 0:
            print(f"    WARNING: 0 rows sampled from {fname} -- lower --sample-every.")

    # copy small reference files
    for f in ["stops_clean.csv", "trips_clean.csv"]:
        shutil.copy2(REAL_DATA_PROCESSED / f, SANDBOX_PROCESSED / f)

    if not (SANDBOX_RAW / WARDS_KML).exists():
        shutil.copy2(REAL_DATA_RAW / WARDS_KML, SANDBOX_RAW / WARDS_KML)

    sandbox_weather = SANDBOX_RAW / "WeatherData" / "mumbai_openmeteo_10km_grid_data"
    if not sandbox_weather.exists():
        print("  Copying weather grid (small, one-time) ...")
        shutil.copytree(WEATHER_SRC, sandbox_weather)

    print("\nSandbox ready at:", SANDBOX)


def purge_sandbox():
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
        print(f"Removed {SANDBOX}")
    else:
        print("No sandbox to remove.")


# Running a stage + safety assertion
def sandbox_env() -> dict:
    env = os.environ.copy()
    env["CITYFLO_ROOT"] = str(SANDBOX)
    return env


def assert_sandbox_wired(env: dict):
    """Re-import config.py in a fresh subprocess and confirm every
    output-bearing path resolves under the sandbox, not the real project.
    Aborts the whole run if this ever fails."""
    code = (
        "import sys; sys.path.insert(0, r'{scripts}');"
        "import config as c;"
        "paths = [c.DATA_PROCESSED, c.DATA_INTERIM, c.OUTPUTS, c.MODELS_DIR, c.TABLES_DIR, c.FIGURES];"
        "root = r'{sandbox}';"
        "bad = [str(p) for p in paths if not str(p).startswith(root)];"
        "print('BAD:' + ','.join(bad) if bad else 'OK')"
    ).format(scripts=str(SCRIPTS_DIR), sandbox=str(SANDBOX))
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    out = result.stdout.strip()
    if out != "OK":
        print("SAFETY ABORT: config.py did not resolve under the sandbox path.")
        print("  stdout:", out)
        print("  stderr:", result.stderr)
        print("Refusing to continue -- this would risk writing into real data.")
        sys.exit(1)


def run_stage(script_name: str, args: list[str], log_name: str) -> tuple[bool, float]:
    env = sandbox_env()
    assert_sandbox_wired(env)
    log_path = LOG_DIR / f"{log_name}.log"
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    t0 = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(
            cmd, env=env, cwd=str(SANDBOX), stdout=logf, stderr=subprocess.STDOUT
        )
    dt = time.time() - t0
    return result.returncode == 0, dt


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# Output validation helpers
def parquet_rowcount(path: Path) -> int:
    import polars as pl

    return pl.scan_parquet(str(path)).select(pl.len()).collect().item()


def parquet_columns(path: Path) -> set[str]:
    import polars as pl

    return set(pl.scan_parquet(str(path)).collect_schema().names())


def check_file_exists(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path.name}"
    if path.stat().st_size == 0:
        return False, f"empty file: {path.name}"
    return True, ""


def check_parquet(
    path: Path, required_cols: list[str] | None = None, min_rows: int = 1
) -> tuple[str, str]:
    ok, msg = check_file_exists(path)
    if not ok:
        return FAIL, msg
    try:
        n = parquet_rowcount(path)
    except Exception as e:
        return FAIL, f"could not read {path.name}: {e}"
    if n < min_rows:
        return (
            WARN,
            f"{path.name}: {n} rows (expected >= {min_rows}) -- likely too small a sample, not necessarily a bug",
        )
    if required_cols:
        cols = parquet_columns(path)
        missing = set(required_cols) - cols
        if missing:
            return FAIL, f"{path.name}: missing expected columns {sorted(missing)}"
    return PASS, f"{path.name}: {n:,} rows"


# Stage definitions and stage-specific sanity checks
def check_01_1():
    files = sorted(SANDBOX_PROCESSED.glob("before_2022-10-22*_bucket0.parquet"))
    if not files:
        return FAIL, "no *_bucket0.parquet outputs found"
    import polars as pl

    total = sum(
        pl.scan_parquet(str(f)).select(pl.len()).collect().item() for f in files
    )
    if total == 0:
        return (
            WARN,
            "0 rows total after ingestion filters -- sample may be too small, or filters may be too strict",
        )
    return PASS, f"{len(files)} bucket file(s), {total:,} rows total"


def check_04_snap_rate():
    path = SANDBOX_PROCESSED / "pings_snapped.parquet"
    status, msg = check_parquet(path, ["snapped_stop_id", "snap_distance_m"])
    if status != PASS:
        return status, msg
    import polars as pl

    df = (
        pl.scan_parquet(str(path))
        .select((pl.col("snapped_stop_id") != -1).mean().alias("snap_rate"))
        .collect()
    )
    rate = df["snap_rate"][0]
    if rate == 0:
        return (
            FAIL,
            f"{msg} -- but 0% of pings snapped to any stop (was PASS on file, FAIL on content)",
        )
    if rate < 0.05:
        return (
            WARN,
            f"{msg} -- only {rate:.1%} snapped, check SNAP_THRESHOLD_M / stop coverage in this sample",
        )
    return PASS, f"{msg}, {rate:.1%} snapped"


def check_05_match_rate():
    path = SANDBOX_PROCESSED / "segments_inferred.parquet"
    status, msg = check_parquet(path, ["template_id", "match_confidence"])
    if status != PASS:
        return status, msg
    import polars as pl

    df = (
        pl.scan_parquet(str(path))
        .select(pl.col("template_id").is_not_null().mean().alias("match_rate"))
        .collect()
    )
    rate = df["match_rate"][0]
    if rate == 0:
        return (
            WARN,
            f"{msg} -- 0% of segments matched a route template (expected on very small/sparse samples)",
        )
    return PASS, f"{msg}, {rate:.1%} matched to a template"


def check_07_reliability():
    hw, sv = (
        SANDBOX_PROCESSED / "headway_stats.parquet",
        SANDBOX_PROCESSED / "stop_visits.parquet",
    )
    s1, m1 = check_parquet(sv, ["stop_id", "headway_min"], min_rows=1)
    s2, m2 = check_parquet(
        hw, min_rows=0
    )  # headway_stats needs >=10 obs/stratum; 0 rows is plausible on a small sample
    if s2 == FAIL:
        return FAIL, f"stop_visits: {m1} | headway_stats: {m2}"
    if s1 != PASS:
        return s1, m1
    note = (
        " (0 rows is expected on a small sample -- the HAVING COUNT(*)>=10 filter is strict)"
        if "0 rows" in m2
        else ""
    )
    return PASS, f"{m1} | headway_stats: {m2}{note}"


def check_metrics_csv(path: Path, model_name: str) -> tuple[str, str]:
    ok, msg = check_file_exists(path)
    if not ok:
        return FAIL, msg
    import csv
    import math

    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return FAIL, f"{path.name}: no rows"
    for row in rows:
        for k in ["MAE", "RMSE", "sMAPE"]:
            if k in row:
                try:
                    v = float(row[k])
                except ValueError:
                    return FAIL, f"{path.name}: {k}='{row[k]}' is not numeric"
                if math.isnan(v) or math.isinf(v):
                    return (
                        FAIL,
                        f"{path.name}: {k} is {row[k]} on split '{row.get('split')}'",
                    )
    return PASS, f"{model_name}: metrics finite across {len(rows)} split(s)"


def check_15_policy():
    import polars as pl

    msgs = []
    for fname, checks in [
        ("mode_shift_scores.parquet", [("mode_shift_score", 0.0, 1.0)]),
        ("co2_savings.parquet", [("co2_saved_kg", 0.0, None)]),
    ]:
        path = SANDBOX_PROCESSED / fname
        status, msg = check_parquet(path, min_rows=0)
        if status == FAIL:
            return FAIL, msg
        if status == PASS:
            df = pl.scan_parquet(str(path)).collect()
            for col, lo, hi in checks:
                if col in df.columns and len(df) > 0:
                    cmin, cmax = df[col].min(), df[col].max()
                    if lo is not None and cmin < lo - 1e-6:
                        return FAIL, f"{fname}: {col} min {cmin} is below {lo}"
                    if hi is not None and cmax > hi + 1e-6:
                        return FAIL, f"{fname}: {col} max {cmax} is above {hi}"
        msgs.append(f"{fname}: {msg}")
    return PASS, " | ".join(msgs)


STAGES = [
    {
        "id": "01_1",
        "desc": "GPS ingestion & quality filtering",
        "script": "01_1_ingest_legacy.py",
        "args": ["--bucket_id", "0", "--bucket_count", "1"],
        "check": check_01_1,
    },
    {
        "id": "01_2",
        "desc": "Dedup + GPS jump filter",
        "script": "01_2_finalize_pings.py",
        "args": ["--bucket_id", "0"],
        "check": lambda: check_parquet(SANDBOX_PROCESSED / "pings_clean_bucket0.parquet"),
    },
    {
        "id": "01_3",
        "desc": "Merge buckets",
        "script": "01_3_merge_buckets.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "pings_clean.parquet",
            ["vehicle_id", "lat", "lng", "timestamp_ist", "ride_date"],
        ),
    },
    {
        "id": "02",
        "desc": "Route catalog construction",
        "script": "02_route_catalog.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "route_catalog.parquet",
            ["template_id", "stop_sequence", "n_stops"],
        ),
    },
    {
        "id": "03",
        "desc": "Trip segmentation",
        "script": "03_trip_segmentation.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "pings_segmented.parquet", ["segment_id"]
        ),
    },
    {
        "id": "04",
        "desc": "Stop snapping",
        "script": "04_stop_snapping.py",
        "args": [],
        "check": check_04_snap_rate,
    },
    {
        "id": "05",
        "desc": "Route inference",
        "script": "05_route_inference.py",
        "args": ["--run_validation", "--validation_sample_n", "200"],
        "check": check_05_match_rate,
    },
    {
        "id": "06",
        "desc": "OD matrix construction",
        "script": "06_od_matrix.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "od_agg.parquet", ["trip_count", "trip_distance_km"]
        ),
    },
    {
        "id": "07",
        "desc": "Reliability metrics",
        "script": "07_reliability.py",
        "args": [],
        "check": check_07_reliability,
    },
    {
        "id": "08",
        "desc": "Weather consolidation",
        "script": "08_weather_consolidate.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "weather_stop_hourly.parquet", ["time", "stop_id"]
        ),
    },
    {
        "id": "09",
        "desc": "Feature engineering",
        "script": "09_feature_engineering.py",
        "args": [],
        "check": lambda: check_parquet(
            SANDBOX_PROCESSED / "features_master.parquet",
            ["trip_count", "hour_sin", "origin_h3"],
        ),
    },
    {
        "id": "10",
        "desc": "Ward aggregation",
        "requires": ["geopandas", "fiona"],
        "script": "10_ward_aggregation.py",
        "args": [],
        "check": lambda: check_parquet(SANDBOX_PROCESSED / "ward_od.parquet", min_rows=0),
    },
    {
        "id": "11",
        "desc": "Negative binomial model",
        "requires": ["statsmodels", "scipy"],
        "script": "11_model_nb.py",
        "args": [],
        "check": lambda: check_metrics_csv(
            SANDBOX_OUTPUTS / "tables" / "nb_metrics.csv", "NB"
        ),
    },
    {
        "id": "12",
        "desc": "XGBoost model",
        "requires": ["xgboost", "shap", "joblib"],
        "script": "12_model_xgboost.py",
        "args": [],
        "check": lambda: check_metrics_csv(
            SANDBOX_OUTPUTS / "tables" / "xgb_metrics.csv", "XGBoost"
        ),
    },
    {
        "id": "13",
        "desc": "ST-GNN model",
        "requires": ["torch", "h3"],
        "script": "13_model_stgnn.py",
        "args": [],
        "check": lambda: check_metrics_csv(
            SANDBOX_OUTPUTS / "tables" / "stgnn_metrics.csv", "ST-GNN"
        ),
    },
    {
        "id": "14",
        "desc": "Analysis / reporting",
        "requires": ["geopandas", "seaborn", "h3"],
        "script": "14_analysis_reporting.py",
        "args": [],
        "check": lambda: check_file_exists(
            SANDBOX_OUTPUTS / "tables" / "network_summary.json"
        ),
    },
    {
        "id": "15",
        "desc": "Policy outputs",
        "script": "15_policy_outputs.py",
        "args": [],
        "check": check_15_policy,
    },
]


# Main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sample-every",
        type=int,
        default=2000,
        help="keep 1 in every N raw GPS rows (default 2000; lower = more data)",
    )
    ap.add_argument(
        "--resample",
        action="store_true",
        help="rebuild the GPS subsample even if it already exists",
    )
    ap.add_argument("--purge", action="store_true", help="delete the sandbox and exit")
    ap.add_argument(
        "--skip-heavy",
        action="store_true",
        help="skip stages 10, 12, 13, 14 (heavier optional dependencies)",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated stage ids to run, e.g. 01_1,01_2,01_3,02",
    )
    args = ap.parse_args()

    if args.purge:
        purge_sandbox()
        return

    preflight()
    build_sandbox(args.sample_every, args.resample)

    only = set(args.only.split(",")) if args.only else None
    skip_heavy_ids = {"10", "12", "13", "14"}

    results = []
    for stage in STAGES:
        sid = stage["id"]
        if only and sid not in only:
            continue
        if args.skip_heavy and sid in skip_heavy_ids:
            results.append((sid, stage["desc"], SKIP, "skipped via --skip-heavy", 0.0))
            continue
        missing = [m for m in stage.get("requires", []) if not has_module(m)]
        if missing:
            results.append(
                (
                    sid,
                    stage["desc"],
                    SKIP,
                    f"missing optional dependency: {', '.join(missing)}",
                    0.0,
                )
            )
            continue

        print(f"[{sid}] {stage['desc']} ...", flush=True)
        ok, dt = run_stage(stage["script"], stage["args"], log_name=sid)
        if not ok:
            results.append(
                (sid, stage["desc"], FAIL, f"non-zero exit -- see logs/{sid}.log", dt)
            )
            print(f"  -> FAIL ({dt:.0f}s) -- see .dryrun_sandbox/logs/{sid}.log")
            print("  Stopping: downstream stages depend on this one's output.")
            break
        status, msg = stage["check"]()
        results.append((sid, stage["desc"], status, msg, dt))
        print(f"  -> {status} ({dt:.0f}s) {msg}")

    print("\n" + "=" * 78)
    print(f"{'Stage':<6} {'Status':<6} {'Time':>7}  Description / detail")
    print("-" * 78)
    for sid, desc, status, msg, dt in results:
        print(f"{sid:<6} {status:<6} {dt:>6.0f}s  {desc}")
        print(f"       {'':<6} {'':>7}  -> {msg}")
    print("=" * 78)

    n_fail = sum(1 for r in results if r[2] == FAIL)
    n_warn = sum(1 for r in results if r[2] == WARN)
    n_skip = sum(1 for r in results if r[2] == SKIP)
    print(
        f"\n{len(results) - n_fail - n_warn - n_skip} passed, "
        f"{n_warn} warned, {n_fail} failed, {n_skip} skipped."
    )
    print(f"Full logs: {LOG_DIR}")
    print(f"Sandbox outputs: {SANDBOX_OUTPUTS}  (real outputs/ untouched)")

    with open(SANDBOX / "summary.json", "w") as fh:
        json.dump(
            [
                {"stage": s, "desc": d, "status": st, "detail": m, "seconds": t}
                for s, d, st, m, t in results
            ],
            fh,
            indent=2,
        )

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
