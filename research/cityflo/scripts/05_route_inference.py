"""
05_route_inference.py

Matches each GPS trip segment to a scheduled route template.

Input:
    pings_snapped.parquet
    route_catalog.parquet
    trips_clean.csv

Output:
    segments_inferred.parquet
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PINGS_SNAPPED,
    ROUTE_CATALOG,
    ROUTE_HIGH_CONFIDENCE,
    ROUTE_MIN_CONFIDENCE,
    ROUTE_MIN_OBS_STOPS,
    ROUTE_TRIP_WINDOW_MIN,
    SEGMENTS_INFERRED,
    TRIPS_FILE,
)

# Tunable parameters
DEFAULT_MIN_SHARED_STOPS = 2
DEFAULT_TOP_N_CANDIDATES = 30
DEFAULT_TRIP_ASSIGN_MIN_CONF = 0.60
DEFAULT_TRIP_ASSIGN_MIN_OVERLAP = 0.60
DEFAULT_VALIDATION_RANDOM_SEED = 42

# Parsing helpers
_ROUTE_PAIR_RE = re.compile(r"(\\d+)\\s*,\\s*(\\d{6}|\\d{2}:\\d{2}:\\d{2})")

def parse_trip_route(route_str: str) -> List[Tuple[int, str]]:
    if pd.isna(route_str):
        return []

    s = str(route_str).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            out = []
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    sid = int(item[0])
                    t = normalize_time_str(item[1])
                    out.append((sid, t))
            return out
        except Exception:
            pass

    matches = _ROUTE_PAIR_RE.findall(s)
    out = []
    for sid_str, t_str in matches:
        out.append((int(sid_str), normalize_time_str(t_str)))
    return out


def normalize_time_str(t: str) -> str:
    t = str(t).strip()
    if ":" in t:
        parts = t.split(":")
        if len(parts) == 3:
            hh, mm, ss = parts
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
    if len(t) == 6 and t.isdigit():
        return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    raise ValueError(f"Unrecognized time format: {t}")


def time_str_to_minutes(t: str) -> float:
    hh, mm, ss = map(int, normalize_time_str(t).split(":"))
    return hh * 60.0 + mm + ss / 60.0


def parse_ts_to_minutes(ts: str) -> float:
    s = str(ts)
    if "T" in s:
        time_part = s.split("T")[1]
    else:
        time_part = s.split(" ")[1] if " " in s else "00:00:00"
    time_part = time_part[:8]
    hh, mm, ss = map(int, time_part.split(":"))
    return hh * 60.0 + mm + ss / 60.0


def ts_to_date_str(ts: str) -> str:
    s = str(ts)
    if "T" in s:
        return s.split("T")[0]
    if " " in s:
        return s.split(" ")[0]
    return s[:10]


# Sequence helpers
def dedup_consecutive(seq: List[int]) -> List[int]:
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def unique_in_order(seq: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def lcs_length(a: List[int], b: List[int]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def matched_template_positions(
    observed: List[int], tmpl_pos_map: Dict[int, int]
) -> List[int]:
    return [tmpl_pos_map[s] for s in observed if s in tmpl_pos_map]


def monotonicity_ratio(observed: List[int], tmpl_pos_map: Dict[int, int]) -> float:
    """
    More robust than forward_steps / meaningful_steps.

    Score = length of longest non-decreasing subsequence of matched template positions
            divided by number of matched positions.

    Examples:
    - [0,1,2,3] -> 1.0
    - [0,1,0,1,0,1] -> 4/6 or lower depending on subsequence, not falsely near-perfect
    - too few matches -> 0.5
    """
    pos = matched_template_positions(observed, tmpl_pos_map)
    if len(pos) < 2:
        return 0.5

    tails = []
    for x in pos:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x

    lnds_len = len(tails)
    return float(lnds_len / len(pos))


def endpoint_proximity_score(
    observed: List[int], tmpl_stops: List[int], tmpl_pos_map: Dict[int, int]
) -> float:
    """
    Diagnostic/tie-breaker only.

    Rewards observed endpoints appearing near the template ends,
    instead of giving full credit merely because both exist.
    """
    if not observed or not tmpl_stops:
        return 0.0

    n = len(tmpl_stops)
    if n == 0:
        return 0.0

    scores = []

    first_obs = observed[0]
    if first_obs in tmpl_pos_map:
        p = tmpl_pos_map[first_obs]
        start_score = 1.0 - (p / max(n - 1, 1))
        scores.append(start_score)

    last_obs = observed[-1]
    if last_obs in tmpl_pos_map:
        p = tmpl_pos_map[last_obs]
        end_score = 1.0 - ((n - 1 - p) / max(n - 1, 1))
        scores.append(end_score)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def score_margin_from_rank_keys(best_key: Tuple, second_key: Tuple) -> float:
    """
    Simple scalar ambiguity summary derived from core lexicographic dimensions.
    Uses first three main components only, all in [0,1].
    """
    if second_key is None:
        return 1.0
    return round(
        float(
            (best_key[0] - second_key[0])
            + (best_key[1] - second_key[1])
            + (best_key[2] - second_key[2])
        ),
        4,
    )


# Template loading and indexing
def load_templates(catalog_path: Path) -> Dict[int, dict]:
    cat = pd.read_parquet(catalog_path)
    templates: Dict[int, dict] = {}

    for _, row in cat.iterrows():
        tid = int(row["template_id"])
        stops = [int(x) for x in json.loads(row["stop_sequence"])]
        stop_set = frozenset(stops)
        pos_map = {sid: i for i, sid in enumerate(stops)}

        templates[tid] = {
            "template_id": tid,
            "stops": stops,
            "stop_set": stop_set,
            "pos_map": pos_map,
            "n_stops": len(stops),
            "first": stops[0] if stops else None,
            "last": stops[-1] if stops else None,
            "n_trips": int(row.get("n_trips_in_catalog", 0)),
            "sched": json.loads(row["median_schedule_json"])
            if pd.notna(row.get("median_schedule_json"))
            else [],
        }

    return templates


def build_stop_to_templates_index(templates: Dict[int, dict]) -> Dict[int, set]:
    idx = defaultdict(set)
    for tid, tmpl in templates.items():
        for sid in tmpl["stop_set"]:
            idx[sid].add(tid)
    return idx


# Candidate generation and scoring
def generate_candidate_templates(
    observed: List[int],
    stop_to_templates: Dict[int, set],
    min_shared_stops: int = DEFAULT_MIN_SHARED_STOPS,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Candidate generation with overlap-count pruning.
    """
    obs_unique = unique_in_order(observed)
    if not obs_unique:
        return [], {}

    counts = defaultdict(int)
    for sid in obs_unique:
        for tid in stop_to_templates.get(sid, ()):
            counts[tid] += 1

    if not counts:
        return [], {}

    kept = [(tid, cnt) for tid, cnt in counts.items() if cnt >= min_shared_stops]
    if not kept:
        kept = list(counts.items())

    kept.sort(key=lambda x: (-x[1], x[0]))
    kept = kept[:top_n]
    return [tid for tid, _ in kept], dict(kept)


def compute_match_score(observed: List[int], tmpl: dict) -> dict:
    """
    Core thesis-defensible metrics.
    Ranking uses lexicographic order, not arbitrary averaging.
    """
    obs_dedup = dedup_consecutive(observed)
    obs_unique = unique_in_order(obs_dedup)

    obs_set = set(obs_unique)
    tpl_set = set(tmpl["stop_set"])

    n_match = len(obs_set & tpl_set)
    overlap_score = n_match / len(obs_set) if obs_set else 0.0
    coverage_score = n_match / len(tpl_set) if tpl_set else 0.0
    jaccard = n_match / len(obs_set | tpl_set) if (obs_set | tpl_set) else 0.0

    lcs_len = lcs_length(obs_unique, tmpl["stops"])
    order_score_val = lcs_len / len(obs_unique) if obs_unique else 0.0

    dir_score = monotonicity_ratio(obs_dedup, tmpl["pos_map"])
    end_score = endpoint_proximity_score(obs_dedup, tmpl["stops"], tmpl["pos_map"])

    is_full_subseq = bool(obs_unique) and lcs_len == len(obs_unique)
    template_len_penalty = abs(len(tmpl["stops"]) - len(obs_unique)) / max(
        len(tmpl["stops"]), 1
    )

    rank_key = (
        round(float(overlap_score), 6),
        round(float(order_score_val), 6),
        round(float(dir_score), 6),
        round(float(end_score), 6),
        round(float(coverage_score), 6),
        -round(float(template_len_penalty), 6),
    )

    confidence = min(
        overlap_score,
        order_score_val,
        max(0.0, dir_score),
    )

    return {
        "n_matched": int(n_match),
        "jaccard": round(float(jaccard), 4),
        "overlap_score": round(float(overlap_score), 4),
        "coverage_score": round(float(coverage_score), 4),
        "order_score": round(float(order_score_val), 4),
        "direction_score": round(float(dir_score), 4),
        "endpoint_score": round(float(end_score), 4),
        "lcs_len": int(lcs_len),
        "is_subseq": bool(is_full_subseq),
        "template_len_penalty": round(float(template_len_penalty), 4),
        "rank_key": rank_key,
        "confidence": round(float(confidence), 4),
    }


# Trip indexing
def assign_trip_templates_by_inference(
    trips_df: pd.DataFrame,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    min_shared_stops: int = DEFAULT_MIN_SHARED_STOPS,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
    min_confidence: float = DEFAULT_TRIP_ASSIGN_MIN_CONF,
    min_overlap: float = DEFAULT_TRIP_ASSIGN_MIN_OVERLAP,
) -> pd.DataFrame:
    """
    Robustly attach template_id to trips_clean rows.
    This replaces dangerous exact tuple(route) -> template assumptions.
    """
    out_rows = []

    for _, row in trips_df.iterrows():
        parsed = row["parsed"]
        stop_seq = [sid for sid, _ in parsed]
        if not stop_seq:
            out_rows.append(
                {
                    **row.to_dict(),
                    "template_id": pd.NA,
                    "template_assign_confidence": 0.0,
                    "template_assign_overlap": 0.0,
                    "template_assign_method": "no_route",
                }
            )
            continue

        candidate_ids, raw_counts = generate_candidate_templates(
            stop_seq,
            stop_to_templates,
            min_shared_stops=min_shared_stops,
            top_n=top_n,
        )

        if not candidate_ids:
            out_rows.append(
                {
                    **row.to_dict(),
                    "template_id": pd.NA,
                    "template_assign_confidence": 0.0,
                    "template_assign_overlap": 0.0,
                    "template_assign_method": "no_candidates",
                }
            )
            continue

        scored = []
        for tid in candidate_ids:
            sc = compute_match_score(stop_seq, templates[tid])
            scored.append((tid, sc, raw_counts.get(tid, 0)))

        scored.sort(key=lambda x: (x[1]["rank_key"], x[2]), reverse=True)
        best_tid, best_sc, _ = scored[0]

        if (
            best_sc["confidence"] >= min_confidence
            and best_sc["overlap_score"] >= min_overlap
        ):
            method = "inferred_template_match"
            template_id = int(best_tid)
        else:
            method = "unassigned_low_conf"
            template_id = pd.NA

        out_rows.append(
            {
                **row.to_dict(),
                "template_id": template_id,
                "template_assign_confidence": float(best_sc["confidence"]),
                "template_assign_overlap": float(best_sc["overlap_score"]),
                "template_assign_method": method,
            }
        )

    out = pd.DataFrame(out_rows)
    return out


def build_trip_template_index(
    trips_path: Path,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    min_shared_stops: int = DEFAULT_MIN_SHARED_STOPS,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], List[dict]]]:
    trips_df = pd.read_csv(trips_path)
    trips_df = trips_df.rename(
        columns={
            "tripid": "trip_id",
            "tripdate": "trip_date",
            "triproute": "trip_route",
        }
    )
    trips_df["trip_date"] = trips_df["trip_date"].astype(str).str[:10]
    trips_df["parsed"] = trips_df["trip_route"].apply(parse_trip_route)

    trips_df["sched_start_min"] = trips_df["parsed"].apply(
        lambda route: time_str_to_minutes(route[0][1]) if route else np.nan
    )

    trips_df = assign_trip_templates_by_inference(
        trips_df=trips_df,
        templates=templates,
        stop_to_templates=stop_to_templates,
        min_shared_stops=min_shared_stops,
        top_n=top_n,
    )

    trips_df = trips_df[trips_df["template_id"].notna()].copy()
    trips_df["template_id"] = trips_df["template_id"].astype(int)

    trip_idx: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for _, row in trips_df.iterrows():
        key = (row["trip_date"], int(row["template_id"]))
        trip_idx[key].append(
            {
                "trip_id": int(row["trip_id"]),
                "sched_start_min": float(row["sched_start_min"]),
                "template_assign_confidence": float(row["template_assign_confidence"]),
            }
        )

    for key in trip_idx:
        trip_idx[key] = sorted(trip_idx[key], key=lambda x: x["sched_start_min"])

    return trips_df, trip_idx


def infer_trip_id(
    template_id: int,
    seg_start_date: str,
    seg_end_date: str,
    seg_start_min: float,
    trip_index: Dict[Tuple[str, int], List[dict]],
    window_min: int,
) -> Tuple[Optional[int], Optional[float]]:
    candidate_dates = [seg_start_date]
    if seg_end_date != seg_start_date:
        candidate_dates.append(seg_end_date)

    best_trip_id = None
    best_diff = float("inf")

    for dt in candidate_dates:
        for trip in trip_index.get((dt, template_id), []):
            diff = abs(seg_start_min - trip["sched_start_min"])
            diff = min(
                diff,
                abs((seg_start_min + 1440) - trip["sched_start_min"]),
                abs(seg_start_min - (trip["sched_start_min"] + 1440)),
            )
            if diff < best_diff:
                best_diff = diff
                best_trip_id = trip["trip_id"]

    if best_trip_id is None or best_diff > window_min:
        return None, None
    return int(best_trip_id), round(float(best_diff), 2)


# Segment extraction
def _column_exists(parquet_path: Path, col: str) -> bool:
    con = duckdb.connect()
    try:
        cols = (
            con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')")
            .df()["column_name"]
            .tolist()
        )
        return col in cols
    finally:
        con.close()


def extract_segment_sequences(snapped_path: Path, min_obs_stops: int) -> pd.DataFrame:
    has_snap_dist = _column_exists(snapped_path, "snap_distance_m")

    con = duckdb.connect()
    try:
        select_snap = ", snap_distance_m" if has_snap_dist else ""
        agg_snap = (
            ", AVG(snap_distance_m) AS avg_snap_distance_m"
            if has_snap_dist
            else ", NULL::DOUBLE AS avg_snap_distance_m"
        )

        con.execute(f"""
            CREATE TABLE snapped AS
            SELECT
                vehicle_id,
                segment_id,
                ride_date,
                snapped_stop_id,
                timestamp_ist
                {select_snap}
            FROM read_parquet('{snapped_path}')
            WHERE snapped_stop_id != -1
              AND segment_id IS NOT NULL
        """)

        segs = con.execute(f"""
            SELECT
                vehicle_id,
                segment_id,
                MIN(timestamp_ist)::VARCHAR AS seg_start_ist,
                MAX(timestamp_ist)::VARCHAR AS seg_end_ist,
                LIST(snapped_stop_id ORDER BY timestamp_ist) AS stop_seq_raw,
                COUNT(*) AS n_obs_pings,
                COUNT(DISTINCT snapped_stop_id) AS n_obs_stops_unique
                {agg_snap}
            FROM snapped
            GROUP BY vehicle_id, segment_id
            HAVING COUNT(DISTINCT snapped_stop_id) >= {min_obs_stops}
        """).df()
    finally:
        con.close()

    segs["stop_seq"] = segs["stop_seq_raw"].apply(
        lambda x: dedup_consecutive([int(v) for v in x])
    )
    segs["n_obs_stops"] = segs["stop_seq"].apply(len)
    segs = segs[segs["n_obs_stops"] >= min_obs_stops].copy()

    segs["seg_start_min"] = segs["seg_start_ist"].apply(parse_ts_to_minutes)
    segs["seg_start_date"] = segs["seg_start_ist"].apply(ts_to_date_str)
    segs["seg_end_date"] = segs["seg_end_ist"].apply(ts_to_date_str)

    return segs.reset_index(drop=True)


# Validation
def simulate_partial_observation(
    stop_seq: List[int],
    keep_fraction: float = 0.6,
    min_keep: int = 3,
    rng: Optional[random.Random] = None,
) -> List[int]:
    if rng is None:
        rng = random.Random(DEFAULT_VALIDATION_RANDOM_SEED)

    seq = dedup_consecutive(stop_seq)
    if len(seq) <= min_keep:
        return seq[:]

    idxs = list(range(len(seq)))
    keep_n = max(min_keep, int(math.ceil(len(seq) * keep_fraction)))
    chosen = sorted(rng.sample(idxs, min(keep_n, len(idxs))))
    sampled = [seq[i] for i in chosen]
    return dedup_consecutive(sampled)


def validate_route_recovery(
    trips_df: pd.DataFrame,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    sample_n: int = 2000,
    keep_fraction: float = 0.6,
    min_obs_stops: int = 3,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
    random_seed: int = DEFAULT_VALIDATION_RANDOM_SEED,
) -> pd.DataFrame:
    rng = random.Random(random_seed)

    valid = trips_df[trips_df["template_id"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    if len(valid) > sample_n:
        valid = valid.sample(sample_n, random_state=random_seed)

    rows = []
    for _, row in valid.iterrows():
        full_seq = [sid for sid, _ in row["parsed"]]
        obs = simulate_partial_observation(
            full_seq, keep_fraction=keep_fraction, min_keep=min_obs_stops, rng=rng
        )
        candidate_ids, raw_counts = generate_candidate_templates(
            obs,
            stop_to_templates,
            min_shared_stops=DEFAULT_MIN_SHARED_STOPS,
            top_n=top_n,
        )

        if not candidate_ids:
            rows.append(
                {
                    "trip_id": row["trip_id"],
                    "true_template_id": int(row["template_id"]),
                    "pred_template_id": pd.NA,
                    "top1_correct": False,
                    "candidate_count": 0,
                    "obs_len": len(obs),
                }
            )
            continue

        scored = []
        for tid in candidate_ids:
            sc = compute_match_score(obs, templates[tid])
            scored.append((tid, sc, raw_counts.get(tid, 0)))

        scored.sort(key=lambda x: (x[1]["rank_key"], x[2]), reverse=True)
        pred_tid = int(scored[0][0])

        rows.append(
            {
                "trip_id": row["trip_id"],
                "true_template_id": int(row["template_id"]),
                "pred_template_id": pred_tid,
                "top1_correct": pred_tid == int(row["template_id"]),
                "candidate_count": len(candidate_ids),
                "obs_len": len(obs),
            }
        )

    res = pd.DataFrame(rows)
    return res


# Main inference
def run_inference(
    snapped_path: Path,
    catalog_path: Path,
    trips_path: Path,
    out_path: Path,
    min_obs_stops: int,
    min_confidence: float,
    trip_window_min: int,
    top_n_candidates: int = DEFAULT_TOP_N_CANDIDATES,
) -> pd.DataFrame:
    print("Loading route templates ...")
    templates = load_templates(catalog_path)
    stop_to_templates = build_stop_to_templates_index(templates)
    print(f"Route templates loaded: {len(templates):,}")

    print("Loading and indexing trips ...")
    trips_df, trip_index = build_trip_template_index(
        trips_path=trips_path,
        templates=templates,
        stop_to_templates=stop_to_templates,
        min_shared_stops=DEFAULT_MIN_SHARED_STOPS,
        top_n=top_n_candidates,
    )
    print(f"Trips indexed: {len(trips_df):,}")

    print(f"Extracting segment sequences (min_obs_stops={min_obs_stops}) ...")
    segs = extract_segment_sequences(snapped_path, min_obs_stops)
    print(f"Segments to match: {len(segs):,}")

    results = []

    for i, row in segs.iterrows():
        obs = row["stop_seq"]
        if len(obs) < min_obs_stops:
            continue

        candidate_ids, raw_counts = generate_candidate_templates(
            obs,
            stop_to_templates,
            min_shared_stops=DEFAULT_MIN_SHARED_STOPS,
            top_n=top_n_candidates,
        )
        candidate_count = len(candidate_ids)

        if candidate_count == 0:
            results.append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "segment_id": row["segment_id"],
                    "seg_start_ist": row["seg_start_ist"],
                    "seg_end_ist": row["seg_end_ist"],
                    "seg_start_date": row["seg_start_date"],
                    "seg_end_date": row["seg_end_date"],
                    "template_id": None,
                    "second_template_id": None,
                    "candidate_trip_id": None,
                    "trip_time_diff_min": None,
                    "match_confidence": 0.0,
                    "second_match_confidence": 0.0,
                    "match_margin": 0.0,
                    "match_jaccard": 0.0,
                    "match_overlap_score": 0.0,
                    "match_coverage_score": 0.0,
                    "match_order_score": 0.0,
                    "match_direction_score": 0.0,
                    "match_endpoint_score": 0.0,
                    "match_lcs_len": 0,
                    "match_is_subseq": False,
                    "match_rank_key": None,
                    "match_method": "unmatched_no_candidates",
                    "candidate_template_count": 0,
                    "n_obs_stops": len(obs),
                    "n_obs_stops_unique": len(set(obs)),
                    "avg_snap_distance_m": row.get("avg_snap_distance_m", np.nan),
                }
            )
            continue

        scored = []
        for tid in candidate_ids:
            sc = compute_match_score(obs, templates[tid])
            scored.append((tid, sc, raw_counts.get(tid, 0)))

        scored.sort(key=lambda x: (x[1]["rank_key"], x[2]), reverse=True)

        best_tid, best_sc, _ = scored[0]
        if len(scored) > 1:
            second_tid, second_sc, _ = scored[1]
        else:
            second_tid, second_sc = None, None

        margin = score_margin_from_rank_keys(
            best_sc["rank_key"], second_sc["rank_key"] if second_sc else None
        )

        if best_sc["confidence"] < min_confidence:
            results.append(
                {
                    "vehicle_id": row["vehicle_id"],
                    "segment_id": row["segment_id"],
                    "seg_start_ist": row["seg_start_ist"],
                    "seg_end_ist": row["seg_end_ist"],
                    "seg_start_date": row["seg_start_date"],
                    "seg_end_date": row["seg_end_date"],
                    "template_id": None,
                    "second_template_id": second_tid,
                    "candidate_trip_id": None,
                    "trip_time_diff_min": None,
                    "match_confidence": float(best_sc["confidence"]),
                    "second_match_confidence": float(second_sc["confidence"])
                    if second_sc
                    else 0.0,
                    "match_margin": margin,
                    "match_jaccard": float(best_sc["jaccard"]),
                    "match_overlap_score": float(best_sc["overlap_score"]),
                    "match_coverage_score": float(best_sc["coverage_score"]),
                    "match_order_score": float(best_sc["order_score"]),
                    "match_direction_score": float(best_sc["direction_score"]),
                    "match_endpoint_score": float(best_sc["endpoint_score"]),
                    "match_lcs_len": int(best_sc["lcs_len"]),
                    "match_is_subseq": bool(best_sc["is_subseq"]),
                    "match_rank_key": json.dumps(best_sc["rank_key"]),
                    "match_method": "below_threshold_lexicographic",
                    "candidate_template_count": candidate_count,
                    "n_obs_stops": len(obs),
                    "n_obs_stops_unique": len(set(obs)),
                    "avg_snap_distance_m": row.get("avg_snap_distance_m", np.nan),
                }
            )
            continue

        trip_id = None
        time_diff = None
        if best_sc["confidence"] >= ROUTE_HIGH_CONFIDENCE:
            trip_id, time_diff = infer_trip_id(
                template_id=best_tid,
                seg_start_date=row["seg_start_date"],
                seg_end_date=row["seg_end_date"],
                seg_start_min=float(row["seg_start_min"]),
                trip_index=trip_index,
                window_min=trip_window_min,
            )

        results.append(
            {
                "vehicle_id": row["vehicle_id"],
                "segment_id": row["segment_id"],
                "seg_start_ist": row["seg_start_ist"],
                "seg_end_ist": row["seg_end_ist"],
                "seg_start_date": row["seg_start_date"],
                "seg_end_date": row["seg_end_date"],
                "template_id": best_tid,
                "second_template_id": second_tid,
                "candidate_trip_id": trip_id,
                "trip_time_diff_min": time_diff,
                "match_confidence": float(best_sc["confidence"]),
                "second_match_confidence": float(second_sc["confidence"])
                if second_sc
                else 0.0,
                "match_margin": margin,
                "match_jaccard": float(best_sc["jaccard"]),
                "match_overlap_score": float(best_sc["overlap_score"]),
                "match_coverage_score": float(best_sc["coverage_score"]),
                "match_order_score": float(best_sc["order_score"]),
                "match_direction_score": float(best_sc["direction_score"]),
                "match_endpoint_score": float(best_sc["endpoint_score"]),
                "match_lcs_len": int(best_sc["lcs_len"]),
                "match_is_subseq": bool(best_sc["is_subseq"]),
                "match_rank_key": json.dumps(best_sc["rank_key"]),
                "match_method": "lexicographic_overlap_order_direction",
                "candidate_template_count": candidate_count,
                "n_obs_stops": len(obs),
                "n_obs_stops_unique": len(set(obs)),
                "avg_snap_distance_m": row.get("avg_snap_distance_m", np.nan),
            }
        )

        if (i + 1) % 10_000 == 0:
            pct = 100 * (i + 1) / max(len(segs), 1)
            print(f"  [{pct:5.1f}%] {i + 1:,}/{len(segs):,}", flush=True)

    inferred = pd.DataFrame(results)

    matched = inferred[inferred["template_id"].notna()].copy()
    with_trip = inferred[inferred["candidate_trip_id"].notna()].copy()

    print("\nRoute inference results:")
    print(f"  Segments processed       : {len(inferred):,}")
    print(f"  Template matched         : {len(matched):,} ({100 * len(matched) / max(len(inferred), 1):.1f}%)")
    print(f"  trip_id assigned         : {len(with_trip):,} ({100 * len(with_trip) / max(len(inferred), 1):.1f}%)")

    if len(matched):
        print(f"  Mean confidence          : {matched['match_confidence'].mean():.3f}")
        print(f"  Mean margin              : {matched['match_margin'].mean():.3f}")
        print(f"  High-conf (>={ROUTE_HIGH_CONFIDENCE:.2f}) : {(matched['match_confidence'] >= ROUTE_HIGH_CONFIDENCE).sum():,}")
        print(f"  Full subsequence matches : {matched['match_is_subseq'].sum():,}")
        print(f"  Mean candidate templates : {matched['candidate_template_count'].mean():.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    inferred.to_parquet(out_path, index=False, compression="zstd")
    print(f"\nWritten -> {out_path}")
    return inferred


# CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--snapped", default=str(PINGS_SNAPPED))
    ap.add_argument("--catalog", default=str(ROUTE_CATALOG))
    ap.add_argument("--trips", default=str(TRIPS_FILE))
    ap.add_argument("--out", default=str(SEGMENTS_INFERRED))
    ap.add_argument("--min_obs_stops", type=int, default=ROUTE_MIN_OBS_STOPS)
    ap.add_argument("--min_conf", type=float, default=ROUTE_MIN_CONFIDENCE)
    ap.add_argument("--trip_window", type=int, default=ROUTE_TRIP_WINDOW_MIN)
    ap.add_argument("--top_n_candidates", type=int, default=DEFAULT_TOP_N_CANDIDATES)
    ap.add_argument("--run_validation", action="store_true")
    ap.add_argument("--validation_sample_n", type=int, default=2000)
    ap.add_argument("--validation_keep_fraction", type=float, default=0.6)
    args = ap.parse_args()

    templates = load_templates(Path(args.catalog))
    stop_to_templates = build_stop_to_templates_index(templates)

    trips_df, _ = build_trip_template_index(
        trips_path=Path(args.trips),
        templates=templates,
        stop_to_templates=stop_to_templates,
        min_shared_stops=DEFAULT_MIN_SHARED_STOPS,
        top_n=args.top_n_candidates,
    )

    if args.run_validation:
        val = validate_route_recovery(
            trips_df=trips_df,
            templates=templates,
            stop_to_templates=stop_to_templates,
            sample_n=args.validation_sample_n,
            keep_fraction=args.validation_keep_fraction,
            min_obs_stops=max(3, args.min_obs_stops),
            top_n=args.top_n_candidates,
        )
        if len(val):
            print("\nValidation summary:")
            print(f"  Samples                 : {len(val):,}")
            print(f"  Top-1 accuracy          : {val['top1_correct'].mean():.4f}")
            print(f"  Mean candidate count    : {val['candidate_count'].mean():.2f}")
            print(f"  Mean observed length    : {val['obs_len'].mean():.2f}")

    run_inference(
        snapped_path=Path(args.snapped),
        catalog_path=Path(args.catalog),
        trips_path=Path(args.trips),
        out_path=Path(args.out),
        min_obs_stops=args.min_obs_stops,
        min_confidence=args.min_conf,
        trip_window_min=args.trip_window,
        top_n_candidates=args.top_n_candidates,
    )
