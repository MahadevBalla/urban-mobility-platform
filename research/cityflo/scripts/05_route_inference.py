"""
05_route_inference.py

Matches each GPS trip segment to a scheduled route template (Stage 1),
then to a specific scheduled trip instance on that template (Stage 2).

Stage 1: candidate templates are generated via an inverted stop->template
index (pruned by minimum shared-stop count), then ranked by LCSS
(Longest Common Subsequence Similarity) over the ordered stop-id
sequences, following Vlachos, Gunopulos & Kollios (2002). LCSS is used
because it is order-aware and tolerant of missing/extra stops, which
matches the failure modes of snapped GPS sequences (missed pings, snap
failures). Other overlap/coverage/direction/endpoint scores are kept as
diagnostics and tie-breakers only.

Stage 2: the candidate trip on the matched template is selected using
the residual between observed and scheduled stop arrival times, after
removing a per-trip constant offset (delay/earliness) estimated from
the shared stops. This follows standard AVL schedule-adherence matching
practice, and avoids penalizing a trip that is running consistently
early/late but otherwise matches the timetable shape. Falls back to a
start-time-only comparison when too few timestamped stops are shared.

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
    DEFAULT_MIN_SHARED_STOPS,
    DEFAULT_TOP_N_CANDIDATES,
    DEFAULT_TRIP_ASSIGN_MIN_CONF,
    DEFAULT_TRIP_ASSIGN_MIN_OVERLAP,
    DEFAULT_VALIDATION_RANDOM_SEED,
    ROUTE_MAX_TRIP_RESIDUAL_MIN,
)


# Parsing helpers
_ROUTE_PAIR_RE = re.compile(r"(\d+)\s*,\s*(\d{6}|\d{2}:\d{2}:\d{2})")


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
        except (SyntaxError, ValueError):
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
    """Normalize any timestamp/date string to YYYY-MM-DD.

    Retained for segment-level date bookkeeping (seg_start_date,
    seg_end_date on the output rows) even though trip-instance lookup no
    longer keys on date.
    """
    s = str(ts).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    try:
        parsed = pd.to_datetime(s, dayfirst=True, errors="raise")
        return parsed.strftime("%Y-%m-%d")
    except Exception:
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
    """Standard O(mn) dynamic-programming LCS length.

    Simple, correct, and easy to verify.  For the sequence lengths
    encountered here (typically 10-40 stops) this is more than fast
    enough and eliminates any risk of bit‑parallel bugs.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0

    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        # Swap rows
        prev, curr = curr, prev

    return prev[n]


def matched_template_positions(
    observed: List[int], tmpl_pos_map: Dict[int, int]
) -> List[int]:
    return [tmpl_pos_map[s] for s in observed if s in tmpl_pos_map]


def monotonicity_ratio(observed: List[int], tmpl_pos_map: Dict[int, int]) -> float:
    """Diagnostic only: fraction of matched stops in a non-decreasing
    template-position order (longest non-decreasing subsequence)."""
    pos = matched_template_positions(observed, tmpl_pos_map)
    if len(pos) < 2:
        return 0.5

    tails: List[int] = []
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

    return float(len(tails) / len(pos))


def endpoint_proximity_score(
    observed: List[int], tmpl_stops: List[int], tmpl_pos_map: Dict[int, int]
) -> float:
    """Diagnostic only: how close observed start/end stops are to the
    template's own start/end."""
    if not observed or not tmpl_stops:
        return 0.0

    n = len(tmpl_stops)
    if n == 0:
        return 0.0

    scores = []
    first_obs = observed[0]
    if first_obs in tmpl_pos_map:
        p = tmpl_pos_map[first_obs]
        scores.append(1.0 - (p / max(n - 1, 1)))

    last_obs = observed[-1]
    if last_obs in tmpl_pos_map:
        p = tmpl_pos_map[last_obs]
        scores.append(1.0 - ((n - 1 - p) / max(n - 1, 1)))

    return float(np.mean(scores)) if scores else 0.0


def score_margin_from_rank_keys(best_key: Tuple, second_key: Tuple) -> float:
    """Scalar ambiguity summary from the rank_key components
    (LCSS similarity, coverage, negative length penalty)."""
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

    for row in cat.itertuples(index=False):
        tid = int(row.template_id)
        stops = [int(x) for x in json.loads(row.stop_sequence)]
        n_trips = int(getattr(row, "n_trips_in_catalog", 0))
        sched = (
            json.loads(row.median_schedule_json)
            if pd.notna(getattr(row, "median_schedule_json", None))
            else []
        )
        templates[tid] = {
            "template_id": tid,
            "stops": stops,
            "stop_set": frozenset(stops),
            "pos_map": {sid: i for i, sid in enumerate(stops)},
            "n_stops": len(stops),
            "first": stops[0] if stops else None,
            "last": stops[-1] if stops else None,
            "n_trips": n_trips,
            "sched": sched,
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
    """Inverted-index candidate generation, pruned by shared-stop count."""
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
    """LCSS similarity (Vlachos, Gunopulos & Kollios, 2002) over ordered
    stop-id sequences, normalized by the longer of the two sequences so
    a short observation cannot trivially score 1.0 against a much
    longer template. This is `confidence` / the primary rank key.

    Overlap / coverage / jaccard / direction / endpoint are diagnostics
    and secondary tie-breakers only.
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
    
    # FIX: Normalize against the template length, not the bloated observed noise length.
    # This prevents GPS jitter inflation from destroying valid confidence scores.
    denom = len(tmpl["stops"])
    lcss_similarity = (lcs_len / denom) if denom else 0.0
    order_score_val = lcs_len / len(obs_unique) if obs_unique else 0.0

    dir_score = monotonicity_ratio(obs_dedup, tmpl["pos_map"])
    end_score = endpoint_proximity_score(obs_dedup, tmpl["stops"], tmpl["pos_map"])

    is_full_subseq = bool(obs_unique) and lcs_len == len(obs_unique)
    template_len_penalty = abs(len(tmpl["stops"]) - len(obs_unique)) / max(
        len(tmpl["stops"]), 1
    )

    rank_key = (
        round(float(lcss_similarity), 6),
        round(float(coverage_score), 6),
        -round(float(template_len_penalty), 6),
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
        "lcss_similarity": round(float(lcss_similarity), 4),
        "is_subseq": bool(is_full_subseq),
        "template_len_penalty": round(float(template_len_penalty), 4),
        "rank_key": rank_key,
        "confidence": round(float(lcss_similarity), 4),
    }


# Trip -> template assignment (for building the trip index)
def assign_trip_templates_by_inference(
    trips_df: pd.DataFrame,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    min_shared_stops: int = DEFAULT_MIN_SHARED_STOPS,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
    min_confidence: float = DEFAULT_TRIP_ASSIGN_MIN_CONF,
    min_overlap: float = DEFAULT_TRIP_ASSIGN_MIN_OVERLAP,
) -> pd.DataFrame:
    """Attach template_id to trips_clean rows using the same LCSS
    scorer used for segment matching."""
    out_rows = []

    for _, row in trips_df.iterrows():
        parsed = row.parsed
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
            stop_seq, stop_to_templates, min_shared_stops=min_shared_stops, top_n=top_n
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

        scored = [
            (tid, compute_match_score(stop_seq, templates[tid]), raw_counts.get(tid, 0))
            for tid in candidate_ids
        ]
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

    return pd.DataFrame(out_rows)


def build_trip_template_index(
    trips_path: Path,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    min_shared_stops: int = DEFAULT_MIN_SHARED_STOPS,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
) -> Tuple[pd.DataFrame, Dict[int, List[dict]]]:
    """
    Builds the trip-instance index used by infer_trip_id.

    Indexed by template_id only.

    Trip dates are retained for bookkeeping and diagnostics but are not
    used during lookup because trip selection is based on time-of-day
    schedule residuals rather than calendar dates.
    """
    trips_df = pd.read_csv(trips_path)
    trips_df = trips_df.rename(
        columns={
            "tripid": "trip_id",
            "tripdate": "trip_date",
            "triproute": "trip_route",
        }
    )

    trips_df["trip_date"] = trips_df["trip_date"].astype(str).apply(ts_to_date_str)
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

    trip_idx: Dict[int, List[dict]] = defaultdict(list)
    for row in trips_df.itertuples(index=False):
        sched_stop_min = {sid: time_str_to_minutes(t) for sid, t in row.parsed}
        trip_idx[int(row.template_id)].append(
            {
                "trip_id": row.trip_id,
                "trip_date": row.trip_date,  # retained for debugging only
                "sched_start_min": row.sched_start_min,
                "sched_stop_min": sched_stop_min,
                "template_assign_confidence": row.template_assign_confidence,
            }
        )

    for key in trip_idx:
        trip_idx[key] = sorted(trip_idx[key], key=lambda x: x["sched_start_min"])

    n_instances = sum(len(v) for v in trip_idx.values())
    print(f"  Trip index keys (template only): {len(trip_idx):,}")
    print(f"  Total trip instances indexed   : {n_instances:,}")
    if trip_idx:
        max_key = max(trip_idx, key=lambda k: len(trip_idx[k]))
        print(
            f"  Largest candidate pool         : template {max_key} "
            f"({len(trip_idx[max_key]):,} trip instances)"
        )
    return trips_df, trip_idx


def infer_trip_id(
    template_id: int,
    seg_start_min: float,
    stop_time_map: Dict[int, float],
    trip_index: Dict[int, List[dict]],
    window_min: int,
    min_shared_timed_stops: int = 2,
    max_residual_min: float = ROUTE_MAX_TRIP_RESIDUAL_MIN,
) -> Tuple[Optional[int], Optional[float]]:
    """Select the scheduled trip instance for a matched route template.

    Candidates are all scheduled trips assigned to `template_id`. When at
    least `min_shared_timed_stops` are available, estimate a constant
    delay/earliness offset from the shared stops and rank candidates by the
    remaining timing residual. Otherwise, fall back to comparing segment
    start time against the scheduled departure time.

    Returns `(trip_id, residual_minutes)` if the best candidate satisfies
    the configured residual threshold; otherwise returns `(None, None)`.
    """

    def wrapped_diff(a: float, b: float) -> float:
        return min(abs(a - b), abs((a + 1440) - b), abs(a - (b + 1440)))

    def wrapped_signed_diff(obs: float, sched: float) -> float:
        options = [obs - sched, obs - sched + 1440, obs - sched - 1440]
        return min(options, key=abs)

    # We store a tuple (residual, |delta|) to break ties.
    best_trip_id = None
    best_score = (float("inf"), float("inf"))
    best_is_timing = False

    for trip in trip_index.get(template_id, []):
        sched_stop_min = trip.get("sched_stop_min", {})
        shared_stops = set(stop_time_map) & set(sched_stop_min)
        if len(shared_stops) >= min_shared_timed_stops:
            signed_diffs = [
                wrapped_signed_diff(stop_time_map[sid], sched_stop_min[sid])
                for sid in shared_stops
            ]
            delta = float(np.median(signed_diffs))
            residual = float(np.mean([abs(d - delta) for d in signed_diffs]))
            is_timing = True
            score = (residual, abs(delta))
        else:
            residual = wrapped_diff(seg_start_min, trip["sched_start_min"])
            is_timing = False
            score = (residual, abs(residual))

        # Compare: prefer timing-based matches, then smaller (residual, |delta|)
        if best_trip_id is None:
            better = True
        elif is_timing and not best_is_timing:
            better = True
        elif not is_timing and best_is_timing:
            better = False
        else:
            better = score < best_score

        if better:
            best_trip_id = trip["trip_id"]
            best_score = score
            best_is_timing = is_timing

    if best_trip_id is None:
        return None, None

    best_residual = best_score[0]
    bound = max_residual_min if best_is_timing else window_min
    if best_residual > bound:
        return None, None

    return int(best_trip_id), round(float(best_residual), 2)


# Segment extraction
def _column_exists(parquet_path: Path, col: str) -> bool:
    con = None
    try:
        con = duckdb.connect()
        cols = (
            con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')")
            .df()["column_name"]
            .tolist()
        )
        return col in cols
    finally:
        if con is not None:
            con.close()


def extract_segment_sequences(snapped_path: Path, min_obs_stops: int) -> pd.DataFrame:
    has_snap_dist = _column_exists(snapped_path, "snap_distance_m")

    con = None
    try:
        con = duckdb.connect()
        select_snap = ", snap_distance_m" if has_snap_dist else ""
        agg_snap = (
            ", AVG(snap_distance_m) AS avg_snap_distance_m"
            if has_snap_dist
            else ", NULL::DOUBLE AS avg_snap_distance_m"
        )

        con.execute(f"""
            CREATE TABLE snapped AS
            SELECT
                vehicle_id, segment_id, ride_date, snapped_stop_id, timestamp_ist
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
                LIST(timestamp_ist ORDER BY timestamp_ist)::VARCHAR[] AS stop_ts_raw,
                COUNT(*) AS n_obs_pings,
                COUNT(DISTINCT snapped_stop_id) AS n_obs_stops_unique
                {agg_snap}
            FROM snapped
            GROUP BY vehicle_id, segment_id
            HAVING COUNT(DISTINCT snapped_stop_id) >= {min_obs_stops}
        """).df()
    finally:
        if con is not None:
            con.close()

    segs["stop_seq"] = segs["stop_seq_raw"].apply(
        lambda x: dedup_consecutive([int(v) for v in x])
    )
    segs["n_obs_stops"] = segs["stop_seq"].apply(len)
    segs = segs[segs["n_obs_stops"] >= min_obs_stops].copy()

    def _build_stop_time_map(stop_ids_raw, ts_raw) -> dict:
        m: Dict[int, float] = {}
        for sid, ts in zip(stop_ids_raw, ts_raw):
            sid = int(sid)
            if sid not in m:
                m[sid] = parse_ts_to_minutes(ts)
        return m

    segs["stop_time_map"] = segs.apply(
        lambda r: _build_stop_time_map(r["stop_seq_raw"], r["stop_ts_raw"]), axis=1
    )

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
    corrupt_prob: float = 0.0,
) -> List[int]:
    """Simulate a partial observation of a full stop sequence.

    With probability `corrupt_prob`, applies a contiguous-block, prefix,
    or suffix removal (closer to real GPS dropout patterns) instead of
    independently-random deletion. Default corrupt_prob=0.0 reproduces
    plain random deletion.
    """
    if rng is None:
        rng = random.Random(DEFAULT_VALIDATION_RANDOM_SEED)

    seq = dedup_consecutive(stop_seq)
    if len(seq) <= min_keep:
        return seq[:]

    if corrupt_prob > 0.0 and rng.random() < corrupt_prob:
        mode = rng.choice(["prefix", "suffix", "block"])
        n = len(seq)
        keep_n = max(min_keep, int(math.ceil(n * keep_fraction)))
        if mode == "prefix":
            return seq[:keep_n]
        if mode == "suffix":
            return seq[n - keep_n :]
        max_start = max(0, n - keep_n)
        start = rng.randint(0, max_start)
        return seq[start : start + keep_n]

    idxs = list(range(len(seq)))
    keep_n = max(min_keep, int(math.ceil(len(seq) * keep_fraction)))
    chosen = sorted(rng.sample(idxs, min(keep_n, len(idxs))))
    return dedup_consecutive([seq[i] for i in chosen])


def validate_route_recovery(
    trips_df: pd.DataFrame,
    templates: Dict[int, dict],
    stop_to_templates: Dict[int, set],
    sample_n: int = 2000,
    keep_fraction: float = 0.6,
    min_obs_stops: int = 3,
    top_n: int = DEFAULT_TOP_N_CANDIDATES,
    random_seed: int = DEFAULT_VALIDATION_RANDOM_SEED,
    corrupt_prob: float = 0.0,
) -> pd.DataFrame:
    rng = random.Random(random_seed)

    valid = trips_df[trips_df["template_id"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    if len(valid) > sample_n:
        valid = valid.sample(sample_n, random_state=random_seed)

    rows = []
    for row in valid.itertuples(index=False):
        full_seq = [sid for sid, _ in row.parsed]
        obs = simulate_partial_observation(
            full_seq,
            keep_fraction=keep_fraction,
            min_keep=min_obs_stops,
            rng=rng,
            corrupt_prob=corrupt_prob,
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
                    "trip_id": row.trip_id,
                    "true_template_id": row.template_id,
                    "pred_template_id": pd.NA,
                    "top1_correct": False,
                    "candidate_count": 0,
                    "obs_len": len(obs),
                }
            )
            continue

        scored = [
            (tid, compute_match_score(obs, templates[tid]), raw_counts.get(tid, 0))
            for tid in candidate_ids
        ]
        scored.sort(key=lambda x: (x[1]["rank_key"], x[2]), reverse=True)
        pred_tid = int(scored[0][0])

        rows.append(
            {
                "trip_id": row.trip_id,
                "true_template_id": int(row.template_id),
                "pred_template_id": pred_tid,
                "top1_correct": pred_tid == int(row.template_id),
                "candidate_count": len(candidate_ids),
                "obs_len": len(obs),
            }
        )

    return pd.DataFrame(rows)


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
    print("segs['n_obs_stops'].describe:")
    print(segs["n_obs_stops"].describe())
    print("\nTemplate length distribution:")
    print(pd.Series([t["n_stops"] for t in templates.values()]).describe())
    print("\nAverage snap distance:")
    print(segs["avg_snap_distance_m"].describe())

    results = []
    total_segments = len(segs)

    for i, row in enumerate(segs.itertuples(index=False), start=1):
        obs = row.stop_seq
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
                    "vehicle_id": row.vehicle_id,
                    "segment_id": row.segment_id,
                    "seg_start_ist": row.seg_start_ist,
                    "seg_end_ist": row.seg_end_ist,
                    "seg_start_date": row.seg_start_date,
                    "seg_end_date": row.seg_end_date,
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
                    "avg_snap_distance_m": getattr(row, "avg_snap_distance_m", np.nan),
                }
            )
            continue

        scored = [
            (tid, compute_match_score(obs, templates[tid]), raw_counts.get(tid, 0))
            for tid in candidate_ids
        ]
        scored.sort(key=lambda x: (x[1]["rank_key"], x[2]), reverse=True)

        best_tid, best_sc, _ = scored[0]
        second_tid, second_sc = (
            (scored[1][0], scored[1][1]) if len(scored) > 1 else (None, None)
        )

        margin = score_margin_from_rank_keys(
            best_sc["rank_key"], second_sc["rank_key"] if second_sc else None
        )

        if best_sc["confidence"] < min_confidence:
            results.append(
                {
                    "vehicle_id": row.vehicle_id,
                    "segment_id": row.segment_id,
                    "seg_start_ist": row.seg_start_ist,
                    "seg_end_ist": row.seg_end_ist,
                    "seg_start_date": row.seg_start_date,
                    "seg_end_date": row.seg_end_date,
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
                    "match_method": "below_threshold_lcss",
                    "candidate_template_count": candidate_count,
                    "n_obs_stops": len(obs),
                    "n_obs_stops_unique": len(set(obs)),
                    "avg_snap_distance_m": getattr(row, "avg_snap_distance_m", np.nan),
                }
            )
            continue

        trip_id = None
        time_diff = None
        if best_sc["confidence"] >= ROUTE_HIGH_CONFIDENCE:
            trip_id, time_diff = infer_trip_id(
                template_id=best_tid,
                seg_start_min=row.seg_start_min,
                stop_time_map=row.stop_time_map,
                trip_index=trip_index,
                window_min=trip_window_min,
            )

        results.append(
            {
                "vehicle_id": row.vehicle_id,
                "segment_id": row.segment_id,
                "seg_start_ist": row.seg_start_ist,
                "seg_end_ist": row.seg_end_ist,
                "seg_start_date": row.seg_start_date,
                "seg_end_date": row.seg_end_date,
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
                "match_method": "lcss_similarity",
                "candidate_template_count": candidate_count,
                "n_obs_stops": len(obs),
                "n_obs_stops_unique": len(set(obs)),
                "avg_snap_distance_m": getattr(row, "avg_snap_distance_m", np.nan),
            }
        )

        if i % 10_000 == 0:
            pct = 100 * i / total_segments
            print(f"  [{pct:5.1f}%] {i:,}/{total_segments:,}", flush=True)

    inferred = pd.DataFrame(results)

    matched = inferred[inferred["template_id"].notna()].copy()
    with_trip = inferred[inferred["candidate_trip_id"].notna()].copy()

    print("\nRoute inference results:")
    print(f"  Segments processed       : {len(inferred):,}")
    print(
        f"  Template matched         : {len(matched):,} ({100 * len(matched) / max(len(inferred), 1):.1f}%)"
    )
    print(
        f"  trip_id assigned         : {len(with_trip):,} ({100 * len(with_trip) / max(len(inferred), 1):.1f}%)"
    )

    if len(matched):
        print(f"  Mean confidence          : {matched['match_confidence'].mean():.3f}")
        print(f"  Mean margin              : {matched['match_margin'].mean():.3f}")
        print(
            f"  High-conf (>={ROUTE_HIGH_CONFIDENCE:.2f}) : {(matched['match_confidence'] >= ROUTE_HIGH_CONFIDENCE).sum():,}"
        )
        print(f"  Full subsequence matches : {matched['match_is_subseq'].sum():,}")
        print(
            f"  Mean candidate templates : {matched['candidate_template_count'].mean():.2f}"
        )

        high_conf_mask = matched["match_confidence"] >= ROUTE_HIGH_CONFIDENCE
        n_high_conf = int(high_conf_mask.sum())
        n_high_conf_with_trip = int(
            matched.loc[high_conf_mask, "candidate_trip_id"].notna().sum()
        )
        if n_high_conf > 0:
            print(
                f"  Of high-conf segments, trip_id resolved for: "
                f"{n_high_conf_with_trip:,}/{n_high_conf:,} "
                f"({100 * n_high_conf_with_trip / n_high_conf:.1f}%)"
            )

    # =====================================================================
    # CITYFLO DEBUG BLOCK: INJECT EXACTLY HERE -- PC
    # =====================================================================
    print("\n--- RUNNING ROUTE INFERENCE DIAGNOSTICS ---")
    
    # 1. The Short-Sequence Illusion Check
    short_matches = inferred[inferred["n_obs_stops"] < min_obs_stops]
    print(f"[TEST 1] Inferred routes with < {min_obs_stops} stops: {len(short_matches):,}")
    assert len(short_matches) == 0, f"CRITICAL FAILURE: LCSS is matching ghost segments. Segments with < {min_obs_stops} stops must be dropped."

    # 2. Midnight Residual Wrap-around Check
    massive_delays = inferred[inferred["trip_time_diff_min"].abs() > 700]
    print(f"[TEST 2] Trips with >12 hour schedule residuals: {len(massive_delays):,}")
    if len(massive_delays) > 0:
        print("WARNING: Midnight residual trap detected! Night buses are failing stage 2 matching because of raw time subtraction.")
        
    # 3. Overall Match Rate
    total_segments = len(inferred)
    matched_segments = inferred["template_id"].notna().sum()
    match_rate = (matched_segments / total_segments) * 100 if total_segments > 0 else 0
    print(f"[TEST 3] Overall Segment Match Rate: {match_rate:.1f}%")
    if match_rate < 30.0:
        print("WARNING: Less than 30% of your segments matched a route. The snapping threshold or LCSS confidence parameters are too strict.")
    # =====================================================================

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
    ap.add_argument("--validation_corrupt_prob", type=float, default=0.0)
    args = ap.parse_args()

    if args.run_validation:
        templates = load_templates(Path(args.catalog))
        stop_to_templates = build_stop_to_templates_index(templates)

        trips_df, _ = build_trip_template_index(
            trips_path=Path(args.trips),
            templates=templates,
            stop_to_templates=stop_to_templates,
            min_shared_stops=DEFAULT_MIN_SHARED_STOPS,
            top_n=args.top_n_candidates,
        )
        val = validate_route_recovery(
            trips_df=trips_df,
            templates=templates,
            stop_to_templates=stop_to_templates,
            sample_n=args.validation_sample_n,
            keep_fraction=args.validation_keep_fraction,
            min_obs_stops=max(3, args.min_obs_stops),
            top_n=args.top_n_candidates,
            corrupt_prob=args.validation_corrupt_prob,
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
