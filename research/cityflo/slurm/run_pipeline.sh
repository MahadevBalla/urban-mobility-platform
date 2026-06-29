#!/bin/bash
#
# run_pipeline.sh
#
# Submits the full Cityflo pipeline as a chain of SLURM jobs with the
# correct dependency graph. Each stage is a separate .slurm file; this
# script only handles sequencing via --dependency.
#
# Usage:
#   export CITYFLO_ROOT=/home/ride/dbarot/urban-mobility-platform/research/cityflo
#   cd "$CITYFLO_ROOT/slurm"
#   ./run_pipeline.sh
#
# To resume from a later stage (e.g. GPS ingestion already done):
#   ./run_pipeline.sh --from 05
#
# To run a single stage and stop:
#   ./run_pipeline.sh --from 17 --to 17
#
set -euo pipefail

if [[ -z "${CITYFLO_ROOT:-}" ]]; then
    echo "CITYFLO_ROOT is not set. Export it before running this script:"
    echo "  export CITYFLO_ROOT=/path/to/urban-mobility-platform/research/cityflo"
    exit 1
fi

export CITYFLO_ROOT
mkdir -p "${CITYFLO_ROOT}/slurm/logs"
cd "${CITYFLO_ROOT}/slurm"

FROM_STAGE="01"
TO_STAGE="17"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STAGE="$2"; shift 2 ;;
        --to)   TO_STAGE="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# stage_ge a b -> true if stage a >= stage b, comparing as integers
stage_ge() { [[ "$((10#$1))" -ge "$((10#$2))" ]]; }
stage_le() { [[ "$((10#$1))" -le "$((10#$2))" ]]; }
in_range() { stage_ge "$1" "$FROM_STAGE" && stage_le "$1" "$TO_STAGE"; }

# submit <stage> <slurm_file> [--dependency=...] -> prints ONLY the job id to
# stdout, so that JOBID[x]=$(submit ...) captures a clean id. Every other
# message goes to stderr, otherwise it would get swallowed into the captured
# value and corrupt downstream --dependency strings.
submit() {
    local stage="$1"
    local file="$2"
    shift 2
    if ! in_range "$stage"; then
        echo "Skipping stage ${stage} (${file}) — outside requested range" >&2
        return 0
    fi
    local jobid
    [[ -f "$file" ]] || {
        echo "Missing SLURM script: $file" >&2
        exit 1
    }
    jobid=$(sbatch --parsable "$@" "$file")
    echo "Submitted stage ${stage} (${file}) -> job ${jobid}" >&2
    echo "$jobid"
}

declare -A JOBID

# join_deps <jobid> [<jobid> ...] -> "afterok:1000:1001:1002", or empty if no args
join_deps() {
    local ids=()
    for id in "$@"; do
        [[ -n "$id" ]] && ids+=("$id")
    done
    [[ ${#ids[@]} -eq 0 ]] && return 0
    local joined
    joined=$(IFS=:; echo "${ids[*]}")
    echo "afterok:${joined}"
}

echo "Pipeline range: stage ${FROM_STAGE} to stage ${TO_STAGE}"
echo "CITYFLO_ROOT  : ${CITYFLO_ROOT}"
echo

# Stage 01-03: GPS ingestion -> finalize -> merge
# Stage 02 is also a job array (one finalize task per bucket) so each task
# only needs to wait on the matching array index of stage 01, not all of it.
# aftercorr expresses exactly that: task N waits on task N of the dependency.
JOBID[01]=$(submit 01 01_1_ingest_legacy.slurm)
dep=""
[[ -n "${JOBID[01]:-}" ]] && dep="aftercorr:${JOBID[01]}"
JOBID[02]=$(submit 02 01_2_finalize_pings.slurm ${dep:+--dependency=$dep})

dep=$(join_deps "${JOBID[02]:-}")
JOBID[03]=$(submit 03 01_3_merge_buckets.slurm ${dep:+--dependency=$dep})

# Stage 04: route catalog, independent of the GPS chain
JOBID[04]=$(submit 04 04_route_catalog.slurm)

# Stage 05-07: segmentation -> snapping -> route inference
dep=$(join_deps "${JOBID[03]:-}")
JOBID[05]=$(submit 05 03_trip_segmentation.slurm ${dep:+--dependency=$dep})

dep=$(join_deps "${JOBID[05]:-}")
JOBID[06]=$(submit 06 04_stop_snapping.slurm ${dep:+--dependency=$dep})

dep=$(join_deps "${JOBID[06]:-}" "${JOBID[04]:-}")
JOBID[07]=$(submit 07 05_route_inference.slurm ${dep:+--dependency=$dep})

# Stage 08-09: OD matrix and reliability, both depend on route inference, run in parallel
dep=$(join_deps "${JOBID[07]:-}")
JOBID[08]=$(submit 08 06_od_matrix.slurm ${dep:+--dependency=$dep})
JOBID[09]=$(submit 09 07_reliability.slurm ${dep:+--dependency=$dep})

# Stage 10: weather, fully independent of the GPS chain
JOBID[10]=$(submit 10 08_weather_consolidate.slurm)

# Stage 11: feature engineering, needs OD + reliability + weather
dep=$(join_deps "${JOBID[08]:-}" "${JOBID[09]:-}" "${JOBID[10]:-}")
JOBID[11]=$(submit 11 09_feature_engineering.slurm ${dep:+--dependency=$dep})

# Stage 12: ward aggregation, needs only OD matrix
dep=$(join_deps "${JOBID[08]:-}")
JOBID[12]=$(submit 12 10_ward_aggregation.slurm ${dep:+--dependency=$dep})

# Stage 13-15: forecasting models run in parallel after feature engineering
dep=$(join_deps "${JOBID[11]:-}")
JOBID[13]=$(submit 13 11_model_nb.slurm ${dep:+--dependency=$dep})
JOBID[14]=$(submit 14 12_model_xgboost.slurm ${dep:+--dependency=$dep})
JOBID[15]=$(submit 15 13_model_stgnn.slurm ${dep:+--dependency=$dep})

# Stage 16: analysis/reporting depends on all model outputs
dep=$(join_deps "${JOBID[13]:-}" "${JOBID[14]:-}" "${JOBID[15]:-}")
JOBID[16]=$(submit 16 14_analysis_reporting.slurm ${dep:+--dependency=$dep})

# Stage 17: policy outputs depend on route catalog, OD matrix, and reliability
dep=$(join_deps "${JOBID[04]:-}" "${JOBID[08]:-}" "${JOBID[09]:-}")
JOBID[17]=$(submit 17 15_policy_outputs.slurm ${dep:+--dependency=$dep})

echo
echo "All requested stages submitted. Track with:"
echo "  squeue --me"
