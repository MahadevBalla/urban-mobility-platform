#!/bin/bash
#SBATCH --job-name=cf_policy
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/15_policy_outputs_%j.out
#SBATCH --error=logs/15_policy_outputs_%j.err

set -euo pipefail

echo "================================================="
echo "15_policy_outputs.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/od_agg.parquet
ls -lh data/processed/service_supply.parquet
ls -lh data/processed/headway_stats.parquet
ls -lh data/processed/route_catalog.parquet

echo
python scripts/15_policy_outputs.py

echo
echo "Output:"
ls -lh data/processed/mode_shift_scores.parquet
ls -lh data/processed/co2_savings.parquet
ls -lh data/processed/service_gaps.parquet

echo
echo "Finished: $(date)"
