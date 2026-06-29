#!/bin/bash
#SBATCH --job-name=cf_reliability
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/07_reliability_%j.out
#SBATCH --error=logs/07_reliability_%j.err

set -euo pipefail

echo "================================================="
echo "07_reliability.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/pings_snapped.parquet
ls -lh data/processed/segments_inferred.parquet
ls -lh data/processed/trips_clean.csv

echo
python scripts/07_reliability.py

echo
echo "Output:"
ls -lh data/processed/headway_stats.parquet
ls -lh data/processed/schedule_adherence_stats.parquet
ls -lh data/processed/stop_visits.parquet
echo
echo "Finished: $(date)"
