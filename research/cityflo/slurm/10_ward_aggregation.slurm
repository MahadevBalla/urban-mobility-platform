#!/bin/bash
#SBATCH --job-name=cf_ward
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/10_ward_aggregation_%j.out
#SBATCH --error=logs/10_ward_aggregation_%j.err

set -euo pipefail

echo "================================================="
echo "10_ward_aggregation.py"
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
ls -lh data/processed/stops_clean.csv

echo
python scripts/10_ward_aggregation.py

echo
echo "Output:"
ls -lh data/processed/ward_od.parquet
ls -lh data/processed/stop_ward_map.csv
echo
echo "Finished: $(date)"
