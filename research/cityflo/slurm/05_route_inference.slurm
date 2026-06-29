#!/bin/bash
#SBATCH --job-name=cf_route
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/05_route_inference_%j.out
#SBATCH --error=logs/05_route_inference_%j.err

set -euo pipefail

echo "================================================="
echo "05_route_inference.py"
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
ls -lh data/processed/route_catalog.parquet
ls -lh data/processed/trips_clean.csv

echo
python scripts/05_route_inference.py

echo
echo "Output:"
ls -lh data/processed/segments_inferred.parquet
echo
echo "Finished: $(date)"
