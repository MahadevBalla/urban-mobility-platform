#!/bin/bash
#SBATCH --job-name=cf_catalog
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/02_route_catalog_%j.out
#SBATCH --error=logs/02_route_catalog_%j.err

set -euo pipefail

echo "================================================="
echo "02_route_catalog.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/trips_clean.csv
ls -lh data/processed/stops_clean.csv

echo
python scripts/02_route_catalog.py

echo
echo "Output:"
ls -lh data/processed/route_catalog.parquet
ls -lh data/processed/route_catalog.json
echo
echo "Finished: $(date)"
