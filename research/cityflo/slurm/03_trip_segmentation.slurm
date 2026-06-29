#!/bin/bash
#SBATCH --job-name=cf_segment
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/03_trip_segmentation_%j.out
#SBATCH --error=logs/03_trip_segmentation_%j.err

set -euo pipefail

echo "================================================="
echo "03_trip_segmentation.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/pings_clean.parquet

echo
python scripts/03_trip_segmentation.py

echo
echo "Output:"
ls -lh data/processed/pings_segmented.parquet
echo
echo "Finished: $(date)"
