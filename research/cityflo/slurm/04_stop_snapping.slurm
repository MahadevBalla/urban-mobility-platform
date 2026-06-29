#!/bin/bash
#SBATCH --job-name=cf_snap
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/04_stop_snapping_%j.out
#SBATCH --error=logs/04_stop_snapping_%j.err

set -euo pipefail

echo "================================================="
echo "04_stop_snapping.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/pings_segmented.parquet

echo
python scripts/04_stop_snapping.py --chunk 5000000

echo
echo "Output:"
ls -lh data/processed/pings_snapped.parquet
echo
echo "Finished: $(date)"
