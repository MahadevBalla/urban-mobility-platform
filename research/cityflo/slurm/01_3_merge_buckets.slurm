#!/bin/bash
#SBATCH --job-name=cf_merge
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/01_3_merge_buckets_%j.out
#SBATCH --error=logs/01_3_merge_buckets_%j.err

set -euo pipefail

echo "================================================="
echo "01_3_merge_buckets.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

python scripts/01_3_merge_buckets.py

echo
echo "Output:"
ls -lh data/processed/pings_clean.parquet
echo
echo "Finished: $(date)"
