#!/bin/bash
#SBATCH --job-name=cf_od
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/06_od_matrix_%j.out
#SBATCH --error=logs/06_od_matrix_%j.err

set -euo pipefail

echo "================================================="
echo "06_od_matrix.py"
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
ls -lh data/processed/route_catalog.parquet

echo
python scripts/06_od_matrix.py

echo
echo "Output:"
ls -lh data/processed/od_tier1.parquet
ls -lh data/processed/od_tier2.parquet
ls -lh data/processed/od_agg.parquet
ls -lh data/processed/service_supply.parquet
ls -lh data/processed/service_frequency.parquet
echo
echo "Finished: $(date)"
