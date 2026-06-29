#!/bin/bash
#SBATCH --job-name=cf_features
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/09_feature_engineering_%j.out
#SBATCH --error=logs/09_feature_engineering_%j.err

set -euo pipefail

echo "================================================="
echo "09_feature_engineering.py"
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
ls -lh data/processed/headway_stats.parquet
ls -lh data/processed/service_supply.parquet
ls -lh data/processed/schedule_adherence_stats.parquet
ls -lh data/processed/weather_stop_hourly.parquet

echo
python scripts/09_feature_engineering.py

echo
echo "Output:"
ls -lh data/processed/features_master.parquet
echo
echo "Finished: $(date)"
