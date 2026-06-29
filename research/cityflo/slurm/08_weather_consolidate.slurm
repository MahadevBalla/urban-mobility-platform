#!/bin/bash
#SBATCH --job-name=cf_weather
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/08_weather_consolidate_%j.out
#SBATCH --error=logs/08_weather_consolidate_%j.err

set -euo pipefail

echo "================================================="
echo "08_weather_consolidate.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/raw/weather/ | head -20
ls -lh data/processed/stops_clean.csv

echo
python scripts/08_weather_consolidate.py

echo
echo "Output:"
ls -lh data/processed/weather_grid_master.parquet
ls -lh data/processed/weather_stop_hourly.parquet
echo
echo "Finished: $(date)"
