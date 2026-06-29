#!/bin/bash
#SBATCH --job-name=cf_report
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/14_analysis_reporting_%j.out
#SBATCH --error=logs/14_analysis_reporting_%j.err

set -euo pipefail

echo "================================================="
echo "14_analysis_reporting.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh outputs/tables/nb_predictions.parquet
ls -lh outputs/tables/xgb_predictions.parquet
ls -lh outputs/tables/stgnn_predictions.parquet

echo
python scripts/14_analysis_reporting.py

echo
echo "Output:"
ls -lh outputs/figures
ls -lh outputs/tables

echo
echo "Finished: $(date)"
