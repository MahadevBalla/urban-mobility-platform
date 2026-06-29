#!/bin/bash
#SBATCH --job-name=cf_nb
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/11_model_nb_%j.out
#SBATCH --error=logs/11_model_nb_%j.err

set -euo pipefail

echo "================================================="
echo "11_model_nb.py"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

echo
echo "Input:"
ls -lh data/processed/features_master.parquet

echo
python scripts/11_model_nb.py

echo
echo "Output:"
ls -lh outputs/models/nb_model_summary.txt
ls -lh outputs/tables/nb_predictions.parquet
ls -lh outputs/tables/nb_metrics.csv
echo
echo "Finished: $(date)"
