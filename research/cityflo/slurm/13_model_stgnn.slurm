#!/bin/bash
#SBATCH --job-name=cf_stgnn
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/13_model_stgnn_%j.out
#SBATCH --error=logs/13_model_stgnn_%j.err

set -euo pipefail

echo "================================================="
echo "13_model_stgnn.py"
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
python scripts/13_model_stgnn.py

echo
echo "Output:"
ls -lh outputs/models/stgnn_model.pt
ls -lh outputs/tables/stgnn_predictions.parquet
ls -lh outputs/tables/stgnn_metrics.csv

echo
echo "Finished: $(date)"
