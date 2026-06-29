#!/bin/bash
#SBATCH --job-name=cf_xgb
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/12_model_xgboost_%j.out
#SBATCH --error=logs/12_model_xgboost_%j.err

set -euo pipefail

echo "================================================="
echo "12_model_xgboost.py"
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
python scripts/12_model_xgboost.py

echo
echo "Output:"
ls -lh outputs/models/xgb_model.pkl
ls -lh outputs/tables/xgb_predictions.parquet
ls -lh outputs/tables/xgb_metrics.csv
ls -lh outputs/figures/xgb_shap_summary.png
ls -lh outputs/figures/xgb_feature_importance.png
echo
echo "Finished: $(date)"
