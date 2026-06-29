#!/bin/bash
#SBATCH --job-name=cf_finalize
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/01_2_finalize_pings_%A_%a.out
#SBATCH --error=logs/01_2_finalize_pings_%A_%a.err

set -euo pipefail

echo "================================================="
echo "01_2_finalize_pings.py  (bucket ${SLURM_ARRAY_TASK_ID})"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load python@3.13.5

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

python scripts/01_2_finalize_pings.py --bucket_id "${SLURM_ARRAY_TASK_ID}"

echo
echo "Finished: $(date)"
