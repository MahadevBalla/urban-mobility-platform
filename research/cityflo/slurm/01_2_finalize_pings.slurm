#!/bin/bash
#SBATCH --job-name=cf_finalize
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/01_2_finalize_pings_%j.out
#SBATCH --error=logs/01_2_finalize_pings_%j.err
# NOTICE: No --array flag here either!

set -euo pipefail

echo "================================================="
echo "01_2_finalize_pings.py  (Sequential Buckets 0-7)"
echo "Started: $(date)"
echo "Node   : $(hostname)"
echo "================================================="

source /lustre-flash/apps/spack/share/spack/setup-env.sh
#spack load python@3.13.5

CITYFLO_ROOT="/home/ride/25d0295/urban-mobility-platform/research/cityflo"

cd "${CITYFLO_ROOT}"
source .venv/bin/activate

# Loop through all 8 buckets sequentially inside this single SLURM job
for BUCKET in {0..7}; do
    echo ""
    echo "---> STARTING BUCKET ${BUCKET} <---"
    python scripts/01_2_finalize_pings.py --bucket_id "${BUCKET}"
done

echo ""
echo "Finished all 8 buckets successfully: $(date)"
