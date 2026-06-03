#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=auracast-verify
#SBATCH --mem=60GB
#SBATCH --time=01:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#
# AuraCast — A100 verification job.
# Loads a lightweight CLIP model under FP16, runs a small synthetic-image
# batch through it, asserts shape/dtype/score-range, logs latency + VRAM.
#
#   sbatch scripts/slurm/verify_a100.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp

echo "[verify-a100] hostname: $(hostname)"
echo "[verify-a100] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# Editable install so changes to the source pick up without re-publish.
pip install -e . >/dev/null

echo "[verify-a100] running auracast.scripts.verify_clip_a100 ..."
python -m auracast.scripts.verify_clip_a100

echo "[verify-a100] done at $(date -Iseconds)"
