#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=auracast-qwen2vl
#SBATCH --mem=120GB
#SBATCH --time=02:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#
# AuraCast — Qwen2-VL A100 verification job.
# Loads Qwen2-VL-7B-Instruct under FP16 on a single A100-80GB, runs 2
# synthetic images through it, asserts non-empty captions + logs latency
# and peak VRAM.
#
# First run will download ~15 GB of weights to ~/.cache/huggingface. Make
# sure HF_HOME points at scratch if your $HOME is /tmp on compute nodes:
#   export HF_HOME=/data/vision/beery/scratch/serena/.cache/huggingface
# (already exported by .bashrc if you set it earlier; otherwise this script
# does it defensively.)
#
#   sbatch scripts/slurm/verify_qwen2vl.sh
#
# For fast iteration without the 15 GB download, override the model:
#   sbatch --export=ALL,VD_QWEN_MODEL=Qwen/Qwen2-VL-2B-Instruct \
#       scripts/slurm/verify_qwen2vl.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp

# Default HF cache to scratch if not already set — /tmp/$USER/.cache evaporates.
export HF_HOME="${HF_HOME:-/data/vision/beery/scratch/serena/.cache/huggingface}"
mkdir -p "$HF_HOME"

echo "[verify-qwen2vl] hostname: $(hostname)"
echo "[verify-qwen2vl] HF_HOME: $HF_HOME"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

pip install -e . >/dev/null

MODEL="${VD_QWEN_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
echo "[verify-qwen2vl] model: $MODEL"

python -m auracast.scripts.verify_qwen2vl_a100 --model "$MODEL" --n-images 2

echo "[verify-qwen2vl] done at $(date -Iseconds)"
