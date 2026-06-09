#!/bin/bash
# ===========================================================================
# run_generate_dataset.sbatch — SLURM wrapper for scripts/generate_dataset.py
# ===========================================================================
# Runs the EmoVecLLM sample-dataset build on a SLURM GPU node (e.g. Snellius).
# It drives generate_dataset.py in --non-interactive mode: every choice is read
# from EMOVEC_*/WANDB_* environment variables (defaults below), so the same job
# script reproduces a run with no prompts.
#
# Submit (defaults):
#     mkdir -p logs && sbatch scripts/run_generate_dataset.sbatch
#
# Override any knob at submit time without editing the file:
#     sbatch --export=ALL,EMOVEC_GENERATOR_MODEL=meta-llama/Llama-3.1-8B-Instruct,\
# EMOVEC_PRECISION=4bit,EMOVEC_BATCH_SIZE=8 scripts/run_generate_dataset.sbatch
#
# Resume is automatic: re-submitting skips job_ids already in stories.jsonl,
# so a job that hits the time limit just needs to be submitted again.
# ---------------------------------------------------------------------------
# NB: no trailing comments on #SBATCH lines — sbatch parses them as extra args.
#SBATCH --job-name=emovec-gen
# --partition: EDIT for your cluster (Snellius: gpu_a100 / gpu_h100)
#SBATCH --partition=gpu_a100
# --gpus: generate_dataset.py uses device_map=auto across all visible GPUs
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# --time: resume on re-submit if this isn't enough
#SBATCH --time=24:00:00
#SBATCH --output=logs/emovec-gen-%j.out
#SBATCH --error=logs/emovec-gen-%j.err
# To get notified, uncomment and set your address:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.org

set -euo pipefail

echo "==========================================================="
echo "job ${SLURM_JOB_ID:-?} on $(hostname)   $(date -Is)"
echo "==========================================================="
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "repo: $(pwd)"

# ── 1. Environment ─────────────────────────────────────────────────────────
# Pick ONE activation path for your cluster (edit as needed). Set EMOVEC_VENV
# to a venv dir, or EMOVEC_CONDA_ENV to a conda env name, at submit time.
if [[ -n "${EMOVEC_VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${EMOVEC_VENV}/bin/activate"
elif [[ -n "${EMOVEC_CONDA_ENV:-}" ]]; then
    # module load Anaconda3 2>/dev/null || true   # uncomment if conda needs a module
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${EMOVEC_CONDA_ENV}"
else
    # Fallback: load a module stack. EDIT to match your site.
    # module load 2023; module load Python/3.11.3-GCCcore-12.3.0; module load CUDA/12.1.1
    echo "(no EMOVEC_VENV / EMOVEC_CONDA_ENV set — using python on PATH)"
fi
echo "python: $(command -v python)  ($(python --version 2>&1))"

# ── 2. Secrets (never hard-code) ───────────────────────────────────────────
# HF_TOKEN     — for gated models (Llama-3). Export it in your submit env or
#                ~/.bashrc, e.g. `export HF_TOKEN=hf_xxx`.
# WANDB_API_KEY — for online wandb. Or `wandb login` once on the login node.
[[ -z "${HF_TOKEN:-}" ]] && echo "warn: HF_TOKEN unset (fine unless using a gated model)"
if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "warn: WANDB_API_KEY unset and WANDB_MODE=online — run 'wandb login' or set WANDB_MODE=offline"
fi

# ── 3. Run parameters (defaults; override via sbatch --export) ──────────────
# generate_dataset.py reads all of these; only set what you want to pin.
: "${EMOVEC_WORK_DIR:=$(pwd)}"                         # where data/ + logs/ live
: "${EMOVEC_GENERATOR_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${EMOVEC_PRECISION:=auto}"                          # auto|bf16|fp16|4bit|8bit
: "${EMOVEC_DEVICE_MAP:=auto}"
: "${EMOVEC_BATCH_SIZE:=8}"
: "${EMOVEC_MAX_JOBS:=0}"                              # 0 = all pending
: "${EMOVEC_RESUME:=1}"
: "${EMOVEC_BUILD_MANIFEST:=1}"                        # run nb02 if manifest absent
: "${WANDB_PROJECT:=emovecllm}"
: "${WANDB_MODE:=online}"                              # offline if compute nodes lack internet
export EMOVEC_WORK_DIR EMOVEC_GENERATOR_MODEL EMOVEC_PRECISION EMOVEC_DEVICE_MAP \
       EMOVEC_BATCH_SIZE EMOVEC_MAX_JOBS EMOVEC_RESUME EMOVEC_BUILD_MANIFEST \
       WANDB_PROJECT WANDB_MODE

echo "model=${EMOVEC_GENERATOR_MODEL}  precision=${EMOVEC_PRECISION}  batch=${EMOVEC_BATCH_SIZE}"
echo "work_dir=${EMOVEC_WORK_DIR}  wandb=${WANDB_MODE}/${WANDB_PROJECT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

# ── 4. Generate ────────────────────────────────────────────────────────────
# --non-interactive: no prompts; everything from the env above.
# EXTRA_ARGS lets you pass one-off flags, e.g. EXTRA_ARGS="--smoke-test".
srun python scripts/generate_dataset.py --non-interactive ${EXTRA_ARGS:-}

echo "done   $(date -Is)"

# If you ran with WANDB_MODE=offline, sync the run from a login node afterwards:
#     wandb sync wandb/offline-run-*
