#!/usr/bin/env bash
# ===========================================================================
# setup_pod.sh — one-shot bootstrap for a fresh cloud GPU pod (RunPod / etc.)
# ===========================================================================
# Run once after `git clone` on the pod:
#
#     cp .env.example .env      # then fill in WANDB_API_KEY (+ HF_TOKEN if gated)
#     bash scripts/setup_pod.sh
#
# It installs deps, loads .env, checks the GPU + keys, builds the prompt
# manifest if needed, and runs a tiny gpt2 smoke test. Then you launch the
# real run with `python scripts/generate_dataset.py`.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
echo "repo: $(pwd)"

# ── 1. Dependencies ─────────────────────────────────────────────────────────
echo "== installing requirements (this is the slow step) =="
pip install -q -r requirements.txt
pip install -q wandb huggingface_hub      # tracker + egress

# ── 2. Environment ──────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    set -a; source .env; set +a
    echo "loaded .env"
else
    echo "WARN: no .env — copy .env.example to .env and fill in your keys, then re-run."
fi

# ── 3. GPU + key checks ─────────────────────────────────────────────────────
echo "== GPU =="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  {p.total_memory/1e9:.0f} GB  × {torch.cuda.device_count()} GPU")
else:
    print("!! no CUDA visible — generation will be extremely slow")
PY

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARN: WANDB_API_KEY unset & WANDB_MODE=online — run 'wandb login' or set it in .env"
fi
[[ -z "${HF_TOKEN:-}" ]] && echo "note: HF_TOKEN unset (only needed for gated models / HF upload)"

# ── 4. Smoke test (tiny, fast, wandb off) ───────────────────────────────────
# Forces gpt2 + manifest build so this is quick regardless of .env settings.
echo "== smoke test: building manifest + generating 4 jobs on gpt2 =="
EMOVEC_SMOKE_TEST=1 EMOVEC_BUILD_MANIFEST=1 EMOVEC_GENERATOR_MODEL=gpt2 \
    EMOVEC_MAX_NEW_TOKENS=64 \
    python scripts/generate_dataset.py --non-interactive --wandb-mode disabled

cat <<'EOF'

bootstrap OK ✓

Pipeline on the pod:
    python scripts/generate_dataset.py     # 1. stories  (nb03) — interactive; add --non-interactive for head-less
    python scripts/extract_features.py     # 2. features (nb04) — emotion vectors + per-step timeseries

Get results off the pod afterwards:
    bash scripts/push_results.sh                       # git: commit text dataset to the repo (default)
    RESULTS_TARGET=hf HF_DATASET_REPO=you/emovecllm-stories bash scripts/push_results.sh   # incl. binary features
EOF
