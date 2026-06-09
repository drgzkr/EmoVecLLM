#!/usr/bin/env bash
# ===========================================================================
# test_run.sh — one-shot, SELF-STOPPING pilot of the full pipeline
# ===========================================================================
# Runs a small end-to-end test (generate → extract → validate) and then STOPS
# THE POD ITSELF, so you never pay for an idle GPU you forgot to shut down. The
# pod is stopped on success AND on failure (via an EXIT trap), and a watchdog
# force-stops it if the run hangs past MAX_RUNTIME_MIN.
#
# Run it detached so an SSH drop can't kill it (the trap still stops the pod):
#     cp .env.example .env   # set WANDB_API_KEY, RUNPOD_API_KEY, AUTO_STOP=1
#     nohup bash scripts/test_run.sh > test_run.out 2>&1 &
#     tail -f test_run.out
#
# Knobs (env or .env) — defaults are a tiny plumbing test (~1-2 min):
#   TEST_MAX_JOBS=12                 generation jobs for the pilot
#   TEST_GEN_MODEL=gpt2              tiny — no big download
#   TEST_TARGET_MODEL=gpt2           tiny — probed model
#   RUN_VALIDATION=1                 run nb06 at the end (small held-out set)
#   AUTO_STOP=1                      stop the pod when done (THE whole point)
#   STOP_MODE=stop                   stop (keep volume) | remove (terminate pod)
#   MAX_RUNTIME_MIN=10               watchdog hard cap (raise it if you point the
#                                    knobs at bigger models that need a download)
#   RUNPOD_API_KEY=...               needed for self-stop (RunPod Settings→API Keys)
#   RUNPOD_POD_ID                    auto-set by RunPod
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

# Defaults are deliberately TINY — this is a plumbing test (does every system
# work + does the pod stop itself?), not a science run. The stories will be junk
# and the metrics near-chance; that's expected. Run the full pipeline straight
# after with generate_dataset.py / extract_features.py (real models, no caps).
TEST_MAX_JOBS="${TEST_MAX_JOBS:-12}"
TEST_GEN_MODEL="${TEST_GEN_MODEL:-gpt2}"
TEST_TARGET_MODEL="${TEST_TARGET_MODEL:-gpt2}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
# Keep every stage fast: short completions + a small held-out set for nb06.
: "${EMOVEC_MAX_NEW_TOKENS:=128}"; export EMOVEC_MAX_NEW_TOKENS
: "${EMOVEC_N_HELDOUT:=50}";        export EMOVEC_N_HELDOUT
AUTO_STOP="${AUTO_STOP:-1}"
STOP_MODE="${STOP_MODE:-stop}"
MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-10}"

STATUS="OK"
_cleaned=0

banner() { echo; echo "============================================================"; echo "$*"; echo "============================================================"; }

stop_pod() {
    if [[ "$AUTO_STOP" != "1" ]]; then
        echo "AUTO_STOP=0 → leaving the pod running. STOP IT MANUALLY when done."
        return
    fi
    if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
        echo "!! RUNPOD_POD_ID unset (not on a RunPod pod?) → cannot self-stop. STOP MANUALLY."
        return
    fi
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "!! runpodctl not found → cannot self-stop pod $RUNPOD_POD_ID. STOP MANUALLY."
        return
    fi
    [[ -n "${RUNPOD_API_KEY:-}" ]] && runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
    banner "stopping pod $RUNPOD_POD_ID  (mode=$STOP_MODE, status=$STATUS)"
    if [[ "$STOP_MODE" == "remove" ]]; then
        runpodctl remove pod "$RUNPOD_POD_ID" || echo "!! remove failed — STOP MANUALLY."
    else
        runpodctl stop pod "$RUNPOD_POD_ID" || echo "!! stop failed — STOP MANUALLY."
    fi
}

cleanup() {
    [[ $_cleaned == 1 ]] && return
    _cleaned=1
    rm -f "${WATCHDOG_SENTINEL:-}" 2>/dev/null || true   # disarm the watchdog first
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        pkill -P "$WATCHDOG_PID" 2>/dev/null || true     # the inner sleep child
        kill "$WATCHDOG_PID" 2>/dev/null || true         # the subshell itself
    fi
    banner "TEST RUN FINISHED — status=$STATUS"
    echo "wandb: check your 'emovecllm' project for the runs + artifacts."
    echo "data : $(pwd)/data/processed   (validation.json under features/...)"
    stop_pod
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Watchdog: force shutdown if the run hangs past the cap. The sentinel file is
# removed by cleanup, so an orphaned watchdog that fires *after* we've exited
# finds no sentinel and does nothing (never signals a since-reused PID).
WATCHDOG_SENTINEL="$(mktemp)"
( sleep $((MAX_RUNTIME_MIN * 60))
  [[ -e "$WATCHDOG_SENTINEL" ]] || exit 0
  echo "!! WATCHDOG: exceeded ${MAX_RUNTIME_MIN} min — forcing shutdown"
  kill -TERM $$ 2>/dev/null ) &
WATCHDOG_PID=$!

die() { STATUS="FAILED: $*"; echo "!! $*"; exit 1; }

# ── 0. Deps (skip if already present) ───────────────────────────────────────
banner "0. environment"
python -c "import torch, transformers" 2>/dev/null || {
    echo "installing requirements…"; pip install -q -r requirements.txt wandb || die "pip install failed"; }
python -c "import torch; print('cuda:', torch.cuda.is_available())" || die "torch import failed"
if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARN: WANDB_API_KEY unset & online — runs may not log. (set it in .env)"
fi

# ── 1. Generate (small) ─────────────────────────────────────────────────────
banner "1. generate ($TEST_MAX_JOBS jobs, $TEST_GEN_MODEL)"
EMOVEC_BUILD_MANIFEST=1 python scripts/generate_dataset.py --non-interactive \
    --model "$TEST_GEN_MODEL" --max-jobs "$TEST_MAX_JOBS" || die "generation failed"

# ── 2. Extract features ─────────────────────────────────────────────────────
banner "2. extract features ($TEST_TARGET_MODEL)"
python scripts/extract_features.py --non-interactive \
    --target-model "$TEST_TARGET_MODEL" || die "extraction failed"

# ── 3. Validate (non-fatal: a dataset hiccup shouldn't block shutdown) ──────
if [[ "$RUN_VALIDATION" == "1" ]]; then
    banner "3. validate (nb06)"
    EMOVEC_TARGET_MODEL="$TEST_TARGET_MODEL" \
        jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
        --output-dir /tmp notebooks/06_validation_held_out.ipynb \
        || echo "WARN: validation failed (non-fatal) — features are still saved."
fi

banner "pilot complete ✓"
# EXIT trap → cleanup → stop_pod
