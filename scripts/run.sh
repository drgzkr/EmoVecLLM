#!/usr/bin/env bash
# ===========================================================================
# run.sh — one-shot, SELF-STOPPING pipeline runner (lite test OR full run)
# ===========================================================================
# Runs the pipeline end-to-end (generate → extract → validate) and then STOPS
# THE POD ITSELF, so you never pay for an idle GPU you forgot to shut down. The
# pod is stopped on success AND on failure (via an EXIT trap), and a watchdog
# force-stops it if the run hangs past MAX_RUNTIME_MIN.
#   PROFILE=lite (default) → tiny plumbing test;  PROFILE=full → the real run.
# (`scripts/test_run.sh` is a back-compat shim that calls this.)
#
# Run it detached so an SSH drop can't kill it (the trap still stops the pod):
#     cp .env.example .env   # set WANDB_API_KEY, RUNPOD_API_KEY, AUTO_STOP=1
#     nohup bash scripts/run.sh > run.out 2>&1 &              # lite test
#     nohup env PROFILE=full bash scripts/run.sh > run.out 2>&1 &   # full run
#     tail -f run.out
#
# Knobs (env or .env):
#   PROFILE=lite|full                lite (default) = tiny plumbing test (~1-2 min);
#                                    full = the real run (Qwen-7B + pythia, all jobs)
#   TEST_MAX_JOBS=12                 generation jobs (lite 12 / full 0 = all)
#   TEST_GEN_MODEL=gpt2              generator (lite gpt2 / full Qwen2.5-7B-Instruct)
#   TEST_TARGET_MODEL=gpt2           probed model (lite gpt2 / full pythia-1.4b)
#   RUN_VALIDATION=1                 run nb06 at the end
#   AUTO_STOP=1                      stop the pod when done (THE whole point)
#   STOP_MODE=stop                   stop (keep volume) | remove (terminate pod)
#   MAX_RUNTIME_MIN                  watchdog cap; default per profile (lite=10,
#                                    full=240). Set to override BOTH profiles.
#   RUNPOD_API_KEY=...               needed for self-stop (RunPod Settings→API Keys)
#   RUNPOD_POD_ID                    auto-set by RunPod
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

# PROFILE=lite (default): tiny plumbing test — does every system work + does the
#   pod stop itself? gpt2 both stages, few jobs, short completions; ~1-2 min. The
#   stories are junk and the metrics near-chance — that's expected.
# PROFILE=full: the real run — Qwen-7B generator, pythia-1.4b target, ALL jobs,
#   spec-default completion length, full held-out set, demo off. Same auto-stop.
# PROFILE=extract: SKIP generation (stories must already exist); extract features
#   on TEST_TARGET_MODEL (default Qwen-7B) + validate. For probing a real model
#   on an already-generated dataset, incl. extra targets later.
PROFILE="${PROFILE:-lite}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
AUTO_STOP="${AUTO_STOP:-1}"
STOP_MODE="${STOP_MODE:-stop}"
SKIP_GENERATE=0
if [[ "$PROFILE" == "full" ]]; then
    TEST_MAX_JOBS="${TEST_MAX_JOBS:-0}"              # 0 = all pending jobs
    TEST_GEN_MODEL="${TEST_GEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    TEST_TARGET_MODEL="${TEST_TARGET_MODEL:-EleutherAI/pythia-1.4b}"
    MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-240}"
    (( MAX_RUNTIME_MIN < 240 )) && { echo "note: raising watchdog ${MAX_RUNTIME_MIN}→240m (full floor; ignoring a smaller .env value)"; MAX_RUNTIME_MIN=240; }
    export EMOVEC_DEMO=0                             # full coverage, no per-emotion cap
    # completion length + held-out size stay at spec / notebook defaults
elif [[ "$PROFILE" == "extract" ]]; then
    SKIP_GENERATE=1                                  # stories already exist; probe only
    TEST_TARGET_MODEL="${TEST_TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-120}"
    (( MAX_RUNTIME_MIN < 120 )) && { echo "note: raising watchdog ${MAX_RUNTIME_MIN}→120m (extract floor; ignoring a smaller .env value)"; MAX_RUNTIME_MIN=120; }
    export EMOVEC_DEMO=0                             # full coverage, no per-emotion cap
else
    TEST_MAX_JOBS="${TEST_MAX_JOBS:-12}"
    TEST_GEN_MODEL="${TEST_GEN_MODEL:-gpt2}"
    TEST_TARGET_MODEL="${TEST_TARGET_MODEL:-gpt2}"
    MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-10}"
    : "${EMOVEC_MAX_NEW_TOKENS:=128}"; export EMOVEC_MAX_NEW_TOKENS   # short completions
    : "${EMOVEC_N_HELDOUT:=50}";        export EMOVEC_N_HELDOUT       # small nb06 set
fi

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
trap 'exit 129' HUP    # closing a terminal w/o nohup still runs cleanup → self-stop

# Watchdog: force shutdown if the run hangs past the cap. The sentinel file is
# removed by cleanup, so an orphaned watchdog that fires *after* we've exited
# finds no sentinel and does nothing (never signals a since-reused PID).
WATCHDOG_SENTINEL="$(mktemp)"
( sleep $((MAX_RUNTIME_MIN * 60))
  [[ -e "$WATCHDOG_SENTINEL" ]] || exit 0
  echo "!! WATCHDOG: exceeded ${MAX_RUNTIME_MIN} min — forcing pod shutdown"
  STATUS="WATCHDOG TIMEOUT (${MAX_RUNTIME_MIN}m)"
  stop_pod ) &     # stop the pod directly — a busy foreground stage would otherwise
WATCHDOG_PID=$!     # defer a parent-signalling trap until that stage finishes

die() { STATUS="FAILED: $*"; echo "!! $*"; exit 1; }

# ── 0. Deps (skip if already present) ───────────────────────────────────────
banner "0. environment"
python -c "import torch, transformers" 2>/dev/null || {
    echo "installing requirements…"; pip install -q -r requirements.txt wandb || die "pip install failed"; }
python -c "import torch; print('cuda:', torch.cuda.is_available())" || die "torch import failed"
if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARN: WANDB_API_KEY unset & online — runs may not log. (set it in .env)"
fi

# ── 1. Generate ─────────────────────────────────────────────────────────────
if [[ "$SKIP_GENERATE" == "1" ]]; then
    banner "1. generate — SKIPPED (PROFILE=extract; using existing stories)"
else
    banner "1. generate ($TEST_MAX_JOBS jobs, $TEST_GEN_MODEL)"
    EMOVEC_BUILD_MANIFEST=1 python scripts/generate_dataset.py --non-interactive \
        --model "$TEST_GEN_MODEL" --max-jobs "$TEST_MAX_JOBS" || die "generation failed"
fi

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

banner "run complete ✓"
# EXIT trap → cleanup → stop_pod
