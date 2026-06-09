#!/usr/bin/env bash
# ===========================================================================
# push_results.sh — get the generated data OFF the pod into durable storage
# ===========================================================================
# RESULTS_TARGET selects where (default: git):
#
#   git  — commit the small TEXT dataset (stories/prompts/spec/manifests) to
#          THIS repo and push. Binary feature .npz are gitignored (use hf/wandb).
#          Needs the repo cloned on the pod with push creds, OR run this from
#          your laptop after the data has synced down.
#
#   hf   — upload all of data/processed (incl. binary features) to a Hugging
#          Face Hub dataset repo. Set HF_DATASET_REPO + a write HF_TOKEN.
#
# Usage:
#     bash scripts/push_results.sh                       # git (default)
#     RESULTS_TARGET=hf HF_DATASET_REPO=you/emovecllm-stories bash scripts/push_results.sh
#
# Reminder: every ONLINE generation run already logs stories.jsonl as a versioned
# wandb Artifact, so this script is for the canonical copy / large feature files.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

WORK_DIR="${EMOVEC_WORK_DIR:-$(pwd)}"
DATA="${WORK_DIR}/data/processed"
TARGET="${RESULTS_TARGET:-git}"

[[ -d "$DATA" ]] || { echo "no data at $DATA — run the generation first."; exit 1; }

case "$TARGET" in
  git)
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -z "$REPO_ROOT" || "$DATA" != "$REPO_ROOT/data/processed" ]]; then
        cat <<EOF
git mode needs the data to live INSIDE this repo's working tree.
  data : $DATA
  repo : ${REPO_ROOT:-<not a git repo>}
Fix: clone the repo onto your persistent volume and leave EMOVEC_WORK_DIR blank
(so data/processed is the repo's own), or use RESULTS_TARGET=hf instead.
EOF
        exit 1
    fi
    echo "staging text dataset (binaries are gitignored)…"
    git add data/processed
    if git diff --cached --quiet; then echo "no new tracked data to commit."; exit 0; fi
    git commit -q -m "data: generation output ($(date -Is))"
    git push origin "$(git rev-parse --abbrev-ref HEAD)"
    echo "pushed text dataset to $(git remote get-url origin)"
    ;;
  hf)
    REPO="${HF_DATASET_REPO:-}"
    [[ -n "$REPO" ]]        || { echo "set HF_DATASET_REPO=<user>/emovecllm-stories"; exit 1; }
    [[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN unset — need a write token"; exit 1; }
    pip install -q huggingface_hub
    echo "uploading $DATA → hf://datasets/$REPO/data/processed"
    huggingface-cli upload "$REPO" "$DATA" "data/processed" \
        --repo-type dataset --token "$HF_TOKEN" --commit-message "EmoVecLLM data upload"
    echo "done → https://huggingface.co/datasets/$REPO"
    ;;
  *)
    echo "unknown RESULTS_TARGET='$TARGET' (use git|hf)"; exit 1 ;;
esac

# Manual one-off alternatives (no setup):
#   runpodctl send "$DATA"                                  # then 'runpodctl receive <code>'
#   rsync -avz -e ssh root@<pod-ip>:"$DATA" ./data_from_pod/
