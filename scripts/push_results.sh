#!/usr/bin/env bash
# ===========================================================================
# push_results.sh — get the generated data OFF the pod and into durable storage
# ===========================================================================
# Primary path: push to a Hugging Face Hub *dataset* repo (private), the
# canonical cross-platform store — nb04 can then pull it from any pod/Colab.
#
#     export HF_DATASET_REPO=drgzkr/emovecllm-stories   # or set in .env
#     bash scripts/push_results.sh
#
# Other ways to grab results are printed below if HF_DATASET_REPO is unset.
# Note: wandb already captures stories.jsonl as a versioned Artifact on each
# online run — this script is for the canonical dataset copy / large feature
# files that you don't want living only in wandb.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }

WORK_DIR="${EMOVEC_WORK_DIR:-$(pwd)}"
DATA="${WORK_DIR}/data/processed"
REPO="${HF_DATASET_REPO:-}"

if [[ ! -d "$DATA" ]]; then
    echo "no data at $DATA — run the generation first."; exit 1
fi

if [[ -z "$REPO" ]]; then
    cat <<EOF
HF_DATASET_REPO is unset, so nothing was uploaded. Pick one:

  1) Hugging Face Hub (recommended, durable, versioned):
       export HF_DATASET_REPO=<user>/emovecllm-stories
       export HF_TOKEN=hf_...        # write-scoped token
       bash scripts/push_results.sh

  2) RunPod built-in transfer (no setup, one-off):
       runpodctl send "$DATA"        # then run 'runpodctl receive <code>' on your laptop

  3) rsync over SSH (RunPod gives an SSH endpoint):
       rsync -avz -e ssh root@<pod-ip>:"$DATA" ./data_from_pod/
EOF
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN unset — need a write token to push to the Hub."; exit 1
fi

pip install -q huggingface_hub
echo "uploading $DATA  →  hf://datasets/$REPO/data/processed"
# Creates the (private) dataset repo on first push; re-runs upload only changes.
huggingface-cli upload "$REPO" "$DATA" "data/processed" \
    --repo-type dataset --token "$HF_TOKEN" --commit-message "EmoVecLLM data upload"
echo "done → https://huggingface.co/datasets/$REPO"
