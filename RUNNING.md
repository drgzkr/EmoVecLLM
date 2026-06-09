# Running the pipeline on a cloud GPU (RunPod)

How to run EmoVecLLM end-to-end on a rented GPU pod: a **lite plumbing test**
first (proves every system works and that the pod stops itself), then the
**full run**. Both stream progress to **Weights & Biases** and **auto-stop the
pod** when finished, so you never leave a GPU idling.

> TL;DR
> ```bash
> # one-time on the pod
> cd /workspace && git clone https://github.com/drgzkr/EmoVecLLM.git && cd EmoVecLLM
> cp .env.example .env && nano .env          # WANDB_API_KEY + RUNPOD_API_KEY
> bash scripts/setup_pod.sh
> # 1) lite test (~1-2 min, self-stops)
> nohup bash scripts/test_run.sh > test_run.out 2>&1 & tail -f test_run.out
> # 2) full run (self-stops)
> nohup env PROFILE=full bash scripts/test_run.sh > full_run.out 2>&1 & tail -f full_run.out
> ```

---

## 0. The pipeline

| Stage | Script | Notebook | Output |
|---|---|---|---|
| Stories | `scripts/generate_dataset.py` | nb03 | `data/processed/stories/{spec}/{model}/stories.jsonl` |
| Features | `scripts/extract_features.py` | nb04 | `data/processed/features/{spec}/{target}/*.npz` |
| Validation | nb06 (`nbconvert`) | nb06 | `features/.../validation.json` |

`scripts/test_run.sh` chains all three and stops the pod afterwards.
`scripts/setup_pod.sh` bootstraps a fresh pod. `scripts/push_results.sh` gets
data off the pod.

---

## 1. One-time accounts + keys

- **RunPod** — add prepaid credit. Get an **API key**: Settings → API Keys
  (needed so the pod can stop *itself*).
- **wandb** — get your **API key** from <https://wandb.ai/authorize>.
  *(There's no OAuth "linking" — putting the key in `.env` is the entire link.)*
- **Hugging Face** token — only if you use a **gated** model (e.g. Llama-3).
  Qwen / pythia / gpt2 are open and need none.

> Never paste these keys into chat or commit them. They live only in `.env` on
> the pod (which is gitignored).

---

## 2. Launch the pod

- **New Pod → RTX 4090** (~$0.4–0.7/hr; fits everything in 4-bit).
- Template: a **PyTorch** image.
- **Attach a Network Volume (~50 GB) mounted at `/workspace`** — Qwen-7B weights
  are ~15 GB, and this keeps data + model cache across restarts.
- Deploy → **Connect** (Web Terminal or SSH).

---

## 3. One-time pod setup

```bash
cd /workspace
git clone https://github.com/drgzkr/EmoVecLLM.git
cd EmoVecLLM
cp .env.example .env
nano .env
```

Fill in at least:

```ini
WANDB_API_KEY=<from wandb.ai/authorize>
RUNPOD_API_KEY=<RunPod Settings → API Keys>     # required for self-stop
AUTO_STOP=1                                       # stop the pod when a run finishes
STOP_MODE=stop                                    # 'stop' keeps the volume; 'remove' terminates
# HF_TOKEN=<only for gated models>
# EMOVEC_WORK_DIR=                                # leave blank: repo root on /workspace
```

Then bootstrap (installs deps, checks GPU + keys, builds the prompt manifest,
runs a gpt2 smoke test):

```bash
bash scripts/setup_pod.sh
```

A clean finish means the wiring is good.

---

## 4. The lite test (do this first)

Pure plumbing check — gpt2 for both stages, 12 jobs, short completions, small
held-out set. Runs in ~1–2 minutes and **stops the pod itself**.

```bash
nohup bash scripts/test_run.sh > test_run.out 2>&1 &
tail -f test_run.out
```

`nohup … &` runs it detached, so an SSH drop can't kill it (the auto-stop still
fires). What it exercises: wandb auth + live logging, generation, extraction,
nb06 validation (incl. the GoEmotions/EmoBank download), artifact logging, and
the **self-stop**.

**Success looks like:**
- runs appear under your `emovecllm` project on wandb;
- the log ends with `pilot complete ✓` then `stopping pod … runpodctl stop pod`;
- the pod transitions to *Stopped* in the RunPod console.

The stories will be gibberish and the metrics near-chance — **that's expected**;
you're testing the plumbing, not the science.

> If the pod does **not** stop: check `RUNPOD_API_KEY` is set in `.env` and that
> `runpodctl` exists on the image (`command -v runpodctl`). The log says exactly
> why if it couldn't self-stop.

---

## 5. The full run

Same script, `PROFILE=full`: Qwen2.5-7B generator, pythia-1.4b target, **all**
613 jobs, spec-default completion length, full held-out validation, demo off,
4-hour watchdog. Still auto-stops.

```bash
nohup env PROFILE=full bash scripts/test_run.sh > full_run.out 2>&1 &
tail -f full_run.out
```

First run downloads Qwen-7B (~15 GB, one-time on the volume). Expect roughly
**1–2 h on a 4090** for generation + a faster extraction pass; watch live on
wandb. It resumes automatically if re-run (done jobs are skipped).

**Prefer to drive the stages yourself** (more control, e.g. a 7–8B target)?
Run them directly in `tmux` and stop the pod when done:

```bash
tmux new -s emovec
python scripts/generate_dataset.py --non-interactive --model Qwen/Qwen2.5-7B-Instruct
python scripts/extract_features.py --non-interactive --target-model EleutherAI/pythia-1.4b --demo 0
EMOVEC_TARGET_MODEL=EleutherAI/pythia-1.4b \
    jupyter nbconvert --to notebook --execute --output-dir /tmp \
    notebooks/06_validation_held_out.ipynb
runpodctl stop pod "$RUNPOD_POD_ID"     # stop billing when finished
```

Either path: both CLIs are interactive if you drop `--non-interactive` (they'll
prompt for model + parameters).

---

## 6. Get the results out

Data lives on the `/workspace` volume (survives `STOP_MODE=stop`). To keep a
durable copy:

```bash
# A) commit the small TEXT dataset (stories/prompts/spec) to this repo — needs git push creds:
RESULTS_TARGET=git bash scripts/push_results.sh
# B) upload everything incl. binary features to a HF dataset repo:
RESULTS_TARGET=hf HF_DATASET_REPO=<you>/emovecllm-stories bash scripts/push_results.sh
# C) quick one-off grab to your laptop:
runpodctl send /workspace/EmoVecLLM/data/processed     # then 'runpodctl receive <code>' locally
```

Every **online** run also logs its outputs as a versioned **wandb Artifact**
(`stories-…` / `emotion-vectors-…`), so results are captured automatically even
if you skip the above.

---

## 7. Resuming / re-running

- **Generation** skips `job_id`s already in `stories.jsonl`, so re-running after
  a time-limit or crash continues where it stopped.
- **Extraction** re-reads the largest `stories.jsonl` and rebuilds the vectors.
- A pod that hit the **watchdog** just needs re-running (raise `MAX_RUNTIME_MIN`
  if a legitimate run needs longer than the cap).

---

## 8. Cost

At this scale the dataset is a few **euros**: a 4090 is ~$0.4–0.7/hr and the
full run is ~1–2 h. The lite test is a few cents. `STOP_MODE=stop` leaves a
small volume-storage charge; `STOP_MODE=remove` terminates the pod entirely
(you lose the volume unless detached).

---

## 9. Key environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `WANDB_API_KEY` | all | wandb auth (the RunPod↔wandb link) |
| `WANDB_PROJECT` / `WANDB_MODE` | all | project (`emovecllm`) / `online`·`offline`·`disabled` |
| `RUNPOD_API_KEY` | `test_run.sh` | lets the pod stop itself |
| `AUTO_STOP` / `STOP_MODE` | `test_run.sh` | self-stop on/off / `stop`·`remove` |
| `MAX_RUNTIME_MIN` | `test_run.sh` | watchdog hard cap |
| `PROFILE` | `test_run.sh` | `lite` (test) / `full` (real run) |
| `EMOVEC_WORK_DIR` | all | data root (blank = repo root) |
| `EMOVEC_GENERATOR_MODEL` / `EMOVEC_TARGET_MODEL` | gen / extract | which model |
| `EMOVEC_PRECISION` | gen / extract | `auto`·`bf16`·`fp16`·`4bit`·`8bit` |
| `EMOVEC_MAX_JOBS` / `EMOVEC_DEMO` | gen / extract | cap jobs / demo caps |
| `HF_TOKEN` | gen / extract | gated models only |
| `RESULTS_TARGET` / `HF_DATASET_REPO` | `push_results.sh` | `git`·`hf` / HF dataset |

See `.env.example` for the full annotated list.
