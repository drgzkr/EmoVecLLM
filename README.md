# EmoVecLLM

> Open-source replication of Anthropic's April 2026 paper [*Emotion Concepts and their Function in a Large Language Model*](https://transformer-circuits.pub/2026/emotions), pushed across multiple open-weight LLM families and run on Google Colab.

---

## What's being replicated

Anthropic's paper extracts **171 emotion vectors** from Claude Sonnet 4.5's residual stream via difference-of-means between emotion-conditioned and neutral generations. The vectors:

- Reproduce the human valence–arousal circumplex (PC1 ↔ valence *r* = 0.81; PC2 ↔ arousal *r* = 0.66, vs. Warriner et al. 2013 norms).
- Causally steer behaviour: a small addition of the *desperate* direction takes blackmail rate 22 % → 72 %; coding reward-hacking 5 % → 70 %.
- Activate **locally** rather than as a persistent affective state — they track the *current generation window*.

This project replicates that pipeline on **open-source LLMs** with a model-agnostic adapter pattern, so the same difference-of-means / probing / steering code runs on Pythia, Llama-3, Qwen-2.5, GPT-2, etc.

---

## What's in here

A set of Jupyter notebooks (`notebooks/01_…` → `notebooks/10_…`) that step through the full replication pipeline.

| # | Notebook | Stage |
|---|---|---|
| 01 | `01_setup_and_models.ipynb` | Load a model via TransformerLens; extract residuals + attention; visualise per-layer / per-head structure |
| 02 | `02_emotion_word_list_and_prompts.ipynb` | Source the 171-emotion list; build emotion / neutral story prompts |
| 03 | `03_story_generation.ipynb` | Per-emotion story generation, cached |
| 04 | `04_activation_extraction.ipynb` | Probe a swappable target model → pooled features + **difference-of-means emotion vectors** (clustered) + per-story **per-step timeseries** (residual at each token × layer, from cumulatively reading the story) |
| 05 | `05_emotion_vectors.ipynb` | Figures: story/emotion coverage, vector geometry (PCA + cosine), **emotion "loading"** (emotion vector projected onto the per-step timeseries), and per-layer **token × feature** + **token × token** correlation panels |
| 06 | `06_validation_held_out.ipynb` | Held-out scoring on EmoBank / GoEmotions |
| 07 | `07_geometry_pca_clustering.ipynb` | PCA → align PC1/PC2 to Warriner valence/arousal; k-means clusters |
| 08 | `08_local_vs_global.ipynb` | Multi-turn vs current-window vector activation |
| 09 | `09_causal_steering.ipynb` | Additive interventions on three case studies |
| 10 | `10_model_comparison.ipynb` | Cross-model summary (Pythia / Llama-3 / Qwen-2.5) |

Notebooks 06–10 are scaffolds; **01–05** are fleshed out.

---

## Run on Colab — no local setup needed

- [![nb01](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/01_setup_and_models.ipynb) — **01** Setup & models
- [![nb02](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/02_emotion_word_list_and_prompts.ipynb) — **02** Emotion word list & prompts
- [![nb03](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/03_story_generation.ipynb) — **03** Story generation
- [![nb04](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/04_activation_extraction.ipynb) — **04** Activation extraction → vectors + timeseries
- [![nb05](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/05_emotion_vectors.ipynb) — **05** Emotion-vector geometry & loading figures

The first cell of each notebook handles installs (`pip install transformer-lens`), GPU detection, and a `WORK_DIR` for caching. For models that require a Hugging Face licence (Llama-3), set `HF_TOKEN` in **Colab Secrets** (Settings → Secrets) before running.

**Persistent output (nb02 → nb03 → nb04 → nb05).** The pipeline notebooks mount **Google Drive** by default on Colab and write under `/content/drive/MyDrive/EmoVecLLM/`, so each stage's artefact survives a runtime reset and is picked up by the next: nb02 writes `prompts.jsonl`; nb03 the resumable `stories/…/stories.jsonl`; nb04 the `features/{spec_hash}/{target_model}/` set (`segment_features.npz`, `emotion_vectors.npz`, and `cumulative_timeseries.npz` = per-story **per-step** residual `(seq, n_layers, d)`, the model's state after reading each prefix); nb05 the figures. Everything is overridable by environment variable for head-less / HPC runs (`EMOVEC_WORK_DIR`, `EMOVEC_MOUNT_DRIVE`, `EMOVEC_GENERATOR_MODEL`, `EMOVEC_TARGET_MODEL`, `EMOVEC_PRECISION`, `EMOVEC_BASELINE`, `EMOVEC_DEMO`, …), so the same notebooks run unchanged via `jupyter nbconvert --execute` or `papermill` on a cluster.

**Demo / development mode.** nb04 and nb05 default to `EMOVEC_DEMO=1`: they consume *whatever* stories are on disk (even a partial generation run), default to a tiny target model (`gpt2`), cap per-emotion stories, and still emit a full set of (rough) vectors and figures. Set `EMOVEC_DEMO=0` with a real target for the science run.

**Baseline / normalisation.** Emotion vectors are difference-of-means against a neutral baseline. nb04 treats any story of kind `neutral_dialogue` *or* `neutral_story` as baseline (`EMOVEC_BASELINE` = `neutral_mean` / `project_pcs` / `global_mean` / `none`), so switching from Anthropic's neutral **dialogues** to style-matched neutral **stories** needs only new rows in nb02's manifest — no code change. See nb04 §6 for the trade-offs.

### Compute footprint

| Model | T4 (free) | L4 / A100 (Pro) |
|---|---|---|
| `gpt2` (124 M) | fp16 ✅ | fp16 ✅ |
| `EleutherAI/pythia-1.4b` | fp16 ✅ | fp16 ✅ |
| `meta-llama/Llama-3.1-8B-Instruct` | 4-bit (bitsandbytes) ✅ | fp16 ✅ |
| `Qwen/Qwen2.5-7B-Instruct` | 4-bit ✅ | fp16 ✅ |

TransformerLens converts HF weights in-place, briefly peaking at ~2× model size. fp16 8B → ~32 GB transient; comfortable on A100, tight on L4. Default to 4-bit on smaller GPUs; promote to fp16 only when geometry / steering results justify it.

---

## Status

**Stimuli → generation → vectors online.** Notebooks 01–05 run end-to-end: 01 (model-loading + feature tour, ~30 s on T4), 02 (171 emotions × 100 topics → a frozen `prompts.jsonl` job manifest), 03 (swappable open-weight generator → emotion stories + neutral baseline, cached resumably to Drive), 04 (probe a swappable target model → pooled features, clustered difference-of-means emotion vectors, and per-story per-step feature timeseries — the residual at each token across layers, from cumulatively reading the story — demo-mode aware), and 05 (preliminary figures: coverage, vector geometry, and emotion-loading curves). Notebooks 06–10 are placeholders being filled in as the pipeline comes online.

---

## License & attribution

Independent academic replication. Models pulled from Hugging Face retain their original licences (Llama-3 is gated; Pythia / Qwen / Gemma / GPT-2 are open). The primary reference is the Anthropic paper at [transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions).

## Author

[Dora Gözükara](https://github.com/drgzkr) — PhD candidate, Donders Institute. The replication feeds into a downstream cognitive-neuroscience programme (naturalistic emotional fMRI encoding with Renée Visser's lab).
