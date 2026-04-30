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
| 04 | `04_activation_extraction.ipynb` | Hook → mean-pool → tensors on disk |
| 05 | `05_emotion_vectors.ipynb` | Difference-of-means → 171 emotion direction vectors per layer |
| 06 | `06_validation_held_out.ipynb` | Held-out scoring on EmoBank / GoEmotions |
| 07 | `07_geometry_pca_clustering.ipynb` | PCA → align PC1/PC2 to Warriner valence/arousal; k-means clusters |
| 08 | `08_local_vs_global.ipynb` | Multi-turn vs current-window vector activation |
| 09 | `09_causal_steering.ipynb` | Additive interventions on three case studies |
| 10 | `10_model_comparison.ipynb` | Cross-model summary (Pythia / Llama-3 / Qwen-2.5) |

Notebooks 02–10 are scaffolds; only **01** is currently fleshed out.

---

## Run on Colab — no local setup needed

[![Open Notebook 01 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/EmoVecLLM/blob/master/notebooks/01_setup_and_models.ipynb)

The first cell of each notebook handles installs (`pip install transformer-lens`), GPU detection, and a `WORK_DIR` for caching. For models that require a Hugging Face licence (Llama-3), set `HF_TOKEN` in **Colab Secrets** (Settings → Secrets) before running.

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

**Pre-implementation.** Notebook 01 — the model-loading + feature-extraction tour — is runnable end-to-end (~30 s on T4). Notebooks 02–10 are placeholders being filled in as the pipeline comes online.

---

## License & attribution

Independent academic replication. Models pulled from Hugging Face retain their original licences (Llama-3 is gated; Pythia / Qwen / Gemma / GPT-2 are open). The primary reference is the Anthropic paper at [transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions).

## Author

[Dora Gözükara](https://github.com/drgzkr) — PhD candidate, Donders Institute. The replication feeds into a downstream cognitive-neuroscience programme (naturalistic emotional fMRI encoding with Renée Visser's lab).
