#!/usr/bin/env python3
"""generate_dataset.py — interactive, wandb-tracked story generation for EmoVecLLM.

A head-less-friendly CLI port of ``notebooks/03_story_generation.ipynb`` for
running the sample-dataset build on a remote GPU server. It:

1. **Asks the operator** which generator model + decoding/runtime parameters to
   use (menu with sensible defaults; every prompt is skippable with
   ``--non-interactive`` and overridable by ``EMOVEC_*`` env vars / CLI flags).
2. **Logs every decision** to stdout, a timestamped ``logs/*.log`` file, and a
   ``run_config.json`` written next to the output, so a run is fully reproducible.
3. **Creates the dataset**: decodes the nb02 prompt manifest into
   ``stories/{spec_hash}/{model}/stories.jsonl`` in resumable batches.
4. **Tracks progress with Weights & Biases**: a run is initialised with the full
   config, per-batch throughput/coverage/leakage metrics are logged live, and a
   sample table + final coverage summary are recorded at the end.

Prerequisite: nb02's ``dataset_spec.json`` + ``prompts.jsonl`` must exist under
``WORK_DIR/data/processed`` (the script offers to build them via nbconvert if
missing). This file lives in ``scripts/`` and is *local-only* per the repo
``.gitignore`` — copy it to the remote box with ``rsync``/``scp``.

Examples
--------
Interactive (recommended first run on a fresh box)::

    python scripts/generate_dataset.py

Fully head-less (e.g. inside a SLURM/sbatch job), all knobs from flags/env::

    python scripts/generate_dataset.py --non-interactive \
        --model Qwen/Qwen2.5-7B-Instruct --precision 4bit \
        --batch-size 8 --wandb-project emovecllm

Quick wiring check (tiny model, few jobs, wandb offline)::

    python scripts/generate_dataset.py --non-interactive --smoke-test \
        --model gpt2 --wandb-mode offline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Torch / transformers are imported lazily in load_generator() so that --help,
# the manifest preflight, and the manifest build don't require the GPU stack.

# ── Model menu offered interactively ────────────────────────────────────────
MODEL_MENU = [
    ("Qwen/Qwen2.5-7B-Instruct",        "Apache-2.0, no gating — project default"),
    ("meta-llama/Llama-3.1-8B-Instruct", "gated — needs HF_TOKEN + accepted licence"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Apache-2.0 cross-family check"),
    ("EleutherAI/pythia-1.4b",          "small base model, fast prototyping"),
    ("gpt2",                            "tiny — smoke-test wiring only"),
]
PRECISION_CHOICES = ["auto", "bf16", "fp16", "4bit", "8bit"]


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════
class RunLogger:
    """Tee log lines to stdout and a timestamped file; collect decisions."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")
        self.decisions: dict = {}

    def log(self, msg: str = "") -> None:
        line = str(msg)
        print(line, flush=True)
        ts = datetime.now().strftime("%H:%M:%S")
        self._fh.write(f"{ts}  {line}\n")
        self._fh.flush()

    def decision(self, key: str, value, note: str = "") -> None:
        """Record a config decision and echo it prominently."""
        self.decisions[key] = value
        suffix = f"   ({note})" if note else ""
        self.log(f"  · {key:<18} = {value}{suffix}")

    def close(self) -> None:
        self._fh.close()


# ════════════════════════════════════════════════════════════════════════════
# Interactive parameter resolution
# ════════════════════════════════════════════════════════════════════════════
class Resolver:
    """Resolve each parameter by precedence: CLI flag > env var > prompt > default.

    In ``--non-interactive`` mode the prompt step is skipped (default/env/flag
    only), so the same code path runs both at a terminal and inside a batch job.
    """

    def __init__(self, args, logger: RunLogger):
        self.args = args
        self.log = logger
        self.interactive = not args.non_interactive and sys.stdin.isatty()
        if not self.interactive and not args.non_interactive:
            logger.log("(no TTY detected → running non-interactively)")

    def _ask(self, prompt: str, default: str) -> str:
        if not self.interactive:
            return default
        try:
            raw = input(f"{prompt} [{default}]: ").strip()
        except EOFError:
            raw = ""
        return raw or default

    def value(self, key, cli_val, env_name, default, note="") -> str:
        """A free-text/typed parameter."""
        if cli_val is not None:
            val, src = str(cli_val), "flag"
        elif os.environ.get(env_name):
            val, src = os.environ[env_name], "env"
        else:
            val = self._ask(f"{key}", str(default))
            src = "input" if self.interactive else "default"
        self.log.decision(key, val, note or src)
        return val

    def flag(self, key, cli_val, env_name, default: bool, note="") -> bool:
        if cli_val is not None:
            val = bool(cli_val)
        elif os.environ.get(env_name):
            val = os.environ[env_name].strip().lower() in ("1", "true", "yes", "on")
        elif self.interactive:
            ans = self._ask(f"{key}? (y/n)", "y" if default else "n")
            val = ans.strip().lower() in ("y", "yes", "1", "true")
        else:
            val = default
        self.log.decision(key, val, note)
        return val

    def choice(self, key, cli_val, env_name, options, default, note="") -> str:
        if cli_val is not None:
            val = str(cli_val)
        elif os.environ.get(env_name):
            val = os.environ[env_name]
        elif self.interactive:
            self.log.log(f"  {key} options: {', '.join(options)}")
            val = self._ask(key, default)
        else:
            val = default
        if val not in options:
            self.log.log(f"  ! '{val}' not in {options}; using default '{default}'")
            val = default
        self.log.decision(key, val, note)
        return val

    def pick(self, key, cli_val, env_name, menu, default, note="") -> str:
        """Menu picker: menu is list[(value, description)]; allows a custom paste."""
        if cli_val is not None:
            val = str(cli_val)
        elif os.environ.get(env_name):
            val = os.environ[env_name]
        elif self.interactive:
            self.log.log(f"\n{key} options:")
            for i, (v, d) in enumerate(menu, 1):
                self.log.log(f"  {i}. {v:<38} — {d}")
            self.log.log(f"  {len(menu)+1}. (enter a custom id)")
            raw = self._ask(f"choose 1-{len(menu)+1} or paste an id", "1")
            if raw.isdigit() and 1 <= int(raw) <= len(menu):
                val = menu[int(raw) - 1][0]
            elif raw.isdigit():
                val = self._ask("custom id", menu[0][0])
            else:
                val = raw
        else:
            val = default
        self.log.decision(key, val, note)
        return val

    def model(self) -> str:
        if self.args.model is not None:
            val = self.args.model
        elif os.environ.get("EMOVEC_GENERATOR_MODEL"):
            val = os.environ["EMOVEC_GENERATOR_MODEL"]
        elif self.interactive:
            self.log.log("\nWhich generator model?")
            for i, (name, desc) in enumerate(MODEL_MENU, 1):
                self.log.log(f"  {i}. {name:<38} — {desc}")
            self.log.log(f"  {len(MODEL_MENU)+1}. (enter a custom HF model id)")
            raw = self._ask("choose 1-%d or paste an id" % (len(MODEL_MENU) + 1), "1")
            if raw.isdigit() and 1 <= int(raw) <= len(MODEL_MENU):
                val = MODEL_MENU[int(raw) - 1][0]
            elif raw.isdigit():
                val = self._ask("custom HF model id", MODEL_MENU[0][0])
            else:
                val = raw  # pasted an id directly
        else:
            val = MODEL_MENU[0][0]
        self.log.decision("model", val)
        return val


# ════════════════════════════════════════════════════════════════════════════
# Work dir + manifest
# ════════════════════════════════════════════════════════════════════════════
def resolve_work_dir(args) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser().resolve()
    if os.environ.get("EMOVEC_WORK_DIR"):
        return Path(os.environ["EMOVEC_WORK_DIR"]).expanduser().resolve()
    # Default: repo root (this file is at <repo>/scripts/generate_dataset.py).
    return Path(__file__).resolve().parent.parent


def build_manifest(repo_root: Path, work_dir: Path, logger: RunLogger) -> None:
    """Run nb02 head-lessly via nbconvert to materialise spec + prompts.jsonl."""
    nb = repo_root / "notebooks" / "02_emotion_word_list_and_prompts.ipynb"
    if not nb.exists():
        raise SystemExit(f"cannot build manifest: {nb} not found")
    out_dir = Path("/tmp") / "emovec_nbexec"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EMOVEC_WORK_DIR=str(work_dir), EMOVEC_MOUNT_DRIVE="0",
               MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
           "--execute", "--ExecutePreprocessor.timeout=1200",
           "--output-dir", str(out_dir), str(nb)]
    logger.log(f"building manifest via: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise SystemExit("nb02 execution failed — build the manifest manually.")


def ensure_manifest(data_dir: Path, repo_root: Path, work_dir: Path,
                    resolver: Resolver, logger: RunLogger) -> None:
    spec_path = data_dir / "dataset_spec.json"
    prompts_path = data_dir / "prompts.jsonl"
    if spec_path.exists() and prompts_path.exists():
        return
    logger.log(f"\n! prompt manifest missing under {data_dir}")
    do_build = resolver.flag("build_manifest", resolver.args.build_manifest,
                             "EMOVEC_BUILD_MANIFEST", default=False,
                             note="run nb02 via nbconvert to create it")
    if not do_build:
        raise SystemExit(
            f"missing {spec_path.name} / {prompts_path.name}. Run nb02 first, or "
            f"re-run with --build-manifest.")
    build_manifest(repo_root, work_dir, logger)
    if not (spec_path.exists() and prompts_path.exists()):
        raise SystemExit("manifest still missing after nb02 — check its output.")


# ════════════════════════════════════════════════════════════════════════════
# Generation helpers (ported from nb03)
# ════════════════════════════════════════════════════════════════════════════
_HEADER_RE = re.compile(
    r"(?im)^[ \t]*\**\[?\s*(?:story|dialogue|example|passage)\s*#?\s*\d+\s*\]?\**"
    r"[ \t]*[:.)\-]?[ \t]*")
_NUM_RE = re.compile(r"(?im)^[ \t]*\d+[ \t]*[.)][ \t]+")


def split_segments(text: str):
    text = text.strip()
    for rx in (_HEADER_RE, _NUM_RE):
        parts = [p.strip() for p in rx.split(text) if p.strip()]
        if len(parts) >= 2:
            return parts
    chunks = [c.strip() for c in re.split(r"\n[ \t]*\n", text) if c.strip()]
    return chunks if len(chunks) >= 2 else [text]


def convert_dialogue_roles(text: str) -> str:
    text = re.sub(r"(?im)^[ \t]*Person[ \t]*:", "Human:", text)
    text = re.sub(r"(?im)^[ \t]*AI[ \t]*:", "Assistant:", text)
    return text


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def resolve_precision(pref: str, device: str, vram_gb: float) -> str:
    if pref != "auto":
        return pref
    if device == "cpu":
        return "fp32"
    return "4bit" if (vram_gb and vram_gb < 24) else "bf16"


def load_generator(name, precision, device_map, device, vram_gb, logger):
    """Model-agnostic loader; mirrors nb03 §3."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        BitsAndBytesConfig = None

    prec = resolve_precision(precision, device, vram_gb)
    hf_token = os.environ.get("HF_TOKEN") or None   # empty string → unauthenticated

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation pads on the left

    kw = dict(trust_remote_code=True, token=hf_token)
    if device == "cpu":
        kw.update(torch_dtype=torch.float32)
    elif prec in ("4bit", "8bit") and BitsAndBytesConfig is not None:
        kw.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=(prec == "4bit"),
                load_in_8bit=(prec == "8bit"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            device_map=device_map,
        )
    else:
        dtype = (torch.bfloat16 if prec == "bf16" and torch.cuda.is_bf16_supported()
                 else torch.float16)
        kw.update(torch_dtype=dtype, device_map=device_map)

    logger.log(f"loading {name} (precision={prec}) …")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    logger.log(f"loaded in {time.time() - t0:.0f}s")
    return tok, model, prec


# ════════════════════════════════════════════════════════════════════════════
# wandb wrapper (graceful no-op if unavailable / disabled)
# ════════════════════════════════════════════════════════════════════════════
class Wandb:
    def __init__(self, mode, project, entity, run_name, config, logger):
        self.run = None
        self.wb = None
        if mode == "disabled":
            logger.log("wandb: disabled")
            return
        try:
            import wandb
        except Exception:
            logger.log("wandb: not installed (`pip install wandb`) → continuing without it")
            return
        self.wb = wandb
        try:
            self.run = wandb.init(project=project, entity=entity or None,
                                  name=run_name, config=config, mode=mode)
            logger.log(f"wandb: {mode} run '{run_name}' → {getattr(self.run, 'url', '(offline)')}")
        except Exception as e:  # auth failure, network, etc. — never block the run
            logger.log(f"wandb: init failed ({e}) → continuing without it")
            self.run = None

    def log(self, metrics, step=None):
        if self.run is not None:
            self.wb.log(metrics, step=step)

    def summary(self, **kv):
        if self.run is not None:
            for k, v in kv.items():
                self.run.summary[k] = v

    def sample_table(self, recs, n=8):
        if self.run is None:
            return
        cols = ["kind", "emotion", "topic", "n_segments", "first_segment"]
        tbl = self.wb.Table(columns=cols)
        for r in recs[:n]:
            seg = (r["segments"][0] if r["segments"] else r["raw"])[:500]
            tbl.add_data(r["kind"], r.get("emotion"), r["topic"], r["n_segments"], seg)
        self.wb.log({"samples": tbl})

    def log_artifact(self, name, atype, files, metadata=None):
        """Version the generated dataset alongside the run (provenance + egress)."""
        if self.run is None:
            return
        art = self.wb.Artifact(name, type=atype, metadata=metadata or {})
        for f in files:
            if Path(f).exists():
                art.add_file(str(f))
        self.run.log_artifact(art)

    def finish(self):
        if self.run is not None:
            self.run.finish()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive, wandb-tracked EmoVecLLM dataset generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--non-interactive", action="store_true",
                   help="skip all prompts; use flags/env/defaults only")
    p.add_argument("--work-dir", help="WORK_DIR (default: EMOVEC_WORK_DIR or repo root)")
    # Generation knobs (None → fall back to env/prompt/default)
    p.add_argument("--model", help="generator HF model id (overrides spec)")
    p.add_argument("--precision", choices=PRECISION_CHOICES)
    p.add_argument("--device-map", help='HF device_map (default "auto")')
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-jobs", type=int, help="cap jobs this run (0 = all pending)")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--kinds", help="csv kind filter (e.g. emotion_story,neutral_dialogue)")
    p.add_argument("--smoke-test", dest="smoke_test", action="store_const", const=True,
                   default=None, help="tiny run to check wiring")
    p.add_argument("--no-resume", dest="resume", action="store_const", const=False,
                   default=None, help="ignore already-done job_ids")
    p.add_argument("--build-manifest", dest="build_manifest", action="store_const",
                   const=True, default=None, help="run nb02 if manifest missing")
    # wandb
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--no-wandb", action="store_true", help="alias for --wandb-mode disabled")
    return p.parse_args(argv)


def detect_device(logger):
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        ngpu = torch.cuda.device_count()
        logger.log(f"device: cuda ({name}, {vram:.0f} GB × {ngpu} GPU)")
        return "cuda", vram
    logger.log("device: cpu (no CUDA visible — generation will be slow)")
    return "cpu", 0.0


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    work_dir = resolve_work_dir(args)
    data_dir = work_dir / "data" / "processed"

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logger = RunLogger(work_dir / "logs" / f"generate_dataset_{stamp}.log")
    logger.log("=" * 72)
    logger.log(f"EmoVecLLM dataset generation — {stamp}")
    logger.log(f"work_dir : {work_dir}")
    logger.log("=" * 72)

    resolver = Resolver(args, logger)

    # ── Preflight: manifest from nb02 ───────────────────────────────────────
    ensure_manifest(data_dir, repo_root, work_dir, resolver, logger)
    spec = json.loads((data_dir / "dataset_spec.json").read_text())
    jobs = [json.loads(l) for l in (data_dir / "prompts.jsonl").read_text().splitlines()
            if l.strip()]
    spec_hash = spec["spec_hash"]
    dec = spec["decoding"]
    temperature, top_p = dec["temperature"], dec["top_p"]
    logger.log(f"\nmanifest : spec_hash={spec_hash}  jobs={len(jobs)}")
    for k, c in sorted(Counter(j["kind"] for j in jobs).items()):
        logger.log(f"  {k:<20} {c:>6}")

    # ── Gather decisions ────────────────────────────────────────────────────
    logger.log("\n── parameters ──")
    model = resolver.model() or spec["generator_model"]
    precision = resolver.choice("precision", args.precision, "EMOVEC_PRECISION",
                                PRECISION_CHOICES, "auto")
    device_map = resolver.value("device_map", args.device_map, "EMOVEC_DEVICE_MAP", "auto")
    batch_size = int(resolver.value("batch_size", args.batch_size, "EMOVEC_BATCH_SIZE", 8))
    max_jobs = int(resolver.value("max_jobs", args.max_jobs, "EMOVEC_MAX_JOBS", 0))
    max_new_tokens = int(resolver.value("max_new_tokens", args.max_new_tokens,
                                        "EMOVEC_MAX_NEW_TOKENS",
                                        dec["target_tokens_per_call"]))
    kind_filter = resolver.value("kinds", args.kinds, "EMOVEC_KINDS", "")
    smoke = resolver.flag("smoke_test", args.smoke_test, "EMOVEC_SMOKE_TEST", False)
    resume = resolver.flag("resume", args.resume, "EMOVEC_RESUME", True)

    if smoke:
        max_jobs = max_jobs or 4
        batch_size = min(batch_size, 2)
        logger.log(f"  (smoke-test → max_jobs={max_jobs}, batch_size={batch_size})")

    # ── wandb decisions ─────────────────────────────────────────────────────
    logger.log("\n── wandb ──")
    if args.no_wandb:
        wandb_mode = "disabled"
        logger.decision("wandb_mode", wandb_mode, "--no-wandb")
    else:
        wandb_mode = resolver.choice("wandb_mode", args.wandb_mode, "WANDB_MODE",
                                     ["online", "offline", "disabled"], "online")
    wandb_project = resolver.value("wandb_project", args.wandb_project,
                                   "WANDB_PROJECT", "emovecllm")
    wandb_entity = resolver.value("wandb_entity", args.wandb_entity, "WANDB_ENTITY", "")
    default_run = f"{safe_name(model)}-{spec_hash}-{stamp}"
    wandb_run_name = resolver.value("wandb_run_name", args.wandb_run_name,
                                    "WANDB_RUN_NAME", default_run)

    # ── Output path + resume ────────────────────────────────────────────────
    if kind_filter:
        keep = {k.strip() for k in kind_filter.split(",") if k.strip()}
        jobs = [j for j in jobs if j["kind"] in keep]

    out_dir = data_dir / "stories" / spec_hash / safe_name(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    stories_path = out_dir / "stories.jsonl"

    done_ids = set()
    if resume and stories_path.exists():
        for line in stories_path.read_text().splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["job_id"])
                except Exception:
                    pass
    pending = [j for j in jobs if j["job_id"] not in done_ids]
    if max_jobs:
        pending = pending[:max_jobs]

    logger.log(f"\noutput     → {stories_path}")
    logger.log(f"already done: {len(done_ids)}   pending now: {len(pending)} "
               f"(of {len(jobs)} filtered jobs)")

    # ── Persist the full run config for provenance ──────────────────────────
    run_config = {
        **logger.decisions,
        "model": model, "spec_hash": spec_hash, "work_dir": str(work_dir),
        "stories_path": str(stories_path), "temperature": temperature, "top_p": top_p,
        "n_jobs_filtered": len(jobs), "n_already_done": len(done_ids),
        "n_pending_this_run": len(pending), "started": stamp,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    logger.log(f"run config → {out_dir / 'run_config.json'}")

    if not pending:
        logger.log("\nnothing to do — all jobs already generated. Exiting.")
        logger.close()
        return

    # ── wandb init ──────────────────────────────────────────────────────────
    wb = Wandb(wandb_mode, wandb_project, wandb_entity, wandb_run_name, run_config, logger)

    # ── Load model ──────────────────────────────────────────────────────────
    import torch
    device, vram = detect_device(logger)
    tokenizer, model_obj, used_precision = load_generator(
        model, precision, device_map, device, vram, logger)
    input_device = model_obj.get_input_embeddings().weight.device
    wb.summary(used_precision=used_precision, device=device,
               input_device=str(input_device))

    # Clamp generation length to the model's context window (gpt2 etc. are 1024)
    # so prompt+new tokens can't overflow it — overflow throws a cryptic CUDA
    # device-side assert (position-embedding index out of range), not a clear error.
    _ctx = (getattr(model_obj.config, "n_positions", None)
            or getattr(model_obj.config, "max_position_embeddings", None))
    if _ctx and max_new_tokens > _ctx - 128:
        _clamped = max(16, _ctx - 256)
        logger.log(f"  clamping max_new_tokens {max_new_tokens} → {_clamped} "
                   f"(model context {_ctx})")
        max_new_tokens = _clamped

    # ── Generation closures (need tokenizer/model in scope) ─────────────────
    def build_chat_text(system_prompt, user):
        if getattr(tokenizer, "chat_template", None):
            for msgs in ([{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user}],
                         [{"role": "user", "content": system_prompt + "\n\n" + user}]):
                try:
                    return tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    continue
        return f"{system_prompt}\n\n{user}\n\n"

    @torch.no_grad()
    def generate_batch(texts, seeds):
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=4096).to(input_device)
        torch.manual_seed(int(seeds[0]) & 0x7fffffff)
        out = model_obj.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        n_new = int((gen != tokenizer.pad_token_id).sum().item())
        return tokenizer.batch_decode(gen, skip_special_tokens=True), n_new

    def record(job, completion):
        is_dialogue = job["kind"] in ("neutral_dialogue", "emotional_dialogue")
        segs = split_segments(completion)
        if is_dialogue:
            segs = [convert_dialogue_roles(s) for s in segs]
        return {
            "job_id": job["job_id"], "kind": job["kind"],
            "emotion": job.get("emotion"), "person_emotion": job.get("person_emotion"),
            "ai_emotion": job.get("ai_emotion"), "topic": job["topic"],
            "topic_idx": job["topic_idx"], "spec_hash": spec_hash,
            "generator_model": model, "n_segments": len(segs),
            "segments": segs, "raw": completion,
        }

    # ── Run ─────────────────────────────────────────────────────────────────
    logger.log(f"\ngenerating {len(pending)} jobs in batches of {batch_size} …")
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    written = seg_total = leak_total = leak_hits = tok_total = 0
    t0 = time.time()

    for bi, batch in enumerate(batches):
        texts = [build_chat_text(j["system_prompt"], j["user"]) for j in batch]
        seeds = [j["seed"] for j in batch]
        try:
            completions, n_new = generate_batch(texts, seeds)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            completions, n_new = [], 0
            for t, s in zip(texts, seeds):  # retry one-at-a-time
                c, nn = generate_batch([t], [s])
                completions += c
                n_new += nn

        with stories_path.open("a", encoding="utf-8") as fout:
            for job, comp in zip(batch, completions):
                rec = record(job, comp)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                seg_total += rec["n_segments"]
                if rec["kind"] == "emotion_story" and rec["emotion"]:
                    for seg in rec["segments"]:
                        leak_total += 1
                        if re.search(rf"\b{re.escape(rec['emotion'])}\b", seg, re.I):
                            leak_hits += 1
        tok_total += n_new

        elapsed = time.time() - t0
        jpm = written / max(elapsed, 1e-9) * 60
        tps = tok_total / max(elapsed, 1e-9)
        remaining = len(pending) - written
        eta_min = remaining / max(jpm, 1e-9)
        frac = (len(done_ids) + written) / max(len(jobs), 1)
        leak_rate = leak_hits / leak_total if leak_total else 0.0

        wb.log({
            "jobs_written": written, "jobs_pending": remaining,
            "fraction_complete": frac, "jobs_per_min": jpm,
            "tokens_per_sec": tps, "segments_total": seg_total,
            "leak_rate": leak_rate, "elapsed_min": elapsed / 60,
            "eta_min": eta_min, "batch": bi + 1,
        }, step=written)

        logger.log(f"  batch {bi+1}/{len(batches)}  written={written}/{len(pending)}  "
                   f"{jpm:.1f} jobs/min  {tps:.0f} tok/s  ETA {eta_min:.0f} min  "
                   f"leak={leak_rate:.1%}")

    dt = time.time() - t0
    logger.log(f"\nwrote {written} job outputs in {dt/60:.1f} min "
               f"({written/max(dt,1)*60:.1f} jobs/min)")

    # ── Summary + run manifest ──────────────────────────────────────────────
    recs = [json.loads(l) for l in stories_path.read_text().splitlines() if l.strip()]
    by_kind = Counter(r["kind"] for r in recs)
    n_seg = sum(r["n_segments"] for r in recs)
    done = len({r["job_id"] for r in recs})
    coverage = done / max(len(jobs), 1)

    run_manifest = {
        "spec_hash": spec_hash, "generator_model": model,
        "precision": used_precision, "stories_file": str(stories_path),
        "n_job_outputs": len(recs), "n_segments_total": n_seg,
        "by_kind": dict(by_kind), "n_jobs_in_manifest": len(jobs),
        "coverage": coverage, "updated": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    final_leak = leak_hits / leak_total if leak_total else 0.0
    wb.summary(coverage=coverage, n_job_outputs=len(recs), n_segments_total=n_seg,
               final_leak_rate=final_leak, runtime_min=dt / 60)
    wb.sample_table([r for r in recs if r["kind"] == "emotion_story"])
    wb.log_artifact(
        f"stories-{spec_hash}-{safe_name(model)}", "dataset",
        [stories_path, out_dir / "run_manifest.json", out_dir / "run_config.json"],
        metadata={"spec_hash": spec_hash, "model": model, "coverage": coverage,
                  "n_job_outputs": len(recs), "n_segments_total": n_seg})
    wb.finish()

    logger.log(f"\ncoverage: {done}/{len(jobs)} jobs ({coverage:.1%})  |  "
               f"{n_seg} segments  |  emotion-word leakage {final_leak:.1%}")
    logger.log(f"run manifest → {out_dir / 'run_manifest.json'}")
    logger.log("done.")
    logger.close()


if __name__ == "__main__":
    main()
