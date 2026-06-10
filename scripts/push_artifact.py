#!/usr/bin/env python3
"""Upload a results directory as a wandb artifact, one file at a time.

Lightweight replacement for `wandb artifact put`, which gets OOM-killed on
multi-GB feature files (it materialises the whole directory before upload).
This adds files individually with `skip_cache=True` (no local cache copy) so
memory and container-disk stay flat.

    python scripts/push_artifact.py                      # default: the Qwen features dir
    python scripts/push_artifact.py <dir> --name my-art  # any directory

Reads WANDB_* from .env if present (no need to `source` it first).
"""
import argparse
import os
import re
from pathlib import Path

DEFAULT_DIR = "data/processed/features/f97f2b0c9968/Qwen_Qwen2.5-7B-Instruct"


def load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=DEFAULT_DIR,
                    help=f"directory to upload (default: {DEFAULT_DIR})")
    ap.add_argument("--name", default=None,
                    help="artifact name (default: derived from the path)")
    ap.add_argument("--type", default="dataset")
    args = ap.parse_args()

    load_dotenv()
    import wandb  # after .env so WANDB_API_KEY is picked up

    root = Path(args.path)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no files under {root}")

    name = args.name or re.sub(r"[^A-Za-z0-9._-]", "_",
                               "-".join(root.parts[-2:]) + "-full")
    project = os.environ.get("WANDB_PROJECT", "emovecllm")

    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"artifact : {project}/{name} ({args.type})")
    print(f"files    : {len(files)}  ({total_mb:.0f} MB)")

    run = wandb.init(project=project, job_type="upload", name=f"upload-{name}")
    art = wandb.Artifact(name, type=args.type)
    for p in files:
        rel = p.relative_to(root).as_posix()
        print(f"  + {rel}  ({p.stat().st_size / 1e6:.0f} MB)")
        art.add_file(str(p), name=rel, skip_cache=True)
    run.log_artifact(art)
    art.wait()  # block until the upload is actually finished
    run.finish()
    print(f"\ndone — download with:\n  wandb artifact get "
          f"\"{run.entity}/{project}/{name}:latest\" --root <dest>")


if __name__ == "__main__":
    main()
