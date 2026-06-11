#!/usr/bin/env python3
"""Upload a results directory as a wandb artifact — safe for huge files on
RAM-starved pods.

`wandb artifact put` (and naive add_file loops) get OOM-killed on multi-GB
feature files: the client accumulates memory across adds within one process.
Here, files larger than --max-part-mb are stream-split into numbered
.partNNNN chunks (constant memory) and EVERY item is uploaded by a fresh
subprocess into one incremental artifact, so client memory resets per part.
Reassemble the parts after download with --reassemble.

    python scripts/push_artifact.py                      # default: the Qwen features dir
    python scripts/push_artifact.py <dir> --name my-art  # any directory
    python scripts/push_artifact.py --reassemble <dir>   # after `wandb artifact get`

Reads WANDB_* from .env if present (no need to `source` it first).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_DIR = "data/processed/features/f97f2b0c9968/Qwen_Qwen2.5-7B-Instruct"
CHUNK = 64 * 1024 * 1024  # streaming copy block size (64 MB)


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


def split_file(path, part_bytes, tmpdir):
    """Stream-split `path` into part files; returns the part paths."""
    parts = []
    with open(path, "rb") as src:
        i = 0
        while True:
            part = tmpdir / f"{path.name}.part{i:04d}"
            written = 0
            with open(part, "wb") as dst:
                while written < part_bytes:
                    buf = src.read(min(CHUNK, part_bytes - written))
                    if not buf:
                        break
                    dst.write(buf)
                    written += len(buf)
            if written == 0:
                part.unlink()
                break
            parts.append(part)
            i += 1
    return parts


def reassemble(root):
    """Join *.partNNNN groups under `root` back into whole files."""
    firsts = sorted(root.rglob("*.part0000"))
    if not firsts:
        print(f"nothing to reassemble under {root}")
        return
    for first in firsts:
        target = first.with_name(first.name[: -len(".part0000")])
        parts = sorted(first.parent.glob(target.name + ".part[0-9][0-9][0-9][0-9]"))
        print(f"  {target.name}  <-  {len(parts)} parts")
        with open(target, "wb") as dst:
            for part in parts:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, dst, CHUNK)
        for part in parts:
            part.unlink()
    print("reassembled ✓")


def upload_one(artifact_name, art_type, file_path, entry_name, project):
    """Child-process body: add ONE file to the (incremental) artifact."""
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    run = wandb.init(project=project, job_type="upload",
                     name=f"up-{Path(entry_name).name}"[:64])
    try:
        art = wandb.Artifact(artifact_name, type=art_type, incremental=True)
    except TypeError:  # very old wandb without incremental
        art = wandb.Artifact(artifact_name, type=art_type)
    art.add_file(str(file_path), name=entry_name, skip_cache=True)
    run.log_artifact(art)
    art.wait()  # block until this part is actually uploaded
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=DEFAULT_DIR,
                    help=f"directory to upload (default: {DEFAULT_DIR})")
    ap.add_argument("--name", default=None,
                    help="artifact name (default: derived from the path)")
    ap.add_argument("--type", default="dataset")
    ap.add_argument("--max-part-mb", type=int, default=256,
                    help="split files larger than this into parts (default 256)")
    ap.add_argument("--reassemble", action="store_true",
                    help="join .partNNNN files under PATH instead of uploading")
    # internal: child-process mode (one file per process to bound memory)
    ap.add_argument("--upload-one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--entry-name", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    load_dotenv()
    project = os.environ.get("WANDB_PROJECT", "emovecllm")

    if args.upload_one:
        upload_one(args.name, args.type, args.upload_one, args.entry_name, project)
        return

    root = Path(args.path)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if args.reassemble:
        reassemble(root)
        return

    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and ".part" not in p.suffix)
    if not files:
        raise SystemExit(f"no files under {root}")

    name = args.name or re.sub(r"[^A-Za-z0-9._-]", "_",
                               "-".join(root.parts[-2:]) + "-full")
    part_bytes = args.max_part_mb * 1024 * 1024

    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"artifact : {project}/{name} ({args.type})")
    print(f"files    : {len(files)}  ({total_mb:.0f} MB)")

    tmpdir = Path(tempfile.mkdtemp(prefix="artparts_", dir=root))
    try:
        items = []  # (path, entry_name)
        for p in files:
            rel = p.relative_to(root).as_posix()
            if p.stat().st_size > part_bytes:
                print(f"splitting {rel} ({p.stat().st_size / 1e6:.0f} MB)…")
                for part in split_file(p, part_bytes, tmpdir):
                    items.append((part, f"{rel}.{part.name.split('.')[-1]}"))
            else:
                items.append((p, rel))

        for i, (path, entry) in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {entry}  "
                  f"({path.stat().st_size / 1e6:.0f} MB)", flush=True)
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--upload-one", str(path), "--entry-name", entry,
                   "--name", name, "--type", args.type]
            for attempt in (1, 2, 3):
                if subprocess.run(cmd).returncode == 0:
                    break
                print(f"    attempt {attempt} failed — retrying…", flush=True)
            else:
                raise SystemExit(f"upload failed repeatedly: {entry}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    try:
        import wandb
        entity = wandb.Api().default_entity
    except Exception:
        entity = "<entity>"
    print(f"\ndone — download with:\n  wandb artifact get "
          f"\"{entity}/{project}/{name}:latest\" --root <dest>\n"
          f"then:  python scripts/push_artifact.py --reassemble <dest>")


if __name__ == "__main__":
    main()
