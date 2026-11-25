#!/usr/bin/env python3
"""Download a size ladder of open-weight models into the HF cache, one after
another, with timestamps, per-model timing and free-disk logging.

Run it inside your venv (so huggingface_hub is available), ideally in tmux:

    cd /path/to/workspace/xmad
    source /path/to/workspace/venvs/vllm/bin/activate
    python download_models.py 2>&1 | tee download.log
    # detach tmux with Ctrl+b then d ; reattach with: tmux attach -t dl

Idempotent: already-cached files are verified and skipped, so re-running is safe
and resumes interrupted downloads.
"""
import os, time, shutil, datetime
from huggingface_hub import snapshot_download

# One family, three sizes (Qwen2.5-Instruct: dense, tool-capable, vLLM 0.11.0-safe)
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",    # ~15 GB  (likely already cached)
    "Qwen/Qwen2.5-32B-Instruct",   # ~65 GB
    "Qwen/Qwen2.5-72B-Instruct",   # ~145 GB
]

# skip formats we don't serve, to avoid wasting space
IGNORE = ["*.gguf", "*.pth", "consolidated*", "original/*"]

LUSTRE = "/path/to/storage"
CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("nan")


def main():
    log("=" * 60)
    log(f"HF_HOME   = {CACHE}")
    log(f"models    = {len(MODELS)}")
    log(f"free disk = {free_gb(LUSTRE):.0f} GB on {LUSTRE}")
    log("=" * 60)

    results = {}
    grand_t0 = time.time()
    for i, repo in enumerate(MODELS, 1):
        log(f"[{i}/{len(MODELS)}] downloading {repo} ...")
        t0 = time.time()
        try:
            path = snapshot_download(repo_id=repo, ignore_patterns=IGNORE)
            dt = (time.time() - t0) / 60.0
            results[repo] = f"OK ({dt:.1f} min)"
            log(f"[{i}/{len(MODELS)}] OK  {repo}  in {dt:.1f} min")
            log(f"    -> {path}")
        except Exception as e:
            results[repo] = f"FAIL: {e}"
            log(f"[{i}/{len(MODELS)}] FAIL {repo}: {e}")
        log(f"    free disk now: {free_gb(LUSTRE):.0f} GB")

    log("=" * 60)
    log(f"all done in {(time.time() - grand_t0) / 60.0:.1f} min")
    for repo, status in results.items():
        log(f"  {repo:35s} {status}")
    log("=" * 60)


if __name__ == "__main__":
    main()
