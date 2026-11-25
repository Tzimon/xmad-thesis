#!/usr/bin/env python3
"""Download the Llama tiers for Phase 2 into the HF cache, one after another,
with timestamps, per-model timing and free-disk logging.

These are UNGATED community mirrors of the official Meta weights (identical
weights, no access request / token needed). If you later get official access,
just swap the repo ids back to meta-llama/... .

Run inside your venv, ideally in tmux:

    cd /path/to/workspace/xmad
    source /path/to/workspace/venvs/vllm/bin/activate
    python download_llama.py 2>&1 | tee download_llama.log
    # detach tmux: Ctrl+b then d

Idempotent: cached files are verified and skipped; interrupted downloads resume.
"""
import os, time, shutil, datetime
from huggingface_hub import snapshot_download

# Ungated mirrors (identical weights to meta-llama/...).
#   8B  -> partner for Qwen2.5-7B
#   70B -> partner for Qwen2.5-72B (newest 70B: Llama 3.3)
MODELS = [
    "unsloth/Meta-Llama-3.1-8B-Instruct",   # ~16 GB   (== meta-llama/Llama-3.1-8B-Instruct)
    "unsloth/Llama-3.3-70B-Instruct",       # ~140 GB  (== meta-llama/Llama-3.3-70B-Instruct)
]

# skip formats we don't serve (mirrors may ship gguf / 4bit variants too)
IGNORE = ["*.gguf", "*.pth", "consolidated*", "original/*", "*bnb-4bit*", "*4bit*"]

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
    log(f"models    = {len(MODELS)}  (ungated mirrors, no token needed)")
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
            results[repo] = f"FAIL: {type(e).__name__}"
            log(f"[{i}/{len(MODELS)}] FAIL {repo}: {type(e).__name__}: {str(e)[:200]}")
        log(f"    free disk now: {free_gb(LUSTRE):.0f} GB")

    log("=" * 60)
    log(f"all done in {(time.time() - grand_t0) / 60.0:.1f} min")
    for repo, status in results.items():
        log(f"  {repo:40s} {status}")
    log("=" * 60)


if __name__ == "__main__":
    main()
