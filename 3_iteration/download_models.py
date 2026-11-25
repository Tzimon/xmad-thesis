#!/usr/bin/env python3
"""
xmad/3_iteration/download_models.py

Automated download of the Phase 3 models (models.yaml) into the HF cache on
Lustre, one model after another, with timestamps, per-model timing, a
free-disk check before every model, retries, and a final status table. The
exit code is non-zero if any selected model failed, so the script can be
chained in a job.

Run it inside the vLLM venv (huggingface_hub is a vLLM dependency), ideally in
tmux, on a node whose route to the HF CDN works (Phase 0: the model CDN was
not throttled, the package index was):

    cd /path/to/workspace/xmad/3_iteration
    source /path/to/workspace/venvs/vllm/bin/activate
    export HF_HOME=/path/to/workspace/hf_cache
    export HF_TOKEN=...            # only needed for gated repos (meta-llama/*)

    python download_models.py --list                  # what is selected, with sizes from models.yaml
    python download_models.py --check                 # ask the Hub: repo exists? gated? real size?
    python download_models.py --priority must 2>&1 | tee download_must.log
    python download_models.py                         # everything in models.yaml
    python download_models.py --keys qwen3.5-27b gemma4-31b
    python download_models.py --vendor alibaba --tier L

Selection flags combine (intersection). --priority selects the models that
appear in runs.yaml at that priority or higher, plus the judge model.

Idempotent: files already in the cache are verified and skipped, so re-running
is safe and resumes interrupted downloads.
"""

import argparse
import datetime
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PRIORITIES = ["must", "should", "could"]

# never needed for vLLM serving
IGNORE_ALWAYS = ["*.gguf", "*.pth", "*.onnx", "*.msgpack", "*.h5", "original/*"]
# Mistral-format files: only skipped for models that are NOT served in Mistral
# format (for those, vLLM reads the HF-format safetensors instead)
IGNORE_NON_MISTRAL = ["consolidated*"]

LUSTRE = "/path/to/storage"
CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def free_gb(path):
    for p in (path, CACHE, "/"):
        try:
            return shutil.disk_usage(p).free / 1e9
        except Exception:
            continue
    return float("nan")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ignore_patterns(m):
    pats = list(IGNORE_ALWAYS)
    if not m.get("mistral_format"):
        pats += IGNORE_NON_MISTRAL
    pats += m.get("ignore", []) or []
    return sorted(set(pats))


def select(models, runs, args):
    keys = list(models)
    if args.priority:
        allowed = PRIORITIES[: PRIORITIES.index(args.priority) + 1]
        needed = {k for r in runs for k in (r["agent_a"], r["agent_b"])
                  if r["priority"] in allowed}
        needed |= {k for k, m in models.items() if m.get("role") == "judge"}  # judge is always needed
        keys = [k for k in keys if k in needed]
    if args.keys:
        unknown = set(args.keys) - set(models)
        if unknown:
            sys.exit(f"unknown model key(s): {sorted(unknown)}")
        keys = [k for k in keys if k in args.keys]
    if args.vendor:
        keys = [k for k in keys if models[k]["vendor"] in args.vendor]
    if args.tier:
        keys = [k for k in keys if models[k]["tier"] in args.tier]
    # small first: a slow or failing giant should not block the rest
    return sorted(keys, key=lambda k: models[k].get("disk_gb", 0))


def print_list(models, keys):
    total = 0
    print(f"{'key':28} {'tier':4} {'ffn':5} {'dtype':5} {'gpus':>4} {'~GB':>5}  hf_id")
    for k in keys:
        m = models[k]
        total += m.get("disk_gb", 0)
        flags = []
        if not m.get("hf_id_verified", False):
            flags.append("id-unverified")
        if m.get("gated"):
            flags.append("gated")
        print(f"{k:28} {m['tier']:4} {m['ffn']:5} {m['dtype']:5} {m['gpus']:>4} "
              f"{m.get('disk_gb', 0):>5}  {m['hf_id']}" + (f"   [{', '.join(flags)}]" if flags else ""))
    print(f"\n{len(keys)} models, ~{total} GB (models.yaml estimate), "
          f"free on {LUSTRE}: {free_gb(LUSTRE):.0f} GB")


def check_hub(models, keys):
    """Ask the Hub for each repo: does it exist, is it gated, how large are the
    files that would actually be downloaded (after the ignore patterns)."""
    from fnmatch import fnmatch
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    api = HfApi()
    ok = True
    grand = 0.0
    for k in keys:
        m = models[k]
        try:
            info = api.model_info(m["hf_id"], files_metadata=True)
            pats = ignore_patterns(m)
            size = sum((s.size or 0) for s in info.siblings
                       if not any(fnmatch(s.rfilename, p) for p in pats)) / 1e9
            grand += size
            gated = getattr(info, "gated", False)
            print(f"OK    {k:28} {m['hf_id']:50} {size:7.1f} GB"
                  + (f"  gated={gated}" if gated else ""))
        except GatedRepoError:
            print(f"GATED {k:28} {m['hf_id']:50} -> accept the license on HF and set HF_TOKEN")
            ok = False
        except RepositoryNotFoundError:
            print(f"MISS  {k:28} {m['hf_id']:50} -> repo id wrong or private; fix models.yaml")
            ok = False
        except Exception as e:
            print(f"ERR   {k:28} {m['hf_id']:50} -> {type(e).__name__}: {str(e)[:120]}")
            ok = False
    print(f"\ntotal to download: {grand:.0f} GB, free on {LUSTRE}: {free_gb(LUSTRE):.0f} GB")
    return ok


def download(models, keys, retries, force):
    from huggingface_hub import snapshot_download

    log("=" * 70)
    log(f"HF_HOME   = {CACHE}")
    log(f"models    = {len(keys)}")
    log(f"free disk = {free_gb(LUSTRE):.0f} GB on {LUSTRE}")
    log("=" * 70)

    results = {}
    grand_t0 = time.time()
    for i, k in enumerate(keys, 1):
        m = models[k]
        repo = m["hf_id"]
        need = m.get("disk_gb", 0) * 1.1
        free = free_gb(LUSTRE)
        if need and free == free and free < need and not force:   # free == free: not NaN
            results[k] = f"SKIP: needs ~{need:.0f} GB, {free:.0f} GB free (use --force)"
            log(f"[{i}/{len(keys)}] SKIP {k}: {results[k]}")
            continue
        if m.get("gated") and not os.environ.get("HF_TOKEN"):
            log(f"[{i}/{len(keys)}] WARN {k} is gated and HF_TOKEN is not set -- trying anyway")

        log(f"[{i}/{len(keys)}] downloading {k} ({repo}, ~{m.get('disk_gb', '?')} GB) ...")
        t0 = time.time()
        for attempt in range(1, retries + 1):
            try:
                path = snapshot_download(repo_id=repo, ignore_patterns=ignore_patterns(m))
                dt = (time.time() - t0) / 60.0
                results[k] = f"OK ({dt:.1f} min)"
                log(f"[{i}/{len(keys)}] OK   {k} in {dt:.1f} min -> {path}")
                break
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < retries:
                    wait = 30 * attempt
                    log(f"[{i}/{len(keys)}] attempt {attempt}/{retries} failed ({msg}); retry in {wait}s")
                    time.sleep(wait)
                else:
                    results[k] = f"FAIL: {msg}"
                    log(f"[{i}/{len(keys)}] FAIL {k}: {msg}")
        log(f"    free disk now: {free_gb(LUSTRE):.0f} GB")

    log("=" * 70)
    log(f"all done in {(time.time() - grand_t0) / 60.0:.1f} min")
    for k, status in results.items():
        log(f"  {k:28} {status}")
    log("=" * 70)
    return all(s.startswith("OK") for s in results.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", choices=PRIORITIES,
                    help="only models used by runs.yaml runs of this priority or higher")
    ap.add_argument("--keys", nargs="+", help="explicit model keys from models.yaml")
    ap.add_argument("--vendor", nargs="+", help="e.g. alibaba google mistral nvidia meta")
    ap.add_argument("--tier", nargs="+", choices=["S", "M", "L"])
    ap.add_argument("--list", action="store_true", help="print the selection and exit")
    ap.add_argument("--check", action="store_true", help="query the Hub (existence, gating, size) and exit")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="download even if the free-disk check fails")
    args = ap.parse_args()

    models = load_yaml(HERE / "models.yaml")["models"]
    runs = load_yaml(HERE / "runs.yaml")["runs"]
    keys = select(models, runs, args)
    if not keys:
        sys.exit("nothing selected")

    if args.list:
        print_list(models, keys)
        return
    if args.check:
        sys.exit(0 if check_hub(models, keys) else 1)
    sys.exit(0 if download(models, keys, args.retries, args.force) else 1)


if __name__ == "__main__":
    main()
