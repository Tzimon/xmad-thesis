"""
xmad/4_iteration/download_benchmarks.py

Downloads the Phase 4 benchmarks from the official releases of their authors
into data/<benchmark>/ and writes data/<benchmark>/SOURCE.json (URL, date,
sha256, number of questions), so every run can be traced to one file.

    python download_benchmarks.py            # both
    python download_benchmarks.py musique    # one of: musique, 2wiki

Sources (official GitHub repositories, read 2026-10-22):
  MuSiQue          github.com/StonyBrookNLP/musique, download_data.sh:
                   gdown --id 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -> musique_v1.0.zip  (CC BY 4.0)
  2WikiMultiHopQA  github.com/Alab-NII/2wikimultihop, README (update 2021-04-07):
                   https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip  (Apache-2.0 repo)
                   = the version with the fixed sentence segmentation

Run on the login node (needs internet). HotpotQA comes from Hugging Face as in
Phase 3 and is not handled here.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SOURCES = {
    "musique": {
        "repo": "https://github.com/StonyBrookNLP/musique",
        "gdrive_id": "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
        "zip": "musique_v1.0.zip",
        "want": ["musique_ans_v1.0_dev.jsonl", "musique_ans_v1.0_train.jsonl"],
        "license": "CC BY 4.0",
    },
    "2wiki": {
        "repo": "https://github.com/Alab-NII/2wikimultihop",
        "url": "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
        "zip": "data_ids_april7.zip",
        "want": ["dev.json", "train.json"],
        "license": "Apache-2.0 (repository license; no separate data license stated)",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_gdrive(file_id, dest):
    """Google Drive download via the gdown Python API. The official MuSiQue script
    calls `gdown --id <id>`; the `--id` option was removed in gdown 5, the API
    call below works with old and new versions."""
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    print(f"  gdown {getattr(gdown, '__version__', '?')}", flush=True)
    try:
        result = gdown.download(id=file_id, output=str(dest), quiet=False)
    except TypeError:                     # very old gdown without the id= argument
        result = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)
    if not result or not Path(dest).exists():
        sys.exit(f"gdown could not download {file_id}. Download it manually from "
                 f"https://drive.google.com/file/d/{file_id} and put it at {dest}.")


def count_questions(path):
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


def fetch(name):
    src, out = SOURCES[name], DATA / name
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / src["zip"]
    if zpath.exists() and not zipfile.is_zipfile(zpath):
        print(f"[{name}] {zpath.name} is incomplete or not a zip (e.g. an HTML error page) -- downloading again")
        zpath.unlink()
    if not zpath.exists():
        print(f"[{name}] downloading {src['zip']} ...", flush=True)
        if "gdrive_id" in src:
            download_gdrive(src["gdrive_id"], zpath)
        else:
            urllib.request.urlretrieve(src["url"], zpath)
        if not zipfile.is_zipfile(zpath):
            bad = zpath.with_suffix(".failed")
            zpath.replace(bad)
            sys.exit(f"[{name}] download did not return a zip file (kept as {bad.name} for inspection). "
                     f"Download it manually from {src.get('url') or 'https://drive.google.com/file/d/' + src['gdrive_id']} "
                     f"and put it at {zpath}, then run this script again.")
    tmp = out / "_unzipped"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    files = {}
    for want in src["want"]:
        hits = [p for p in tmp.rglob(want) if "__MACOSX" not in p.parts]
        if not hits:
            print(f"[{name}] WARNING: {want} not found in {src['zip']}")
            continue
        dest = out / want
        shutil.copy2(hits[0], dest)
        files[want] = {"from_archive_path": str(hits[0].relative_to(tmp)), "sha256": sha256(dest),
                       "questions": count_questions(dest)}
        print(f"[{name}] {want}: {files[want]['questions']} questions")
    shutil.rmtree(tmp, ignore_errors=True)
    (out / "SOURCE.json").write_text(json.dumps({
        "benchmark": name, "repository": src["repo"], "download": src.get("url") or f"gdrive:{src['gdrive_id']}",
        "archive": src["zip"], "archive_sha256": sha256(zpath), "license": src["license"],
        "downloaded": datetime.now().isoformat(timespec="seconds"), "files": files}, indent=2), encoding="utf-8")
    print(f"[{name}] -> {out}/SOURCE.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmarks", nargs="*", metavar="BENCHMARK",
                    help=f"one or more of {', '.join(SOURCES)} (default: all)")
    args = ap.parse_args()
    unknown = [b for b in args.benchmarks if b not in SOURCES]
    if unknown:
        ap.error(f"unknown benchmark(s) {unknown}; choose from {list(SOURCES)}")
    for b in args.benchmarks or list(SOURCES):
        fetch(b)
