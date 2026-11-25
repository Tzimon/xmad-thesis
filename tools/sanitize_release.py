"""Remove infrastructure details from a copy of the result files before publishing them.

Replaces given path prefixes, node names, the Slurm partition and Slurm job IDs in every
text file below DIR, in place. Run it only on a copy (see tools/pack_release.sh).
Afterwards it checks that none of the given names is left and exits with code 1 otherwise.

Usage:
    python3 tools/sanitize_release.py DIR \
        --replace /cluster/home/user /path/to/workspace \
        --replace /cluster /path/to/storage \
        --node-prefix node --partition gpu_partition
"""
import argparse
import os
import re
import sys


def text_files(root):
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            try:
                raw = open(p, "rb").read()
            except OSError:
                continue
            if b"\0" in raw[:4096]:
                continue
            try:
                yield p, raw.decode("utf-8")
            except UnicodeDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--replace", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
                    help="literal text to replace, e.g. a path prefix (repeatable)")
    ap.add_argument("--node-prefix", action="append", default=[],
                    help="node name prefix, NAME followed by digits becomes gpu-node (repeatable)")
    ap.add_argument("--partition", action="append", default=[],
                    help="Slurm partition name, also with suffixes, becomes <partition> (repeatable)")
    a = ap.parse_args()

    rules = []
    for old, new in sorted(a.replace, key=lambda r: -len(r[0])):  # longest prefix first
        rules.append((re.compile(re.escape(old)), new.replace("\\", r"\\")))
    for n in a.node_prefix:
        rules.append((re.compile(r"\b" + re.escape(n) + r"\d+\b"), "gpu-node"))
    for p in a.partition:
        rules.append((re.compile(r"\b" + re.escape(p) + r"\w*"), "<partition>"))
    # Slurm job IDs
    rules.append((re.compile(r'("slurm_job_id":\s*")\d+(")'), r"\1\2"))
    rules.append((re.compile(r"_job\d{4,}"), ""))
    rules.append((re.compile(r"\b([Jj][Oo][Bb][ =:])\d{4,}\b"), r"\1<id>"))

    hits = changed = 0
    for path, text in text_files(a.dir):
        n = 0
        for pat, rep in rules:
            text, k = pat.subn(rep, text)
            n += k
        if n:
            changed += 1
            hits += n
            open(path, "w", encoding="utf-8", newline="").write(text)
    print(f"changed {hits} occurrences in {changed} files")

    leftovers = [re.compile(re.escape(old)) for old, _ in a.replace]
    leftovers += [re.compile(r"\b" + re.escape(n) + r"\d+\b") for n in a.node_prefix]
    leftovers += [re.compile(re.escape(p)) for p in a.partition]
    bad = [path for path, text in text_files(a.dir) if any(r.search(text) for r in leftovers)]
    if bad:
        print("NOT CLEAN, names still found in:")
        for p in bad[:20]:
            print("  " + p)
        sys.exit(1)
    print("check passed: no given name left")


if __name__ == "__main__":
    main()
