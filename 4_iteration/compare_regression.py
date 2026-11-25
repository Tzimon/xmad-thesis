"""
xmad/4_iteration/compare_regression.py

Regression check of the Phase 4 harness: run P4-HPQ-001 (Qwen2.5-7B with
itself on HotpotQA, as P3-001) on the first n questions of the sample and
compare it question by question with the Phase 3 result.

    python phase4.py run --only P4-HPQ-001 --n 30
    python compare_regression.py            # -> agreement of EM, F1, answers, rounds

Same model, same prompts, same decoding, same questions: the results should be
identical. Small differences in single answers can come from GPU
non-determinism (Phase 3 replicates: <= 0.1 pp over 1000 questions); a
systematic difference (other EM, other questions, errors) points to a bug.
"""

import csv
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    csv.field_size_limit(10**9)
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): r for r in csv.DictReader(f)}


def main():
    p4 = glob.glob(str(HERE / "results" / "P4-HPQ-001_*_test" / "debate_results.csv")) \
        or glob.glob(str(HERE / "results" / "P4-HPQ-001_*" / "debate_results.csv"))
    p3 = glob.glob(str(HERE.parent / "3_iteration" / "results" / "P3-001_qwen2.5-7b__qwen2.5-7b" / "debate_results.csv"))
    if not p4 or not p3:
        sys.exit(f"missing results: Phase 4 {p4 or 'not found'}, Phase 3 {p3 or 'not found'}")
    new, old = rows(p4[0]), rows(p3[0])
    common = sorted(set(new) & set(old))
    if len(common) < len(new):
        print(f"WARNING: {len(new) - len(common)} Phase 4 questions are not in the Phase 3 sample -> sampling differs")
    same = {k: 0 for k in ("pred", "em", "source", "rounds")}
    diffs = []
    for i in common:
        a, b = new[i], old[i]
        for k in same:
            same[k] += a[k] == b[k]
        if a["pred"] != b["pred"] or a["em"] != b["em"]:
            diffs.append((i, b["pred"], a["pred"], b["em"], a["em"]))
    n = len(common)
    em_new = sum(int(new[i]["em"]) for i in common) / n
    em_old = sum(int(old[i]["em"]) for i in common) / n
    print(f"{n} common questions (Phase 4 file: {Path(p4[0]).parent.name})")
    print(f"EM on these questions: Phase 3 {em_old:.3f} | Phase 4 {em_new:.3f}")
    for k, v in same.items():
        print(f"  same {k:7}: {v}/{n}")
    errors = [i for i in common if new[i]["error"]]
    print(f"  errors in Phase 4: {len(errors)}")
    if diffs:
        print("\nquestions with a different answer (idx | Phase 3 | Phase 4 | EM3 | EM4):")
        for d in diffs:
            print(f"  {d[0]:5} | {d[1][:40]:40} | {d[2][:40]:40} | {d[3]} | {d[4]}")
    pending = sum(new[i]["source"] == "pending_judge" for i in common)
    if pending:
        print(f"\nnote: {pending} Phase 4 debates wait for the judge (Phase 3 rows are judged) -- "
              "compare EM only on consensus questions, or run `python phase4.py judge --only P4-HPQ-001` first")
    ok = not errors and len(diffs) <= max(1, n // 30)
    print("\nRESULT:", "OK -- harness reproduces Phase 3" if ok else "CHECK the differences above")


if __name__ == "__main__":
    main()
