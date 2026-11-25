"""Split the tokens of every Phase 3 run into prompt (input) and completion (output) tokens.

The split is needed for API-equivalent prices (input and output tokens are priced differently,
as in Phase 1). It is only stored in the per-question session files (results/<run>/sessions/,
not in git), so this script runs on the cluster and writes one small CSV that can be committed:

    python token_split.py            # -> analysis/tables/06_token_split.csv

Rows are de-duplicated per question (last session file wins, as in the analysis). Judge tokens
are not in the session files and are not included.
"""
import csv
import glob
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "analysis" / "tables" / "06_token_split.csv"


def main():
    rows = []
    for run_dir in sorted(glob.glob(str(HERE / "results" / "P3-*"))):
        name = Path(run_dir).name
        if name.endswith("_test"):
            continue
        files = glob.glob(str(Path(run_dir) / "sessions" / "*.json"))
        if not files:
            continue
        tot = {k: 0 for k in ("a_prompt", "a_completion", "b_prompt", "b_completion")}
        for f in files:
            try:
                s = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            for r in s.get("rounds", []):
                for ag, key in (("agent_a", "a"), ("agent_b", "b")):
                    t = r.get(ag) or {}
                    tot[f"{key}_prompt"] += int(t.get("prompt_tokens") or 0)
                    tot[f"{key}_completion"] += int(t.get("completion_tokens") or 0)
        rows.append({"run_id": name.split("_")[0], "folder": name, "questions": len(files), **tot})
        print(f"{name:60s} {len(files):5d} q  in {tot['a_prompt'] + tot['b_prompt']:>12,}  "
              f"out {tot['a_completion'] + tot['b_completion']:>10,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", OUT, f"({len(rows)} runs)")


if __name__ == "__main__":
    main()
