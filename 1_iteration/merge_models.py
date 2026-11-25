"""
xmad/1_iteration/merge_models.py

Chunk-aware cross-model comparison for the preemptible sweep.

A model may be run in several chunks (each chunk = one logs/baseline_<label>_*/
folder) because a low-priority job can be preempted and resumed. This script
groups the chunks by label, merges their per-question rows (deduplicated by idx),
recomputes EM/F1 over the whole set, sums tokens and GPU energy across the chunks,
and writes the comparison table.

Energy is integrated directly from each chunk's gpu_metrics.csv (so even a
preempted chunk without a summary.json still contributes its energy).

    python merge_models.py

Writes logs/comparison.csv and logs/comparison.tex.
"""

import csv, sys
csv.field_size_limit(10_000_000)

import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

USD_EUR = 0.86  # [please verify]
API_PRICE_USD = {           # USD per million tokens (in, out); [please verify]
    "0.5B": (0.04, 0.04),
    "7B":   (0.10, 0.20),
    "32B":  (0.18, 0.18),
    "72B":  (0.36, 0.40),
}
ORDER = ["0.5B", "7B", "32B", "72B"]
KWH_PRICE_EUR = 0.069        # [please verify] must match the sweep


def chunk_energy_wh(gpu_csv):
    """Integrate GPU energy (Wh) from one chunk's gpu_metrics.csv over active GPUs."""
    try:
        per = {}
        with open(gpu_csv) as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) < 7:
                    continue
                try:
                    idx = int(row[1]); mem = float(row[4]); power = float(row[6])
                    ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                except (ValueError, IndexError):
                    continue
                g = per.setdefault(idx, {"ts": [], "mem": [], "power": []})
                g["ts"].append(ts); g["mem"].append(mem); g["power"].append(power)
        if not per:
            return 0.0
        active = [i for i in per if max(per[i]["mem"]) > 2000] \
            or [max(per, key=lambda k: max(per[k]["mem"]))]
        ts0 = per[active[0]]["ts"]
        dur_s = (ts0[-1] - ts0[0]).total_seconds() if len(ts0) > 1 else 0.0
        sum_mean_power = sum(sum(per[i]["power"]) / len(per[i]["power"]) for i in active)
        return sum_mean_power * dur_s / 3600
    except FileNotFoundError:
        return 0.0


def collect(label):
    """Merge all chunks for one label. Returns a dict of merged metrics or None."""
    chunks = sorted(glob.glob(f"logs/baseline_{label}_*"))
    if not chunks:
        return None
    rows = {}          # idx -> row (dedup; later chunk wins on overlap)
    energy_wh = 0.0
    load_time = None
    for ch in chunks:
        energy_wh += chunk_energy_wh(Path(ch) / "gpu_metrics.csv")
        # load time: take the first available from any chunk's summary
        sp = Path(ch) / "summary.json"
        if load_time is None and sp.exists():
            try:
                load_time = json.load(open(sp))["config"].get("load_time_s")
            except Exception:
                pass
        cp = Path(ch) / "baseline_results.csv"
        if not cp.exists():
            continue
        with open(cp, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows[int(row["idx"])] = row
                except (ValueError, KeyError):
                    pass
    if not rows:
        return None
    items = list(rows.values())
    n = len(items)
    em = sum(int(r["em"]) for r in items) / n
    f1 = sum(float(r["f1"]) for r in items) / n
    pt = sum(int(r["prompt_tokens"]) for r in items)
    ct = sum(int(r["completion_tokens"]) for r in items)
    lat = sum(float(r["latency_s"]) for r in items) / n
    errs = Counter(r.get("error", "") for r in items if r.get("error"))
    return {
        "label": label, "n": n, "em": round(em, 4), "f1": round(f1, 4),
        "load_s": load_time, "lat_mean_s": round(lat, 4),
        "tok_prompt": pt, "tok_completion": ct, "tok_total": pt + ct,
        "energy_wh": round(energy_wh, 2),
        "energy_cost_eur": round(energy_wh / 1000 * KWH_PRICE_EUR, 5),
        "errors": sum(errs.values()), "error_rate": round(sum(errs.values()) / n, 4),
        "error_types": dict(errs),
        "chunks": len(chunks),
    }


def with_costs(r):
    pin, pout = API_PRICE_USD.get(r["label"], (None, None))
    api_eur = None
    if pin is not None:
        api_eur = (r["tok_prompt"] / 1e6 * pin + r["tok_completion"] / 1e6 * pout) * USD_EUR
    local = r["energy_cost_eur"]
    r["api_cost_eur"] = None if api_eur is None else round(api_eur, 4)
    r["local_cost_eur"] = local
    r["cheaper"] = "" if api_eur is None else ("local" if local < api_eur else "API")
    return r


def main():
    labels = [l for l in ORDER if glob.glob(f"logs/baseline_{l}_*")]
    labels += [m.group(1) for p in glob.glob("logs/baseline_*")
               if (m := re.match(r"logs/baseline_(.+?)_\d", p)) and m.group(1) not in labels]
    seen, uniq = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l); uniq.append(l)

    rows = [with_costs(collect(l)) for l in uniq if collect(l)]
    if not rows:
        print("No chunks found under logs/baseline_*/"); return

    print(f"\n{'model':>6} {'n':>5} {'EM':>6} {'F1':>6} {'load_s':>7} {'err%':>6} "
          f"{'tokens':>11} {'energy_Wh':>9} {'local_EUR':>9} {'API_EUR':>8} {'cheaper':>7} {'chunks':>6}")
    for r in rows:
        print(f"{r['label']:>6} {r['n']:>5} {r['em']:>6.3f} {r['f1']:>6.3f} "
              f"{(r['load_s'] or 0):>7.1f} {r['error_rate']*100:>5.1f}% "
              f"{r['tok_total']:>11,} {r['energy_wh']:>9.1f} "
              f"{(r['local_cost_eur'] or 0):>9.4f} "
              f"{(r['api_cost_eur'] if r['api_cost_eur'] is not None else 0):>8.3f} "
              f"{r['cheaper']:>7} {r['chunks']:>6}")

    cols = ["label", "n", "em", "f1", "load_s", "lat_mean_s", "errors", "error_rate",
            "tok_prompt", "tok_completion", "tok_total", "energy_wh",
            "local_cost_eur", "api_cost_eur", "cheaper", "chunks"]
    with open("logs/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    def fmt(x, s):
        return "--" if x is None else format(x, s)
    with open("logs/comparison.tex", "w") as f:
        f.write("\\begin{tabular}{@{}l r r r r r r r r l@{}}\n\\toprule\n")
        f.write("Model & $n$ & EM & F1 & Load (s) & Err.\\,\\% & Tokens & Energy (Wh) & "
                "Local (\\euro) & API (\\euro) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"Qwen2.5-{r['label']} & {r['n']} & {r['em']:.3f} & {r['f1']:.3f} & "
                    f"{fmt(r['load_s'], '.0f')} & {r['error_rate']*100:.1f} & "
                    f"{r['tok_total']:,} & {r['energy_wh']:.1f} & "
                    f"{fmt(r['local_cost_eur'], '.4f')} & {fmt(r['api_cost_eur'], '.3f')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\n-> logs/comparison.csv\n-> logs/comparison.tex")
    print("[reminder] verify API_PRICE_USD (0.5B, 32B), USD_EUR, KWH_PRICE_EUR before quoting.")


if __name__ == "__main__":
    main()
