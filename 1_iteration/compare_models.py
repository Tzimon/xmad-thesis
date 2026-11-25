"""
xmad/1_iteration/compare_models.py

Reads every logs/baseline_<label>_*/summary.json produced by the sweep and builds
a cross-model comparison: quality (EM/F1), model load time, token totals, GPU
energy and its electricity cost (local), and the API-equivalent token cost. It
then marks which was cheaper per model.

    python compare_models.py

Writes logs/comparison.csv and logs/comparison.tex (LaTeX table).
"""

import csv
import glob
import json
import os
from pathlib import Path

USD_EUR = 0.86  # [please verify] current rate; see thesis footnote

# Commercial API list prices, USD per MILLION tokens (input, output).
# 7B and 72B are taken from provider listings; 0.5B and 32B are estimates.
# [please verify] all four before quoting in the thesis.
API_PRICE_USD = {
    "0.5B": (0.04, 0.04),   # [please verify]
    "7B":   (0.10, 0.20),   # OpenRouter Qwen2.5-7B
    "32B":  (0.18, 0.18),   # [please verify]
    "72B":  (0.36, 0.40),   # OpenRouter Qwen2.5-72B
}

ORDER = ["0.5B", "7B", "32B", "72B"]  # display order; unknown labels appended


def load_summaries(logs_dir):
    out = {}
    for path in glob.glob(str(logs_dir / "baseline_*" / "summary.json")):
        with open(path) as f:
            s = json.load(f)
        label = s.get("config", {}).get("label", Path(path).parent.name)
        out[label] = s  # last run of a label wins
    return out


def row_for(label, s):
    m = s["metrics"]; g = s.get("gpu") or {}; c = s["config"]
    pt = m["tokens"]["prompt_total"]; ct = m["tokens"]["completion_total"]
    pin, pout = API_PRICE_USD.get(label, (None, None))
    api_eur = None
    if pin is not None:
        api_usd = pt / 1e6 * pin + ct / 1e6 * pout
        api_eur = api_usd * USD_EUR
    local_eur = g.get("energy_cost_eur")
    cheaper = ""
    if api_eur is not None and local_eur is not None:
        cheaper = "local" if local_eur < api_eur else "API"
    return {
        "model": label,
        "n": m["questions"],
        "em": m["exact_match"],
        "f1": m["f1"],
        "load_s": c.get("load_time_s"),
        "lat_mean_s": m["latency_s"]["mean"],
        "tok_total": m["tokens"]["total"],
        "tok_prompt": pt,
        "tok_completion": ct,
        "gpu_energy_wh": g.get("energy_wh"),
        "gpu_power_mean_w": g.get("power_mean_w"),
        "local_cost_eur": local_eur,
        "api_cost_eur": None if api_eur is None else round(api_eur, 4),
        "cheaper": cheaper,
    }


def main():
    logs = Path("logs")
    summaries = load_summaries(logs)
    if not summaries:
        print("No summaries found under logs/baseline_*/summary.json"); return

    labels = [l for l in ORDER if l in summaries] + \
             [l for l in summaries if l not in ORDER]
    rows = [row_for(l, summaries[l]) for l in labels]

    # console
    print(f"\n{'model':>6} {'EM':>5} {'F1':>5} {'load_s':>7} {'lat_s':>6} "
          f"{'tokens':>11} {'kWh_Wh':>7} {'local_EUR':>9} {'API_EUR':>8} {'cheaper':>7}")
    for r in rows:
        print(f"{r['model']:>6} {r['em']:>5.3f} {r['f1']:>5.3f} "
              f"{(r['load_s'] or 0):>7.1f} {r['lat_mean_s']:>6.2f} "
              f"{r['tok_total']:>11,} {(r['gpu_energy_wh'] or 0):>7.1f} "
              f"{(r['local_cost_eur'] or 0):>9.4f} "
              f"{(r['api_cost_eur'] if r['api_cost_eur'] is not None else 0):>8.3f} "
              f"{r['cheaper']:>7}")

    # CSV
    cols = ["model", "n", "em", "f1", "load_s", "lat_mean_s", "tok_total",
            "tok_prompt", "tok_completion", "gpu_power_mean_w", "gpu_energy_wh",
            "local_cost_eur", "api_cost_eur", "cheaper"]
    with open(logs / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    # LaTeX
    def fmt(x, spec):
        return "--" if x is None else format(x, spec)
    with open(logs / "comparison.tex", "w") as f:
        f.write("\\begin{tabular}{@{}l r r r r r r r l@{}}\n\\toprule\n")
        f.write("Model & EM & F1 & Load (s) & Tokens & Energy (Wh) & "
                "Local (\\euro) & API (\\euro) & Cheaper \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"Qwen2.5-{r['model']} & {r['em']:.3f} & {r['f1']:.3f} & "
                    f"{fmt(r['load_s'], '.0f')} & {r['tok_total']:,} & "
                    f"{fmt(r['gpu_energy_wh'], '.1f')} & "
                    f"{fmt(r['local_cost_eur'], '.4f')} & "
                    f"{fmt(r['api_cost_eur'], '.3f')} & {r['cheaper']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"\n-> {logs/'comparison.csv'}\n-> {logs/'comparison.tex'}")
    print("\n[reminder] verify API_PRICE_USD (esp. 0.5B, 32B) and USD_EUR before quoting.")


if __name__ == "__main__":
    main()
