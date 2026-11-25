"""Phase 4 cost figure, built on the cost model of 3_iteration/plot_cost_phase3.py.

Local cost (electricity + GPU depreciation) vs. API-equivalent per run and pair type, one panel per benchmark
(counterpart of Phase 3 Fig. 10c). Output: analysis/merged/figures/cost_per_run.png and tables/M4_cost.csv.

Cost model: electricity 0.069 EUR/kWh, depreciation 0.50 EUR per GPU hour, GPU hours = debate time x GPUs of
the run, API-equivalent = list prices of analysis/tables/07_api_prices.csv of Phase 3 at 0.86 EUR/USD.
Judge cost is not included (separate pass, as in Phase 3). Means per run use only runs with reliable energy.

    python plot_cost_phase4.py
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
P3 = HERE.parent / "3_iteration"
sys.path.insert(0, str(P3))
import analyze_phase3 as A  # noqa: E402

KWH_EUR, USD_EUR, DEPR_EUR = 0.069, 0.86, 0.50
LOCAL_C, HW_C, API_C = "#14A38B", "#9FB8D8", "#E07050"      # as in Phase 3 Fig. 10
BENCH = [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue"), ("2wiki", "2WikiMultiHopQA")]
OUT = HERE / "analysis" / "merged"
GROUPS = ["Homog.", "DxD", "DxM", "MxM"]


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load():
    prices = {r["model"]: (float(r["usd_in"]), float(r["usd_out"]))
              for r in rd(P3 / "analysis/tables/07_api_prices.csv") if r["usd_in"]}
    p3_split = {r["run_id"]: r for r in rd(P3 / "analysis/tables/06_token_split.csv")}
    info = {"hotpotqa": rd(P3 / "analysis/tables/02_pairs.csv"),
            "musique": rd(HERE / "analysis/musique/tables/02_pairs.csv"),
            "2wiki": rd(HERE / "analysis/2wiki/tables/02_pairs.csv")}
    info = {b: {r["run_id"]: r for r in v} for b, v in info.items()}
    roots = {"hotpotqa": (P3 / "results", "P3-*"), "musique": (HERE / "results", "P4-MSQ-*"),
             "2wiki": (HERE / "results", "P4-2WK-*")}
    runs = []
    for b, (root, pat) in roots.items():
        for f in sorted(glob.glob(str(root / pat / "summary.json"))):
            if "_test" in f or "_rep" in f:
                continue
            s = json.load(open(f, encoding="utf-8"))
            gpus = set(s["energy"]["gpus_agent_a"]) | set(s["energy"]["gpus_agent_b"])
            a, bb = s["agent_a"], s["agent_b"]
            t = p3_split[s["run_id"]] if b == "hotpotqa" else s["tokens_split"]
            api = None
            if a in prices and bb in prices:
                api = (int(t["a_prompt"]) * prices[a][0] + int(t["a_completion"]) * prices[a][1] +
                       int(t["b_prompt"]) * prices[bb][0] + int(t["b_completion"]) * prices[bb][1]) / 1e6 * USD_EUR
            p = info[b].get(s["run_id"])
            r = dict(bench=b, group="Homog." if a == bb else p["pair_type"], wh=s["energy"]["total_wh"],
                     gpu_h=s["wall_time_s"]["total"] / 3600 * len(gpus), api=api,
                     reliable=p is None or p["cost_reliable"] == "True")
            r["elec"] = r["wh"] / 1000 * KWH_EUR
            r["depr"] = r["gpu_h"] * DEPR_EUR
            r["local"] = r["elec"] + r["depr"]
            runs.append(r)
    return runs


def totals(v):
    a = [r["api"] for r in v if r["api"] is not None]
    return dict(runs=len(v), kwh=sum(r["wh"] for r in v) / 1000, gpu_h=sum(r["gpu_h"] for r in v),
                elec=sum(r["elec"] for r in v), depr=sum(r["depr"] for r in v), api=sum(a), api_runs=len(a))


def per_group(runs, b, g):
    return [r for r in runs if r["bench"] == b and r["group"] == g and r["reliable"]]


def figure(plt, runs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True, gridspec_kw={"wspace": 0.08})
    w, top = 0.36, 0
    for ax, (b, name) in zip(axes, BENCH):
        A._thesis_axes(ax, 2, 1)
        gs = [per_group(runs, b, g) for g in GROUPS]
        x = np.arange(4)
        el = [np.mean([r["elec"] for r in v]) for v in gs]
        de = [np.mean([r["depr"] for r in v]) for v in gs]
        ap = [np.mean([r["api"] for r in v if r["api"] is not None]) for v in gs]
        ax.bar(x - w / 2, el, w, color=LOCAL_C, edgecolor="white", linewidth=1, label="Local electricity", zorder=3)
        ax.bar(x - w / 2, de, w, bottom=el, color=HW_C, edgecolor="white", linewidth=1, label="Local GPU depreciation", zorder=3)
        ax.bar(x + w / 2, ap, w, color=API_C, edgecolor="white", linewidth=1, label="API-equivalent", zorder=3)
        for xi, loc, api in zip(x, np.add(el, de), ap):
            ax.text(xi - w / 2, loc, f"{loc:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            ax.text(xi + w / 2, api, f"{api:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            top = max(top, loc, api)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({len(v)})" for g, v in zip(GROUPS, gs)], fontsize=12, fontweight="bold")
        ax.set_title(name, fontsize=15, loc="left")
        A._bold_ticks(fig, ax)
        if b == "musique":
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=11, frameon=False, ncol=3)
    axes[0].set_ylabel("cost per run (€)", fontsize=13)
    axes[0].set_ylim(0, top * 1.14)
    fig.savefig(OUT / "figures/cost_per_run.png")
    plt.close(fig)


def main():
    plt = A._plt()
    runs = load()
    tot = {b: totals([r for r in runs if r["bench"] == b]) for b, _ in BENCH}
    tot["phase4"] = totals([r for r in runs if r["bench"] != "hotpotqa"])
    tot["all"] = totals(runs)
    figure(plt, runs)
    with open(OUT / "tables/M4_cost.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        keys = ["runs", "kwh", "gpu_h", "elec", "depr", "api", "api_runs"]
        w.writerow(["scope"] + keys)
        for k, v in tot.items():
            w.writerow([k] + [round(v[c], 2) for c in keys])
        w.writerow([])
        w.writerow(["bench", "group", "n", "elec_eur", "depr_eur", "local_eur", "api_eur", "gpu_h"])
        for b, _ in BENCH:
            for g in GROUPS:
                v = per_group(runs, b, g)
                a = [r["api"] for r in v if r["api"] is not None]
                w.writerow([b, g, len(v)] + [round(float(np.mean([r[c] for r in v])), 3) for c in ("elec", "depr", "local")] +
                           [round(float(np.mean(a)), 3), round(float(np.mean([r["gpu_h"] for r in v])), 2)])
    for k, v in tot.items():
        print(k, {c: round(x, 1) for c, x in v.items()}, "local", round(v["elec"] + v["depr"], 1))
    for b, _ in BENCH:
        for g in GROUPS:
            v = per_group(runs, b, g)
            print(b, g, len(v), "elec", round(float(np.mean([r["elec"] for r in v])), 2), "local",
                  round(float(np.mean([r["local"] for r in v])), 2), "api",
                  round(float(np.mean([r["api"] for r in v if r["api"] is not None])), 2))


if __name__ == "__main__":
    main()
