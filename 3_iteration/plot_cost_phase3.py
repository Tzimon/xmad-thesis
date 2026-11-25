"""Phase 3 quality-vs-cost figure (analysis/figures/10_quality_vs_cost.png), built like the
Phase 1 single-agent figure (thesis Fig. "Phase 1 Single-Agent Results"):

  (a) quality vs. GPU energy for the Qwen3.5 ladder: homogeneous pairs as EM / F1 lines
      (labelled with the model size), the mixed Qwen3.5 pairs as single points
  (b) EM and F1 per pair type (bars) with the mean GPU energy per run (dotted line)
  (c) cost per run per pair type: local electricity vs. API-equivalent

Only mixed pairs with reliable energy are used (cost_reliable in 02_pairs.csv).
API-equivalent needs analysis/tables/06_token_split.csv (token_split.py, run on the cluster)
and analysis/tables/07_api_prices.csv (USD per million input / output tokens per model);
without them panel (c) shows local electricity only.

    python plot_cost_phase3.py
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

import analyze_phase3 as A

HERE = Path(__file__).parent
TAB = HERE / "analysis" / "tables"
KWH_PRICE_EUR = 0.069          # identical to Phase 1 (1_iteration/merge_models.py) and summary.json
USD_EUR = 0.86                 # identical to Phase 1 (1_iteration/merge_models.py)
# GPU depreciation (hardware wear), GPUs only, straight-line over the service life at full use.
# ASSUMPTIONS, to be verified / sourced before quoting:
H200_PRICE_EUR = 30000.0       # purchase price of one NVIDIA H200 141 GB (author's assumption)
H200_LIFETIME_H = 60000.0      # service life in operating hours (author's assumption)
HW_EUR_PER_GPU_H = H200_PRICE_EUR / H200_LIFETIME_H
EM_C, F1_C, WH_C = "#2F80ED", "#14A38B", "#6C5CE7"      # colours of the Phase 1 figure
LOCAL_C, HW_C, API_C = "#14A38B", "#9FB8D8", "#E07050"
MIX_C = "#7B5EA7"
LADDER = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"]
SHORT = {"qwen3.5-4b": "4B", "qwen3.5-9b": "9B", "qwen3.5-27b": "27B", "qwen3.5-35b-a3b": "35B",
         "qwen3.5-122b-a10b": "122B", "qwen3.5-397b-a17b": "397B"}      # total parameters
TYPES = [("DxD", "DxD"), ("DxM", "DxM"), ("MxM", "MxM")]   # dense+dense, dense+MoE, MoE+MoE


def summary(run_id):
    return json.load(open(glob.glob(str(HERE / f"results/{run_id}_*/summary.json"))[0], encoding="utf-8"))


def load():
    pairs = [p for p in csv.DictReader(open(TAB / "02_pairs.csv", encoding="utf-8"))]
    homog = {r["model"]: r for r in csv.DictReader(open(TAB / "01_models_homogeneous.csv", encoding="utf-8"))}
    for f in glob.glob(str(HERE / "results/P3-*/summary.json")):
        s = json.load(open(f, encoding="utf-8"))
        if s.get("agent_a") == s.get("agent_b") and "_rep" not in f and "_test" not in f and s["agent_a"] in homog:
            homog[s["agent_a"]].update(f1=s["f1"], run_id=s["run_id"])
    for p in pairs:
        s = summary(p["run_id"])
        gpus = set(s["energy"]["gpus_agent_a"]) | set(s["energy"]["gpus_agent_b"])
        p.update(em=float(p["em_pair"]), f1=s["f1"], wh=float(p["energy_wh"]), tok=float(p["tokens_total"]),
                 reliable=p["cost_reliable"] == "True",
                 gpu_h=s["wall_time_s"]["total"] / 3600 * len(gpus))   # debate time x GPUs of the run
    return pairs, homog


def api_cost(run_id, a, b, split, prices):
    """API-equivalent EUR of one run, or None if a split or a price is missing."""
    r = split.get(run_id)
    if r is None or a not in prices or b not in prices:
        return None
    (ai, ao), (bi, bo) = prices[a], prices[b]
    usd = (int(r["a_prompt"]) * ai + int(r["a_completion"]) * ao +
           int(r["b_prompt"]) * bi + int(r["b_completion"]) * bo) / 1e6
    return usd * USD_EUR


def main():
    plt = A._plt()
    pairs, homog = load()
    split = {r["run_id"]: r for r in csv.DictReader(open(TAB / "06_token_split.csv"))} \
        if (TAB / "06_token_split.csv").exists() else {}
    prices = {r["model"]: (float(r["usd_in"]), float(r["usd_out"])) for r in csv.DictReader(open(TAB / "07_api_prices.csv"))
              if r.get("usd_in")} if (TAB / "07_api_prices.csv").exists() else {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"wspace": 0.48})

    # (a) Qwen3.5 ladder ----------------------------------------------------------------------
    ax = axes[0]
    A._thesis_axes(ax, 0.05, 0.025)
    lad = sorted((homog[m] for m in LADDER), key=lambda h: float(h["energy_wh"]))
    xs = [float(h["energy_wh"]) for h in lad]
    ax.plot(xs, [float(h["em_debate"]) for h in lad], "-o", color=EM_C, lw=2.6, ms=10,
            markeredgecolor="white", label="EM, homogeneous Qwen3.5", zorder=4)
    ax.plot(xs, [h["f1"] for h in lad], "--o", color=F1_C, lw=2.6, ms=10,
            markeredgecolor="white", label="F1, homogeneous Qwen3.5", zorder=4)
    # size labels at the EM points (as in the Phase 1 figure), placed by hand to avoid overlaps
    offs = {"qwen3.5-35b-a3b": (-6, 12, "right"), "qwen3.5-4b": (0, -14, "center"), "qwen3.5-9b": (10, -12, "left"),
            "qwen3.5-122b-a10b": (0, -16, "center"), "qwen3.5-27b": (0, -14, "center"),
            "qwen3.5-397b-a17b": (-8, 10, "right")}
    for h, x in zip(lad, xs):
        dx, dy, ha = offs[h["model"]]
        ax.annotate(SHORT[h["model"]], (x, float(h["em_debate"])), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom" if dy > 0 else "top", fontsize=10.5, fontweight="bold", color=EM_C, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    mixed = [p for p in pairs if p["agent_a"] in LADDER and p["agent_b"] in LADDER and p["reliable"]]
    ax.scatter([p["wh"] for p in mixed], [p["em"] for p in mixed], s=70, marker="D", color=MIX_C,
               edgecolor="white", linewidth=0.8, zorder=5, label=f"EM, mixed Qwen3.5 pair ({len(mixed)})")
    ax.scatter([p["wh"] for p in mixed], [p["f1"] for p in mixed], s=70, marker="D", facecolor="white",
               edgecolor=MIX_C, linewidth=1.6, zorder=5, label="F1, mixed Qwen3.5 pair")
    ax.set_ylim(0.55, 0.85); ax.set_xlim(0, 3100)
    far = max(mixed, key=lambda p: p["wh"])
    ax.annotate(f"{SHORT[far['agent_a']]} + {SHORT[far['agent_b']]}", (far["wh"], far["em"]), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top", fontsize=10, fontweight="bold", color=MIX_C)
    ax.set_xlabel("GPU energy per run (Wh)  →  cost", fontsize=12.5)
    ax.set_ylabel("EM / F1", fontsize=13)
    ax.set_title("(a)  Quality vs. energy, Qwen3.5 ladder", fontsize=14, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=10, frameon=False, ncol=2)
    A._bold_ticks(fig, ax)

    # (b) EM / F1 per pair type + energy ------------------------------------------------------
    rel = [p for p in pairs if p["reliable"]]
    groups = [(t, lab, [p for p in rel if p["pair_type"] == t]) for t, lab in TYPES]
    names = [f"{lab}\n({len(v)})" for _, lab, v in groups]
    x = np.arange(len(groups)); w = 0.36
    mean = lambda v, k: float(np.mean([p[k] for p in v]))
    ax = axes[1]
    A._thesis_axes(ax, 0.2, 0.1)
    for off, key, col, lab in ((-w / 2, "em", EM_C, "EM"), (w / 2, "f1", F1_C, "F1")):
        bars = ax.bar(x + off, [mean(v, key) for _, _, v in groups], w, color=col, label=lab, zorder=3)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("EM / F1", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, [mean(v, "wh") for _, _, v in groups], ":o", color=WH_C, lw=2.6, ms=10,
             markeredgecolor="white", label="Energy (Wh)", zorder=5)
    ax2.set_ylim(0, 3000); ax2.set_ylabel("GPU energy per run (Wh)", fontsize=12.5, color=WH_C)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_linewidth(1.8)
    ax2.tick_params(axis="y", width=1.6, labelsize=11, colors=WH_C)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=10, frameon=False, ncol=3)
    ax.set_title("(b)  Quality and energy per pair type", fontsize=14, loc="left")
    A._bold_ticks(fig, ax)

    # (c) local vs API-equivalent per pair type -----------------------------------------------
    ax = axes[2]
    A._thesis_axes(ax, 0.5, 0.25)
    for p in rel:
        p["eur_local"] = p["wh"] / 1000 * KWH_PRICE_EUR
        p["eur_api"] = api_cost(p["run_id"], p["agent_a"], p["agent_b"], split, prices)
    for p in rel:
        p["eur_hw"] = p["gpu_h"] * HW_EUR_PER_GPU_H
    loc = [mean(v, "eur_local") for _, _, v in groups]
    hw = [mean(v, "eur_hw") for _, _, v in groups]
    have_api = bool(split) and any(p["eur_api"] is not None for p in rel)
    api = [float(np.mean([p["eur_api"] for p in v if p["eur_api"] is not None])) for _, _, v in groups] if have_api else None
    xl = x - w / 2 if have_api else x
    ax.bar(xl, loc, w, color=LOCAL_C, label="Local electricity", zorder=3)
    bh = ax.bar(xl, hw, w, bottom=loc, color=HW_C, label="Local GPU depreciation", zorder=3)
    bars = []
    for b, l in zip(bh, loc):
        ax.text(b.get_x() + b.get_width() / 2, l + b.get_height(), f"{l + b.get_height():.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    if have_api:
        ba = ax.bar(x + w / 2, api, w, color=API_C, label="API-equivalent", zorder=3)
        for b in ba:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    top = max([l + h for l, h in zip(loc, hw)] + (api or []))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylabel("cost per run (€)", fontsize=13)
    ax.set_ylim(0, top * 1.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=10, frameon=False, ncol=2)
    ax.set_title("(c)  Local vs. API cost", fontsize=14, loc="left")
    A._bold_ticks(fig, ax)

    out = A.FIG / "10_quality_vs_cost.png"
    fig.savefig(out); plt.close(fig)

    # numbers for the text ---------------------------------------------------------------------
    print("(a) Qwen3.5 ladder, homogeneous:")
    for h in lad:
        print(f"   {SHORT[h['model']]:10s} EM {float(h['em_debate']):.3f} F1 {h['f1']:.3f} Wh {float(h['energy_wh']):.0f}")
    print(f"    mixed Qwen3.5 pairs: {len(mixed)}, EM {min(p['em'] for p in mixed):.3f}-{max(p['em'] for p in mixed):.3f}, "
          f"Wh {min(p['wh'] for p in mixed):.0f}-{max(p['wh'] for p in mixed):.0f}")
    print(f"GPU depreciation: {HW_EUR_PER_GPU_H:.3f} EUR per GPU-hour (H200 {H200_PRICE_EUR:.0f} EUR / {H200_LIFETIME_H:.0f} h)")
    for (t, lab, v), l, hh in zip(groups, loc, hw):
        print(f"(b/c) {lab:12s} n={len(v):2d} EM {mean(v,'em'):.3f} F1 {mean(v,'f1'):.3f} Wh {mean(v,'wh'):.0f} "
              f"GPU-h {mean(v,'gpu_h'):.2f} € elec {l:.3f} € hw {hh:.3f}"
              + (f" API € {np.mean([p['eur_api'] for p in v if p['eur_api'] is not None]):.3f}"
                 f" (n={sum(p['eur_api'] is not None for p in v)})" if have_api else ""))
    hw = [float(h["energy_wh"]) for h in homog.values()]
    allwh = sum(float(p["energy_wh"]) for p in pairs)
    print(f"homogeneous runs: {len(hw)}, total {sum(hw)/1000:.1f} kWh = {sum(hw)/1000*KWH_PRICE_EUR:.2f} EUR, "
          f"mean {np.mean(hw):.0f} Wh = {np.mean(hw)/1000*KWH_PRICE_EUR:.3f} EUR per run")
    print(f"mixed runs (46, incl. overstated): {allwh/1000:.1f} kWh; reliable 39: {sum(p['wh'] for p in rel)/1000:.1f} kWh")
    print("API-equivalent:", "included" if have_api else "MISSING (needs 06_token_split.csv and 07_api_prices.csv)")
    print("->", out)


if __name__ == "__main__":
    main()
