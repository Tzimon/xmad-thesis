"""
xmad/1_iteration/plot_comparison.py

Renders the cross-model results as one modern three-panel figure for the thesis
(section 4.4.3). Reads logs/comparison.csv (produced by merge_models.py or
compare_models.py) and writes logs/model_comparison.pdf and .png.

    python plot_comparison.py [path/to/comparison.csv]

Panels:
  (a) quality-vs-cost trade-off    -- EM and F1 against GPU energy per run
  (b) quality and energy per model -- EM/F1 bars with energy overlaid (purple)
  (c) local vs. API cost           -- electricity cost vs. API-equivalent cost
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ORDER = ["0.5B", "7B", "32B", "72B"]
MODEL_COLOR = {
    "0.5B": "#E17055",   # coral
    "7B":   "#2D82F2",   # blue
    "32B":  "#12A594",   # teal
    "72B":  "#6C5CE7",   # indigo
}
C_EM, C_F1 = "#2D82F2", "#12A594"
C_LOCAL, C_API = "#12A594", "#E17055"
C_ENERGY = "#6C5CE7"                  # purple (kept)
INK, GRID = "#000000", "#CBD0D6"      # all text black; light grid only

ALIASES = {
    "energy_wh": ["energy_wh", "gpu_energy_wh"],
    "em": ["em"], "f1": ["f1"],
    "local_cost_eur": ["local_cost_eur"], "api_cost_eur": ["api_cost_eur"],
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "font.weight": "bold",
        "axes.edgecolor": INK, "axes.linewidth": 1.8,
        "axes.labelcolor": INK, "axes.labelsize": 14, "axes.labelweight": "bold",
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2,
        "axes.axisbelow": True, "figure.dpi": 120,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            lbl = (r.get("label") or r.get("model") or "").strip()
            if lbl:
                rows[lbl] = r
    models = [m for m in ORDER if m in rows] + [m for m in rows if m not in ORDER]

    def num(m, k):
        for key in ALIASES.get(k, [k]):
            v = rows.get(m, {}).get(key)
            if v not in (None, "", "--"):
                try:
                    return float(v)
                except ValueError:
                    pass
        return None
    return models, num


def vlabel(ax, xpos, y, text):
    if y is not None:
        ax.text(xpos, y, text, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/comparison.csv")
    if not path.exists():
        print(f"not found: {path}"); return
    style()
    models, num = load(path)
    x = list(range(len(models)))
    w = 0.38

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    energy = [num(m, "energy_wh") for m in models]
    em = [num(m, "em") for m in models]
    f1 = [num(m, "f1") for m in models]

    # ---- (a) quality vs cost trade-off ------------------------------------
    valid = [i for i in range(len(models)) if None not in (energy[i], em[i], f1[i])]
    order = sorted(valid, key=lambda i: energy[i])
    ex = [energy[i] for i in order]
    axA.plot(ex, [em[i] for i in order], "-", color=C_EM, lw=3, zorder=1)
    axA.plot(ex, [f1[i] for i in order], "--", color=C_F1, lw=3, zorder=1)
    for i in valid:
        axA.scatter(energy[i], em[i], s=170, color=C_EM, edgecolor="white",
                    linewidth=2, zorder=3)
        axA.scatter(energy[i], f1[i], s=170, color=C_F1, edgecolor="white",
                    linewidth=2, zorder=3)
        axA.annotate(models[i], (energy[i], max(em[i], f1[i])),
                     textcoords="offset points", xytext=(0, 13), ha="center",
                     fontsize=13, fontweight="bold",
                     color=MODEL_COLOR.get(models[i], INK))
    axA.set_xlabel("GPU energy per run (Wh)  \u2192  cost")
    axA.set_ylabel("score")
    axA.set_title("(a)  Quality vs. cost", loc="left")
    axA.set_ylim(0, 1)
    axA.plot([], [], "-", color=C_EM, lw=3, label="Exact match")
    axA.plot([], [], "--", color=C_F1, lw=3, label="F1")
    axA.legend(loc="lower right", frameon=True, framealpha=0.95,
               edgecolor=INK, prop={"weight": "bold", "size": 12})
    despine(axA)

    # ---- (b) quality + energy per model -----------------------------------
    emv = [v if v is not None else 0.0 for v in em]
    f1v = [v if v is not None else 0.0 for v in f1]
    axB.bar([i - w / 2 for i in x], emv, w, color=C_EM, label="Exact match")
    axB.bar([i + w / 2 for i in x], f1v, w, color=C_F1, label="F1")
    for i in x:
        vlabel(axB, i - w / 2, em[i], f"{em[i]:.2f}" if em[i] is not None else "")
        vlabel(axB, i + w / 2, f1[i], f"{f1[i]:.2f}" if f1[i] is not None else "")
    axB.set_xticks(x); axB.set_xticklabels(models)
    axB.set_ylabel("score"); axB.set_ylim(0, 1)
    axB.set_title("(b)  Quality and energy per model", loc="left")
    despine(axB)
    axBe = axB.twinx()
    ev = [v if v is not None else float("nan") for v in energy]
    axBe.plot(x, ev, "o-", color=C_ENERGY, lw=3, markersize=11,
              markeredgecolor="white", markeredgewidth=2, label="Energy (Wh)")
    axBe.set_ylabel("GPU energy (Wh)", color=C_ENERGY, fontweight="bold")
    axBe.tick_params(axis="y", colors=C_ENERGY, width=1.6, labelsize=13)
    for lbl in axBe.get_yticklabels():
        lbl.set_fontweight("bold")
    axBe.spines["top"].set_visible(False)
    axBe.spines["right"].set_color(C_ENERGY); axBe.spines["right"].set_linewidth(1.8)
    axBe.grid(False)
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBe.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, framealpha=0.95,
               edgecolor=INK, prop={"weight": "bold", "size": 12})

    # ---- (c) local vs API cost --------------------------------------------
    local = [num(m, "local_cost_eur") for m in models]
    api = [num(m, "api_cost_eur") for m in models]
    localv = [v if v is not None else 0.0 for v in local]
    apiv = [v if v is not None else 0.0 for v in api]
    axC.bar([i - w / 2 for i in x], localv, w, color=C_LOCAL, label="Local electricity")
    axC.bar([i + w / 2 for i in x], apiv, w, color=C_API, label="API-equivalent")
    for i in x:
        vlabel(axC, i - w / 2, local[i], f"{local[i]:.2f}" if local[i] is not None else "")
        vlabel(axC, i + w / 2, api[i], f"{api[i]:.2f}" if api[i] is not None else "")
    axC.set_xticks(x); axC.set_xticklabels(models)
    axC.set_ylabel("cost per run (\u20ac)")
    axC.set_title("(c)  Local vs. API cost", loc="left")
    axC.legend(loc="upper left", frameon=True, framealpha=0.95,
               edgecolor=INK, prop={"weight": "bold", "size": 12})
    despine(axC)

    fig.suptitle("Single-agent baseline across Qwen2.5 model sizes (7405 questions)",
                 fontsize=16, fontweight="bold", x=0.01, ha="left")

    for ext in ("pdf", "png"):
        out = path.parent / f"model_comparison.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
