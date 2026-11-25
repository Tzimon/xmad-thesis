"""Phase 4 figure for Chapter 5: gain in evidence recall against the gain in EM, one panel per benchmark.

Recall gain = recall_all_rounds of the mixed pair minus the larger of recall_all_rounds of its two models.
HotpotQA has no recall data. The Jaccard contrast is not drawn, since it is already in the figures
msq_07 and 2wk_07 of Chapter 4.

Input: analysis/{musique,2wiki}/tables/{01_models_homogeneous,02_pairs}.csv (read-only).
Output: images/ch5/recall_gain_vs_delta_em.png in the thesis folder.

    python plot_recall_gain_phase4.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "3_iteration"))
import analyze_phase3 as A  # noqa: E402

OUT = Path(r"C:\KIT\Claude_Masterarbeit\images\ch5\recall_gain_vs_delta_em.png")
BENCH = [("musique", "MSQ"), ("2wiki", "2WK")]
TYPE_C = {"DxD": "#3F7FBF", "DxM": "#8E6BB8", "MxM": "#E8914A"}
RING_C, ALL_C, REST_C = "#C0392B", "#333333", "#1B8A6B"
NAMES = {"llama3.1-8b": "Llama-3.1-8B", "llama3.3-70b": "Llama-3.3-70B", "qwen2.5-7b": "Qwen2.5-7B"}
L70, QWEN = frozenset(("llama3.1-8b", "llama3.3-70b")), frozenset(("qwen2.5-7b", "llama3.1-8b"))
# hand-placed labels of the two extreme pairs, kept off the trend lines: (dx, dy in points, ha, va)
LABEL_AT = {("musique", L70): (-13, 9, "right", "bottom"), ("musique", QWEN): (13, -14, "right", "top"),
            ("2wiki", L70): (0, -17, "center", "top"), ("2wiki", QWEN): (-13, 9, "right", "bottom")}


def rd(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load(bench):
    tables = HERE / "analysis" / bench / "tables"
    recall = {r["model"]: float(r["recall_all_rounds"]) for r in rd(tables / "01_models_homogeneous.csv")}
    pairs = []
    for r in rd(tables / "02_pairs.csv"):
        best = max(recall[r["agent_a"]], recall[r["agent_b"]])
        pairs.append(dict(a=r["agent_a"], b=r["agent_b"], type=r["pair_type"], sig=r["significant"] == "True",
                          gain=(float(r["recall_all_rounds"]) - best) * 100, dem=float(r["delta_em"]) * 100))
    return pairs


def trend(ax, x, y, color, ls, label):
    k, d = np.polyfit(x, y, 1)
    xx = np.array([min(x), max(x)])
    r = float(np.corrcoef(x, y)[0, 1])
    ax.plot(xx, k * xx + d, color=color, lw=2.4, ls=ls, zorder=2, label=f"{label}: r = {r:+.2f}".replace("-", "−"))
    return r


def main():
    plt = A._plt()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True, gridspec_kw={"wspace": 0.07})
    for ax, (bench, tag) in zip(axes, BENCH):
        pairs = load(bench)
        A._thesis_axes(ax, 2, 1)
        ax.xaxis.set_major_locator(MultipleLocator(2)); ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.axhline(0, color="#555555", lw=1.2, ls=":", zorder=1)
        ax.set_xlim(-0.5, 13); ax.set_ylim(-5, 10)
        for p in pairs:
            ax.scatter(p["gain"], p["dem"], s=95, color=TYPE_C[p["type"]], edgecolor="white", linewidth=0.9, zorder=3)
            if p["sig"]:
                ax.scatter(p["gain"], p["dem"], s=250, facecolor="none", edgecolor=RING_C, linewidth=2.2, zorder=4)
            at = LABEL_AT.get((bench, frozenset((p["a"], p["b"]))))
            if at:
                ax.annotate(f"{NAMES[p['a']]} +\n{NAMES[p['b']]}", (p["gain"], p["dem"]), xytext=at[:2],
                            textcoords="offset points", ha=at[2], va=at[3], fontsize=10, fontweight="bold",
                            color="#111111", linespacing=1.05, zorder=5)
        x = [p["gain"] for p in pairs]
        y = [p["dem"] for p in pairs]
        rest = [p for p in pairs if not p["sig"]]
        r_all = trend(ax, x, y, ALL_C, (0, (5, 2.5)), "all pairs")
        r_rest = trend(ax, [p["gain"] for p in rest], [p["dem"] for p in rest], REST_C, "-", "excl. significant pairs")
        ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95, edgecolor="#D0D3D8", handlelength=2.6,
                  prop={"weight": "bold", "size": 11})
        ax.set_title(f"{tag} Recall gain", fontsize=15)
        ax.set_xlabel("Evidence recall gain (pp)", fontsize=12.5)
        A._bold_ticks(fig, ax)
        print(bench, "n", len(pairs), "r all", round(r_all, 2), "r excl. significant", round(r_rest, 2),
              "gain>0", sum(v > 0 for v in x), "significant", len(pairs) - len(rest))
    axes[0].set_ylabel("ΔEM (pp)", fontsize=13)
    handles = [Line2D([], [], marker="o", ls="", ms=10, mfc=c, mec="white", label=t) for t, c in TYPE_C.items()]
    handles.append(Line2D([], [], marker="o", ls="", ms=15, mfc="none", mec=RING_C, mew=2.2, label="significant (BH)"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=4, fontsize=12, frameon=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
