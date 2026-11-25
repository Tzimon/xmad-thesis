"""
xmad/3_iteration/analyze_phase3.py

Evaluation of Phase 3. Reads the finished runs in results/ and writes everything
into analysis/ -- figures, tables and a short report. Runs locally (laptop),
needs only numpy, matplotlib and pyyaml:

    pip install numpy matplotlib pyyaml
    python analyze_phase3.py

Output (analysis/README.md explains every file):

    analysis/
      README.md                 what is where, how to read the figures
      REPORT.md                 the results in numbers and sentences (generated)
      figures/                  heatmaps per model group + summary plots
      tables/                   the same numbers as CSV (for the thesis tables)

Central measure (Phase 3 plan):
    ΔEM(A,B) = EM(A+B) - max(EM(A+A), EM(B+B))
i.e. a mixed pair is compared with the BETTER of its two homogeneous pairs.
A cell counts as a real difference only if McNemar's exact test on the paired
per-question answers is significant AFTER Benjamini-Hochberg correction over all
mixed pairs (q < 0.05; with ~40 tests about two p < 0.05 are expected by chance)
AND |ΔEM| exceeds the run-to-run noise measured with the replicate runs.
Uncorrected p < 0.05 is reported separately as "nominal".

If debates still wait for the judge, every output is marked PRELIMINARY,
because those debates count as wrong until the judge has decided them.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "analysis"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Model groups shown as one heatmap each: (file stem, title, models in display order).
# Dense models first, then MoE, each by size -- so the quadrants of a heatmap read
# dense x dense (top left), dense x MoE (off-diagonal blocks), MoE x MoE (bottom right).
MATRICES = [
    ("01_cross_vendor_30B", "~30B, three vendors",
     ["qwen3.5-27b", "gemma4-31b", "muse-glimmer-30b", "gemma4-26b-a4b", "qwen3.5-35b-a3b"]),
    ("02_qwen35_ladder", "Qwen3.5 size ladder",
     ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b", "qwen3.5-35b-a3b", "qwen3.5-122b-a10b", "qwen3.5-397b-a17b"]),
    ("03_large_models", "Large models, three vendors",
     ["llama3.3-70b", "mistral-small4-119b", "qwen3.5-122b-a10b"]),
    ("04_gemma4_ladder", "Gemma 4 size ladder",
     ["gemma4-e4b", "gemma4-12b", "gemma4-31b", "gemma4-26b-a4b"]),
    ("05_mistral_ladder", "Mistral size ladder",
     ["ministral3-8b", "ministral3-14b", "mistral-small4-119b"]),
]
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(p):
    csv.field_size_limit(10**9)
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def folder(run):
    name = f"{run['run_id']}_{run['label']}"
    return name + (f"_rep{run['replicate']}" if run.get("replicate", 1) > 1 else "")


def recompute(s, rows, duplicates):
    """Quality metrics from the (deduplicated) per-question rows, so that the
    analysis never depends on a summary.json that may contain duplicates."""
    s = dict(s)
    n = len(rows) or 1
    num = lambda v: float(v) if v not in ("", None) else 0.0
    jac = [num(x["r1_jaccard"]) for x in rows if x["r1_jaccard"] not in ("", None)]
    s.update({
        "questions": len(rows),
        "exact_match": round(sum(num(x["em"]) for x in rows) / n, 4),
        "f1": round(sum(num(x["f1"]) for x in rows) / n, 4),
        "consensus_rate": round(sum(1 for x in rows if x["source"] == "consensus") / n, 4),
        "judge_rate": round(sum(1 for x in rows if x["source"] in ("judge", "pending_judge")) / n, 4),
        "judge_pending": sum(1 for x in rows if x["source"] == "pending_judge"),
        "evidence_jaccard_round1": round(sum(jac) / len(jac), 4) if jac else None,
        "rounds_mean": round(sum(num(x["rounds"]) for x in rows) / n, 3),
        "round1_single_agent_em": {"agent_a": round(sum(num(x["a_r1_em"]) for x in rows) / n, 4),
                                   "agent_b": round(sum(num(x["b_r1_em"]) for x in rows) / n, 4)},
        "duplicate_rows": duplicates,
    })
    return s


def load_runs():
    """All finished runs: plan entry + summary + per-question correctness."""
    plan = load_yaml(HERE / "runs.yaml")["runs"]
    out = []
    for r in plan:
        d = RESULTS / folder(r)
        st = json.loads((d / "status.json").read_text(encoding="utf-8")) if (d / "status.json").exists() else {}
        if st.get("state") != "done" or not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        rows_all = read_csv(d / "debate_results.csv")
        # one row per question: if a question was written twice (two processes ran the
        # same run, 2026-10-09), the LAST row is kept -- it belongs to the session file.
        last = {}
        for x in rows_all:
            last[int(x["idx"])] = x
        rows = list(last.values())
        s = recompute(s, rows, duplicates=len(rows_all) - len(rows))
        em = {int(x["idx"]): int(x["em"] or 0) for x in rows}
        f1 = {int(x["idx"]): float(x["f1"] or 0) for x in rows}
        src = {int(x["idx"]): x["source"] for x in rows}
        out.append({**r, "summary": s, "em_per_q": em, "f1_per_q": f1, "src_per_q": src, "folder": d.name,
                    "pending": s.get("judge_pending", 0), "duplicates": s["duplicate_rows"]})
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def mcnemar(em_x, em_y):
    """Exact two-sided McNemar test on paired 0/1 answers (same questions).
    Returns (only_x_right, only_y_right, p)."""
    common = sorted(set(em_x) & set(em_y))
    b = sum(1 for q in common if em_x[q] == 1 and em_y[q] == 0)
    c = sum(1 for q in common if em_x[q] == 0 and em_y[q] == 1)
    n, k = b + c, min(b, c)
    if n == 0:
        return b, c, 1.0
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return b, c, p


def benjamini_hochberg(pvals):
    """q-values (BH-adjusted p-values), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q, prev = [0.0] * m, 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


def ols(X, y, names):
    """Plain least squares with standard errors (small n -- read as descriptive)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return [{"term": n, "coef": round(float(b), 4), "std_err": round(float(s), 4)}
            for n, b, s in zip(["intercept"] + names, beta, se)]


# --------------------------------------------------------------------------- #
# Assemble the tables
# --------------------------------------------------------------------------- #
def build(models, runs):
    homo, reps = {}, {}
    for r in runs:
        if r["agent_a"] == r["agent_b"]:
            reps.setdefault(r["agent_a"], []).append(r)
            if r.get("replicate", 1) == 1:
                homo[r["agent_a"]] = r
    noise = max((max(x["summary"]["exact_match"] for x in v) - min(x["summary"]["exact_match"] for x in v)
                 for v in reps.values() if len(v) > 1), default=0.0)

    pairs = []
    for r in runs:
        a, b = r["agent_a"], r["agent_b"]
        if a == b or a not in homo or b not in homo:
            continue
        s, ha, hb = r["summary"], homo[a], homo[b]
        best = ha if ha["summary"]["exact_match"] >= hb["summary"]["exact_match"] else hb
        delta = s["exact_match"] - best["summary"]["exact_match"]
        only_pair, only_best, p = mcnemar(r["em_per_q"], best["em_per_q"])
        f1_mean = lambda x: sum(x["f1_per_q"].values()) / len(x["f1_per_q"])
        delta_f1 = f1_mean(r) - max(f1_mean(ha), f1_mean(hb))
        qs = set(r["em_per_q"]) & set(ha["em_per_q"]) & set(hb["em_per_q"])
        oracle = sum(1 for q in qs if ha["em_per_q"][q] or hb["em_per_q"][q]) / len(qs)
        judged = [q for q, v in r["src_per_q"].items() if v == "judge"]
        ma, mb = models[a], models[b]
        pairs.append({
            "run_id": r["run_id"], "agent_a": a, "agent_b": b,
            "pair_type": r["pair_type"], "same_vendor": r["same_vendor"],
            "em_pair": s["exact_match"], "em_homog_a": ha["summary"]["exact_match"],
            "em_homog_b": hb["summary"]["exact_match"], "better_homog": best["agent_a"],
            "delta_em": round(delta, 4), "only_pair_right": only_pair, "only_best_right": only_best,
            "p_mcnemar": round(p, 5), "nominal": p < ALPHA and abs(delta) > noise,
            "delta_f1": round(delta_f1, 4),
            "oracle_em": round(oracle, 4), "gap_to_oracle": round(oracle - s["exact_match"], 4),
            "judge_decided": len(judged),
            "em_judge_decided": round(sum(r["em_per_q"][q] for q in judged) / len(judged), 3) if judged else "",
            "jaccard_r1": s.get("evidence_jaccard_round1"),
            "consensus_rate": s["consensus_rate"], "judge_rate": s["judge_rate"],
            "rounds_mean": s["rounds_mean"],
            "active_b_ratio": round(max(ma["params_active_b"], mb["params_active_b"])
                                    / min(ma["params_active_b"], mb["params_active_b"]), 2),
            "tokens_total": s["tokens"]["agent_a"] + s["tokens"]["agent_b"],
            "energy_wh": s["energy"]["total_wh"], "judge_pending": r["pending"],
            "duplicate_rows_removed": r["duplicates"],
            "cost_reliable": r["duplicates"] == 0,
        })

    qvals = benjamini_hochberg([p["p_mcnemar"] for p in pairs])
    for p, q in zip(pairs, qvals):
        p["q_bh"] = round(q, 5)
        p["significant"] = q < ALPHA and abs(p["delta_em"]) > noise

    homog_rows = []
    for k, r in sorted(homo.items(), key=lambda kv: -kv[1]["summary"]["exact_match"]):
        m, s = models[k], r["summary"]
        homog_rows.append({
            "model": k, "vendor": m["vendor"], "ffn": m["ffn"], "params_total_b": m["params_total_b"],
            "params_active_b": m["params_active_b"], "tier": m["tier"],
            "em_debate": s["exact_match"], "em_single_agent_round1": s["round1_single_agent_em"]["agent_a"],
            "jaccard_r1": s.get("evidence_jaccard_round1"), "consensus_rate": s["consensus_rate"],
            "forced_answer_share": round(s.get("forced_final_answers", {}).get("agent_a", 0) / s["questions"], 3),
            "truncated": sum(s["truncated_replies"].values()),
            "sec_per_question": s["wall_time_s"]["mean"], "energy_wh": s["energy"]["total_wh"],
            "judge_pending": r["pending"]})

    rep_rows = [{"model": k, "em_replicates": " / ".join(str(x["summary"]["exact_match"]) for x in v),
                 "spread": round(max(x["summary"]["exact_match"] for x in v)
                                 - min(x["summary"]["exact_match"] for x in v), 4)}
                for k, v in reps.items() if len(v) > 1]
    return homo, pairs, homog_rows, rep_rows, noise


def by_group(pairs):
    out = []
    for key in (("pair_type",), ("same_vendor",), ("pair_type", "same_vendor")):
        groups = {}
        for p in pairs:
            groups.setdefault(tuple(p[k] for k in key), []).append(p)
        for g, v in sorted(groups.items(), key=lambda kv: str(kv[0])):
            d = np.array([x["delta_em"] for x in v])
            j = np.array([x["jaccard_r1"] for x in v if x["jaccard_r1"] is not None])
            out.append({"grouping": "+".join(key), "group": " / ".join(str(x) for x in g), "n_pairs": len(v),
                        "delta_em_mean": round(float(d.mean()), 4), "delta_em_min": round(float(d.min()), 4),
                        "delta_em_max": round(float(d.max()), 4),
                        "share_pairs_better": round(float((d > 0).mean()), 2),
                        "delta_f1_mean": round(float(np.mean([x["delta_f1"] for x in v])), 4),
                        "n_nominal_better": sum(1 for x in v if x["nominal"] and x["delta_em"] > 0),
                        "n_significant_better": sum(1 for x in v if x["significant"] and x["delta_em"] > 0),
                        "n_significant_worse": sum(1 for x in v if x["significant"] and x["delta_em"] < 0),
                        "jaccard_r1_mean": round(float(j.mean()), 3) if len(j) else ""})
    return out


def regression(pairs, homo):
    names = ["different_vendor", "dense_x_moe", "moe_x_moe", "log2_active_param_ratio", "mean_homog_em"]
    X = np.array([[0.0 if p["same_vendor"] else 1.0,
                   1.0 if p["pair_type"] == "DxM" else 0.0,
                   1.0 if p["pair_type"] == "MxM" else 0.0,
                   math.log2(p["active_b_ratio"]),
                   (p["em_homog_a"] + p["em_homog_b"]) / 2] for p in pairs])
    y = np.array([p["delta_em"] for p in pairs])
    res_d = ols(X, y, names) if len(pairs) > len(names) + 2 else []
    yj = np.array([p["jaccard_r1"] for p in pairs])
    res_j = ols(X, yj, names) if len(pairs) > len(names) + 2 else []
    return ([{"outcome": "delta_em", **r} for r in res_d] +
            [{"outcome": "jaccard_r1", **r} for r in res_j])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# One consistent style for every figure: sans-serif, larger type, light grid,
# no top/right spines, explanations as a caption below the plot (not in the
# title), automatic black/white text depending on the cell colour.
DENSE_C, MOE_C = "#2F6DB5", "#E07A2E"          # dense = blue, MoE = orange
TYPE_C = {"DxD": "#2F6DB5", "DxM": "#7B5EA7", "MxM": "#E07A2E"}
SIG_C = "#C0392B"
GREY = "#8A8F98"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 10,
        "axes.labelsize": 11.5, "axes.labelweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": "#E3E5E8", "grid.linewidth": 0.8,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


def _caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=9.5, color="#444444", wrap=True)


def _text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#1A1A1A"


def _label(k, models):
    m = models[k]
    return f"{k}\n{'MoE' if m['ffn'] == 'moe' else 'dense'} · {m['params_active_b']:g}B active"


def _tick_colors(ax, keys, models):
    for lab, k in zip(ax.get_xticklabels(), keys):
        lab.set_color(MOE_C if models[k]["ffn"] == "moe" else DENSE_C)
    for lab, k in zip(ax.get_yticklabels(), keys):
        lab.set_color(MOE_C if models[k]["ffn"] == "moe" else DENSE_C)


def _light_cmap(name, lo, hi):
    """Only the light part of a colour map, so that dark text stays readable."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_light", base(np.linspace(lo, hi, 256)))


def _light_diverging():
    """Diverging red-white-blue map that stays light at both ends."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "rdbu_light", ["#D6604D", "#F4A582", "#FDDBC7", "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3"])


def heatmap(stem, title, keys, homo, pairs, models, noise, prelim):
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm
    keys = [k for k in keys if k in homo]
    n = len(keys)
    if n < 2:
        return None
    look = {frozenset((p["agent_a"], p["agent_b"])): p for p in pairs}
    D = np.full((n, n), np.nan)
    J = np.full((n, n), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                J[i, j] = homo[a]["summary"].get("evidence_jaccard_round1") or np.nan
            elif frozenset((a, b)) in look:
                p = look[frozenset((a, b))]
                D[i, j], J[i, j] = p["delta_em"] * 100, p["jaccard_r1"]

    cell = 1.15
    fig, axes = plt.subplots(1, 2, figsize=(2 * (3.4 + cell * n), 1.9 + cell * n), layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    lim = max(2.0, float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 2.0)
    cmap_d = _light_diverging()
    norm_d = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

    ax = axes[0]
    im = ax.imshow(np.where(np.isfinite(D), D, 0), cmap=cmap_d, norm=norm_d)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#EDEEF0", zorder=2))
                ax.text(j, i, f"EM\n{homo[a]['summary']['exact_match']:.3f}", ha="center", va="center",
                        fontsize=10, color="#333333", zorder=3)
            elif np.isfinite(D[i, j]):
                p = look[frozenset((a, b))]
                mark = "*" if p["significant"] else ("(*)" if p["nominal"] else "")
                ax.text(j, i, f"{D[i, j]:+.1f}{mark}", ha="center", va="center", zorder=3,
                        fontsize=12 if p["significant"] else 11,
                        fontweight="bold" if (p["significant"] or p["nominal"]) else "normal",
                        color=_text_color(cmap_d(norm_d(D[i, j]))))
            else:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#D5D7DB", zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="#666666", zorder=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("ΔEM (pp)", fontweight="bold")
    ax.set_title("Accuracy gain ΔEM vs. better homogeneous pair")

    ax2 = axes[1]
    cmap_j = _light_cmap("YlGnBu_r", 0.35, 0.98)   # light yellow-green (high overlap) to mid blue
    im2 = ax2.imshow(J, cmap=cmap_j, vmin=0.45, vmax=1.0)
    for i in range(n):
        for j in range(n):
            if np.isfinite(J[i, j]):
                ax2.text(j, i, f"{J[i, j]:.3f}" if i == j else f"{J[i, j]:.2f}", ha="center", va="center",
                         fontsize=10.5, fontweight="bold" if i != j else "normal",
                         color=_text_color(cmap_j((J[i, j] - 0.45) / 0.55)))
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("Jaccard round 1", fontweight="bold")
    ax2.set_title("Evidence overlap in round 1 (Jaccard)")

    for a in axes:
        a.grid(False)
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels([_label(k, models) for k in keys], rotation=40, ha="right", fontsize=9.5)
        a.set_yticklabels([_label(k, models) for k in keys], fontsize=9.5)
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
        _tick_colors(a, keys, models)
    fig.suptitle(title + ("  ·  PRELIMINARY" if prelim else ""), fontsize=15, fontweight="bold")
    _caption(fig, f"Left: EM of the mixed pair minus EM of the better homogeneous pair (diagonal, grey). "
                  f"Blue = mix better, red = worse. * significant after Benjamini-Hochberg correction (q < {ALPHA}), "
                  f"(*) nominal p < {ALPHA} only; run-to-run noise ≤ {noise*100:.1f} pp.\n"
                  f"Right: overlap of the passages both agents retrieved in round 1 (1 = identical evidence; "
                  f"diagonal = noise floor of a model debating itself). Labels: blue = dense, orange = MoE.",
             y=-0.04)
    path = FIG / f"{stem}.png"
    fig.savefig(path); plt.close(fig)
    return path


def _repel(ax, xs, ys, labels, fontsize=9):
    """Place point labels without overlap (greedy, in display coordinates)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    placed = []
    for x, y in zip(xs, ys):              # the markers themselves are obstacles too
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox([[px - 9, py - 9], [px + 9, py + 9]]))
    order = sorted(range(len(xs)), key=lambda i: -ys[i])
    offsets = [(8, 5), (8, -14), (-8, 5), (-8, -14), (10, 18), (10, -26), (-10, 18), (-10, -26),
               (16, 32), (16, -40), (-16, 32), (-16, -40), (28, 0), (-28, 0), (24, 46), (-24, -52)]
    for i in order:
        for dx, dy in offsets:
            t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha="left" if dx > 0 else "right", va="bottom",
                            color="#333333")
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                if True:                      # thin leader line for every label: no ambiguity
                    t.set_text(""); t.remove()
                    t = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                                    fontsize=fontsize, ha="left" if dx > 0 else "right", va="bottom",
                                    color="#333333", arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7))
                break
            t.remove()


def _trend(ax, x, y, color="#555555"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k, d = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, k * xx + d, color=color, lw=1.6, ls="--", alpha=0.8, zorder=1)
    r = np.corrcoef(x, y)[0, 1]
    return r


NICE = {"qwen2.5-7b": "Qwen2.5-7B", "llama3.1-8b": "Llama-3.1-8B", "llama3.3-70b": "Llama-3.3-70B"}
TREND_ALL_C, TREND_SUB_C = "#333333", "#1B8A6B"


def _thesis_axes(ax, major, minor, ygrid=True):
    """Common thesis styling: bold dark axes, large bold tick labels, major + minor grid."""
    from matplotlib.ticker import MultipleLocator
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.8); ax.spines[sp].set_color("#111111")
    ax.tick_params(axis="both", width=1.6, length=5, labelsize=11.5, colors="#111111")
    ax.yaxis.set_major_locator(MultipleLocator(major)); ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(False)
    if ygrid:
        ax.grid(axis="y", which="major", color="#BFC4CA", linewidth=1.0)
        ax.grid(axis="y", which="minor", color="#E1E4E8", linewidth=0.8)
    ax.set_axisbelow(True)


def _bold_ticks(fig, ax):
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    plt.setp(ax.get_yticklabels(), fontweight="bold"); plt.setp(ax.get_xticklabels(), fontweight="bold")


def _trend_labelled(ax, x, y, color, ls, text, at=0.82, above=True, span=None):
    """Least-squares line (fitted on x, y, drawn over `span` or the data range), label written along it.
    `at` = position of the label as a fraction of the drawn x-range."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k, d = np.polyfit(x, y, 1)
    lo, hi = span if span else (x.min(), x.max())
    xx = np.linspace(lo, hi, 50)
    ax.plot(xx, k * xx + d, color=color, lw=2.4, ls=ls, zorder=2)
    r = float(np.corrcoef(x, y)[0, 1])
    xa = lo + at * (hi - lo); ya = k * xa + d
    # angle of the line on screen (axes limits are frozen before), so the label follows the line
    dx = (hi - lo) * 0.05
    p1 = ax.transData.transform((xa, ya)); p2 = ax.transData.transform((xa + dx, ya + k * dx))
    ang = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
    ax.annotate(text.format(r=r).replace("-", "−"), (xa, ya), xytext=(0, 6 if above else -6),
                textcoords="offset points", ha="center", va="bottom" if above else "top",
                rotation=ang, rotation_mode="anchor", fontsize=11.5, fontweight="bold", color=color, zorder=2.5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    return r


def _gain_panels(plt, pairs, xkey, xlabel, fname, xmajor, xminor, label_pos, sub_text="excl. significant pairs: r = {r:+.2f}"):
    """Figures 07 and 09: ΔEM and ΔF1 against one pair property, two trend lines (all pairs / without
    the two significant pairs), r written on each line. No suptitle and no caption (the thesis text explains)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, gridspec_kw={"wspace": 0.24})
    rest = [p for p in pairs if not p["significant"]]
    for ax, key, ylab, ttl in ((axes[0], "delta_em", "ΔEM (pp)", "Exact match (ΔEM)"),
                              (axes[1], "delta_f1", "ΔF1 (pp)", "Token F1 (ΔF1)")):
        _thesis_axes(ax, 1, 0.5)
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(xmajor)); ax.xaxis.set_minor_locator(MultipleLocator(xminor))
        _pair_scatter(ax, pairs, xkey, key, xlabel, ylab, nice=True)
        ax.set_title(ttl, fontsize=14); ax.set_xlabel(xlabel, fontsize=12.5); ax.set_ylabel(ylab, fontsize=13)
        ax.set_ylim(-2.4, 6.2)
        xs = [p[xkey] for p in pairs]
        span = (min(xs), max(xs))
        ax.set_xlim(span[0] - 0.015, span[1] + 0.015)    # freeze limits before the label angles are computed
        fig.canvas.draw()
        (a_all, up_all), (a_sub, up_sub) = label_pos
        _trend_labelled(ax, xs, [p[key] * 100 for p in pairs], TREND_ALL_C, "--",
                        "all pairs: r = {r:+.2f}", at=a_all, above=up_all, span=span)
        _trend_labelled(ax, [p[xkey] for p in rest], [p[key] * 100 for p in rest], TREND_SUB_C, "-",
                        sub_text, at=a_sub, above=up_sub, span=span)
        _bold_ticks(fig, ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10), fontsize=10.5)
    fig.savefig(FIG / fname); plt.close(fig)


def _pair_scatter(ax, pairs, xkey, ykey, xlabel, ylabel, annotate_sig=True, nice=False):
    for t in ("DxD", "DxM", "MxM"):
        for same, mk in ((True, "o"), (False, "D")):
            v = [p for p in pairs if p["pair_type"] == t and p["same_vendor"] == same]
            if not v:
                continue
            ax.scatter([p[xkey] for p in v], [p[ykey] * 100 if ykey.startswith("delta") else p[ykey] for p in v],
                       s=70, marker=mk, color=TYPE_C[t], alpha=0.85, edgecolor="white", linewidth=0.8,
                       label=f"{t}, {'same' if same else 'different'} vendor", zorder=3)
    sig = [p for p in pairs if p["significant"]]
    ax.scatter([p[xkey] for p in sig], [p[ykey] * 100 if ykey.startswith("delta") else p[ykey] for p in sig],
               s=190, facecolor="none", edgecolor=SIG_C, linewidth=2, zorder=4, label="significant (BH)")
    if annotate_sig:
        for p in sig:
            y = p[ykey] * 100 if ykey.startswith("delta") else p[ykey]
            name = (f"{NICE.get(p['agent_a'], p['agent_a'])} + {NICE.get(p['agent_b'], p['agent_b'])}"
                    if nice else f"{p['agent_a']} + {p['agent_b']}")
            ax.annotate(name, (p[xkey], y), xytext=(10, 4), textcoords="offset points",
                        fontsize=10 if nice else 9, color=SIG_C, fontweight="bold")
    if ykey.startswith("delta"):
        ax.axhline(0, color="#555555" if nice else GREY, lw=1.4 if nice else 1, ls=":", zorder=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def summary_plots(pairs, homog_rows, prelim):
    plt = _plt()
    tag = "  ·  PRELIMINARY" if prelim else ""
    order = [("DxD", True), ("DxD", False), ("DxM", True), ("DxM", False), ("MxM", True), ("MxM", False)]
    labels = [f"{t}\n{'same' if s else 'different'}\nvendor" for t, s in order]
    rng = np.random.default_rng(0)

    # 06 -- distribution per pair type
    # sized for a full text-width figure in the thesis: no suptitle / caption (the text explains it),
    # bold axes, larger fonts and a denser grid so that ΔEM differences can be read off
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"wspace": 0.26})
    for ax, key, ylab, ttl in ((axes[0], "delta_em", "ΔEM (pp)", "Accuracy gain over the better homogeneous pair"),
                              (axes[1], "jaccard_r1", "Jaccard round 1", "Evidence overlap (lower = more divergent)")):
        ttl = {"delta_em": "Accuracy gain (ΔEM)", "jaccard_r1": "Evidence overlap (lower = more divergent)"}[key]
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.8); ax.spines[sp].set_color("#111111")
        ax.tick_params(axis="both", width=1.6, length=5, labelsize=11.5, colors="#111111")
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")
        if key == "delta_em":
            ax.yaxis.set_major_locator(MultipleLocator(1)); ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.1)); ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(axis="y", which="major", color="#BFC4CA", linewidth=1.0)
        ax.grid(axis="y", which="minor", color="#E1E4E8", linewidth=0.8)
        ax.set_axisbelow(True)
        means = []
        for x, (t, s) in enumerate(order):
            v = [p for p in pairs if p["pair_type"] == t and p["same_vendor"] == s and p[key] is not None]
            vals = [p[key] * (100 if key == "delta_em" else 1) for p in v]
            if not vals:
                continue
            jit = x + rng.uniform(-.13, .13, len(vals))
            ax.scatter(jit, vals, s=55, color=TYPE_C[t], alpha=.8, edgecolor="white", linewidth=.7, zorder=3)
            sig = [(jx, val) for jx, val, p in zip(jit, vals, v) if p["significant"]]
            ax.scatter([q[0] for q in sig], [q[1] for q in sig], s=170, facecolor="none", edgecolor=SIG_C,
                       linewidth=2, zorder=4)
            m = float(np.mean(vals))
            ax.hlines(m, x - .3, x + .3, color="#222222", lw=2.4, zorder=5)
            means.append((x, m))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=13); ax.set_title(ttl, fontsize=14); ax.grid(axis="x", visible=False)
        for lab in ax.get_yticklabels():
            lab.set_fontweight("bold")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.18)         # room for the mean labels at the top
        for x, m in means:                              # group means in a row above the dots
            ax.text(x, hi + (hi - lo) * 0.09, f"mean\n{m:+.1f}" if key == "delta_em" else f"mean\n{m:.2f}",
                    ha="center", va="center", fontsize=10.5, fontweight="bold", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#F2F3F5", ec="none"))
        if key == "delta_em":
            ax.axhline(0, color="#555555", lw=1.4, ls="--", zorder=2)
        fig.canvas.draw()
        plt.setp(ax.get_yticklabels(), fontweight="bold")
    if prelim:
        fig.suptitle("PRELIMINARY", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(FIG / "06_by_pair_type.png"); plt.close(fig)

    # 07 -- ΔEM and ΔF1 against divergence
    _gain_panels(plt, pairs, "jaccard_r1", "Jaccard round 1  (← more divergent)",
                 "07_gain_vs_divergence_em_f1.png", 0.1, 0.05, ((0.33, True), (0.30, False)),
                 sub_text="excl. significant pairs\nr = {r:+.2f}")

    # 08 -- quality vs energy (labels without overlap)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    xs = [h["energy_wh"] for h in homog_rows]; ys = [h["em_debate"] for h in homog_rows]
    for h in homog_rows:
        ax.scatter(h["energy_wh"], h["em_debate"], s=90, marker="o" if h["ffn"] == "dense" else "s",
                   color=DENSE_C if h["ffn"] == "dense" else MOE_C, edgecolor="white", linewidth=.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("GPU energy per 1000 questions (Wh, log scale)"); ax.set_ylabel("EM (homogeneous pair)")
    ax.set_title("Model quality vs. energy" + tag)
    ax.scatter([], [], marker="o", color=DENSE_C, label="dense"); ax.scatter([], [], marker="s", color=MOE_C, label="MoE")
    ax.legend(loc="lower right")
    ax.margins(x=0.12, y=0.08)
    _repel(ax, xs, ys, [h["model"] for h in homog_rows], fontsize=9)
    _caption(fig, "Each point = one model debating itself on the 1000 questions. Energy = measured GPU energy of "
                  "the GPUs of that run (nvidia-smi), not the whole node.", y=-0.01)
    fig.savefig(FIG / "08_em_vs_energy_homogeneous.png"); plt.close(fig)

    # 09 -- the key relation: strength of the pair vs gain
    for p in pairs:
        p["_mean_homog"] = (p["em_homog_a"] + p["em_homog_b"]) / 2
    _gain_panels(plt, pairs, "_mean_homog", "Mean EM of both models alone  (stronger →)",
                 "09_gain_vs_model_strength.png", 0.05, 0.025, ((0.20, True), (0.21, True)),
                 sub_text="excl. significant pairs\nr = {r:+.2f}")
    for p in pairs:
        p.pop("_mean_homog", None)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(name, rows):
    if not rows:
        return
    with open(TAB / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_readme(figs):
    fig_lines = "\n".join(f"| `figures/{p.name}` | {t} |" for p, t in figs)
    (OUT / "README.md").write_text(f"""# Phase 3 -- analysis

Generated by `python analyze_phase3.py` (do not edit; re-run the script instead).
Start with **REPORT.md**; the figures and tables below are the evidence behind it.

## How to read a heatmap

Each heatmap is one group of models, dense models first, then MoE, each by size.
**Left panel:** every cell is one mixed pair (row model + column model).
The number is **ΔEM in percentage points** = EM of the pair minus the EM of the
better of the two homogeneous pairs. Blue = the mix is better, red = worse.
`*` marks a real difference: McNemar's test on the 1000 paired answers stays
significant after Benjamini-Hochberg correction over all mixed pairs (q < 0.05)
**and** the difference is larger than the run-to-run noise of the replicate runs.
`(*)` = only nominally significant (uncorrected p < 0.05) -- with ~40 tests about
two such cells are expected by chance. The diagonal (grey) shows the EM of the homogeneous pair.
**Right panel:** overlap of the passages both agents retrieved in round 1
(Jaccard; 1 = identical evidence, lower = more divergent). The diagonal is the
noise floor of a model debating itself.

## Files

| File | Content |
|---|---|
| `REPORT.md` | key results in numbers and sentences |
| `PHASE3_RESULTS_BRIEF.md` | detailed results brief for the presentation (written by hand from these numbers, not regenerated) |
{fig_lines}
| `figures/06_by_pair_type.png` | ΔEM and Jaccard of all mixed pairs, grouped by dense/MoE and vendor |
| `figures/07_gain_vs_divergence_em_f1.png` | ΔEM and ΔF1 against evidence divergence, one dot per pair |
| `figures/09_gain_vs_model_strength.png` | ΔEM and ΔF1 against the strength of the two models (key result) |
| `figures/08_em_vs_energy_homogeneous.png` | quality vs. GPU energy of every model (homogeneous pairs) |
| `tables/01_models_homogeneous.csv` | one row per model: EM, single-agent EM (round 1), divergence floor, cost |
| `tables/02_pairs.csv` | one row per mixed pair: ΔEM, McNemar counts and p, Jaccard, cost |
| `tables/03_by_pair_type.csv` | ΔEM and Jaccard averaged per pair type / vendor |
| `tables/04_regression.csv` | which pair properties go with ΔEM and Jaccard (least squares, descriptive) |
| `tables/05_replicates.csv` | the replicate runs = run-to-run noise |
""", encoding="utf-8")


def write_report(homog_rows, pairs, groups, reg, rep_rows, noise, prelim, n_pending):
    def fmt(x, pp=False):
        return f"{x*100:+.1f} pp" if pp else f"{x:.3f}"
    nom_better = [p for p in pairs if p["nominal"] and p["delta_em"] > 0]
    sig_better = [p for p in pairs if p["significant"] and p["delta_em"] > 0]
    sig_worse = [p for p in pairs if p["significant"] and p["delta_em"] < 0]
    best_pairs = sorted(pairs, key=lambda p: -p["delta_em"])[:5]
    lines = [f"# Phase 3 -- results report", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from {len(homog_rows)} homogeneous and {len(pairs)} mixed runs.", ""]
    if prelim:
        lines += [f"> **PRELIMINARY:** {n_pending} debates still wait for the judge and count as wrong. "
                  "Mixed pairs are affected most. Re-run this script after `phase3.py judge`.", ""]
    lines += ["## 1. Noise", "",
              f"Replicate runs differ by at most **{noise*100:.1f} pp EM**. Differences below this are not "
              "interpreted, whatever their p-value.", "",
              "## 2. Models on their own (homogeneous pairs)", "",
              "| Model | Vendor | FFN | Active B | EM debate | EM single agent (round 1) | Jaccard floor | Forced answers |",
              "|---|---|---|---|---|---|---|---|"]
    for h in homog_rows:
        lines.append(f"| {h['model']} | {h['vendor']} | {h['ffn']} | {h['params_active_b']} | {h['em_debate']:.3f} | "
                     f"{h['em_single_agent_round1']:.3f} | {h['jaccard_r1']} | {h['forced_answer_share']*100:.0f} % |")
    lines += ["", "## 3. Mixed pairs", "",
              f"{len(pairs)} mixed pairs. Better than the better homogeneous pair: "
              f"**{sum(1 for p in pairs if p['delta_em'] > 0)}** (nominal p<0.05: **{len(nom_better)}**, significant after BH correction: **{len(sig_better)}**). "
              f"Worse: **{sum(1 for p in pairs if p['delta_em'] < 0)}** (significant: **{len(sig_worse)}**).", "",
              f"Mean ΔEM over all mixed pairs: **{np.mean([p['delta_em'] for p in pairs])*100:+.2f} pp**; "
              f"mean ΔF1: **{np.mean([p['delta_f1'] for p in pairs])*100:+.2f} pp**. "
              f"Mean headroom to the oracle (question solved by at least one homogeneous pair): "
              f"**{np.mean([p['gap_to_oracle'] for p in pairs])*100:.1f} pp**.", "",
              f"Judge: decided {sum(p['judge_decided'] for p in pairs)} debates of the mixed pairs "
              f"(EM there {np.mean([p['em_judge_decided'] for p in pairs if p['em_judge_decided'] != '']):.3f}; "
              "these are the questions on which the agents disagreed, i.e. the hard ones).", "",
              "Largest gains:", "", "| Pair | Type | ΔEM | ΔF1 | p | q (BH) | Jaccard r1 |",
              "|---|---|---|---|---|---|---|"]
    for p in best_pairs:
        lines.append(f"| {p['agent_a']} + {p['agent_b']} | {p['pair_type']}, "
                     f"{'same' if p['same_vendor'] else 'diff.'} vendor | {fmt(p['delta_em'], True)}"
                     f"{' *' if p['significant'] else ''} | {fmt(p['delta_f1'], True)} | {p['p_mcnemar']} | "
                     f"{p['q_bh']} | {p['jaccard_r1']} |")
    lines += ["", "## 4. Where do differences come from?", "",
              "| Grouping | Group | Pairs | mean ΔEM | mean ΔF1 | better / nominal / BH-sig. better / BH-sig. worse | mean Jaccard r1 |",
              "|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['grouping']} | {g['group']} | {g['n_pairs']} | {fmt(g['delta_em_mean'], True)} | "
                     f"{fmt(g['delta_f1_mean'], True)} | "
                     f"{g['share_pairs_better']*100:.0f} % / {g['n_nominal_better']} / {g['n_significant_better']} / "
                     f"{g['n_significant_worse']} | "
                     f"{g['jaccard_r1_mean']} |")
    if reg:
        lines += ["", "Least-squares fit over all mixed pairs (descriptive, small n):", "",
                  "| Outcome | Term | Coefficient | Std. error |", "|---|---|---|---|"]
        lines += [f"| {r['outcome']} | {r['term']} | {r['coef']} | {r['std_err']} |" for r in reg]
    lines += ["", "## 5. Replicates", "", "| Model | EM per replicate | spread |", "|---|---|---|"]
    lines += [f"| {r['model']} | {r['em_replicates']} | {r['spread']} |" for r in rep_rows]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    models = load_yaml(HERE / "models.yaml")["models"]
    runs = load_runs()
    homo, pairs, homog_rows, rep_rows, noise = build(models, runs)
    n_pending = sum(r["pending"] for r in runs)
    prelim = n_pending > 0
    groups = by_group(pairs)
    reg = regression(pairs, homo)

    write_csv("01_models_homogeneous.csv", homog_rows)
    write_csv("02_pairs.csv", pairs)
    write_csv("03_by_pair_type.csv", groups)
    write_csv("04_regression.csv", reg)
    write_csv("05_replicates.csv", rep_rows)

    figs = []
    for stem, title, keys in MATRICES:
        p = heatmap(stem, title, keys, homo, pairs, models, noise, prelim)
        if p:
            figs.append((p, f"heatmap: {title}"))
    summary_plots(pairs, homog_rows, prelim)
    write_readme(figs)
    write_report(homog_rows, pairs, groups, reg, rep_rows, noise, prelim, n_pending)
    print(f"{len(homog_rows)} homogeneous, {len(pairs)} mixed runs; noise {noise*100:.1f} pp"
          + (f"; PRELIMINARY ({n_pending} debates wait for the judge)" if prelim else ""))
    print(f"-> {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
