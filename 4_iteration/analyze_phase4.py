"""
xmad/4_iteration/analyze_phase4.py

Evaluation of Phase 4. Reads the finished runs in results/ and writes everything
into analysis/. Runs locally (laptop), needs numpy, matplotlib and pyyaml:

    python analyze_phase4.py

Two parts:

  1. Per benchmark (MuSiQue, 2WikiMultiHopQA) the Phase 3 evaluation, unchanged:
     the same tables, heatmaps and Fig. 06-09, drawn by the functions of
     3_iteration/analyze_phase3.py (imported, not copied, so the style is identical).
         analysis/musique/   analysis/2wiki/   (figures/, tables/, REPORT.md each)

  2. Merged over the three benchmarks (HotpotQA = Phase 3 tables, same 46 mixed pairs):
         analysis/merged/
           figures/M1_models_across_benchmarks.png   EM of every model alone on each benchmark
           figures/M2_gain_across_benchmarks.png     ΔEM per pair type and benchmark | ΔEM per pair vs. HotpotQA
           tables/M1..M3 *.csv                        the numbers behind both figures + judge statistics
           REPORT.md

Measure, test and thresholds as in Phase 3:
    ΔEM(A,B) = EM(A+B) - max(EM(A+A), EM(B+B)),  McNemar exact test per pair,
    Benjamini-Hochberg over the 46 mixed pairs of ONE benchmark (as Phase 3 over its pairs).
Phase 4 has no replicate runs; the noise threshold is the run-to-run noise measured
in Phase 3 (3_iteration/analysis/tables/05_replicates.csv).
"""

import importlib.util
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3 = HERE.parent / "3_iteration"
RESULTS = HERE / "results"
OUT = HERE / "analysis"


def _load_phase3_module():
    spec = importlib.util.spec_from_file_location("analyze_phase3", P3 / "analyze_phase3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a3 = _load_phase3_module()          # Phase 3 statistics, tables and figure functions
ALPHA = a3.ALPHA

BENCHMARKS = {"musique": "MuSiQue", "2wiki": "2WikiMultiHopQA"}
BENCH_CODE = {"musique": "MSQ", "2wiki": "2WK"}             # as in the run IDs (P4-MSQ-..., P4-2WK-...)
BENCH_ORDER = ["hotpotqa", "musique", "2wiki"]
BENCH_NAME = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "2wiki": "2Wiki"}
BENCH_C = {"hotpotqa": "#6B7280", "musique": "#1B8A6B", "2wiki": "#B0457B"}
BENCH_M = {"hotpotqa": "o", "musique": "s", "2wiki": "D"}
PAIR_TYPES = ["DxD", "DxM", "MxM"]

# Debates the judge could not decide. They count as wrong (EM 0) and do not make
# the analysis PRELIMINARY. 2026-11-01: one debate, judge pass and retry failed.
UNJUDGEABLE = {
    ("P4-MSQ-003", 136): "6 rounds, 8 truncated replies (8192 tokens each); the judge request "
                         "failed, most likely because the transcript exceeds the judge context (32,768 tokens)",
}


def phase3_noise():
    return max(float(r["spread"]) for r in a3.read_csv(P3 / "analysis" / "tables" / "05_replicates.csv"))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def judge_stats(rows):
    """Questions decided by the judge: how often the judge is right, and how often at
    least one agent was right in round 1 (what a perfect judge could at best reach
    from the round-1 answers)."""
    judged = [x for x in rows if x["source"] == "judge"]
    return {"n_judged": len(judged),
            "em_judged": sum(int(x["em"] or 0) for x in judged),
            "either_r1_right_judged": sum(max(int(x["a_r1_em"] or 0), int(x["b_r1_em"] or 0)) for x in judged)}


def load_runs(benchmark):
    """Finished runs of one benchmark, in the format of analyze_phase3.load_runs."""
    out = []
    for r in a3.load_yaml(HERE / "runs.yaml")["runs"]:
        if r["benchmark"] != benchmark:
            continue
        d = RESULTS / a3.folder(r)
        st = json.loads((d / "status.json").read_text(encoding="utf-8")) if (d / "status.json").exists() else {}
        if st.get("state") != "done" or not (d / "summary.json").exists():
            continue
        rows = a3.read_csv(d / "debate_results.csv")
        if len({x["idx"] for x in rows}) != len(rows):
            raise SystemExit(f"{d.name}: duplicate questions -- run `python phase4.py dedupe` first")
        s = a3.recompute(json.loads((d / "summary.json").read_text(encoding="utf-8")), rows, duplicates=0)
        unjudgeable = sum(1 for x in rows if x["source"] == "pending_judge"
                          and (r["run_id"], int(x["idx"])) in UNJUDGEABLE)
        out.append({**r, "summary": s, "folder": d.name, "duplicates": 0,
                    "em_per_q": {int(x["idx"]): int(x["em"] or 0) for x in rows},
                    "f1_per_q": {int(x["idx"]): float(x["f1"] or 0) for x in rows},
                    "src_per_q": {int(x["idx"]): x["source"] for x in rows},
                    "pending": s["judge_pending"] - unjudgeable, "unjudgeable": unjudgeable,
                    "judge": judge_stats(rows)})
    return out


def build(models, runs, noise):
    """Phase 3 tables; significance with the Phase 3 noise threshold; evidence recall added."""
    homo, pairs, homog_rows, _, _ = a3.build(models, runs)
    for p in pairs:
        p["nominal"] = p["p_mcnemar"] < ALPHA and abs(p["delta_em"]) > noise
        p["significant"] = p["q_bh"] < ALPHA and abs(p["delta_em"]) > noise
    by_id = {r["run_id"]: r for r in runs}
    for p in pairs:
        rec = by_id[p["run_id"]]["summary"].get("evidence_recall") or {}
        p["recall_r1_union"], p["recall_all_rounds"] = rec.get("union_round1"), rec.get("all_rounds")
    for h in homog_rows:
        rec = homo[h["model"]]["summary"].get("evidence_recall") or {}
        h["recall_r1"], h["recall_all_rounds"] = rec.get("agent_a_round1"), rec.get("all_rounds")
    return homo, pairs, homog_rows


# --------------------------------------------------------------------------- #
# Per benchmark: the Phase 3 figures
# --------------------------------------------------------------------------- #
def _gain_panels(plt, pairs, xkey, xlabel, fname, xmajor, xminor, label_pos,
                 sub_text="excl. significant pairs: r = {r:+.2f}"):
    """Phase 3 Fig. 07/09 (analyze_phase3._gain_panels) with three changes: the y-range
    follows the data (Phase 3: fixed -2.4..6.2 pp); significant pairs are ringed but not
    named (Phase 4 has up to 14 per benchmark; the heatmaps name them); the r of both
    trend lines stands in a small legend above the points (the Phase 3 label
    positions along the lines overlap with the Phase 4 data). label_pos and sub_text are
    accepted for the call in analyze_phase3.summary_plots and not used."""
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, gridspec_kw={"wspace": 0.24})
    rest = [p for p in pairs if not p["significant"]]
    vals = [p[k] * 100 for p in pairs for k in ("delta_em", "delta_f1")]
    ylim = (np.floor(min(vals) * 2) / 2 - 0.5, np.ceil(max(vals) * 2) / 2 + 2.5)   # top: room for the r legend
    for ax, key, ylab, ttl in ((axes[0], "delta_em", "ΔEM (pp)", "Exact match (ΔEM)"),
                              (axes[1], "delta_f1", "ΔF1 (pp)", "Token F1 (ΔF1)")):
        a3._thesis_axes(ax, 1, 0.5)
        ax.xaxis.set_major_locator(MultipleLocator(xmajor)); ax.xaxis.set_minor_locator(MultipleLocator(xminor))
        a3._pair_scatter(ax, pairs, xkey, key, xlabel, ylab, annotate_sig=False, nice=True)
        if key == "delta_em":
            h, l = ax.get_legend_handles_labels()              # point legend below the figure
        ax.set_title(ttl, fontsize=14); ax.set_xlabel(xlabel, fontsize=12.5); ax.set_ylabel(ylab, fontsize=13)
        ax.set_ylim(*ylim)
        span = (min(p[xkey] for p in pairs), max(p[xkey] for p in pairs))
        ax.set_xlim(span[0] - 0.015, span[1] + 0.015)
        trend = []
        for sub, color, ls, name in ((pairs, a3.TREND_ALL_C, "--", "all pairs"),
                                     (rest, a3.TREND_SUB_C, "-", "excl. significant pairs")):
            x, y = np.array([p[xkey] for p in sub]), np.array([p[key] * 100 for p in sub])
            k, d = np.polyfit(x, y, 1)
            xx = np.linspace(*span, 50)
            r = f"{np.corrcoef(x, y)[0, 1]:+.2f}".replace("-", "−")
            trend += ax.plot(xx, k * xx + d, color=color, lw=2.4, ls=ls, zorder=2, label=f"{name}: r = {r}")
        leg = ax.legend(handles=trend, loc="upper right", fontsize=11, frameon=True, framealpha=0.92,
                        edgecolor="#BFC4CA", handlelength=2.6)
        for t, ln in zip(leg.get_texts(), trend):
            t.set_color(ln.get_color()); t.set_fontweight("bold")
        a3._bold_ticks(fig, ax)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10), fontsize=10.5)
    fig.savefig(a3.FIG / fname); plt.close(fig)


@contextmanager
def benchmark_title(code):
    """Every figure saved inside this block gets the benchmark code in front of each
    panel title ("MSQ Exact match (ΔEM)"), so that the MuSiQue and 2Wiki versions of
    Fig. 06-09 can be told apart without an extra title line."""
    from matplotlib.figure import Figure
    save = Figure.savefig

    def savefig(fig, *args, **kwargs):
        for ax in fig.axes:
            if ax.get_title():
                ax.set_title(f"{code} {ax.get_title()}")
        return save(fig, *args, **kwargs)
    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = save


def write_report(out, bench, runs, homog_rows, pairs, groups, reg, noise, prelim, n_pending):
    fmt = lambda x: f"{x*100:+.1f} pp"
    nom_better = [p for p in pairs if p["nominal"] and p["delta_em"] > 0]
    sig_better = [p for p in pairs if p["significant"] and p["delta_em"] > 0]
    sig_worse = [p for p in pairs if p["significant"] and p["delta_em"] < 0]
    lines = [f"# Phase 4 -- {BENCHMARKS[bench]} -- results report", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from {len(homog_rows)} homogeneous and "
             f"{len(pairs)} mixed runs (1000 questions each).", ""]
    if prelim:
        lines += [f"> **PRELIMINARY:** {n_pending} debates still wait for the judge and count as wrong.", ""]
    unj = [(r["run_id"], i, t) for r in runs for (rid, i), t in UNJUDGEABLE.items() if rid == r["run_id"]]
    for rid, i, t in unj:
        lines += [f"> Not judgeable, counted as wrong: {rid}, question idx {i} ({t}).", ""]
    lines += ["## 1. Noise", "",
              f"No replicate runs in Phase 4. Threshold = run-to-run noise of Phase 3: **{noise*100:.1f} pp EM**.", "",
              "## 2. Models on their own (homogeneous pairs)", "",
              "| Model | FFN | Active B | EM debate | EM single agent (round 1) | Jaccard floor | Recall r1 | Recall all rounds | Forced answers |",
              "|---|---|---|---|---|---|---|---|---|"]
    for h in homog_rows:
        lines.append(f"| {h['model']} | {h['ffn']} | {h['params_active_b']} | {h['em_debate']:.3f} | "
                     f"{h['em_single_agent_round1']:.3f} | {h['jaccard_r1']} | {h['recall_r1']} | "
                     f"{h['recall_all_rounds']} | {h['forced_answer_share']*100:.0f} % |")
    js = [r["judge"] for r in runs if r["agent_a"] != r["agent_b"]]
    nj = sum(j["n_judged"] for j in js)
    lines += ["", "## 3. Mixed pairs", "",
              f"{len(pairs)} mixed pairs. Better than the better homogeneous pair: "
              f"**{sum(1 for p in pairs if p['delta_em'] > 0)}** (nominal p<0.05: **{len(nom_better)}**, "
              f"significant after BH correction: **{len(sig_better)}**). "
              f"Worse: **{sum(1 for p in pairs if p['delta_em'] < 0)}** (significant: **{len(sig_worse)}**).", "",
              f"Mean ΔEM: **{np.mean([p['delta_em'] for p in pairs])*100:+.2f} pp**; "
              f"mean ΔF1: **{np.mean([p['delta_f1'] for p in pairs])*100:+.2f} pp**. "
              f"Mean headroom to the oracle (question solved by at least one homogeneous pair): "
              f"**{np.mean([p['gap_to_oracle'] for p in pairs])*100:.1f} pp**.", "",
              f"Judge (mixed pairs): decided {nj} debates; right on **{sum(j['em_judged'] for j in js) / nj:.3f}** of them, "
              f"while at least one agent was right in round 1 on **{sum(j['either_r1_right_judged'] for j in js) / nj:.3f}**.", "",
              "All pairs by ΔEM:", "", "| Pair | Type | ΔEM | ΔF1 | p | q (BH) | Jaccard r1 | Recall all rounds |",
              "|---|---|---|---|---|---|---|---|"]
    for p in sorted(pairs, key=lambda p: -p["delta_em"]):
        mark = " *" if p["significant"] else (" (*)" if p["nominal"] else "")
        lines.append(f"| {p['agent_a']} + {p['agent_b']} | {p['pair_type']}, "
                     f"{'same' if p['same_vendor'] else 'diff.'} vendor | {fmt(p['delta_em'])}{mark} | "
                     f"{fmt(p['delta_f1'])} | {p['p_mcnemar']} | {p['q_bh']} | {p['jaccard_r1']} | "
                     f"{p['recall_all_rounds']} |")
    lines += ["", "## 4. Where do differences come from?", "",
              "| Grouping | Group | Pairs | mean ΔEM | mean ΔF1 | better / nominal / BH-sig. better / BH-sig. worse | mean Jaccard r1 |",
              "|---|---|---|---|---|---|---|"]
    for g in groups:
        lines.append(f"| {g['grouping']} | {g['group']} | {g['n_pairs']} | {fmt(g['delta_em_mean'])} | "
                     f"{fmt(g['delta_f1_mean'])} | {g['share_pairs_better']*100:.0f} % / {g['n_nominal_better']} / "
                     f"{g['n_significant_better']} / {g['n_significant_worse']} | {g['jaccard_r1_mean']} |")
    if reg:
        lines += ["", "Least-squares fit over all mixed pairs (descriptive, small n):", "",
                  "| Outcome | Term | Coefficient | Std. error |", "|---|---|---|---|"]
        lines += [f"| {r['outcome']} | {r['term']} | {r['coef']} | {r['std_err']} |" for r in reg]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_benchmark(bench, models, noise):
    out = OUT / bench
    a3.FIG, a3.TAB = out / "figures", out / "tables"       # the Phase 3 functions write here
    a3.FIG.mkdir(parents=True, exist_ok=True); a3.TAB.mkdir(parents=True, exist_ok=True)
    runs = load_runs(bench)
    homo, pairs, homog_rows = build(models, runs, noise)
    n_pending = sum(r["pending"] for r in runs)
    prelim = n_pending > 0
    groups = a3.by_group(pairs)
    reg = a3.regression(pairs, homo)
    a3.write_csv("01_models_homogeneous.csv", homog_rows)
    a3.write_csv("02_pairs.csv", pairs)
    a3.write_csv("03_by_pair_type.csv", groups)
    a3.write_csv("04_regression.csv", reg)
    for stem, title, keys in a3.MATRICES:
        a3.heatmap(stem, f"{title} · {BENCHMARKS[bench]}", keys, homo, pairs, models, noise, prelim)
    a3._gain_panels = _gain_panels                          # Fig. 07/09: y-range from the data
    with benchmark_title(BENCH_CODE[bench]):
        a3.summary_plots(pairs, homog_rows, prelim)
    write_report(out, bench, runs, homog_rows, pairs, groups, reg, noise, prelim, n_pending)
    print(f"{BENCHMARKS[bench]}: {len(homog_rows)} homogeneous, {len(pairs)} mixed runs"
          + (f"; PRELIMINARY ({n_pending} debates wait for the judge)" if prelim else ""))
    return {"runs": runs, "pairs": pairs, "homog_rows": homog_rows}


# --------------------------------------------------------------------------- #
# Merged over the three benchmarks
# --------------------------------------------------------------------------- #
def hotpotqa_phase3():
    """HotpotQA = Phase 3 tables (homogeneous models, mixed pairs) + judge statistics
    of the Phase 3 runs."""
    homog = {r["model"]: r for r in a3.read_csv(P3 / "analysis" / "tables" / "01_models_homogeneous.csv")}
    pairs = a3.read_csv(P3 / "analysis" / "tables" / "02_pairs.csv")
    for p in pairs:
        for k in ("delta_em", "delta_f1", "q_bh", "p_mcnemar"):
            p[k] = float(p[k])
        for k in ("significant", "nominal", "same_vendor"):
            p[k] = p[k] == "True"
    judge = {}
    for r in a3.load_yaml(P3 / "runs.yaml")["runs"]:
        d = P3 / "results" / a3.folder(r)
        if r.get("replicate", 1) == 1 and (d / "debate_results.csv").exists():
            last = {int(x["idx"]): x for x in a3.read_csv(d / "debate_results.csv")}
            judge[r["run_id"]] = judge_stats(list(last.values()))
    return homog, pairs, judge


def _pearson(x, y):
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def _verdict(delta, nominal, significant):
    if significant:
        return "significant better" if delta > 0 else "significant worse"
    if nominal:
        return "nominal better" if delta > 0 else "nominal worse"
    return "better" if delta > 0 else ("worse" if delta < 0 else "equal")


def merged_tables(models, res, hpq_homog, hpq_pairs, hpq_judge):
    model_rows = []
    for k, h in sorted(hpq_homog.items(), key=lambda kv: -float(kv[1]["em_debate"])):
        rows = {b: next((x for x in res[b]["homog_rows"] if x["model"] == k), None) for b in BENCHMARKS}
        if None in rows.values():
            continue
        m = models[k]
        model_rows.append({"model": k, "vendor": m["vendor"], "ffn": m["ffn"],
                           "params_total_b": m["params_total_b"], "params_active_b": m["params_active_b"],
                           "em_hotpotqa": float(h["em_debate"]),
                           **{f"em_{b}": rows[b]["em_debate"] for b in BENCHMARKS},
                           **{f"recall_all_rounds_{b}": rows[b]["recall_all_rounds"] for b in BENCHMARKS}})

    hpq = {frozenset((p["agent_a"], p["agent_b"])): p for p in hpq_pairs}
    by = {b: {frozenset((p["agent_a"], p["agent_b"])): p for p in res[b]["pairs"]} for b in BENCHMARKS}
    pair_rows = []
    for key, pm in by["musique"].items():
        if key not in hpq or key not in by["2wiki"]:
            continue
        src = {"hotpotqa": hpq[key], **{b: by[b][key] for b in BENCHMARKS}}
        row = {"agent_a": pm["agent_a"], "agent_b": pm["agent_b"], "pair_type": pm["pair_type"],
               "same_vendor": pm["same_vendor"], "run_id_hotpotqa": hpq[key]["run_id"],
               "run_id_musique": pm["run_id"], "run_id_2wiki": by["2wiki"][key]["run_id"]}
        for b in BENCH_ORDER:
            p = src[b]
            row.update({f"delta_em_{b}": round(p["delta_em"], 4), f"q_bh_{b}": p["q_bh"],
                        f"verdict_{b}": _verdict(p["delta_em"], p["nominal"], p["significant"])})
        pair_rows.append(row)
    pair_rows.sort(key=lambda r: -r["delta_em_musique"])

    run_judge = {b: {r["run_id"]: r["judge"] for r in res[b]["runs"]} for b in BENCHMARKS}
    run_judge["hotpotqa"] = hpq_judge
    type_rows = []
    for b in BENCH_ORDER:
        for t in PAIR_TYPES + ["all"]:
            v = [r for r in pair_rows if t == "all" or r["pair_type"] == t]
            d = np.array([r[f"delta_em_{b}"] for r in v])
            js = [run_judge[b][r[f"run_id_{b}"]] for r in v]
            nj = sum(j["n_judged"] for j in js)
            type_rows.append({
                "benchmark": BENCH_NAME[b], "pair_type": t, "n_pairs": len(v),
                "delta_em_mean": round(float(d.mean()), 4), "share_pairs_better": round(float((d > 0).mean()), 2),
                "n_significant_better": sum(1 for r in v if r[f"verdict_{b}"] == "significant better"),
                "n_significant_worse": sum(1 for r in v if r[f"verdict_{b}"] == "significant worse"),
                "judge_decided_share": round(nj / (1000 * len(v)), 4),
                "em_judge_decided": round(sum(j["em_judged"] for j in js) / nj, 3),
                "either_agent_right_r1_judge_decided": round(sum(j["either_r1_right_judged"] for j in js) / nj, 3)})
    return model_rows, pair_rows, type_rows


def _style(ax, axis, major, minor):
    """Thesis styling of analyze_phase3 (bold axes, major + minor grid) for the given axis."""
    from matplotlib.ticker import MultipleLocator
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.8); ax.spines[sp].set_color("#111111")
    ax.tick_params(axis="both", width=1.6, length=5, labelsize=11.5, colors="#111111")
    loc = ax.xaxis if axis == "x" else ax.yaxis
    loc.set_major_locator(MultipleLocator(major)); loc.set_minor_locator(MultipleLocator(minor))
    ax.grid(False)
    ax.grid(axis=axis, which="major", color="#BFC4CA", linewidth=1.0)
    ax.grid(axis=axis, which="minor", color="#E1E4E8", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_models(model_rows, fig_dir):
    """M1: EM of every model debating itself, on each benchmark (models ordered by HotpotQA)."""
    plt = a3._plt()
    fig, ax = plt.subplots(figsize=(11, 8.6))
    y = np.arange(len(model_rows))[::-1]
    for yi, r in zip(y, model_rows):
        v = [r[f"em_{b}"] for b in BENCH_ORDER]
        ax.hlines(yi, min(v), max(v), color="#C9CDD2", lw=2.2, zorder=1)
    for b in BENCH_ORDER:
        lab = BENCH_NAME[b] + ("  (Phase 3)" if b == "hotpotqa" else
                               f"  (r = {_pearson([r['em_' + b] for r in model_rows], [r['em_hotpotqa'] for r in model_rows]):.2f} vs. HotpotQA)")
        ax.scatter([r[f"em_{b}"] for r in model_rows], y, s=85, marker=BENCH_M[b], color=BENCH_C[b],
                   edgecolor="white", linewidth=0.9, zorder=3, label=lab)
    _style(ax, "x", 0.05, 0.025)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['model']}  ({'MoE' if r['ffn'] == 'moe' else 'dense'}, {r['params_active_b']:g}B act.)"
                        for r in model_rows], fontsize=10.5)
    for lab, r in zip(ax.get_yticklabels(), model_rows):
        lab.set_color(a3.MOE_C if r["ffn"] == "moe" else a3.DENSE_C)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("EM (homogeneous pair, 1000 questions)", fontsize=12.5)
    ax.set_ylim(-0.7, len(model_rows) - 0.3)
    fig.canvas.draw()
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=10.5)
    fig.savefig(fig_dir / "M1_models_across_benchmarks.png"); plt.close(fig)


def fig_gain(pair_rows, type_rows, fig_dir):
    """M2 left: ΔEM of the 46 mixed pairs per pair type and benchmark (as Fig. 06).
    M2 right: ΔEM of each pair on MuSiQue / 2Wiki against its ΔEM on HotpotQA."""
    plt = a3._plt()
    rng = np.random.default_rng(0)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 6.0), gridspec_kw={"wspace": 0.22, "width_ratios": [1.25, 1]})
    groups = PAIR_TYPES + ["all"]
    off = {"hotpotqa": -0.27, "musique": 0.0, "2wiki": 0.27}
    means = []
    for gx, t in enumerate(groups):
        v = [r for r in pair_rows if t == "all" or r["pair_type"] == t]
        for b in BENCH_ORDER:
            vals = np.array([r[f"delta_em_{b}"] * 100 for r in v])
            x = gx + off[b] + rng.uniform(-0.07, 0.07, len(vals))
            axl.scatter(x, vals, s=26, marker=BENCH_M[b], color=BENCH_C[b], alpha=0.75, edgecolor="white",
                        linewidth=0.5, zorder=3, label=BENCH_NAME[b] if gx == 0 else None)
            sig = [i for i, r in enumerate(v) if r[f"verdict_{b}"].startswith("significant")]
            axl.scatter(x[sig], vals[sig], s=95, facecolor="none", edgecolor=a3.SIG_C, linewidth=1.6, zorder=4,
                        label="significant (BH)" if gx == 0 and b == "hotpotqa" else None)
            m = float(vals.mean())
            axl.hlines(m, gx + off[b] - 0.11, gx + off[b] + 0.11, color="#222222", lw=2.4, zorder=5)
            means.append((gx + off[b], m, b))
    _style(axl, "y", 1, 0.5)
    axl.axhline(0, color="#555555", lw=1.4, ls="--", zorder=2)
    axl.set_xticks(range(len(groups)))
    axl.set_xticklabels([f"{t}\n(n = {sum(1 for r in pair_rows if t == 'all' or r['pair_type'] == t)})"
                         for t in groups], fontsize=11, fontweight="bold")
    axl.grid(axis="x", visible=False)
    axl.set_ylabel("ΔEM (pp)", fontsize=13); axl.set_title("Accuracy gain per pair type", fontsize=14)
    lo, hi = axl.get_ylim()
    axl.set_ylim(lo, hi + (hi - lo) * 0.24)
    row = {"hotpotqa": 0.05, "musique": 0.12, "2wiki": 0.19}   # one row of mean labels per benchmark
    for x, m, b in means:
        axl.text(x, hi + (hi - lo) * row[b], f"{m:+.1f}", ha="center", va="center", fontsize=9.5, fontweight="bold",
                 color=BENCH_C[b], bbox=dict(boxstyle="round,pad=0.18", fc="#F2F3F5", ec="none"))

    for b in ("musique", "2wiki"):
        xs = [r["delta_em_hotpotqa"] * 100 for r in pair_rows]
        ys = [r[f"delta_em_{b}"] * 100 for r in pair_rows]
        axr.scatter(xs, ys, s=55, marker=BENCH_M[b], color=BENCH_C[b], alpha=0.8, edgecolor="white", linewidth=0.7,
                    zorder=3, label=f"{BENCH_NAME[b]}  (r = {_pearson(xs, ys):+.2f})")
    _style(axr, "y", 1, 0.5)
    from matplotlib.ticker import MultipleLocator
    axr.xaxis.set_major_locator(MultipleLocator(1)); axr.xaxis.set_minor_locator(MultipleLocator(0.5))
    axr.grid(axis="x", which="major", color="#BFC4CA", linewidth=1.0)
    axr.grid(axis="x", which="minor", color="#E1E4E8", linewidth=0.8)
    lim = [min(axr.get_xlim()[0], axr.get_ylim()[0]), max(axr.get_xlim()[1], axr.get_ylim()[1])]
    axr.plot(lim, lim, color="#888888", lw=1.2, ls=":", zorder=1)
    axr.axhline(0, color="#555555", lw=1.2, ls="--", zorder=2); axr.axvline(0, color="#555555", lw=1.2, ls="--", zorder=2)
    axr.set_xlim(lim); axr.set_ylim(lim)
    axr.set_xlabel("ΔEM on HotpotQA (pp, Phase 3)", fontsize=12.5); axr.set_ylabel("ΔEM on the new benchmark (pp)", fontsize=13)
    axr.set_title("Same pair, other benchmark", fontsize=14)
    axr.legend(loc="upper left", fontsize=10.5)
    for ax in (axl, axr):
        a3._bold_ticks(fig, ax)
    h, l = axl.get_legend_handles_labels()
    order = [l.index(BENCH_NAME[b]) for b in BENCH_ORDER] + [l.index("significant (BH)")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=4,
               bbox_to_anchor=(0.32, -0.08), fontsize=10.5, markerscale=1.4)
    fig.savefig(fig_dir / "M2_gain_across_benchmarks.png"); plt.close(fig)


def write_merged_report(out, model_rows, pair_rows, type_rows):
    r = lambda b, c: _pearson([x[f"em_{b}"] for x in model_rows], [x[f"em_{c}"] for x in model_rows])
    rd = lambda b, c: _pearson([x[f"delta_em_{b}"] for x in pair_rows], [x[f"delta_em_{c}"] for x in pair_rows])
    sign = lambda b: sum(1 for x in pair_rows if (x[f"delta_em_{b}"] > 0) == (x["delta_em_hotpotqa"] > 0))
    lines = ["# Phase 4 -- merged over HotpotQA, MuSiQue and 2WikiMultiHopQA", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. HotpotQA = Phase 3 runs; the same {len(model_rows)} "
             f"models and the same {len(pair_rows)} mixed pairs on all three benchmarks.", "",
             "## 1. Models on their own (Fig. M1)", "",
             "Pearson r of the EM of the homogeneous pairs between the benchmarks:", "",
             "| | HotpotQA | MuSiQue | 2Wiki |", "|---|---|---|---|"]
    for b in BENCH_ORDER:
        lines.append(f"| {BENCH_NAME[b]} | " + " | ".join(f"{r(b, c):.2f}" for c in BENCH_ORDER) + " |")
    lines += ["", "| Benchmark | EM min | EM max |", "|---|---|---|"]
    for b in BENCH_ORDER:
        v = [x[f"em_{b}"] for x in model_rows]
        lines.append(f"| {BENCH_NAME[b]} | {min(v):.3f} | {max(v):.3f} |")
    lines += ["", "## 2. Debate gain per pair type (Fig. M2 left)", "",
              "| Benchmark | Pair type | Pairs | mean ΔEM | better | sig. better | sig. worse | judge-decided | "
              "EM judge-decided | ≥1 agent right r1 (judge-decided) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in type_rows:
        lines.append(f"| {t['benchmark']} | {t['pair_type']} | {t['n_pairs']} | {t['delta_em_mean']*100:+.2f} pp | "
                     f"{t['share_pairs_better']*100:.0f} % | {t['n_significant_better']} | {t['n_significant_worse']} | "
                     f"{t['judge_decided_share']*100:.1f} % | {t['em_judge_decided']:.3f} | "
                     f"{t['either_agent_right_r1_judge_decided']:.3f} |")
    lines += ["", "## 3. Same pair, other benchmark (Fig. M2 right)", "",
              "| | r of ΔEM per pair | same sign as HotpotQA |", "|---|---|---|",
              f"| MuSiQue vs. HotpotQA | {rd('musique', 'hotpotqa'):+.2f} | {sign('musique')} / {len(pair_rows)} |",
              f"| 2Wiki vs. HotpotQA | {rd('2wiki', 'hotpotqa'):+.2f} | {sign('2wiki')} / {len(pair_rows)} |",
              f"| MuSiQue vs. 2Wiki | {rd('musique', '2wiki'):+.2f} | -- |", "",
              "Pairs significantly better on at least two benchmarks:", ""]
    for x in pair_rows:
        n = sum(1 for b in BENCH_ORDER if x[f"verdict_{b}"] == "significant better")
        if n >= 2:
            lines.append(f"- {x['agent_a']} + {x['agent_b']} ({x['pair_type']}): "
                         + ", ".join(f"{BENCH_NAME[b]} {x[f'delta_em_{b}']*100:+.1f} pp ({x[f'verdict_{b}']})"
                                     for b in BENCH_ORDER))
    lines += ["", "Pairs significantly worse on at least one benchmark:", ""]
    for x in pair_rows:
        bad = [b for b in BENCH_ORDER if x[f"verdict_{b}"] == "significant worse"]
        if bad:
            lines.append(f"- {x['agent_a']} + {x['agent_b']} ({x['pair_type']}): "
                         + ", ".join(f"{BENCH_NAME[b]} {x[f'delta_em_{b}']*100:+.1f} pp" for b in BENCH_ORDER))
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme():
    (OUT / "README.md").write_text("""# Phase 4 -- analysis

Generated by `python analyze_phase4.py` (do not edit; re-run the script instead).

| Folder | Content |
|---|---|
| `musique/`, `2wiki/` | the Phase 3 evaluation per benchmark: `REPORT.md`, `figures/01-09`, `tables/01-04` (same files and reading as `3_iteration/analysis/README.md`; no replicate table -- the noise threshold is the Phase 3 one) |
| `merged/` | the three benchmarks together (HotpotQA from Phase 3): `REPORT.md`, `figures/M1`, `figures/M2`, `tables/M1-M3` |
| `crash_logs_2026-10-31.txt` | log extract of the 4 runs that lost the MCP connection (see README of 4_iteration) |
| `retriever_recall.csv` | recall@k of the bge retriever per benchmark (`retriever_check.py`; rerun 2026-11 with the corrected script) |
| `retriever_recall_bm25.csv` | first run of `retriever_check.py`, which measured the BM25 retriever of Phase 1 by mistake (kept for the record) |

Merged figures:

| File | Content |
|---|---|
| `merged/figures/M1_models_across_benchmarks.png` | EM of each model debating itself on HotpotQA, MuSiQue, 2Wiki; r against HotpotQA in the legend |
| `merged/figures/M2_gain_across_benchmarks.png` | left: ΔEM of the 46 mixed pairs per pair type and benchmark (mean on top, red ring = significant after BH); right: ΔEM of each pair on MuSiQue / 2Wiki against its ΔEM on HotpotQA |
| `merged/tables/M1_models.csv` | EM and evidence recall per model and benchmark |
| `merged/tables/M2_pairs.csv` | ΔEM, q and verdict per pair and benchmark |
| `merged/tables/M3_by_pair_type.csv` | mean ΔEM, better / significant counts and judge statistics per benchmark and pair type |
""", encoding="utf-8")


def main():
    np.random.seed(0)
    models = a3.load_yaml(HERE / "models.yaml")["models"]
    noise = phase3_noise()
    res = {b: analyze_benchmark(b, models, noise) for b in BENCHMARKS}

    out = OUT / "merged"
    a3.TAB = out / "tables"
    (out / "figures").mkdir(parents=True, exist_ok=True); a3.TAB.mkdir(parents=True, exist_ok=True)
    hpq_homog, hpq_pairs, hpq_judge = hotpotqa_phase3()
    model_rows, pair_rows, type_rows = merged_tables(models, res, hpq_homog, hpq_pairs, hpq_judge)
    a3.write_csv("M1_models.csv", model_rows)
    a3.write_csv("M2_pairs.csv", pair_rows)
    a3.write_csv("M3_by_pair_type.csv", type_rows)
    fig_models(model_rows, out / "figures")
    fig_gain(pair_rows, type_rows, out / "figures")
    write_merged_report(out, model_rows, pair_rows, type_rows)
    write_readme()
    print(f"merged: {len(model_rows)} models, {len(pair_rows)} pairs on all three benchmarks -> {OUT}")


if __name__ == "__main__":
    main()
