#!/usr/bin/env python3
"""One publication-quality figure for a Phase-0 run.

Usage:  python plot_phase0.py logs/run_<stamp>_job<id>

Writes a single combined figure into the run directory:
  - phase0_figure.png : left  = request-latency distribution (mean/median/p90/p99)
                        right = GPU utilisation (top) and VRAM (bottom) over the
                                run, with an "endpoint ready" marker and the
                                benchmark window. The idle start-up head is
                                cropped so the active phase fills the panel.

Needs numpy + matplotlib.
"""
import sys, os, csv, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_GPU, C_VRAM = "#4F46E5", "#F59E0B"
C_MEAN, C_MEDIAN, C_TAIL = "#DC2626", "#059669", "#6B7280"
C_BAR, C_BAREDGE = "#93C5FD", "#1D4ED8"
C_EVENT, C_BENCH, C_LOAD = "#111827", "#10B981", "#9CA3AF"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.size": 13, "font.family": "DejaVu Sans",
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13.5, "axes.labelcolor": "#111827",
    "axes.edgecolor": "#9CA3AF", "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#D1D5DB",
    "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "xtick.color": "#374151", "ytick.color": "#374151",
    "legend.frameon": True, "legend.fontsize": 12,
})

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def _hms(iso):
    h, m, s = iso.split("T")[1].split(":")[:3]
    return int(h) * 3600 + int(m) * 60 + int(float(s))

def parse_events(run, rows):
    ev = {}
    log = os.path.join(run, "phase0.log")
    if os.path.exists(log) and rows:
        base = [(_hms(r["timestamp"]), float(r["elapsed_s"])) for r in rows]
        to_el = lambda sec: min(base, key=lambda b: abs(b[0] - sec))[1]
        pat = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)")
        for line in open(log):
            m = pat.match(line)
            if not m:
                continue
            hh, mm, ss, rest = m.groups()
            sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
            if ("endpoint ready" in rest or "R0.3 PASS" in rest) and "ready" not in ev:
                ev["ready"] = to_el(sec)
            if rest.startswith("--- Benchmark") or "Benchmark:" in rest:
                ev["bench_start"] = to_el(sec)
    if "ready" not in ev and rows:
        t = np.array([float(r["elapsed_s"]) for r in rows])
        v = np.array([float(r["vram_used_mb"]) for r in rows])
        if np.nanmax(v) > 5000:
            ev["ready"] = float(t[np.argmax(v >= 0.92 * np.nanmax(v))])
    bs = os.path.join(run, "bench_summary.json")
    if "bench_start" in ev and os.path.exists(bs):
        try:
            ev["bench_end"] = ev["bench_start"] + float(json.load(open(bs)).get("wall_time_s", 0))
        except Exception:
            pass
    return ev

def combined(run):
    mp = os.path.join(run, "metrics.csv")
    bp = os.path.join(run, "bench_results.csv")
    if not (os.path.exists(mp) and os.path.exists(bp)):
        print("need metrics.csv and bench_results.csv"); return
    rows = read_csv(mp)
    t   = np.array([float(r["elapsed_s"]) for r in rows])
    gpu = np.array([float(r["gpu_util_pct"]) for r in rows])
    vram_gb = np.array([float(r["vram_used_mb"]) for r in rows]) / 1024.0
    vtot_gb = np.nanmax([float(r["vram_total_mb"]) for r in rows]) / 1024.0
    ev = parse_events(run, rows)

    lat = np.array([float(r["latency_s"]) for r in read_csv(bp) if r["ok"] == "True"])
    mean, median = lat.mean(), np.median(lat)
    p90, p99 = np.percentile(lat, 90), np.percentile(lat, 99)

    active = (gpu > 5) | (vram_gb > vram_gb.min() + 2)
    t_start = max(t.min(), float(t[active.argmax()]) - 8) if active.any() else t.min()

    fig = plt.figure(figsize=(13.5, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.18],
                          height_ratios=[1, 1], wspace=0.24, hspace=0.16)
    ax_lat  = fig.add_subplot(gs[:, 0])
    ax_gpu  = fig.add_subplot(gs[0, 1])
    ax_vram = fig.add_subplot(gs[1, 1], sharex=ax_gpu)

    n_bins = min(28, max(12, int(np.sqrt(lat.size) * 2)))
    ax_lat.hist(lat, bins=n_bins, color=C_BAR, edgecolor=C_BAREDGE,
                linewidth=0.6, alpha=0.9, zorder=2)
    for x, c, lab in [(mean, C_MEAN, f"mean {mean:.3f}s"),
                      (median, C_MEDIAN, f"median {median:.3f}s")]:
        ax_lat.axvline(x, color=c, ls="--", lw=2.0, zorder=4, label=lab)
    for x, lab in [(p90, f"p90 {p90:.3f}s"), (p99, f"p99 {p99:.3f}s")]:
        ax_lat.axvline(x, color=C_TAIL, ls=":", lw=1.6, zorder=3, label=lab)
    ax_lat.set_xlabel("latency per request (s)")
    ax_lat.set_ylabel("number of requests")
    ax_lat.set_title("Request latency  (100 prompts)")
    ax_lat.legend(loc="upper right", framealpha=0.95, edgecolor="#E5E7EB")
    stats = (f"n = {lat.size}\n"
             f"mean   {mean:.3f} s\nmedian {median:.3f} s\n"
             f"p90    {p90:.3f} s\np99    {p99:.3f} s\n"
             f"min    {lat.min():.3f} s\nmax    {lat.max():.3f} s")
    ax_lat.text(0.03, 0.97, stats, transform=ax_lat.transAxes, ha="left", va="top",
                fontsize=11, family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.5", fc="#F9FAFB", ec="#D1D5DB"))

    ax_gpu.fill_between(t, gpu, color=C_GPU, alpha=0.12, zorder=1)
    ax_gpu.plot(t, gpu, color=C_GPU, lw=2.0, zorder=3)
    ax_gpu.set_ylabel("GPU util. (%)")
    ax_gpu.set_ylim(0, 105); ax_gpu.set_yticks([0, 50, 100])
    ax_gpu.set_title("Resource utilisation over the run")
    plt.setp(ax_gpu.get_xticklabels(), visible=False)

    ax_vram.fill_between(t, vram_gb, color=C_VRAM, alpha=0.14, zorder=1)
    ax_vram.plot(t, vram_gb, color=C_VRAM, lw=2.0, zorder=3)
    ax_vram.axhline(vtot_gb, color=C_TAIL, ls=":", lw=1.4, zorder=2)
    ax_vram.text(t_start, vtot_gb, f" capacity {vtot_gb:.0f} GB",
                 va="bottom", ha="left", fontsize=11, color=C_TAIL)
    ax_vram.set_ylabel("VRAM used (GB)")
    ax_vram.set_ylim(0, vtot_gb * 1.14)
    ax_vram.set_xlabel("elapsed time (s)")
    ax_vram.set_xlim(t_start, t.max())

    for ax in (ax_gpu, ax_vram):
        if "ready" in ev:
            ax.axvspan(t_start, ev["ready"], color=C_LOAD, alpha=0.10, zorder=0)
            ax.axvline(ev["ready"], color=C_EVENT, ls="--", lw=1.5, zorder=4)
        if "bench_start" in ev and "bench_end" in ev:
            ax.axvspan(ev["bench_start"], ev["bench_end"],
                       color=C_BENCH, alpha=0.18, zorder=0)
    if "ready" in ev:
        ax_gpu.annotate(f"endpoint ready\n(model loaded, {ev['ready']:.0f}s)",
                        xy=(ev["ready"], 52), xytext=(ev["ready"] - 3, 62),
                        ha="right", va="center", fontsize=11, color=C_EVENT,
                        arrowprops=dict(arrowstyle="->", color=C_EVENT, lw=1.0))

    handles = [
        plt.Line2D([], [], color=C_GPU, lw=2.0, label="GPU utilisation"),
        plt.Line2D([], [], color=C_VRAM, lw=2.0, label="VRAM used"),
        Patch(facecolor=C_LOAD, alpha=0.10, label="model load / start-up"),
        Patch(facecolor=C_BENCH, alpha=0.18, label="benchmark"),
    ]
    ax_gpu.legend(handles=handles, loc="upper left", ncol=1,
                  fontsize=10.5, framealpha=0.95, edgecolor="#E5E7EB")

    out = os.path.join(run, "phase0_figure.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    combined(sys.argv[1])