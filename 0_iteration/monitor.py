#!/usr/bin/env python3
"""Lightweight resource sampler for the Phase-0 run.

Records CPU%, RAM, GPU-util and VRAM once per interval into a CSV, so the run's
infrastructure utilisation can be plotted afterwards. Uses only the standard
library plus `nvidia-smi` (no psutil / no pip installs needed).

Note: CPU and RAM are read from /proc and are therefore *node-level*. On a shared
node they include other users' load; on an exclusive allocation they are the job.
GPU/VRAM come from the GPU(s) nvidia-smi reports (one row for a gpu:1 job).
"""
import argparse, time, subprocess, signal, datetime

def read_cpu():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = list(map(int, parts[1:]))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return sum(vals), idle

def read_mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.strip().split()[0])  # kB
    total = info["MemTotal"] / 1024.0            # MB
    avail = info.get("MemAvailable", info["MemFree"]) / 1024.0
    return total - avail, total

def read_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        u, mu, mt = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return float(u), float(mu), float(mt)
    except Exception:
        return float("nan"), float("nan"), float("nan")

running = True
def _stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()

    t0 = time.time()
    prev = read_cpu()
    with open(a.out, "w", buffering=1) as f:
        f.write("timestamp,elapsed_s,cpu_pct,ram_used_mb,ram_total_mb,ram_pct,"
                "gpu_util_pct,vram_used_mb,vram_total_mb,vram_pct\n")
        while running:
            time.sleep(a.interval)
            cur = read_cpu()
            d_tot, d_idle = cur[0] - prev[0], cur[1] - prev[1]
            cpu = 100.0 * (1 - d_idle / d_tot) if d_tot > 0 else float("nan")
            prev = cur
            ru, rt = read_mem()
            rp = 100 * ru / rt if rt else float("nan")
            gu, vu, vt = read_gpu()
            vp = 100 * vu / vt if vt == vt and vt else float("nan")
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(f"{ts},{time.time()-t0:.1f},{cpu:.1f},{ru:.0f},{rt:.0f},"
                    f"{rp:.1f},{gu:.1f},{vu:.0f},{vt:.0f},{vp:.1f}\n")

if __name__ == "__main__":
    main()
