# Phase 0 — automated environment + single-model hosting

Automated proof that a node can be allocated, a model served with vLLM, and
inference run end-to-end — while verifying requirements **R0.1–R0.4**,
benchmarking throughput/latency over N prompts, and recording CPU/RAM/GPU/VRAM
utilisation for later plotting.

## Files
| File | Role |
|------|------|
| `run_phase0.slurm` | Main driver: allocates a node (or runs on the current one), sets up the env, serves the model, checks R0.1–R0.4, runs the benchmark, and writes **all** logs incl. `phase0_summary.json`. |
| `monitor.py` | Samples CPU/RAM (from `/proc`) and GPU/VRAM (from `nvidia-smi`) once per second → `metrics.csv`. Stdlib only. |
| `bench_phase0.py` | Fires N prompts at the endpoint; records per-request latency + tokens → `bench_results.csv` + `bench_summary.json`. Needs `requests` (already a vLLM dep). |
| `plot_phase0.py` | Optional: turns one run's CSVs into `resource_timeline.png` + `latency_hist.png`. Run **locally** (cluster venv has no matplotlib). |

## Placement
Put the whole folder on the cluster at:
```
/path/to/workspace/xmad/0_iteration/
```
All four files must sit together — the driver calls `monitor.py` and
`bench_phase0.py` from its own directory.

## Run

There are two ways to start it. **Option B is the safe fallback** when the
scheduler queue is full (which is common on this cluster).

### Option A — as a SLURM job (only if a node is free)
```bash
cd /path/to/workspace/xmad/0_iteration
mkdir -p logs
sbatch run_phase0.slurm
```
Watch it:
```bash
squeue -u $USER
tail -f logs/phase0_<jobid>.out
```
> **If `sbatch` fails with "Invalid account or account/partition combination":**
> the partition in the script is not available to your account. Either set
> `#SBATCH --partition=<partition>` to a partition you may use in the script, or comment the `#SBATCH --partition=` line out entirely, or just use
> Option B.

### Option B — directly on a node you already hold (recommended when all nodes are `alloc`)
```bash
hostname                      # must show a GPU node, not the login node
cd /path/to/workspace/xmad/0_iteration
mkdir -p logs
pkill -f "vllm serve"         # clear any leftover server on port 7555
bash run_phase0.slurm 2>&1 | tee logs/manual_run.out
```
> The `#SBATCH` lines are just comments in this mode and are ignored — so the
> partition/account error cannot occur.
> **Do not `exit` the node before the run finishes** — if you leave, you may not
> get a new allocation. For a run that survives an SSH drop, wrap the last
> command in `tmux` (`tmux new -s phase0`, run, detach with `Ctrl+b` then `d`).
> Live output in this mode is `logs/manual_run.out` (there is no `phase0_<jobid>.out`).

### Optional overrides (defaults in brackets)
```bash
MODEL=Qwen/Qwen2.5-7B-Instruct   # [Qwen/Qwen2.5-7B-Instruct]
ALIAS=qwen                       # [qwen]
PORT=7555                        # [7555]
NPROMPTS=100                     # [100]
CONCURRENCY=8                    # [8]
MAXTOKENS=128                    # [128]
# e.g. (Option A):  NPROMPTS=200 CONCURRENCY=16 sbatch run_phase0.slurm
# e.g. (Option B):  NPROMPTS=200 CONCURRENCY=16 bash run_phase0.slurm
```

## GPU / resource notes
- Each node has 8x NVIDIA H200. The script does **not** pin a GPU; vLLM
  automatically uses GPU 0, the rest stay idle. No `nvidia-smi` step needed.
- CPU and RAM in `metrics.csv` are read from `/proc` and are therefore
  **node-level** (on a shared node they include other users). GPU/VRAM are exact.

## Output (one folder per run)
```
logs/run_<timestamp>_job<id>/
├── phase0.log            # human-readable log incl. R0.1–R0.4 verdicts
├── r0_1_env.txt          # python / torch / vllm / CUDA versions
├── r0_2_gpu.txt          # torch.cuda check + device name
├── r0_3_models.json      # /v1/models response (endpoint alive)
├── r0_4_response.json    # the single validation chat-completion answer
├── metrics.csv           # per-second CPU/RAM/GPU/VRAM  <- for diagrams
├── bench_results.csv     # per-request latency + tokens <- for diagrams
├── bench_summary.json    # benchmark only: throughput, latency p50/p90/p99
├── vllm_server.log       # full server log
├── vllm_throughput.txt   # vLLM's own throughput lines
└── phase0_summary.json   # everything: R0.1-R0.4 status + embedded benchmark
```
> `phase0_summary.json` is the one-file "truth" for the whole phase (it embeds
> `bench_summary.json`). Use it when comparing phases automatically later.
> `r0_4_response.json` contains **one** answer on purpose — R0.4 only proves the
> chain works; the 100-prompt run is measured in `bench_results.csv` (numbers,
> not answer texts).

## Requirement mapping
| Req | Checked by | Evidence file |
|-----|------------|---------------|
| **R0.1** env installed & persists on Lustre | stack imports; venv path under `/path/to/storage` | `r0_1_env.txt` |
| **R0.2** GPU visible from PyTorch | `torch.cuda.is_available()` on the node | `r0_2_gpu.txt` |
| **R0.3** single model served, stable endpoint | poll `/v1/models` until `200` | `r0_3_models.json` |
| **R0.4** valid chat-completion response | one request, parsed & non-empty | `r0_4_response.json` |

## Diagrams (later, locally)
Copy a run folder off the cluster, then:
```bash
python plot_phase0.py logs/run_<timestamp>_job<id>
```
Or hand the two CSVs (`metrics.csv`, `bench_results.csv`) over and the figures
can be generated directly.

## Reuse for Phases 1–4
`monitor.py` and `bench_phase0.py` are phase-agnostic. For later phases, copy
`run_phase0.slurm` to `run_phaseN.slurm`, start the additional server(s) /
components, and point the same benchmark + monitor at them — the CSV/JSON format
stays identical, so results are directly comparable across phases.