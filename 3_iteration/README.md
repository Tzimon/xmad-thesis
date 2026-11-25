# Phase 3 — cross-model debates (dense vs. MoE)

Phase 3 runs the Phase 2 debate for **22 models from 5 vendors** (Qwen, Gemma 4,
Mistral, Nemotron, Meta) as 85 runs (`runs.yaml`) on the same 1000 HotpotQA
questions. Everything runs on **one node with 8 H200**. Several runs share the
node at the same time, each on its own GPUs.

## Files

| File | What it is |
|---|---|
| `models.yaml` | all models: architecture (dense/MoE, size, vendor), GPUs, vLLM flags, how reasoning is switched off |
| `runs.yaml` | all planned runs with fixed IDs `P3-001` … `P3-085` |
| `config/base.yaml` | the fixed conditions of every run (decoding, rounds, judge, serving) |
| `download_models.py` | downloads the models into the HF cache |
| `setup_venv.sh` | creates the Phase 3 venv (new vLLM) |
| `smoke_test.py` | 5-minute check per model before it is used |
| `phase3.py` | runs everything: `run`, `judge`, `status`, `remaining` |
| `run_phase3.sbatch` | batch job around `phase3.py run` (re-submits itself until done) |
| `run_debate.py` | the debate harness for one run (started by `phase3.py`) |
| `servers.py`, `serve_mcp.py` | start/stop vLLM and the MCP tool server |
| `xmad_debate/` | debate logic (Phase 2, plus reasoning switch, retries, judge as separate step) |
| `prompts/`, `templates/` | prompts (unchanged from Phase 2) and chat templates |
| `smoke/SMOKE_RESULTS.md` | result of the smoke tests (one line per model) |
| `results/OVERVIEW.md` | **the overview of all runs** — status, EM, ΔEM, energy, link to each folder |
| `tests/test_offline.py` | offline tests (no GPU): `python tests/test_offline.py` |

## Step by step

Long commands in tmux on the **login node** (`tmux new -s p3`; leave with the
command `tmux detach`; back with `tmux attach -t p3`). The experiment itself runs
as a **batch job** (step 5), so it does not depend on your SSH session.

```bash
cd /path/to/workspace/xmad && git pull
cd 3_iteration
export HF_HOME=/path/to/workspace/hf_cache
```

**1. Environment (once)** — on the login node
```bash
bash setup_venv.sh
source /path/to/workspace/venvs/vllm-p3/bin/activate
```

**2. Downloads** — login node; `--priority must` includes the judge model
```bash
unset HF_HUB_OFFLINE
python download_models.py --priority must 2>&1 | tee -a download_must.log
python download_models.py --priority should 2>&1 | tee -a download_should.log   # later
```

**3. Smoke tests** — on a GPU node (interactive allocation); result in `smoke/SMOKE_RESULTS.md`
```bash
python smoke_test.py --priority must --missing --gpus 0,1,2,3 2>&1 | tee -a smoke.log
```
Only models with a PASS are used; runs with an untested model show as `blocked`.

**4. Pre-test** — all homogeneous pairs with 30 questions (folders `results/*_test`),
then a plausibility table. Do not start the real runs before it is clean.
```bash
sbatch --export=ALL,MODE=pretest run_phase3.sbatch   # or interactively:
python phase3.py run --priority must --homogeneous --n 30
python phase3.py sanity --test                       # result also in pretest_sanity.txt
```
Flags: missing answers in round 1 > 1 %, errors > 0.5 %, truncated replies > 2 %,
round-1 EM < 0.30. A flagged model needs a look at its `debate_results.csv`.

**Repair of the first runs (once, 2026-10-06).** The first runs used an older
protocol (forced answer with `tool_choice="none"`, which gave empty replies for
Qwen/Gemma; two missing answers counted as consensus; invalid tool calls failed
the question). `repair` keeps every question the change did not touch, backs up
the old CSV (`debate_results_v1.csv`) and marks the run `pending`, so the next
`run` re-computes only the affected questions (list in `repair.json`).
```bash
cp -r results results_old            # full backup, not in git
python phase3.py repair --dry-run    # how many questions per run
python phase3.py repair
```

**5. The experiment** — as a batch job (whole node, re-submits itself)
```bash
python phase3.py run --dry-run --priority must    # what would run
sbatch run_phase3.sbatch                          # must-runs (51)
sbatch --export=ALL,PRIORITY=should run_phase3.sbatch   # later: must + should
squeue -u $USER                                   # is it running?
tail -f p3_<jobid>.out                            # progress
python phase3.py status && cat results/OVERVIEW.md
```
Time limit or preemption is handled: unfinished runs show as `interrupted`, the
job submits itself again and continues where it stopped. Stop for good with
`touch STOP` and `scancel --signal=TERM --batch <jobid>`. Failed runs (e.g. a
server that did not start) show the reason in `OVERVIEW.md`; start them again
with `python phase3.py run --retry-failed --only <IDs>`.

**6. Judge pass** — after the debates, GPU node with 4 free GPUs
```bash
python phase3.py judge --gpus 0,1,2,3
```
Debates without consensus wait as `pending_judge` until then. Only after this
step are EM and ΔEM in the overview final.

## One run folder

`results/P3-012_qwen3.5-27b__gemma4-31b/`

| File | Content |
|---|---|
| `run.yaml` | exact configuration (models, GPUs, ports, decoding, budgets) |
| `status.json` | state, start/end, duration, GPUs per agent, runs that ran at the same time |
| `debate.log` | progress of the run |
| `debate_results.csv` | one row per question: answer, EM/F1, rounds, round-1 answers, tokens … |
| `summary.json` | metrics: EM, F1, consensus, Jaccard, energy per agent, tokens, errors |
| `server_logs/` | vLLM and MCP logs (not in git) |
| `sessions/` | full debate per question (not in git) |

## Fixed conditions (as in Phase 2 unless noted)

- same 1000 questions as Phase 2 (seed 42), same prompts, temperature 0, top-p 1
- reasoning OFF for every debater (server-, request- or prompt-side, see `models.yaml`)
- **new:** `--generation-config vllm` on every server — no hidden model sampling defaults
- **new:** `max_tokens` 8192, judge as a separate pass (DeepSeek-V4-Flash, reasoning on)
- **new:** energy only on the GPUs of the run, reported per agent
- models that issue one tool call per reply get 12 instead of 6 turns per round
