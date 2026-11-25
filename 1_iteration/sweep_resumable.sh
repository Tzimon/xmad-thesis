#!/bin/bash
#SBATCH --job-name=xmad-sweep
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=sweep_%j.out
# ---------------------------------------------------------------------------
# Preemptible, resumable cross-model baseline sweep for low-priority nodes.
#
# Submit ONCE:   sbatch sweep_resumable.sh
# It then queues, starts when a node is free, and for each Qwen model runs only
# the questions not yet done (resuming after a preemption). If it is preempted or
# hits the time limit before all models are complete, it resubmits itself, so it
# eventually finishes across as many runs as needed with no manual action.
# When every model has all N questions, it runs merge_models.py for the table.
#
# The completed work is safe regardless: run_baseline writes each question to its
# chunk CSV immediately, and merge_models.py stitches the chunks together.
# ---------------------------------------------------------------------------
set -u

# --- config (edit to your environment) -------------------------------------
SELF=/path/to/workspace/xmad/1_iteration/sweep_resumable.sh   # for self-resubmit
VENV=/path/to/workspace/venvs/vllm
PROJECT=/path/to/workspace/xmad/1_iteration
export HF_HOME=/path/to/workspace/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
KWH_PRICE_EUR=0.30
TOTAL=7405                 # questions per model (whole dev split)
MAX_RESUBMIT=50            # safety cap against endless no-progress resubmits

MODELS=(
  "72B|Qwen/Qwen2.5-72B-Instruct|2"
)

source "$VENV/bin/activate"
cd "$PROJECT"

# --- how many questions of a label are already done (next start index) ------
done_count() {
  python3 - "$1" << 'PY'
import sys, csv, glob
label = sys.argv[1]; mx = -1
for f in glob.glob(f"logs/baseline_{label}_*/baseline_results.csv"):
    try:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                try: mx = max(mx, int(row["idx"]))
                except (ValueError, KeyError): pass
    except FileNotFoundError: pass
print(mx + 1)
PY
}

total_done() {
  local sum=0
  for entry in "${MODELS[@]}"; do
    IFS="|" read -r L _ _ <<< "$entry"; sum=$(( sum + $(done_count "$L") ))
  done
  echo "$sum"
}

DONE_AT_START=$(total_done)
RESUBMITTED=0

maybe_resubmit() {
  # all complete? -> merge and stop. else resubmit (with progress-aware cap).
  local incomplete=0
  for entry in "${MODELS[@]}"; do
    IFS="|" read -r L _ _ <<< "$entry"
    [ "$(done_count "$L")" -lt "$TOTAL" ] && incomplete=1
  done
  if [ "$incomplete" -eq 0 ]; then
    echo "[sweep] all models complete -> merging"; python merge_models.py; return
  fi
  mkdir -p logs
  local cnt=0; [ -f logs/.resubmit_count ] && cnt=$(cat logs/.resubmit_count)
  if [ "$(total_done)" -gt "$DONE_AT_START" ]; then cnt=0; fi   # progress -> reset cap
  if [ "$cnt" -ge "$MAX_RESUBMIT" ]; then
    echo "[sweep] resubmit cap reached without progress; stopping. Resubmit: sbatch $SELF"; return
  fi
  echo $((cnt + 1)) > logs/.resubmit_count
  echo "[sweep] work remains -> resubmitting ($((cnt + 1))/$MAX_RESUBMIT)"
  sbatch "$SELF"
}

on_term() {   # preemption / time limit sends SIGTERM first (grace period)
  if [ "$RESUBMITTED" -eq 0 ]; then
    RESUBMITTED=1
    echo "[sweep] SIGTERM received -> attempting resubmit before shutdown"
    kill "${VLLM_PID:-}" 2>/dev/null
    maybe_resubmit
  fi
  exit 1
}
cleanup() { kill "${MCP_PID:-}" "${VLLM_PID:-}" 2>/dev/null; }
trap on_term TERM
trap cleanup EXIT

# --- MCP server once for this run -------------------------------------------
echo "[sweep] starting MCP server ..."
python -u -m xmad_rag.mcp_server > mcp_sweep.log 2>&1 &
MCP_PID=$!
sleep 20

# --- loop over models, running only what's left -----------------------------
for entry in "${MODELS[@]}"; do
  IFS="|" read -r LABEL MODEL_ID TP <<< "$entry"
  DONE=$(done_count "$LABEL")
  REMAIN=$(( TOTAL - DONE ))
  echo "=================================================================="
  if [ "$REMAIN" -le 0 ]; then
    echo "[sweep] $LABEL already complete ($DONE/$TOTAL) -> skip"; continue
  fi
  echo "[sweep] $LABEL: $DONE/$TOTAL done, running $REMAIN more (TP=$TP)"

  LOAD_START=$(date +%s.%N)
  vllm serve "$MODEL_ID" \
      --served-model-name qwen --port 7555 \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --tensor-parallel-size "$TP" \
      > "vllm_${LABEL}.log" 2>&1 &
  VLLM_PID=$!

  READY=0
  for i in $(seq 1 900); do
    if curl -s http://127.0.0.1:7555/v1/models 2>/dev/null | grep -q '"id"'; then READY=1; break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "[sweep] vLLM $LABEL died; see vllm_${LABEL}.log"; break; fi
    sleep 2
  done
  if [ "$READY" -ne 1 ]; then echo "[sweep] $LABEL not ready -> skip this pass"; kill "$VLLM_PID" 2>/dev/null; sleep 10; continue; fi
  LOAD_TIME=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $LOAD_START}")
  echo "[sweep] $LABEL ready in ${LOAD_TIME}s"

  python -u -m xmad_rag.run_baseline --n "$REMAIN" --start "$DONE" \
      --label "$LABEL" --load-time-s "$LOAD_TIME" \
      --tp "$TP" --kwh-price-eur "$KWH_PRICE_EUR"

  echo "[sweep] stopping vLLM ($LABEL) ..."
  kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; sleep 15
done

# --- reached the end without being killed: merge or resubmit ----------------
RESUBMITTED=1
maybe_resubmit
echo "[sweep] pass finished."
