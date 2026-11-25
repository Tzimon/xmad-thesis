#!/bin/bash
#SBATCH --job-name=xmad-sweep
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=sweep_%j.out
# ---------------------------------------------------------------------------
# Cross-model single-agent baseline sweep.
#
# For each Qwen model: start vLLM (under the fixed alias 'qwen'), measure how
# long it took to become ready, run the FULL baseline over the dev split with
# GPU power/energy sampling, then tear vLLM down and move to the next model.
# The MCP server (model-independent, holds the bge retriever) runs once for the
# whole sweep.
#
# Run either as an sbatch job:   sbatch sweep_models.sh
# or interactively on the node:  bash sweep_models.sh
# ---------------------------------------------------------------------------
set -u

# --- paths / config (edit to your environment) -----------------------------
VENV=/path/to/workspace/venvs/vllm
PROJECT=/path/to/workspace/xmad/1_iteration
export HF_HOME=/path/to/workspace/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
KWH_PRICE_EUR=0.30          # electricity price for the local-cost estimate
N_ARG=""                    # "" = whole dev split (7405); e.g. "--n 500" to test

# model label | HF id | tensor-parallel size (72B needs 2 GPUs)
MODELS=(
  "32B|Qwen/Qwen2.5-32B-Instruct|1"
  "72B|Qwen/Qwen2.5-72B-Instruct|2"
)


source "$VENV/bin/activate"
cd "$PROJECT"

# --- MCP server once for the whole sweep -----------------------------------
echo "[sweep] starting MCP server ..."
python -m xmad_rag.mcp_server > mcp_sweep.log 2>&1 &
MCP_PID=$!
sleep 20   # let bge load

cleanup() {
  echo "[sweep] cleanup"
  kill "$MCP_PID" 2>/dev/null
  kill "${VLLM_PID:-}" 2>/dev/null
}
trap cleanup EXIT

# --- loop over models ------------------------------------------------------
for entry in "${MODELS[@]}"; do
  IFS="|" read -r LABEL MODEL_ID TP <<< "$entry"
  echo "=================================================================="
  echo "[sweep] model $LABEL ($MODEL_ID)  TP=$TP"

  # start vLLM, timing the load
  LOAD_START=$(date +%s.%N)
  vllm serve "$MODEL_ID" \
      --served-model-name qwen --port 7555 \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --tensor-parallel-size "$TP" \
      > "vllm_${LABEL}.log" 2>&1 &
  VLLM_PID=$!

  # wait until the endpoint answers, then record load time
  echo "[sweep] waiting for vLLM ($LABEL) ..."
  READY=0
  for i in $(seq 1 900); do   # up to 30 min (72B load is slow)
    if curl -s http://127.0.0.1:7555/v1/models 2>/dev/null | grep -q '"id"'; then
      READY=1; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[sweep] ERROR: vLLM ($LABEL) died during load; see vllm_${LABEL}.log"; break
    fi
    sleep 2
  done
  if [ "$READY" -ne 1 ]; then
    echo "[sweep] skipping $LABEL (never became ready)"; kill "$VLLM_PID" 2>/dev/null; sleep 10; continue
  fi
  LOAD_TIME=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $LOAD_START}")
  echo "[sweep] $LABEL ready in ${LOAD_TIME}s"

  # run the full baseline for this model
  python -u -m xmad_rag.run_baseline $N_ARG \
      --label "$LABEL" --load-time-s "$LOAD_TIME" \
      --tp "$TP" --kwh-price-eur "$KWH_PRICE_EUR"

  # tear vLLM down and wait for the GPUs to free before the next model
  echo "[sweep] stopping vLLM ($LABEL) ..."
  kill "$VLLM_PID" 2>/dev/null
  wait "$VLLM_PID" 2>/dev/null
  sleep 15
done

echo "[sweep] all models done. Building comparison ..."
python compare_models.py
echo "[sweep] finished."
