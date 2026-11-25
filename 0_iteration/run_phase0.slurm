#!/bin/bash
#SBATCH --job-name=xmad-p0
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err
#
# Phase 0 - Environment Setup and Single-Model Hosting (automated proof).
# Submit from inside xmad/0_iteration:   mkdir -p logs && sbatch run_phase0.slurm
# Optional overrides:  MODEL=... ALIAS=... PORT=... NPROMPTS=... sbatch run_phase0.slurm
#
# Verifies requirements R0.1-R0.4, benchmarks throughput/latency over N prompts,
# and records CPU/RAM/GPU/VRAM utilisation per second. All artefacts land in
# logs/run_<timestamp>_job<id>/ for later plotting.

# ----------------------------- configuration -----------------------------
VENV=${VENV:-/path/to/workspace/venvs/vllm}
export HF_HOME=${HF_HOME:-/path/to/workspace/hf_cache}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
ALIAS=${ALIAS:-qwen}
PORT=${PORT:-7555}
NPROMPTS=${NPROMPTS:-100}
CONCURRENCY=${CONCURRENCY:-8}
MAXTOKENS=${MAXTOKENS:-128}
READY_TIMEOUT=${READY_TIMEOUT:-1200}   # seconds to wait for the server to come up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------- run directory + log -------------------------
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/logs/run_${STAMP}_job${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR"
MAIN_LOG="$RUN_DIR/phase0.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }
declare -A REQ   # requirement pass/fail table

# ------------------------------- cleanup ---------------------------------
SRV_PID=""; MON_PID=""
cleanup() {
  log "Cleanup: stopping background processes."
  [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
  # give vLLM a moment to release the GPU
  [ -n "$SRV_PID" ] && sleep 5 && kill -9 "$SRV_PID" 2>/dev/null
}
trap cleanup EXIT

log "=============================================================="
log "X-MAD Phase 0 automated run"
log "node=$(hostname)  job=${SLURM_JOB_ID:-NA}  model=$MODEL  port=$PORT"
log "run_dir=$RUN_DIR"
log "=============================================================="

# --------------------------- environment setup ---------------------------
module load cray-python/3.11.7 2>/dev/null || log "WARN: module load failed (continuing)"
# shellcheck disable=SC1090
source "$VENV/bin/activate" || { log "FATAL: cannot activate venv at $VENV"; exit 1; }

# ============================ R0.1 : environment =========================
# "Isolated, reproducible env with the full inference stack, persists on Lustre."
log "--- R0.1: inference environment ---"
python - > "$RUN_DIR/r0_1_env.txt" 2>&1 <<'PY'
import sys
try:
    import torch, vllm
    print("python", sys.version.split()[0])
    print("torch", torch.__version__)
    print("torch_cuda_build", torch.version.cuda)
    print("vllm", vllm.__version__)
    print("import_ok", True)
except Exception as e:
    print("import_ok", False, "err", e)
PY
cat "$RUN_DIR/r0_1_env.txt" | tee -a "$MAIN_LOG"
if grep -q "import_ok True" "$RUN_DIR/r0_1_env.txt" && [[ "$VENV" == /path/to/storage/* ]]; then
  REQ[R0.1]=PASS; log "R0.1 PASS (stack imports; env on shared Lustre path: $VENV)"
else
  REQ[R0.1]=FAIL; log "R0.1 FAIL"
fi

# ============================ R0.2 : GPU visible =========================
log "--- R0.2: accelerator visible to PyTorch ---"
python - > "$RUN_DIR/r0_2_gpu.txt" 2>&1 <<'PY'
import torch
ok = torch.cuda.is_available()
print("cuda_available", ok)
if ok:
    print("device", torch.cuda.get_device_name(0))
    print("device_count", torch.cuda.device_count())
PY
cat "$RUN_DIR/r0_2_gpu.txt" | tee -a "$MAIN_LOG"
if grep -q "cuda_available True" "$RUN_DIR/r0_2_gpu.txt"; then
  REQ[R0.2]=PASS; log "R0.2 PASS ($(grep device\  "$RUN_DIR/r0_2_gpu.txt"))"
else
  REQ[R0.2]=FAIL; log "R0.2 FAIL (no GPU visible)"
fi

# ------------------- start resource monitor (background) -----------------
log "Starting resource monitor -> $RUN_DIR/metrics.csv"
python "$SCRIPT_DIR/monitor.py" --out "$RUN_DIR/metrics.csv" --interval 1 &
MON_PID=$!

# ============================ R0.3 : serve model =========================
# "A single open-weight model can be served locally behind a stable endpoint."
log "--- R0.3: serving $MODEL on port $PORT ---"
vllm serve "$MODEL" --served-model-name "$ALIAS" --port "$PORT" \
      > "$RUN_DIR/vllm_server.log" 2>&1 &
SRV_PID=$!
log "vLLM server PID=$SRV_PID; waiting for readiness (timeout ${READY_TIMEOUT}s)..."

ready=0
elapsed=0
while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    log "R0.3 FAIL: server process exited early. See vllm_server.log"; break
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/v1/models" || true)
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 3; elapsed=$((elapsed+3))
done
if [ "$ready" = "1" ]; then
  REQ[R0.3]=PASS; log "R0.3 PASS (endpoint ready after ~${elapsed}s)"
  curl -s "http://localhost:$PORT/v1/models" > "$RUN_DIR/r0_3_models.json"
else
  REQ[R0.3]=FAIL; log "R0.3 FAIL (endpoint not ready within ${READY_TIMEOUT}s)"
fi

# ============================ R0.4 : valid answer ========================
# "A chat-completion request returns a valid response (chain works end to end)."
if [ "${REQ[R0.3]}" = "PASS" ]; then
  log "--- R0.4: single chat-completion request ---"
  curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one short sentence: what is a large language model?\"}],\"max_tokens\":64}" \
    > "$RUN_DIR/r0_4_response.json" 2>&1
  if python - "$RUN_DIR/r0_4_response.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    c=d["choices"][0]["message"]["content"].strip()
    print("ANSWER:", c[:200])
    sys.exit(0 if c else 1)
except Exception as e:
    print("parse_error", e); sys.exit(1)
PY
  then
    REQ[R0.4]=PASS; log "R0.4 PASS (valid response received)"
    grep "ANSWER:" "$MAIN_LOG" | tail -1
  else
    REQ[R0.4]=FAIL; log "R0.4 FAIL (no valid response)"
  fi
else
  REQ[R0.4]=SKIP; log "R0.4 SKIP (server not ready)"
fi

# ============================ benchmark N prompts ========================
if [ "${REQ[R0.4]}" = "PASS" ]; then
  log "--- Benchmark: $NPROMPTS prompts, concurrency=$CONCURRENCY, max_tokens=$MAXTOKENS ---"
  python "$SCRIPT_DIR/bench_phase0.py" \
      --base-url "http://localhost:$PORT/v1" \
      --model "$ALIAS" \
      --n "$NPROMPTS" --concurrency "$CONCURRENCY" \
      --max-tokens "$MAXTOKENS" --warmup 5 \
      --out-csv "$RUN_DIR/bench_results.csv" \
      --out-summary "$RUN_DIR/bench_summary.json" 2>&1 | tee -a "$MAIN_LOG"
  # capture vLLM's own throughput reporting
  grep -E "throughput|Avg" "$RUN_DIR/vllm_server.log" | tail -5 > "$RUN_DIR/vllm_throughput.txt" 2>/dev/null || true
else
  log "Benchmark SKIPPED (R0.4 not passed)."
fi

# ------------------------------ summary ----------------------------------
log "=============================================================="
log "Requirement results:"
for r in R0.1 R0.2 R0.3 R0.4; do log "  $r : ${REQ[$r]:-NA}"; done
python - "$RUN_DIR" "${REQ[R0.1]:-NA}" "${REQ[R0.2]:-NA}" "${REQ[R0.3]:-NA}" "${REQ[R0.4]:-NA}" <<'PY'
import json,sys,os
run,*st=sys.argv[1:]
keys=["R0.1","R0.2","R0.3","R0.4"]
summary={"run_dir":run,"requirements":dict(zip(keys,st))}
bs=os.path.join(run,"bench_summary.json")
if os.path.exists(bs):
    summary["benchmark"]=json.load(open(bs))
json.dump(summary, open(os.path.join(run,"phase0_summary.json"),"w"), indent=2)
print("Wrote", os.path.join(run,"phase0_summary.json"))
PY
log "All artefacts in: $RUN_DIR"
log "=============================================================="

# non-zero exit if any hard requirement failed
for r in R0.1 R0.2 R0.3 R0.4; do
  [ "${REQ[$r]}" = "FAIL" ] && exit 2
done
exit 0
