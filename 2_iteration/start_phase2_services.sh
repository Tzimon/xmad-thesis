#!/bin/bash
# xmad/2_iteration/start_phase2_services.sh
#
# Starts everything a debate run needs, in the background, in one command:
#   - agent A's vLLM server (Qwen, hermes tool parser)
#   - agent B's vLLM server (Llama, llama3_json tool parser)
#   - ONE MCP server (holds the shared retriever/corpus)
#
# Only ONE MCP server is needed, not two: the corpus it holds is global server
# state, so both agents' MCP client CONNECTIONS (opened by run_debate.py) read
# from the same loaded corpus regardless of which connection asks. A second
# server process would just duplicate the retriever in memory and risk the
# two servers' corpora going out of sync.
#
# System prompts are NOT set here -- they are sent per API request (from
# config/debate.yaml's prompt files), not baked into the server at start-up.
# This script only sets each model's TOOL-CALLING configuration, which is a
# genuine per-model-family difference.
#
# Usage:
#   bash start_phase2_services.sh          # start everything
#   bash stop_phase2_services.sh           # stop everything
#
# ASSUMPTION TO VERIFY: agent B (Llama) is started WITHOUT an explicit
# --chat-template flag below, matching what worked in manual testing. If tool
# calls come back malformed, uncomment AGENT_B_CHAT_TEMPLATE and point it at
# vLLM's examples/tool_chat_template_llama3.1_json.jinja for your vLLM version.
set -u

# --- paths / config (edit to your environment) ------------------------------
VENV=/path/to/workspace/venvs/vllm
PROJECT_1=/path/to/workspace/xmad/1_iteration   # MCP server + retriever (Phase 1 code, reused)
PROJECT_2=/path/to/workspace/xmad/2_iteration
export HF_HOME=/path/to/workspace/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

AGENT_A_MODEL="Qwen/Qwen2.5-7B-Instruct"
AGENT_A_ALIAS="qwen"
AGENT_A_PORT=7555
AGENT_A_GPU=0
AGENT_A_PARSER="hermes"

AGENT_B_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"
AGENT_B_ALIAS="llama"
AGENT_B_PORT=7556
AGENT_B_GPU=1
AGENT_B_PARSER="llama3_json"
# AGENT_B_CHAT_TEMPLATE="/path/to/workspace/xmad/2_iteration/tool_chat_template_llama3.1_json.jinja"

MCP_PORT=7600

source "$VENV/bin/activate"
mkdir -p "$PROJECT_2/logs" "$PROJECT_2/pids"
cd "$PROJECT_2"

wait_ready() {
  local port=$1 name=$2
  echo "[start] waiting for $name on :$port ..."
  for i in $(seq 1 300); do
    if curl -s "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q '"id"'; then
      echo "[start] $name ready"
      return 0
    fi
    sleep 2
  done
  echo "[start] WARNING: $name did not become ready in time -- check its log"
  return 1
}

echo "[start] launching agent A (${AGENT_A_ALIAS}) on GPU ${AGENT_A_GPU}, port ${AGENT_A_PORT} ..."
CUDA_VISIBLE_DEVICES=$AGENT_A_GPU nohup vllm serve "$AGENT_A_MODEL" \
    --served-model-name "$AGENT_A_ALIAS" --port "$AGENT_A_PORT" \
    --enable-auto-tool-choice --tool-call-parser "$AGENT_A_PARSER" \
    > "$PROJECT_2/logs/agent_a.log" 2>&1 &
echo $! > "$PROJECT_2/pids/agent_a.pid"

echo "[start] launching agent B (${AGENT_B_ALIAS}) on GPU ${AGENT_B_GPU}, port ${AGENT_B_PORT} ..."
CUDA_VISIBLE_DEVICES=$AGENT_B_GPU nohup vllm serve "$AGENT_B_MODEL" \
    --served-model-name "$AGENT_B_ALIAS" --port "$AGENT_B_PORT" \
    --enable-auto-tool-choice --tool-call-parser "$AGENT_B_PARSER" \
    ${AGENT_B_CHAT_TEMPLATE:+--chat-template "$AGENT_B_CHAT_TEMPLATE"} \
    > "$PROJECT_2/logs/agent_b.log" 2>&1 &
echo $! > "$PROJECT_2/pids/agent_b.pid"

echo "[start] launching MCP server on port ${MCP_PORT} ..."
( cd "$PROJECT_1" && nohup python -u -m xmad_rag.mcp_server \
    > "$PROJECT_2/logs/mcp_server.log" 2>&1 & echo $! > "$PROJECT_2/pids/mcp_server.pid" )

wait_ready "$AGENT_A_PORT" "agent A (${AGENT_A_ALIAS})"
wait_ready "$AGENT_B_PORT" "agent B (${AGENT_B_ALIAS})"

echo "[start] waiting for MCP server on :${MCP_PORT} ..."
sleep 8   # the retriever (bge) needs a few seconds to load; no /v1/models-style endpoint to poll
if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${MCP_PORT}/mcp" | grep -qE "^(200|400|406)$"; then
  echo "[start] MCP server responding"
else
  echo "[start] WARNING: MCP server may not be ready -- check logs/mcp_server.log"
fi

echo ""
echo "[start] all services launched. PIDs in ${PROJECT_2}/pids/, logs in ${PROJECT_2}/logs/"
echo "[start] agent A (${AGENT_A_ALIAS}): http://127.0.0.1:${AGENT_A_PORT}/v1"
echo "[start] agent B (${AGENT_B_ALIAS}): http://127.0.0.1:${AGENT_B_PORT}/v1"
echo "[start] MCP server:                 http://127.0.0.1:${MCP_PORT}/mcp"
echo ""
echo "[start] next: python run_debate.py --n 20      # sanity check"
echo "[start] stop everything with:       bash stop_phase2_services.sh"
