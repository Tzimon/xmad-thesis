#!/bin/bash
# xmad/2_iteration/stop_phase2_services.sh
#
# Stops the services started by start_phase2_services.sh. Prefers PID files,
# but falls back to matching each service's process pattern directly -- a
# pidfile can go missing (never written, removed, or from a since-ended
# allocation) without the underlying process actually being gone.
set -u

PROJECT_2=/path/to/workspace/xmad/2_iteration

stop_by_pidfile_or_pattern() {
  local name=$1 pattern=$2
  local pidfile="$PROJECT_2/pids/${name}.pid"
  local stopped=0

  if [ -f "$pidfile" ]; then
    local pid; pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop] stopping $name via pidfile (pid $pid)"
      kill "$pid"
      stopped=1
    else
      echo "[stop] pidfile for $name points to a dead pid ($pid); ignoring"
    fi
    rm -f "$pidfile"
  fi

  # fallback: match by process pattern regardless of whether the pidfile
  # worked, in case an older/duplicate instance is still running
  local matches
  matches=$(pgrep -f "$pattern" 2>/dev/null)
  if [ -n "$matches" ]; then
    echo "[stop] found $name process(es) by pattern, stopping: $matches"
    pkill -f "$pattern"
    stopped=1
  fi

  if [ "$stopped" -eq 0 ]; then
    echo "[stop] no running $name process found"
  fi
}

stop_by_pidfile_or_pattern "agent_a"    "vllm serve.*--port 7555"
stop_by_pidfile_or_pattern "agent_b"    "vllm serve.*--port 7556"
stop_by_pidfile_or_pattern "mcp_server" "xmad_rag.mcp_server"

echo "[stop] done. Verify with: ps aux | grep -E 'vllm serve|mcp_server' | grep -v grep"