#!/bin/bash
# xmad/3_iteration/setup_venv.sh
#
# Creates the Phase 3 Python environment: a NEW venv next to the old one, so
# Phase 0-2 (vLLM 0.11.0 in /path/to/workspace/venvs/vllm) stay reproducible.
# The Phase 3 models need vLLM >= 0.29 (Qwen3.8-Flash-Next 0.29, Muse and
# DeepSeek-V4 0.28, Gemma 4 0.25).
#
#   bash setup_venv.sh                       # newest vLLM release
#   VLLM_VERSION=0.29.1 bash setup_venv.sh   # a fixed version
#
# Writes ENVIRONMENT.txt (versions, for the thesis appendix) and env_freeze.txt.
set -euo pipefail

VENV=${VENV:-/path/to/workspace/venvs/vllm-p3}
VLLM_VERSION=${VLLM_VERSION:-}
HERE="$(cd "$(dirname "$0")" && pwd)"

module load cray-python/3.11.7 2>/dev/null || echo "WARN: module load cray-python/3.11.7 failed -- using $(python3 --version)"

if [ -d "$VENV" ]; then
  echo "venv $VENV exists already -- reusing it (delete it first for a clean install)"
else
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# one resolver call, so vLLM's pins and the other packages are resolved together
pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}" fastmcp datasets openai pyyaml
pip install -e "$HERE/../1_iteration"          # xmad_rag (retriever, MCP server, metrics)

VER=$(python -c "import vllm; print(vllm.__version__)")
python - "$VER" <<'PY'
import sys
v = tuple(int(x) for x in sys.argv[1].split("+")[0].split(".")[:2])
print(f"vLLM {sys.argv[1]}")
if v < (0, 29):
    sys.exit("ERROR: vLLM >= 0.29 is required for the Phase 3 models")
PY

# Gemma 4 tool-call chat template (ships with the vLLM sources, not with the model)
mkdir -p "$HERE/templates"
URL="https://raw.githubusercontent.com/vllm-project/vllm/v${VER}/examples/tool_chat_template_gemma4.jinja"
if curl -fsSL "$URL" -o "$HERE/templates/tool_chat_template_gemma4.jinja"; then
  echo "Gemma 4 chat template -> templates/tool_chat_template_gemma4.jinja"
else
  echo "WARN: could not download $URL -- get examples/tool_chat_template_gemma4.jinja for vLLM $VER by hand"
fi

pip check || echo "WARN: pip check reports conflicts (see above) -- usually harmless, smoke_test.py will tell"
python -c "import vllm, fastmcp, datasets, openai, yaml, xmad_rag.metrics; print('imports OK')"

{
  echo "Phase 3 environment, created $(date -Iseconds) on $(hostname)"
  echo "venv: $VENV"
  python --version
  python -c "import vllm, torch, transformers, fastmcp, openai; print('vllm', vllm.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('fastmcp', fastmcp.__version__); print('openai', openai.__version__)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 \
    || echo "GPU/driver: (no nvidia-smi here -- login node; run on a GPU node: nvidia-smi --query-gpu=name,driver_version --format=csv >> ENVIRONMENT.txt)"
} | tee "$HERE/ENVIRONMENT.txt"
pip freeze > "$HERE/env_freeze.txt"
echo
echo "done. Activate with:  source $VENV/bin/activate"
