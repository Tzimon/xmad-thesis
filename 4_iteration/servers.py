"""
xmad/4_iteration/servers.py (copied from 3_iteration, only comments changed)

Starting and stopping the processes a run needs: vLLM model servers and the
MCP tool server. Shared by smoke_test.py and phase4.py.

Every vLLM server is started as
    vllm serve <hf_id> --served-model-name <key> --port <p>
        --tensor-parallel-size <gpus> --generation-config vllm --seed 0
        --max-model-len ... --gpu-memory-utilization ...
        --enable-auto-tool-choice --tool-call-parser <parser>
        <model-specific vllm_args from models.yaml>
with CUDA_VISIBLE_DEVICES set to exactly the GPUs assigned to it.

--generation-config vllm: vLLM otherwise loads the model's generation_config.json
and silently uses its sampling defaults (e.g. a repetition penalty) for every
parameter a request does not set -- different per model, which would break the
identical-decoding requirement (C1.6).
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PROJECT_1 = HERE.parent / "1_iteration"


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models():
    return load_yaml(HERE / "models.yaml")["models"]


def load_base():
    return load_yaml(HERE / "config" / "base.yaml")


def _resolve_arg(arg, model):
    """templates/... -> absolute path in 4_iteration; hf:<file> -> file in the HF snapshot."""
    if isinstance(arg, str) and arg.startswith("templates/"):
        return str(HERE / arg)
    if isinstance(arg, str) and arg.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model["hf_id"], arg[3:])
    return str(arg)


def vllm_command(key, model, port, tool_parser, serving):
    cmd = ["vllm", "serve", model["hf_id"],
           "--served-model-name", key,
           "--host", "127.0.0.1", "--port", str(port),
           "--tensor-parallel-size", str(model["gpus"]),
           "--generation-config", "vllm",
           "--seed", str(serving["vllm_seed"]),
           "--max-model-len", str(serving["max_model_len"]),
           "--gpu-memory-utilization", str(serving["gpu_memory_utilization"])]
    if tool_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    cmd += [_resolve_arg(a, model) for a in model.get("vllm_args", [])]
    return cmd


def _env(gpus):
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"          # same numbering as nvidia-smi
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env.setdefault("HF_HUB_OFFLINE", "1")             # everything is already downloaded
    env["PYTHONUNBUFFERED"] = "1"
    return env


def start_vllm(key, model, gpus, port, tool_parser, serving, log_path):
    """Start one model server in its own process group; returns the Popen."""
    assert len(gpus) == model["gpus"], f"{key} needs {model['gpus']} GPUs, got {gpus}"
    cmd = vllm_command(key, model, port, tool_parser, serving)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    log.write(f"# GPUs {gpus}\n# {' '.join(cmd)}\n\n"); log.flush()
    return subprocess.Popen(cmd, env=_env(gpus), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def start_mcp(gpu, port, log_path):
    """Start the MCP tool server (Phase 1 retriever) on one GPU of the run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    env = _env([gpu])
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_1), env.get("PYTHONPATH", "")])
    return subprocess.Popen([sys.executable, str(HERE / "serve_mcp.py"), "--port", str(port)],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_vllm_ready(port, proc, timeout_s):
    """True once /v1/models answers; False if the process died or timed out."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if b'"id"' in r.read():
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def wait_port_open(port, proc, timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            s.settimeout(2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(3)
    return False


def stop(proc, grace_s=30):
    """Terminate a process group started by this module (vLLM spawns workers)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def tail(path, n=15):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return ""
