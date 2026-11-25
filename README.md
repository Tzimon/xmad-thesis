# X-MAD: Cross-Model Multi-Agent Debate with a Shared Tool Pool via MCP

Code, configurations and results of the master's thesis
*X-MAD: Cross-Model Multi-Agent Debate with a Shared Tool Pool via MCP*
(Simon Spitzer, Karlsruhe Institute of Technology, Institute AIFB, 2026).

X-MAD lets two role-symmetric agents debate multi-hop questions. The two agents run on
**different open-weight LLMs** but share everything else: the system prompt, the decoding
settings and one tool pool that is connected through the **Model Context Protocol (MCP)**.
The model is therefore the only variable between the agents. The thesis asks whether
different foundation models diverge in their search on their own and whether this
divergence improves the answers.

## Main results

* 20 open-weight models from five vendors, 46 mixed pairs against the 20 homogeneous pairs.
* HotpotQA, MuSiQue and 2WikiMultiHopQA with 1000 questions each.
* Mixed pairs retrieve clearly different evidence, but their exact match changes by only
  +0.52, +0.35 and −0.67 percentage points on average over the better homogeneous pair.
* A mixed run occupies twice as many GPUs and needs 1.6 to 2.0 times the energy.
* All models and benchmarks were connected without changes to the agent or debate code.

The numbers behind these statements are in
[`3_iteration/results/OVERVIEW.md`](3_iteration/results/OVERVIEW.md) (Phase 3, HotpotQA) and
[`4_iteration/analysis/merged/REPORT.md`](4_iteration/analysis/merged/REPORT.md) (Phase 4, all
three benchmarks).

## Repository structure

The work was built in design science research cycles. Each folder is one phase and
corresponds to one section of Chapter 4 of the thesis.

| Folder | Phase | What it contains | Thesis |
|---|---|---|---|
| [`0_iteration/`](0_iteration/) | Phase 0, setup | automated check that a node can serve a model with vLLM, plus a latency and throughput benchmark | 4.3 |
| [`1_iteration/`](1_iteration/) | Phase 1, single agent | agent loop, MCP tool server, retriever comparison (BM25, dense), single-agent baseline over several model sizes | 4.4 |
| [`2_iteration/`](2_iteration/) | Phase 2, debate | round-based debate of one heterogeneous pair (Qwen2.5-7B + Llama-3.1-8B) on HotpotQA, judge, evidence divergence | 4.5 |
| [`3_iteration/`](3_iteration/) | Phase 3, cross-model | model registry, automated runs of 20 homogeneous and 46 mixed pairs on one 8×H200 node, judge pass, analysis | 4.6 |
| [`4_iteration/`](4_iteration/) | Phase 4, more benchmarks | benchmark adapters for MuSiQue and 2WikiMultiHopQA, the Phase 3 debate unchanged on both, merged analysis | 4.7 |

Phases 3 and 4 have their own README with all steps:
[`3_iteration/README.md`](3_iteration/README.md) and [`4_iteration/README.md`](4_iteration/README.md).

Inside each run folder (`3_iteration/results/P3-*`, `4_iteration/results/P4-*`) you find the
resolved configuration (`run.yaml`), the status, the summary and the per-question results
(`debate_results.csv`).

## Reproducing the experiments

The experiments ran on an on-premise HPC cluster with SLURM and nodes of eight NVIDIA H200
GPUs. The scripts expect such an environment.

* **Serving:** vLLM, one OpenAI-compatible server per model.
* **Tools:** an MCP server (FastMCP) that exposes the knowledge base of each question
  through one search tool, with `bge-base-en-v1.5` as dense retriever.
* **Environment:** the exact package versions of Phases 3 and 4 are in
  [`3_iteration/env_freeze.txt`](3_iteration/env_freeze.txt), and
  [`3_iteration/setup_venv.sh`](3_iteration/setup_venv.sh) creates the environment.
* **Models:** [`3_iteration/download_models.py`](3_iteration/download_models.py) downloads
  all models from Hugging Face. Gated models (Llama) need `HF_TOKEN` set in the environment.
* **Benchmarks:** [`4_iteration/download_benchmarks.py`](4_iteration/download_benchmarks.py)
  downloads the official releases and records URL, date and checksums in `SOURCE.json`.

Placeholder paths such as `/path/to/workspace/...` in the scripts and READMEs have to be adapted to
your own environment.

## Debate transcripts and GPU measurements

The full debate transcripts and the per-second GPU measurements are not versioned in Git
because of their size (about 2.2 GB uncompressed). They are attached as archives to the
release [`v1.0`](https://github.com/Tzimon/xmad-thesis/releases/tag/v1.0):

| Archive | Content |
|---|---|
| `xmad_phase2_transcripts.tar.gz` | all Phase 2 debate sessions (`2_iteration/logs/debate_*`) and the divergence comparison |
| `xmad_phase3_transcripts_gpu.tar.gz` | `3_iteration/results/*/sessions/` and `gpu_metrics_*.csv` of all scored runs |
| `xmad_phase4_transcripts_gpu.tar.gz` | `4_iteration/results/*/sessions/` and `gpu_metrics_*.csv` of all scored runs |
| `SHA256SUMS.txt` | checksums of the three archives |

Unpacked in the repository root, the archives restore the files at their original paths.
The transcripts contain passages from HotpotQA (CC BY-SA 4.0), MuSiQue (CC BY 4.0) and
2WikiMultiHopQA (Apache 2.0) and are shared under the terms of these licenses.
The vLLM server logs, the benchmark files and the model weights are not redistributed.
Benchmarks and models can be obtained from their official sources with the download
scripts above.

## Citation

```bibtex
@mastersthesis{spitzer2026xmad,
  author = {Spitzer, Simon},
  title  = {X-MAD: Cross-Model Multi-Agent Debate with a Shared Tool Pool via MCP},
  school = {Karlsruhe Institute of Technology (KIT)},
  type   = {Master's Thesis},
  year   = {2026}
}
```

## License

See [`LICENSE`](LICENSE).
