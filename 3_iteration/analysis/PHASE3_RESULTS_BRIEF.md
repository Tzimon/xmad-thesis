# X-MAD Phase 3 — Results Brief for a Presentation

**Purpose of this document.** Complete, self-contained summary of the Phase 3
results of the Master's thesis *"X-MAD: Cross-Model Multi-Agent Debate with a
Shared Tool Pool via MCP"* (KIT, M.Sc. Information Systems). It is written to be
handed to a presentation designer: every number is final and taken from the
analysis output (`3_iteration/analysis/`), every figure is listed with what it
shows and where it fits. Suggested slide structure at the end (Section 10).

**Status:** final. All 72 runs finished, all debates without consensus decided by
the judge, duplicates removed (see Section 9). Generated 2026-10-15.

---

## 1. One-slide summary (the story in four sentences)

1. Mixing two *different* LLMs in a debate makes them retrieve **measurably different
   evidence** (evidence overlap drops from 0.98 to 0.76 on average) — the effect that
   Phase 2 showed is real and holds across 46 model pairs from 5 vendors.
2. That divergence **does not turn into better answers** for current models: on
   average a mixed pair is only **+0.5 pp EM** better than the better of its two
   models on their own; 30 of 46 pairs are slightly better, none is significantly worse.
3. A significant gain appears **only for weak models**: the two significant pairs are
   Qwen2.5-7B + Llama-3.1-8B (**+4.7 pp**, the Phase 2 pair, reproduced) and
   Llama-3.1-8B + Llama-3.3-70B (**+3.9 pp**). The stronger the models, the smaller the
   gain (correlation **r = −0.50**).
4. **Dense vs. Mixture-of-Experts is not a lever**: pairs that mix dense and MoE models do
   not gain more than dense+dense or MoE+MoE pairs (regression coefficients ≈ 0).

---

## 2. Research context

- **Research question RQ1:** Does the natural divergence between architecturally
  different LLMs improve the accuracy of a knowledge-augmented multi-agent debate?
- **Phase 2 (previous iteration):** one heterogeneous pair (Qwen2.5-7B + Llama-3.1-8B)
  beat both homogeneous pairs significantly (EM 0.522 vs. 0.472 / 0.412).
- **Phase 3 (this document):** scales the question to 20 current open-weight models from
  5 vendors and asks **where the difference comes from**: dense vs. MoE architecture,
  model size, vendor (training lineage), model strength.

## 3. Experimental setup (identical for every run)

| Aspect | Setting |
|---|---|
| Benchmark | HotpotQA, distractor setting, validation split — the **same 1000 questions** as Phase 2 (random sample, seed 42) |
| Debate | 2 agents, no roles, identical system prompt, parallel rounds, max. 10 rounds, stop on exact-match consensus or stalemate |
| Tools | one shared MCP tool pool: dense retriever (bge-base) over the 10 paragraphs of each question |
| Decoding | temperature 0.0, top-p 1.0, max 8192 tokens; no model-specific sampling defaults (`--generation-config vllm`) |
| Reasoning | switched off for every debater (Muse Glimmer: lowest effort "low", it has no off switch) |
| Search budget | 6 tool turns per agent and round (12 for models that issue one call per reply); if exhausted, the agent is asked once without tools for its answer ("forced answer", counted) |
| Judge | DeepSeek-V4-Flash (vendor not among the debaters), reasoning on, run as a separate pass; only for debates without consensus |
| Serving | vLLM 0.31.0 on one node with 8× NVIDIA H200 (141 GB); several runs in parallel, each on its own GPUs |
| Measures | Exact match (EM) and token F1 vs. gold answer; evidence overlap = Jaccard of the passage titles both agents retrieved in round 1; GPU energy (nvidia-smi) |
| Baseline per pair | the **better of the two homogeneous pairs** (each model debating itself): ΔEM = EM(A+B) − max(EM(A+A), EM(B+B)) |
| Significance | McNemar's exact test on the 1000 paired answers, **Benjamini-Hochberg correction** over all 46 mixed pairs (q < 0.05); run-to-run noise from 3 replicate runs: **≤ 0.1 pp** |

**Scale of the experiment:** 72 runs (20 homogeneous incl. anchors, 46 mixed, 6
replicates), 72 000 debated questions, ~766 M tokens, ~89 kWh measured GPU energy (slightly
overstated, see Section 9), 67 run-hours in total — several runs shared the 8-GPU
node at the same time, so the wall-clock time was shorter.

## 4. Model pool

20 models from 5 vendors: **Alibaba** (Qwen2.5, Qwen3.5, Qwen3.8), **Google** (Gemma 4),
**Mistral** (Ministral 3, Mistral Small 4), **NVIDIA** (Nemotron 3), **Meta** (Llama 3.x,
Muse Glimmer). 14 dense, 6 MoE; from 4 B to 397 B parameters (3–70 B active).

Excluded after testing (disclosed): Nemotron-3-Nano-30B and Nemotron-3-Super-120B (with
reasoning off they never answer — they keep calling the search tool), Nemotron-Nano-9B-v2
(tool parser incompatible with vLLM 0.31), Llama 4 (license excludes multimodal models in
the EU), Mistral-Medium-3.5 (needs a vLLM nightly build).

## 5. Results — models on their own (homogeneous pairs)

Each model debating itself. "EM single agent" = the agent's independent answer in round 1
(before any exchange), i.e. the matched single-agent baseline. "Jaccard floor" = evidence
overlap of a model with itself (noise floor). "Forced answers" = share of questions in
which the agent used up its search budget and had to be asked for its answer.

| # | Model | Vendor | FFN | Total B | Active B | EM debate | EM single agent | Jaccard floor | Consensus | Forced answers | s/question | GPU energy (Wh) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | qwen3.5-397b-a17b | alibaba | moe | 397 | 17 | 0.652 | 0.644 | 0.976 | 99.1 % | 4 % | 3.2 | 1295 |
| 2 | qwen3.8-flash-next | alibaba | moe | 176 | 6 | 0.648 | 0.644 | 0.900 | 98.6 % | 69 % | 4.4 | 1873 |
| 3 | muse-glimmer-30b | meta | dense | 29.6 | 29.6 | 0.643 | 0.639 | 0.974 | 99.1 % | 10 % | 12.1 | 2069 |
| 4 | qwen3.5-122b-a10b | alibaba | moe | 122 | 10 | 0.640 | 0.637 | 0.970 | 98.4 % | 17 % | 3.1 | 741 |
| 5 | qwen3.5-27b | alibaba | dense | 27 | 27 | 0.630 | 0.630 | 0.998 | 99.9 % | 8 % | 6.1 | 1057 |
| 6 | gemma4-31b | google | dense | 31 | 31 | 0.628 | 0.626 | 0.999 | 99.5 % | 2 % | 3.2 | 525 |
| 7 | qwen3.8-27b | alibaba | dense | 27 | 27 | 0.626 | 0.625 | 0.997 | 99.8 % | 18 % | 4.2 | 729 |
| 8 | gemma4-26b-a4b | google | moe | 25.2 | 3.8 | 0.621 | 0.610 | 0.987 | 99.8 % | 2 % | 2.6 | 279 |
| 9 | qwen3.5-35b-a3b | alibaba | moe | 35 | 3 | 0.620 | 0.618 | 0.987 | 99.7 % | 5 % | 1.9 | 201 |
| 10 | qwen3.5-9b | alibaba | dense | 9 | 9 | 0.618 | 0.618 | 0.997 | 99.7 % | 3 % | 2.6 | 443 |
| 11 | gemma4-12b | google | dense | 12 | 12 | 0.613 | 0.613 | 0.989 | 99.8 % | 6 % | 2.1 | 272 |
| 12 | gemma4-e4b | google | dense | 4 | 4 | 0.598 | 0.589 | 0.974 | 98.0 % | 3 % | 3.7 | 337 |
| 13 | qwen3.5-4b | alibaba | dense | 4 | 4 | 0.597 | 0.595 | 0.996 | 99.9 % | 12 % | 2.0 | 290 |
| 14 | mistral-small4-119b | mistral | moe | 119 | 6.5 | 0.575 | 0.560 | 0.910 | 97.8 % | 17 % | 2.8 | 519 |
| 15 | ministral3-14b | mistral | dense | 14 | 14 | 0.555 | 0.540 | 0.981 | 97.9 % | 3 % | 1.6 | 255 |
| 16 | ministral3-8b | mistral | dense | 8 | 8 | 0.544 | 0.527 | 0.977 | 98.6 % | 9 % | 1.2 | 134 |
| 17 | llama3.3-70b | meta | dense | 70 | 70 | 0.497 | 0.486 | 0.992 | 99.8 % | 0 % | 2.6 | 837 |
| 18 | qwen2.5-7b | alibaba | dense | 7.6 | 7.6 | 0.480 | 0.478 | 0.989 | 98.9 % | 0 % | 1.0 | 155 |
| 19 | nemotron3-nano-4b | nvidia | dense | 4 | 4 | 0.456 | 0.456 | 0.996 | 99.4 % | 0 % | 0.5 | 68 |
| 20 | llama3.1-8b | meta | dense | 8 | 8 | 0.408 | 0.398 | 0.994 | 97.2 % | 0 % | 0.7 | 117 |

**Key observations (homogeneous):**
- Best: Qwen3.5-397B (0.652), Qwen3.8-Flash-Next (0.648), Muse-Glimmer-30B (0.643). The top
  models cluster at **EM ≈ 0.62–0.65** — the benchmark appears to saturate around 0.65 under
  strict exact match.
- Size buys little: Qwen3.5-9B (0.618) is within 3.4 pp of Qwen3.5-397B (0.652).
- Debating with itself adds almost nothing over the single answer: **+0.6 pp on average**
  (max. +1.7 pp).
- Weakest: Llama-3.1-8B (0.408), Nemotron-3-Nano-4B (0.456), Qwen2.5-7B (0.480),
  Llama-3.3-70B (0.497). Llama-70B is penalised by strict EM (e.g. "1922" vs. gold
  "October 1922"); its F1 is 0.638.
- Qwen3.8-Flash-Next searches excessively: 69 % of its answers had to be forced after the
  search budget — still the 2nd-best model.
- Energy: best quality per energy at ~200–450 Wh per 1000 questions (Qwen3.5-35B-A3B MoE:
  0.620 EM at 201 Wh; Gemma-4-26B MoE: 0.621 at 279 Wh). The largest models need 4–10× the
  energy for +2–3 pp.

## 6. Results — mixed pairs

46 mixed pairs. ΔEM / ΔF1 = gain over the better homogeneous pair, in percentage points.
"sig." = significant after Benjamini-Hochberg correction; "nominal" = p < 0.05 only.
"Oracle gap" = how much higher EM would be if, per question, the right one of the two
homogeneous answers were always picked (upper bound of what a debate could gain).

| Run | Agent A | Agent B | Type | Vendor | EM pair | EM homog A | EM homog B | ΔEM (pp) | ΔF1 (pp) | p | q (BH) | sig. | Jaccard r1 | Consensus | Oracle gap (pp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P3-018 | qwen2.5-7b | llama3.1-8b | DxD | diff. | 0.527 | 0.480 | 0.408 | +4.7 | +5.0 | 0.0000 | 0.0005 | yes | 0.700 | 78 % | 3.0 |
| P3-083 | llama3.1-8b | llama3.3-70b | DxD | same | 0.536 | 0.408 | 0.497 | +3.9 | +4.3 | 0.0002 | 0.0048 | yes | 0.602 | 80 % | 2.3 |
| P3-040 | qwen3.5-9b | qwen3.5-35b-a3b | DxM | same | 0.642 | 0.618 | 0.620 | +2.2 | +0.9 | 0.0062 | 0.0911 | nominal | 0.790 | 95 % | 2.3 |
| P3-071 | gemma4-e4b | gemma4-26b-a4b | DxM | same | 0.639 | 0.598 | 0.621 | +1.8 | +1.6 | 0.0079 | 0.0911 | nominal | 0.800 | 90 % | 2.7 |
| P3-068 | gemma4-e4b | gemma4-12b | DxD | same | 0.628 | 0.598 | 0.613 | +1.5 | +1.8 | 0.0489 | 0.3073 | nominal | 0.803 | 92 % | 3.5 |
| P3-085 | qwen3.5-397b-a17b | qwen3.8-flash-next | MxM | same | 0.667 | 0.652 | 0.648 | +1.5 | +0.9 | 0.0167 | 0.1534 | nominal | 0.733 | 95 % | 1.7 |
| P3-079 | qwen3.5-9b | gemma4-12b | DxD | diff. | 0.632 | 0.618 | 0.613 | +1.4 | +1.4 | 0.0925 | 0.4656 |  | 0.778 | 95 % | 3.7 |
| P3-019 | qwen3.5-27b | qwen3.5-35b-a3b | DxM | same | 0.643 | 0.630 | 0.620 | +1.3 | +1.1 | 0.0295 | 0.2258 | nominal | 0.773 | 96 % | 1.8 |
| P3-037 | qwen3.5-4b | qwen3.5-9b | DxD | same | 0.631 | 0.597 | 0.618 | +1.3 | +0.9 | 0.1112 | 0.4656 |  | 0.810 | 93 % | 2.8 |
| P3-084 | qwen3.5-27b | qwen3.8-27b | DxD | same | 0.642 | 0.630 | 0.626 | +1.2 | +0.7 | 0.1189 | 0.4656 |  | 0.820 | 94 % | 3.4 |
| P3-027 | qwen3.5-35b-a3b | gemma4-26b-a4b | MxM | diff. | 0.631 | 0.620 | 0.621 | +1.0 | +0.3 | 0.1934 | 0.5738 |  | 0.710 | 94 % | 3.8 |
| P3-038 | qwen3.5-9b | qwen3.5-27b | DxD | same | 0.640 | 0.618 | 0.630 | +1.0 | +1.0 | 0.1742 | 0.5738 |  | 0.812 | 96 % | 2.5 |
| P3-020 | qwen3.8-27b | qwen3.8-flash-next | DxM | same | 0.657 | 0.626 | 0.648 | +0.9 | +0.8 | 0.1996 | 0.5738 |  | 0.783 | 95 % | 1.8 |
| P3-022 | ministral3-14b | mistral-small4-119b | DxM | same | 0.584 | 0.555 | 0.575 | +0.9 | +0.5 | 0.3620 | 0.8765 |  | 0.743 | 93 % | 4.7 |
| P3-024 | qwen3.5-27b | gemma4-31b | DxD | diff. | 0.639 | 0.630 | 0.628 | +0.9 | +0.7 | 0.2717 | 0.6943 |  | 0.808 | 94 % | 2.7 |
| P3-061 | llama3.3-70b | mistral-small4-119b | DxM | diff. | 0.583 | 0.497 | 0.575 | +0.8 | +0.5 | 0.4160 | 0.9324 |  | 0.533 | 87 % | 5.9 |
| P3-074 | ministral3-8b | mistral-small4-119b | DxM | same | 0.582 | 0.544 | 0.575 | +0.7 | +0.3 | 0.4887 | 0.9324 |  | 0.758 | 92 % | 5.0 |
| P3-041 | qwen3.5-27b | qwen3.5-122b-a10b | DxM | same | 0.646 | 0.630 | 0.640 | +0.6 | +0.0 | 0.4514 | 0.9324 |  | 0.788 | 96 % | 1.9 |
| P3-072 | gemma4-12b | gemma4-26b-a4b | DxM | same | 0.627 | 0.613 | 0.621 | +0.6 | +1.3 | 0.4614 | 0.9324 |  | 0.825 | 96 % | 3.5 |
| P3-082 | muse-glimmer-30b | llama3.3-70b | DxD | same | 0.648 | 0.643 | 0.497 | +0.5 | -0.2 | 0.5601 | 0.9324 |  | 0.512 | 88 % | 3.5 |
| P3-039 | qwen3.5-4b | qwen3.5-27b | DxD | same | 0.634 | 0.597 | 0.630 | +0.4 | +0.6 | 0.6271 | 0.9324 |  | 0.814 | 94 % | 2.7 |
| P3-069 | gemma4-12b | gemma4-31b | DxD | same | 0.632 | 0.613 | 0.628 | +0.4 | +1.1 | 0.5966 | 0.9324 |  | 0.845 | 95 % | 3.3 |
| P3-073 | ministral3-8b | ministral3-14b | DxD | same | 0.559 | 0.544 | 0.555 | +0.4 | +0.5 | 0.6989 | 0.9324 |  | 0.798 | 95 % | 4.2 |
| P3-035 | muse-glimmer-30b | gemma4-26b-a4b | DxM | diff. | 0.646 | 0.643 | 0.621 | +0.3 | -0.5 | 0.7754 | 0.9324 |  | 0.742 | 93 % | 3.4 |
| P3-063 | qwen3.5-4b | qwen3.5-35b-a3b | DxM | same | 0.623 | 0.597 | 0.620 | +0.3 | -0.0 | 0.7877 | 0.9324 |  | 0.776 | 94 % | 3.5 |
| P3-066 | qwen3.5-9b | qwen3.5-122b-a10b | DxM | same | 0.643 | 0.618 | 0.640 | +0.3 | -0.7 | 0.7754 | 0.9324 |  | 0.781 | 95 % | 3.0 |
| P3-078 | qwen3.5-4b | nemotron3-nano-4b | DxD | diff. | 0.600 | 0.597 | 0.456 | +0.3 | +1.0 | 0.7754 | 0.9324 |  | 0.701 | 83 % | 4.0 |
| P3-021 | gemma4-31b | gemma4-26b-a4b | DxM | same | 0.630 | 0.628 | 0.621 | +0.2 | +0.1 | 0.7905 | 0.9324 |  | 0.823 | 95 % | 2.6 |
| P3-070 | gemma4-e4b | gemma4-31b | DxD | same | 0.630 | 0.598 | 0.628 | +0.2 | +0.5 | 0.8555 | 0.9839 |  | 0.812 | 90 % | 4.4 |
| P3-030 | qwen3.5-27b | gemma4-26b-a4b | DxM | diff. | 0.631 | 0.630 | 0.621 | +0.1 | -0.4 | 1.0000 | 1.0000 |  | 0.761 | 95 % | 3.2 |
| P3-025 | qwen3.5-27b | muse-glimmer-30b | DxD | diff. | 0.643 | 0.630 | 0.643 | +0.0 | -0.6 | 1.0000 | 1.0000 |  | 0.812 | 93 % | 3.8 |
| P3-034 | muse-glimmer-30b | qwen3.5-35b-a3b | DxM | diff. | 0.642 | 0.643 | 0.620 | -0.1 | -0.3 | 1.0000 | 1.0000 |  | 0.771 | 93 % | 3.6 |
| P3-067 | qwen3.5-35b-a3b | qwen3.5-122b-a10b | MxM | same | 0.639 | 0.620 | 0.640 | -0.1 | -0.4 | 1.0000 | 1.0000 |  | 0.822 | 95 % | 3.1 |
| P3-077 | qwen3.5-9b | ministral3-8b | DxD | diff. | 0.617 | 0.618 | 0.544 | -0.1 | -0.4 | 1.0000 | 1.0000 |  | 0.768 | 91 % | 4.2 |
| P3-080 | ministral3-14b | gemma4-12b | DxD | diff. | 0.612 | 0.555 | 0.613 | -0.1 | +0.3 | 1.0000 | 1.0000 |  | 0.761 | 90 % | 4.2 |
| P3-060 | llama3.3-70b | qwen3.5-122b-a10b | DxM | diff. | 0.637 | 0.497 | 0.640 | -0.3 | -1.2 | 0.7493 | 0.9324 |  | 0.486 | 89 % | 4.1 |
| P3-032 | gemma4-31b | qwen3.5-35b-a3b | DxM | diff. | 0.624 | 0.628 | 0.620 | -0.4 | -0.1 | 0.5716 | 0.9324 |  | 0.750 | 93 % | 4.9 |
| P3-045 | qwen3.5-122b-a10b | qwen3.5-397b-a17b | MxM | same | 0.648 | 0.640 | 0.652 | -0.4 | -0.5 | 0.6271 | 0.9324 |  | 0.776 | 96 % | 2.6 |
| P3-057 | qwen3.5-122b-a10b | mistral-small4-119b | MxM | diff. | 0.636 | 0.640 | 0.575 | -0.4 | -0.9 | 0.6778 | 0.9324 |  | 0.749 | 93 % | 3.4 |
| P3-064 | qwen3.5-4b | qwen3.5-122b-a10b | DxM | same | 0.636 | 0.597 | 0.640 | -0.4 | -0.6 | 0.6587 | 0.9324 |  | 0.774 | 92 % | 3.1 |
| P3-044 | qwen3.5-35b-a3b | qwen3.5-397b-a17b | MxM | same | 0.647 | 0.620 | 0.652 | -0.5 | -0.4 | 0.5327 | 0.9324 |  | 0.773 | 94 % | 3.2 |
| P3-042 | qwen3.5-9b | qwen3.5-397b-a17b | DxM | same | 0.646 | 0.618 | 0.652 | -0.6 | -1.1 | 0.4709 | 0.9324 |  | 0.820 | 96 % | 3.5 |
| P3-065 | qwen3.5-4b | qwen3.5-397b-a17b | DxM | same | 0.643 | 0.597 | 0.652 | -0.9 | -0.7 | 0.1996 | 0.5738 |  | 0.793 | 91 % | 4.1 |
| P3-043 | qwen3.5-27b | qwen3.5-397b-a17b | DxM | same | 0.642 | 0.630 | 0.652 | -1.0 | -1.0 | 0.1215 | 0.4656 |  | 0.832 | 96 % | 3.4 |
| P3-026 | gemma4-31b | muse-glimmer-30b | DxD | diff. | 0.632 | 0.628 | 0.643 | -1.1 | -1.5 | 0.2145 | 0.5805 |  | 0.773 | 92 % | 5.2 |
| P3-081 | llama3.1-8b | muse-glimmer-30b | DxD | same | 0.628 | 0.408 | 0.643 | -1.5 | -1.8 | 0.0534 | 0.3073 |  | 0.642 | 72 % | 4.7 |

**Aggregate over all 46 mixed pairs**

| Measure | Value |
|---|---|
| Mean ΔEM | **+0.52 pp** |
| Mean ΔF1 | **+0.32 pp** |
| Pairs better than their better homogeneous pair | **30 / 46** (65 %) |
| … nominally significant (p < 0.05) | 7 |
| … significant after BH correction | **2** (both pairs of weak models) |
| Pairs significantly worse | **0** |
| Mean oracle headroom (what a perfect selector could still gain) | 3.4 pp |
| Consensus rate: homogeneous / mixed pairs | 99.0 % / 92.0 % |
| Mean rounds per debate (mixed) | 1.28 |
| Debates decided by the judge (mixed pairs) | 3 659 (EM there 0.286 — the hard questions) |
| Energy of a mixed run vs. the mean of its two homogeneous runs (median) | 2.3× over the 39 pairs with reliable energy (2.4× incl. the 7 overstated runs) |

## 7. Where do differences come from?

**7.1 Evidence divergence (Jaccard, round 1) — heterogeneity *does* create divergence**

| Pairs | Mean Jaccard r1 |
|---|---|
| Homogeneous (model with itself) | 0.979 |
| All mixed pairs | 0.757 (min. 0.486) |
| Same vendor, dense+dense / dense+MoE / MoE+MoE | 0.752 / 0.791 / 0.776 |
| Different vendor, dense+dense / dense+MoE / MoE+MoE | 0.763 / **0.674** / 0.730 |
| Pairs with Llama-3.3-70B | 0.49–0.53 (most divergent) |

Least-squares fit of Jaccard over all mixed pairs (coefficient ± std. error):
different vendor **−0.040 ± 0.019**, dense×MoE −0.026 ± 0.022, MoE×MoE −0.053 ± 0.030,
log2 size ratio −0.017 ± 0.009, mean model strength **+1.15 ± 0.22** (stronger models
retrieve more similar evidence). → Vendor (training lineage) is the clearest source of
divergence; architecture adds a smaller, uncertain amount.

**7.2 Accuracy gain (ΔEM) — divergence does *not* translate into accuracy**

| Pair type | Pairs | Mean ΔEM | Mean ΔF1 | better / nominal / BH-significant |
|---|---|---|---|---|
| dense + dense | 19 | +0.8 pp | +0.8 pp | 74 % / 3 / 2 |
| dense + MoE | 21 | +0.3 pp | +0.0 pp | 67 % / 3 / 0 |
| MoE + MoE | 6 | +0.2 pp | −0.2 pp | 33 % / 1 / 0 |
| same vendor | 30 | +0.6 pp | +0.4 pp | 73 % / 6 / 1 |
| different vendor | 16 | +0.4 pp | +0.2 pp | 50 % / 1 / 1 |

Least-squares fit of ΔEM over all mixed pairs: different vendor −0.003 ± 0.003,
dense×MoE −0.001 ± 0.004, MoE×MoE 0.000 ± 0.005, size ratio 0.000 ± 0.002 — **all ≈ 0**.
The only clear term is **mean model strength: −0.127 ± 0.037** (≈ −1.3 pp gain per +10 pp
of model accuracy).

**7.3 Correlations (46 pairs)**

| Relation | Pearson r |
|---|---|
| Model strength (mean EM of the two models) vs. ΔEM | **−0.50** |
| Model strength vs. ΔF1 | **−0.55** |
| Evidence overlap (Jaccard) vs. ΔEM | −0.13 (no relation) |
| Evidence overlap vs. ΔF1 | −0.03 (no relation) |

**7.4 Weak vs. strong**

| Pairs by mean model strength | Pairs | Mean ΔEM |
|---|---|---|
| weak (mean EM < 0.55) | 6 | **+1.4 pp** |
| strong (mean EM ≥ 0.60) | 34 | +0.4 pp |

## 8. Interpretation (for the conclusion slide)

- **RQ1, nuanced answer:** Architectural/vendor heterogeneity produces genuine evidence
  divergence, but for current models it does **not** improve answer accuracy beyond the
  better single model. The accuracy gain of Phase 2 is reproduced (+4.7 pp) but is specific
  to weak models.
- **Dense vs. MoE:** no specific effect on accuracy; a small, uncertain effect on divergence.
- **Why little gain?** (i) Strong models already agree on most answers (92 % consensus) —
  there is little disagreement left to resolve; (ii) when they disagree, the questions are
  the hard ones (judge EM 0.29); (iii) the benchmark saturates near EM 0.65 under strict
  exact match; (iv) the debate exploits only a small part of the oracle headroom (3.4 pp).
- **Cost:** a mixed pair costs ≈ 2.3× the energy of a homogeneous run for +0.5 pp on
  average — not worth it for strong models; worth testing for weak/cheap models.
- **Outlook (Phase 4):** test whether the pattern holds on other multi-hop benchmarks and
  with metrics less strict than exact match.

## 9. Methodological notes and limitations (disclose)

- **Protocol correction during the experiment:** a first version of the "forced answer"
  returned empty answers for Qwen/Gemma and counted two missing answers as consensus.
  Fixed on 2026-10-06; the 4 020 affected questions were re-run, all others kept
  (deterministic decoding, each question independent).
- **Duplicates:** in 7 late runs (P3-077 … P3-084) two processes ran the same run
  simultaneously; questions were written twice. One row per question was kept (the last);
  quality numbers are unaffected, **energy of these 7 runs is overstated** (excluded from
  cost statements).
- **Excluded models:** Nemotron-3 MoE models (no answers without reasoning) → no NVIDIA
  dense×MoE pair; the ~30B matrix has 2 instead of 3 MoE models.
- **Single benchmark, strict metric:** HotpotQA distractor (10 paragraphs per question),
  English, Wikipedia-based; EM underestimates correct paraphrases (F1 reported alongside).
- **Judge:** one fixed judge for all runs; it decides 8 % of the mixed-pair debates.
- **Reasoning off** for all debaters: results describe non-reasoning use of the models.
- **Noise:** replicate runs differ by ≤ 0.1 pp — differences above that are not GPU noise,
  but question-sample uncertainty is handled by McNemar + BH.

## 10. Figures (all in `3_iteration/analysis/figures/`) and suggested slides

| File | What it shows | Suggested slide |
|---|---|---|
| `09_gain_vs_model_strength.png` | ΔEM and ΔF1 vs. mean strength of the two models; trend line, r = −0.50 / −0.55; the two significant pairs highlighted | **Key result slide**: "Mixing helps weak models, not strong ones" |
| `07_gain_vs_divergence_em_f1.png` | ΔEM and ΔF1 vs. evidence overlap; r = −0.13 / −0.03 | "Divergence ≠ accuracy" |
| `06_by_pair_type.png` | ΔEM and Jaccard per pair type (dense/MoE × same/different vendor), group means | "Dense vs. MoE is not a lever" |
| `01_cross_vendor_30B.png` | heatmaps (ΔEM + Jaccard) for the ~30B models of Qwen, Gemma, Muse | Matrix slide 1 (cross-vendor) |
| `02_qwen35_ladder.png` | heatmaps for the Qwen3.5 ladder 4B–397B (dense + MoE) | Matrix slide 2 (size within one family) |
| `03_large_models.png` | heatmaps for Llama-70B / Mistral-Small-4 / Qwen-122B | Matrix slide 3 (large models, most divergent) |
| `04_gemma4_ladder.png`, `05_mistral_ladder.png` | heatmaps for the Gemma 4 and Mistral ladders | backup / appendix |
| `08_em_vs_energy_homogeneous.png` | EM vs. GPU energy for every model (dense = circle, MoE = square) | "Quality vs. cost" |

**Suggested deck (10–12 slides):** 1 Title · 2 Question & recap Phase 2 · 3 Setup (Section 3
table) · 4 Model pool (Section 4) · 5 Models on their own + energy (fig. 08, Section 5
highlights) · 6 Divergence is real (Section 7.1, fig. 06 right) · 7 Matrices (fig. 01/02/03) ·
8 Gain is small (Section 6 aggregate table) · 9 Key result: weak vs. strong (fig. 09) ·
10 Divergence ≠ accuracy (fig. 07) · 11 Dense vs. MoE (fig. 06 left) · 12 Conclusion,
limitations, outlook Phase 4 (Sections 8–9).

**Colour code used in all figures:** dense = blue (#2F6DB5), MoE = orange (#E07A2E),
dense+MoE pairs = purple (#7B5EA7), significant = red ring (#C0392B); heatmaps: blue = mixed
pair better, red = worse.
