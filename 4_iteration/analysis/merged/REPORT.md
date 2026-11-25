# Phase 4 -- merged over HotpotQA, MuSiQue and 2WikiMultiHopQA

Generated 2026-11-03 10:43. HotpotQA = Phase 3 runs; the same 20 models and the same 46 mixed pairs on all three benchmarks.

## 1. Models on their own (Fig. M1)

Pearson r of the EM of the homogeneous pairs between the benchmarks:

| | HotpotQA | MuSiQue | 2Wiki |
|---|---|---|---|
| HotpotQA | 1.00 | 0.95 | 0.96 |
| MuSiQue | 0.95 | 1.00 | 0.92 |
| 2Wiki | 0.96 | 0.92 | 1.00 |

| Benchmark | EM min | EM max |
|---|---|---|
| HotpotQA | 0.408 | 0.652 |
| MuSiQue | 0.194 | 0.598 |
| 2Wiki | 0.414 | 0.751 |

## 2. Debate gain per pair type (Fig. M2 left)

| Benchmark | Pair type | Pairs | mean ΔEM | better | sig. better | sig. worse | judge-decided | EM judge-decided | ≥1 agent right r1 (judge-decided) |
|---|---|---|---|---|---|---|---|---|---|
| HotpotQA | DxD | 19 | +0.81 pp | 74 % | 2 | 0 | 10.2 % | 0.362 | 0.513 |
| HotpotQA | DxM | 21 | +0.35 pp | 67 % | 0 | 0 | 6.6 % | 0.283 | 0.495 |
| HotpotQA | MxM | 6 | +0.18 pp | 33 % | 0 | 0 | 5.5 % | 0.250 | 0.518 |
| HotpotQA | all | 46 | +0.52 pp | 65 % | 2 | 0 | 8.0 % | 0.322 | 0.506 |
| MuSiQue | DxD | 19 | +1.21 pp | 68 % | 6 | 2 | 21.0 % | 0.238 | 0.344 |
| MuSiQue | DxM | 21 | +0.09 pp | 48 % | 2 | 2 | 13.5 % | 0.208 | 0.349 |
| MuSiQue | MxM | 6 | -1.42 pp | 17 % | 0 | 2 | 12.4 % | 0.175 | 0.382 |
| MuSiQue | all | 46 | +0.35 pp | 52 % | 8 | 6 | 16.5 % | 0.220 | 0.350 |
| 2Wiki | DxD | 19 | -0.35 pp | 32 % | 2 | 2 | 11.6 % | 0.358 | 0.501 |
| 2Wiki | DxM | 21 | -0.89 pp | 29 % | 0 | 3 | 6.8 % | 0.257 | 0.513 |
| 2Wiki | MxM | 6 | -0.95 pp | 0 % | 0 | 1 | 5.7 % | 0.260 | 0.594 |
| 2Wiki | all | 46 | -0.67 pp | 26 % | 2 | 6 | 8.6 % | 0.313 | 0.513 |

## 3. Same pair, other benchmark (Fig. M2 right)

| | r of ΔEM per pair | same sign as HotpotQA |
|---|---|---|
| MuSiQue vs. HotpotQA | +0.61 | 32 / 46 |
| 2Wiki vs. HotpotQA | +0.49 | 20 / 46 |
| MuSiQue vs. 2Wiki | +0.31 | -- |

Pairs significantly better on at least two benchmarks:

- llama3.1-8b + llama3.3-70b (DxD): HotpotQA +3.9 pp (significant better), MuSiQue +5.6 pp (significant better), 2Wiki +5.1 pp (significant better)
- qwen2.5-7b + llama3.1-8b (DxD): HotpotQA +4.7 pp (significant better), MuSiQue +2.2 pp (better), 2Wiki +3.9 pp (significant better)

Pairs significantly worse on at least one benchmark:

- gemma4-31b + qwen3.5-35b-a3b (DxM): HotpotQA -0.4 pp, MuSiQue +2.3 pp, 2Wiki -3.1 pp
- qwen3.5-35b-a3b + gemma4-26b-a4b (MxM): HotpotQA +1.0 pp, MuSiQue +0.8 pp, 2Wiki -2.9 pp
- qwen3.5-27b + gemma4-26b-a4b (DxM): HotpotQA +0.1 pp, MuSiQue +0.8 pp, 2Wiki -2.9 pp
- qwen3.5-27b + gemma4-31b (DxD): HotpotQA +0.9 pp, MuSiQue +0.6 pp, 2Wiki -2.4 pp
- qwen3.5-4b + nemotron3-nano-4b (DxD): HotpotQA +0.3 pp, MuSiQue +0.2 pp, 2Wiki -2.8 pp
- ministral3-8b + mistral-small4-119b (DxM): HotpotQA +0.7 pp, MuSiQue -0.2 pp, 2Wiki -2.7 pp
- gemma4-31b + muse-glimmer-30b (DxD): HotpotQA -1.1 pp, MuSiQue -2.8 pp, 2Wiki -1.6 pp
- qwen3.5-9b + qwen3.5-397b-a17b (DxM): HotpotQA -0.6 pp, MuSiQue -2.8 pp, 2Wiki -1.3 pp
- muse-glimmer-30b + gemma4-26b-a4b (DxM): HotpotQA +0.3 pp, MuSiQue -2.9 pp, 2Wiki -1.2 pp
- qwen3.5-122b-a10b + qwen3.5-397b-a17b (MxM): HotpotQA -0.4 pp, MuSiQue -3.2 pp, 2Wiki +0.0 pp
- llama3.1-8b + muse-glimmer-30b (DxD): HotpotQA -1.5 pp, MuSiQue -3.7 pp, 2Wiki -1.4 pp
- qwen3.5-122b-a10b + mistral-small4-119b (MxM): HotpotQA -0.4 pp, MuSiQue -3.9 pp, 2Wiki -0.4 pp
