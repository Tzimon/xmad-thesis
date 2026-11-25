# Phase 4 -- MuSiQue -- results report

Generated 2026-11-03 10:43 from 20 homogeneous and 46 mixed runs (1000 questions each).

> Not judgeable, counted as wrong: P4-MSQ-003, question idx 136 (6 rounds, 8 truncated replies (8192 tokens each); the judge request failed, most likely because the transcript exceeds the judge context (32,768 tokens)).

## 1. Noise

No replicate runs in Phase 4. Threshold = run-to-run noise of Phase 3: **0.1 pp EM**.

## 2. Models on their own (homogeneous pairs)

| Model | FFN | Active B | EM debate | EM single agent (round 1) | Jaccard floor | Recall r1 | Recall all rounds | Forced answers |
|---|---|---|---|---|---|---|---|---|
| qwen3.8-flash-next | moe | 6 | 0.598 | 0.584 | 0.8512 | 0.9745 | 0.9825 | 110 % |
| qwen3.5-397b-a17b | moe | 17 | 0.563 | 0.545 | 0.9422 | 0.9187 | 0.9403 | 37 % |
| muse-glimmer-30b | dense | 29.6 | 0.530 | 0.523 | 0.9163 | 0.9343 | 0.9522 | 43 % |
| qwen3.8-27b | dense | 27 | 0.510 | 0.508 | 0.9903 | 0.9543 | 0.955 | 74 % |
| qwen3.5-27b | dense | 27 | 0.509 | 0.509 | 0.9932 | 0.9193 | 0.9213 | 55 % |
| qwen3.5-122b-a10b | moe | 10 | 0.502 | 0.472 | 0.9316 | 0.9233 | 0.9461 | 73 % |
| gemma4-31b | dense | 31 | 0.492 | 0.482 | 0.9941 | 0.8866 | 0.8898 | 32 % |
| qwen3.5-35b-a3b | moe | 3 | 0.485 | 0.470 | 0.9752 | 0.9133 | 0.9236 | 36 % |
| gemma4-26b-a4b | moe | 3.8 | 0.452 | 0.425 | 0.9713 | 0.8539 | 0.8841 | 35 % |
| qwen3.5-4b | dense | 4 | 0.431 | 0.424 | 0.9922 | 0.9007 | 0.9062 | 51 % |
| qwen3.5-9b | dense | 9 | 0.418 | 0.410 | 0.9843 | 0.9013 | 0.9091 | 35 % |
| gemma4-12b | dense | 12 | 0.412 | 0.388 | 0.9842 | 0.8383 | 0.8582 | 45 % |
| gemma4-e4b | dense | 4 | 0.399 | 0.387 | 0.9199 | 0.8243 | 0.8548 | 18 % |
| mistral-small4-119b | moe | 6.5 | 0.393 | 0.370 | 0.8467 | 0.8955 | 0.94 | 56 % |
| ministral3-14b | dense | 14 | 0.349 | 0.332 | 0.9438 | 0.8339 | 0.8847 | 32 % |
| ministral3-8b | dense | 8 | 0.316 | 0.292 | 0.9308 | 0.8344 | 0.8795 | 60 % |
| llama3.3-70b | dense | 70 | 0.282 | 0.278 | 0.991 | 0.6003 | 0.6157 | 0 % |
| qwen2.5-7b | dense | 7.6 | 0.270 | 0.243 | 0.9602 | 0.7062 | 0.7509 | 1 % |
| llama3.1-8b | dense | 8 | 0.197 | 0.189 | 0.984 | 0.6823 | 0.7038 | 0 % |
| nemotron3-nano-4b | dense | 4 | 0.194 | 0.194 | 0.9935 | 0.7203 | 0.7238 | 0 % |

## 3. Mixed pairs

46 mixed pairs. Better than the better homogeneous pair: **24** (nominal p<0.05: **10**, significant after BH correction: **8**). Worse: **22** (significant: **6**).

Mean ΔEM: **+0.35 pp**; mean ΔF1: **+0.46 pp**. Mean headroom to the oracle (question solved by at least one homogeneous pair): **6.5 pp**.

Judge (mixed pairs): decided 7571 debates; right on **0.220** of them, while at least one agent was right in round 1 on **0.350**.

All pairs by ΔEM:

| Pair | Type | ΔEM | ΔF1 | p | q (BH) | Jaccard r1 | Recall all rounds |
|---|---|---|---|---|---|---|---|
| llama3.1-8b + llama3.3-70b | DxD, same vendor | +5.6 pp * | +5.1 pp | 0.0 | 0.0 | 0.5227 | 0.8157 |
| qwen3.5-9b + gemma4-12b | DxD, diff. vendor | +4.0 pp * | +4.3 pp | 0.00099 | 0.01138 | 0.6871 | 0.9555 |
| qwen3.5-4b + qwen3.5-9b | DxD, same vendor | +3.8 pp * | +3.8 pp | 0.00209 | 0.01871 | 0.746 | 0.9623 |
| ministral3-14b + gemma4-12b | DxD, diff. vendor | +3.3 pp * | +3.5 pp | 0.00373 | 0.02268 | 0.6623 | 0.9418 |
| gemma4-e4b + gemma4-12b | DxD, same vendor | +3.2 pp * | +3.5 pp | 0.00446 | 0.02268 | 0.6874 | 0.9185 |
| llama3.3-70b + mistral-small4-119b | DxM, diff. vendor | +3.1 pp * | +4.2 pp | 0.00493 | 0.02268 | 0.4223 | 0.9403 |
| qwen3.5-27b + qwen3.8-27b | DxD, same vendor | +3.1 pp * | +2.6 pp | 0.00752 | 0.0299 | 0.7501 | 0.9762 |
| qwen3.5-27b + qwen3.5-35b-a3b | DxM, same vendor | +2.6 pp * | +2.8 pp | 0.00487 | 0.02268 | 0.735 | 0.9675 |
| qwen3.5-9b + ministral3-8b | DxD, diff. vendor | +2.4 pp (*) | +1.4 pp | 0.03078 | 0.08849 | 0.6561 | 0.9628 |
| gemma4-31b + qwen3.5-35b-a3b | DxM, diff. vendor | +2.3 pp (*) | +2.9 pp | 0.01783 | 0.05468 | 0.6913 | 0.9631 |
| qwen2.5-7b + llama3.1-8b | DxD, diff. vendor | +2.2 pp | +2.6 pp | 0.08607 | 0.21996 | 0.5963 | 0.8721 |
| qwen3.5-27b + qwen3.5-122b-a10b | DxM, same vendor | +1.9 pp | +1.8 pp | 0.07849 | 0.21238 | 0.7656 | 0.9694 |
| ministral3-14b + mistral-small4-119b | DxM, same vendor | +1.4 pp | +1.7 pp | 0.23909 | 0.52372 | 0.6511 | 0.9559 |
| qwen3.5-4b + qwen3.5-35b-a3b | DxM, same vendor | +1.2 pp | +0.9 pp | 0.30288 | 0.58052 | 0.7021 | 0.9674 |
| qwen3.5-9b + qwen3.5-35b-a3b | DxM, same vendor | +1.1 pp | +0.6 pp | 0.38235 | 0.69918 | 0.7273 | 0.9596 |
| qwen3.5-9b + qwen3.5-27b | DxD, same vendor | +1.0 pp | +0.5 pp | 0.39519 | 0.69918 | 0.7535 | 0.9647 |
| qwen3.5-35b-a3b + gemma4-26b-a4b | MxM, diff. vendor | +0.8 pp | +1.2 pp | 0.55427 | 0.7135 | 0.6666 | 0.9534 |
| qwen3.5-27b + gemma4-26b-a4b | DxM, diff. vendor | +0.8 pp | +0.6 pp | 0.49265 | 0.7135 | 0.7058 | 0.9578 |
| qwen3.5-4b + qwen3.5-27b | DxD, same vendor | +0.8 pp | +1.7 pp | 0.50468 | 0.7135 | 0.7461 | 0.9682 |
| ministral3-8b + ministral3-14b | DxD, same vendor | +0.8 pp | -0.1 pp | 0.51949 | 0.7135 | 0.7048 | 0.9327 |
| muse-glimmer-30b + qwen3.5-35b-a3b | DxM, diff. vendor | +0.7 pp | +0.1 pp | 0.5296 | 0.7135 | 0.7119 | 0.9682 |
| qwen3.5-27b + gemma4-31b | DxD, diff. vendor | +0.6 pp | +1.3 pp | 0.65618 | 0.7362 | 0.7481 | 0.9569 |
| gemma4-31b + gemma4-26b-a4b | DxM, same vendor | +0.5 pp | +0.7 pp | 0.63531 | 0.73061 | 0.7709 | 0.9364 |
| qwen3.5-4b + nemotron3-nano-4b | DxD, diff. vendor | +0.2 pp | -0.5 pp | 0.92478 | 0.95189 | 0.5669 | 0.9501 |
| gemma4-12b + gemma4-31b | DxD, same vendor | -0.1 pp | -0.0 pp | 1.0 | 1.0 | 0.7568 | 0.9392 |
| ministral3-8b + mistral-small4-119b | DxM, same vendor | -0.2 pp | -0.4 pp | 0.9312 | 0.95189 | 0.6704 | 0.9558 |
| muse-glimmer-30b + llama3.3-70b | DxD, same vendor | -0.2 pp | -1.0 pp | 0.9132 | 0.95189 | 0.3807 | 0.9586 |
| qwen3.5-35b-a3b + qwen3.5-122b-a10b | MxM, same vendor | -0.3 pp | +0.9 pp | 0.83756 | 0.91733 | 0.7495 | 0.9634 |
| qwen3.8-27b + qwen3.8-flash-next | DxM, same vendor | -0.6 pp | +0.5 pp | 0.62414 | 0.73061 | 0.729 | 0.988 |
| gemma4-e4b + gemma4-31b | DxD, same vendor | -0.6 pp | -1.2 pp | 0.62414 | 0.73061 | 0.6964 | 0.9321 |
| gemma4-12b + gemma4-26b-a4b | DxM, same vendor | -0.6 pp | -0.4 pp | 0.62414 | 0.73061 | 0.7717 | 0.9194 |
| qwen3.5-27b + muse-glimmer-30b | DxD, diff. vendor | -0.7 pp | -0.9 pp | 0.51041 | 0.7135 | 0.7392 | 0.9693 |
| qwen3.5-397b-a17b + qwen3.8-flash-next | MxM, same vendor | -0.7 pp | -0.2 pp | 0.55839 | 0.7135 | 0.6946 | 0.9848 |
| qwen3.5-27b + qwen3.5-397b-a17b | DxM, same vendor | -0.8 pp | -0.6 pp | 0.45574 | 0.7135 | 0.7919 | 0.9663 |
| llama3.3-70b + qwen3.5-122b-a10b | DxM, diff. vendor | -0.9 pp | -0.4 pp | 0.43069 | 0.7135 | 0.3807 | 0.962 |
| gemma4-e4b + gemma4-26b-a4b | DxM, same vendor | -0.9 pp | -0.6 pp | 0.48802 | 0.7135 | 0.6885 | 0.9198 |
| qwen3.5-35b-a3b + qwen3.5-397b-a17b | MxM, same vendor | -1.2 pp | -0.5 pp | 0.22986 | 0.52372 | 0.745 | 0.9605 |
| qwen3.5-4b + qwen3.5-122b-a10b | DxM, same vendor | -1.2 pp | -0.2 pp | 0.28069 | 0.56138 | 0.7197 | 0.9706 |
| qwen3.5-4b + qwen3.5-397b-a17b | DxM, same vendor | -1.2 pp | -1.2 pp | 0.27601 | 0.56138 | 0.7455 | 0.9703 |
| qwen3.5-9b + qwen3.5-122b-a10b | DxM, same vendor | -1.7 pp | -1.3 pp | 0.10377 | 0.25123 | 0.7383 | 0.9663 |
| gemma4-31b + muse-glimmer-30b | DxD, diff. vendor | -2.8 pp * | -2.1 pp | 0.01338 | 0.04396 | 0.7066 | 0.9598 |
| qwen3.5-9b + qwen3.5-397b-a17b | DxM, same vendor | -2.8 pp * | -2.6 pp | 0.0078 | 0.0299 | 0.756 | 0.9661 |
| muse-glimmer-30b + gemma4-26b-a4b | DxM, diff. vendor | -2.9 pp * | -3.2 pp | 0.00934 | 0.03305 | 0.6775 | 0.9618 |
| qwen3.5-122b-a10b + qwen3.5-397b-a17b | MxM, same vendor | -3.2 pp * | -2.5 pp | 0.00244 | 0.01871 | 0.7651 | 0.9662 |
| llama3.1-8b + muse-glimmer-30b | DxD, same vendor | -3.7 pp * | -4.7 pp | 0.00045 | 0.0069 | 0.5048 | 0.9596 |
| qwen3.5-122b-a10b + mistral-small4-119b | MxM, diff. vendor | -3.9 pp * | -3.5 pp | 0.00018 | 0.00414 | 0.6679 | 0.9751 |

## 4. Where do differences come from?

| Grouping | Group | Pairs | mean ΔEM | mean ΔF1 | better / nominal / BH-sig. better / BH-sig. worse | mean Jaccard r1 |
|---|---|---|---|---|---|---|
| pair_type | DxD | 19 | +1.2 pp | +1.0 pp | 68 % / 7 / 6 / 2 | 0.664 |
| pair_type | DxM | 21 | +0.1 pp | +0.3 pp | 48 % / 3 / 2 / 2 | 0.693 |
| pair_type | MxM | 6 | -1.4 pp | -0.8 pp | 17 % / 0 / 0 / 2 | 0.715 |
| same_vendor | False | 16 | +0.6 pp | +0.7 pp | 69 % / 5 / 3 / 3 | 0.643 |
| same_vendor | True | 30 | +0.2 pp | +0.3 pp | 43 % / 5 / 5 / 3 | 0.706 |
| pair_type+same_vendor | DxD / False | 8 | +1.1 pp | +1.2 pp | 75 % / 3 / 2 / 1 | 0.67 |
| pair_type+same_vendor | DxD / True | 11 | +1.2 pp | +0.9 pp | 64 % / 4 / 4 / 1 | 0.659 |
| pair_type+same_vendor | DxM / False | 6 | +0.5 pp | +0.7 pp | 67 % / 2 / 1 / 1 | 0.598 |
| pair_type+same_vendor | DxM / True | 15 | -0.1 pp | +0.1 pp | 40 % / 1 / 1 / 1 | 0.731 |
| pair_type+same_vendor | MxM / False | 2 | -1.6 pp | -1.2 pp | 50 % / 0 / 0 / 1 | 0.667 |
| pair_type+same_vendor | MxM / True | 4 | -1.4 pp | -0.6 pp | 0 % / 0 / 0 / 1 | 0.739 |

Least-squares fit over all mixed pairs (descriptive, small n):

| Outcome | Term | Coefficient | Std. error |
|---|---|---|---|
| delta_em | intercept | 0.0402 | 0.019 |
| delta_em | different_vendor | 0.0007 | 0.0062 |
| delta_em | dense_x_moe | -0.0099 | 0.007 |
| delta_em | moe_x_moe | -0.019 | 0.0101 |
| delta_em | log2_active_param_ratio | 0.0036 | 0.0028 |
| delta_em | mean_homog_em | -0.079 | 0.0433 |
| jaccard_r1 | intercept | 0.3673 | 0.0719 |
| jaccard_r1 | different_vendor | -0.0463 | 0.0234 |
| jaccard_r1 | dense_x_moe | -0.0032 | 0.0264 |
| jaccard_r1 | moe_x_moe | -0.0314 | 0.0385 |
| jaccard_r1 | log2_active_param_ratio | -0.0215 | 0.0105 |
| jaccard_r1 | mean_homog_em | 0.8293 | 0.1645 |
