# Phase 3 -- results report

Generated 2026-10-20 16:56 from 20 homogeneous and 46 mixed runs.

## 1. Noise

Replicate runs differ by at most **0.1 pp EM**. Differences below this are not interpreted, whatever their p-value.

## 2. Models on their own (homogeneous pairs)

| Model | Vendor | FFN | Active B | EM debate | EM single agent (round 1) | Jaccard floor | Forced answers |
|---|---|---|---|---|---|---|---|
| qwen3.5-397b-a17b | alibaba | moe | 17 | 0.652 | 0.644 | 0.9764 | 4 % |
| qwen3.8-flash-next | alibaba | moe | 6 | 0.648 | 0.644 | 0.9001 | 69 % |
| muse-glimmer-30b | meta | dense | 29.6 | 0.643 | 0.639 | 0.9742 | 10 % |
| qwen3.5-122b-a10b | alibaba | moe | 10 | 0.640 | 0.637 | 0.9703 | 17 % |
| qwen3.5-27b | alibaba | dense | 27 | 0.630 | 0.630 | 0.9984 | 8 % |
| gemma4-31b | google | dense | 31 | 0.628 | 0.626 | 0.9987 | 2 % |
| qwen3.8-27b | alibaba | dense | 27 | 0.626 | 0.625 | 0.997 | 18 % |
| gemma4-26b-a4b | google | moe | 3.8 | 0.621 | 0.610 | 0.9872 | 2 % |
| qwen3.5-35b-a3b | alibaba | moe | 3 | 0.620 | 0.618 | 0.9872 | 5 % |
| qwen3.5-9b | alibaba | dense | 9 | 0.618 | 0.618 | 0.9973 | 3 % |
| gemma4-12b | google | dense | 12 | 0.613 | 0.613 | 0.989 | 6 % |
| gemma4-e4b | google | dense | 4 | 0.598 | 0.589 | 0.9742 | 3 % |
| qwen3.5-4b | alibaba | dense | 4 | 0.597 | 0.595 | 0.996 | 12 % |
| mistral-small4-119b | mistral | moe | 6.5 | 0.575 | 0.560 | 0.9097 | 17 % |
| ministral3-14b | mistral | dense | 14 | 0.555 | 0.540 | 0.9808 | 3 % |
| ministral3-8b | mistral | dense | 8 | 0.544 | 0.527 | 0.9765 | 9 % |
| llama3.3-70b | meta | dense | 70 | 0.497 | 0.486 | 0.9923 | 0 % |
| qwen2.5-7b | alibaba | dense | 7.6 | 0.480 | 0.478 | 0.9894 | 0 % |
| nemotron3-nano-4b | nvidia | dense | 4 | 0.456 | 0.456 | 0.9957 | 0 % |
| llama3.1-8b | meta | dense | 8 | 0.408 | 0.398 | 0.9939 | 0 % |

## 3. Mixed pairs

46 mixed pairs. Better than the better homogeneous pair: **30** (nominal p<0.05: **7**, significant after BH correction: **2**). Worse: **15** (significant: **0**).

Mean ΔEM over all mixed pairs: **+0.52 pp**; mean ΔF1: **+0.32 pp**. Mean headroom to the oracle (question solved by at least one homogeneous pair): **3.4 pp**.

Judge: decided 3659 debates of the mixed pairs (EM there 0.286; these are the questions on which the agents disagreed, i.e. the hard ones).

Largest gains:

| Pair | Type | ΔEM | ΔF1 | p | q (BH) | Jaccard r1 |
|---|---|---|---|---|---|---|
| qwen2.5-7b + llama3.1-8b | DxD, diff. vendor | +4.7 pp * | +5.0 pp | 1e-05 | 0.00046 | 0.7002 |
| llama3.1-8b + llama3.3-70b | DxD, same vendor | +3.9 pp * | +4.3 pp | 0.00021 | 0.00483 | 0.6023 |
| qwen3.5-9b + qwen3.5-35b-a3b | DxM, same vendor | +2.2 pp | +0.9 pp | 0.00622 | 0.09108 | 0.7898 |
| gemma4-e4b + gemma4-26b-a4b | DxM, same vendor | +1.8 pp | +1.6 pp | 0.00792 | 0.09108 | 0.8003 |
| gemma4-e4b + gemma4-12b | DxD, same vendor | +1.5 pp | +1.8 pp | 0.04887 | 0.30728 | 0.8035 |

## 4. Where do differences come from?

| Grouping | Group | Pairs | mean ΔEM | mean ΔF1 | better / nominal / BH-sig. better / BH-sig. worse | mean Jaccard r1 |
|---|---|---|---|---|---|---|
| pair_type | DxD | 19 | +0.8 pp | +0.8 pp | 74 % / 3 / 2 / 0 | 0.757 |
| pair_type | DxM | 21 | +0.4 pp | +0.0 pp | 67 % / 3 / 0 / 0 | 0.757 |
| pair_type | MxM | 6 | +0.2 pp | -0.2 pp | 33 % / 1 / 0 / 0 | 0.761 |
| same_vendor | False | 16 | +0.4 pp | +0.2 pp | 50 % / 1 / 1 / 0 | 0.725 |
| same_vendor | True | 30 | +0.6 pp | +0.4 pp | 73 % / 6 / 1 / 0 | 0.774 |
| pair_type+same_vendor | DxD / False | 8 | +0.8 pp | +0.7 pp | 50 % / 1 / 1 / 0 | 0.763 |
| pair_type+same_vendor | DxD / True | 11 | +0.9 pp | +0.9 pp | 91 % / 2 / 1 / 0 | 0.752 |
| pair_type+same_vendor | DxM / False | 6 | +0.1 pp | -0.3 pp | 50 % / 0 / 0 / 0 | 0.674 |
| pair_type+same_vendor | DxM / True | 15 | +0.5 pp | +0.2 pp | 73 % / 3 / 0 / 0 | 0.791 |
| pair_type+same_vendor | MxM / False | 2 | +0.3 pp | -0.3 pp | 50 % / 0 / 0 / 0 | 0.73 |
| pair_type+same_vendor | MxM / True | 4 | +0.1 pp | -0.1 pp | 25 % / 1 / 0 / 0 | 0.776 |

Least-squares fit over all mixed pairs (descriptive, small n):

| Outcome | Term | Coefficient | Std. error |
|---|---|---|---|
| delta_em | intercept | 0.0838 | 0.0221 |
| delta_em | different_vendor | -0.0031 | 0.0033 |
| delta_em | dense_x_moe | -0.0008 | 0.0037 |
| delta_em | moe_x_moe | -0.0001 | 0.0052 |
| delta_em | log2_active_param_ratio | -0.0004 | 0.0015 |
| delta_em | mean_homog_em | -0.1272 | 0.037 |
| jaccard_r1 | intercept | 0.1237 | 0.1282 |
| jaccard_r1 | different_vendor | -0.0404 | 0.0189 |
| jaccard_r1 | dense_x_moe | -0.0256 | 0.0215 |
| jaccard_r1 | moe_x_moe | -0.0527 | 0.0302 |
| jaccard_r1 | log2_active_param_ratio | -0.0174 | 0.0085 |
| jaccard_r1 | mean_homog_em | 1.1474 | 0.2152 |

## 5. Replicates

| Model | EM per replicate | spread |
|---|---|---|
| qwen3.5-9b | 0.618 / 0.618 / 0.617 | 0.001 |
| qwen3.5-35b-a3b | 0.62 / 0.62 / 0.62 | 0.0 |
| gemma4-31b | 0.628 / 0.628 / 0.628 | 0.0 |
