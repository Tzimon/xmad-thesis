# Phase 4 -- 2WikiMultiHopQA -- results report

Generated 2026-11-03 10:43 from 20 homogeneous and 46 mixed runs (1000 questions each).

## 1. Noise

No replicate runs in Phase 4. Threshold = run-to-run noise of Phase 3: **0.1 pp EM**.

## 2. Models on their own (homogeneous pairs)

| Model | FFN | Active B | EM debate | EM single agent (round 1) | Jaccard floor | Recall r1 | Recall all rounds | Forced answers |
|---|---|---|---|---|---|---|---|---|
| qwen3.8-flash-next | moe | 6 | 0.751 | 0.755 | 0.9121 | 0.9995 | 0.9995 | 72 % |
| qwen3.5-27b | dense | 27 | 0.727 | 0.728 | 0.9992 | 0.9965 | 0.9965 | 18 % |
| qwen3.5-35b-a3b | moe | 3 | 0.727 | 0.728 | 0.9864 | 0.9918 | 0.9928 | 10 % |
| qwen3.5-122b-a10b | moe | 10 | 0.721 | 0.729 | 0.9809 | 0.9965 | 0.9968 | 23 % |
| qwen3.5-397b-a17b | moe | 17 | 0.720 | 0.722 | 0.9868 | 0.991 | 0.9928 | 8 % |
| muse-glimmer-30b | dense | 29.6 | 0.713 | 0.716 | 0.9887 | 0.999 | 0.999 | 17 % |
| qwen3.5-4b | dense | 4 | 0.712 | 0.709 | 0.9963 | 0.9955 | 0.9965 | 18 % |
| qwen3.5-9b | dense | 9 | 0.693 | 0.693 | 0.9967 | 0.9935 | 0.994 | 8 % |
| mistral-small4-119b | moe | 6.5 | 0.693 | 0.676 | 0.9354 | 0.9812 | 0.9898 | 17 % |
| gemma4-31b | dense | 31 | 0.689 | 0.690 | 0.9991 | 0.9915 | 0.9915 | 7 % |
| qwen3.8-27b | dense | 27 | 0.681 | 0.679 | 0.9982 | 0.997 | 0.997 | 31 % |
| gemma4-26b-a4b | moe | 3.8 | 0.677 | 0.666 | 0.995 | 0.9705 | 0.9768 | 3 % |
| gemma4-12b | dense | 12 | 0.671 | 0.671 | 0.9933 | 0.9792 | 0.9822 | 14 % |
| ministral3-14b | dense | 14 | 0.645 | 0.617 | 0.9818 | 0.9567 | 0.969 | 6 % |
| gemma4-e4b | dense | 4 | 0.643 | 0.638 | 0.9901 | 0.9805 | 0.9832 | 3 % |
| ministral3-8b | dense | 8 | 0.625 | 0.602 | 0.9751 | 0.9643 | 0.9745 | 10 % |
| llama3.3-70b | dense | 70 | 0.571 | 0.561 | 0.9944 | 0.8373 | 0.8482 | 0 % |
| qwen2.5-7b | dense | 7.6 | 0.528 | 0.505 | 0.9905 | 0.8985 | 0.9125 | 0 % |
| nemotron3-nano-4b | dense | 4 | 0.448 | 0.446 | 0.9984 | 0.8998 | 0.8998 | 0 % |
| llama3.1-8b | dense | 8 | 0.414 | 0.407 | 0.9966 | 0.8488 | 0.851 | 0 % |

## 3. Mixed pairs

46 mixed pairs. Better than the better homogeneous pair: **12** (nominal p<0.05: **2**, significant after BH correction: **2**). Worse: **32** (significant: **6**).

Mean ΔEM: **-0.67 pp**; mean ΔF1: **-0.65 pp**. Mean headroom to the oracle (question solved by at least one homogeneous pair): **4.5 pp**.

Judge (mixed pairs): decided 3973 debates; right on **0.313** of them, while at least one agent was right in round 1 on **0.513**.

All pairs by ΔEM:

| Pair | Type | ΔEM | ΔF1 | p | q (BH) | Jaccard r1 | Recall all rounds |
|---|---|---|---|---|---|---|---|
| llama3.1-8b + llama3.3-70b | DxD, same vendor | +5.1 pp * | +5.2 pp | 1e-05 | 0.00046 | 0.6471 | 0.9517 |
| qwen2.5-7b + llama3.1-8b | DxD, diff. vendor | +3.9 pp * | +4.3 pp | 0.00151 | 0.01158 | 0.7273 | 0.955 |
| qwen3.5-27b + qwen3.5-35b-a3b | DxM, same vendor | +0.6 pp | -0.1 pp | 0.41769 | 0.71162 | 0.8872 | 0.998 |
| qwen3.5-4b + qwen3.5-27b | DxD, same vendor | +0.6 pp | -0.2 pp | 0.41769 | 0.71162 | 0.8917 | 0.9995 |
| gemma4-12b + gemma4-26b-a4b | DxM, same vendor | +0.6 pp | +1.0 pp | 0.46139 | 0.75048 | 0.8802 | 0.9912 |
| qwen3.5-4b + qwen3.5-122b-a10b | DxM, same vendor | +0.5 pp | -0.3 pp | 0.58313 | 0.83825 | 0.8806 | 0.9995 |
| ministral3-8b + ministral3-14b | DxD, same vendor | +0.4 pp | +0.9 pp | 0.72755 | 0.90452 | 0.8604 | 0.987 |
| ministral3-14b + gemma4-12b | DxD, diff. vendor | +0.4 pp | +0.6 pp | 0.65874 | 0.88172 | 0.8254 | 0.9912 |
| muse-glimmer-30b + llama3.3-70b | DxD, same vendor | +0.3 pp | +0.4 pp | 0.69004 | 0.88172 | 0.5384 | 0.999 |
| llama3.3-70b + qwen3.5-122b-a10b | DxM, diff. vendor | +0.1 pp | -0.3 pp | 1.0 | 1.0 | 0.5336 | 0.9972 |
| qwen3.5-4b + qwen3.5-35b-a3b | DxM, same vendor | +0.1 pp | -1.0 pp | 1.0 | 1.0 | 0.8776 | 1.0 |
| qwen3.5-4b + qwen3.5-397b-a17b | DxM, same vendor | +0.1 pp | +0.1 pp | 1.0 | 1.0 | 0.8784 | 1.0 |
| qwen3.5-4b + qwen3.5-9b | DxD, same vendor | +0.0 pp | +0.3 pp | 1.0 | 1.0 | 0.8767 | 0.9995 |
| qwen3.5-122b-a10b + qwen3.5-397b-a17b | MxM, same vendor | +0.0 pp | -0.2 pp | 1.0 | 1.0 | 0.9102 | 0.9978 |
| qwen3.5-9b + gemma4-12b | DxD, diff. vendor | -0.2 pp | -0.1 pp | 0.89568 | 1.0 | 0.8479 | 0.9965 |
| qwen3.5-35b-a3b + qwen3.5-397b-a17b | MxM, same vendor | -0.3 pp | -0.9 pp | 0.81755 | 0.94287 | 0.9127 | 0.9968 |
| llama3.3-70b + mistral-small4-119b | DxM, diff. vendor | -0.3 pp | +0.4 pp | 0.81989 | 0.94287 | 0.5398 | 0.9958 |
| qwen3.5-35b-a3b + qwen3.5-122b-a10b | MxM, same vendor | -0.3 pp | -0.6 pp | 0.79484 | 0.94287 | 0.8958 | 0.998 |
| gemma4-31b + gemma4-26b-a4b | DxM, same vendor | -0.4 pp | -0.7 pp | 0.54126 | 0.82993 | 0.8711 | 0.9945 |
| qwen3.5-27b + qwen3.5-122b-a10b | DxM, same vendor | -0.4 pp | -0.5 pp | 0.60759 | 0.84694 | 0.9096 | 0.9975 |
| qwen3.5-122b-a10b + mistral-small4-119b | MxM, diff. vendor | -0.4 pp | -0.8 pp | 0.67181 | 0.88172 | 0.8262 | 1.0 |
| qwen3.5-27b + qwen3.5-397b-a17b | DxM, same vendor | -0.5 pp | -0.4 pp | 0.58313 | 0.83825 | 0.9022 | 0.998 |
| gemma4-12b + gemma4-31b | DxD, same vendor | -0.5 pp | -0.4 pp | 0.47313 | 0.75048 | 0.879 | 0.995 |
| qwen3.5-9b + qwen3.5-35b-a3b | DxM, same vendor | -0.9 pp | -1.8 pp | 0.33556 | 0.61743 | 0.8885 | 0.998 |
| gemma4-e4b + gemma4-12b | DxD, same vendor | -0.9 pp | +0.0 pp | 0.28061 | 0.53784 | 0.8383 | 0.9938 |
| ministral3-14b + mistral-small4-119b | DxM, same vendor | -1.1 pp | -0.2 pp | 0.26641 | 0.53282 | 0.8231 | 0.997 |
| qwen3.5-9b + qwen3.5-27b | DxD, same vendor | -1.1 pp | -1.3 pp | 0.14387 | 0.30082 | 0.8821 | 0.9975 |
| muse-glimmer-30b + gemma4-26b-a4b | DxM, diff. vendor | -1.2 pp | -0.9 pp | 0.12635 | 0.2928 | 0.8122 | 0.999 |
| qwen3.5-9b + qwen3.5-122b-a10b | DxM, same vendor | -1.2 pp | -1.6 pp | 0.13367 | 0.2928 | 0.8873 | 0.9975 |
| gemma4-e4b + gemma4-26b-a4b | DxM, same vendor | -1.2 pp | +0.2 pp | 0.13367 | 0.2928 | 0.8383 | 0.9938 |
| qwen3.5-9b + qwen3.5-397b-a17b | DxM, same vendor | -1.3 pp (*) | -1.0 pp | 0.02945 | 0.10421 | 0.9045 | 0.9968 |
| qwen3.5-9b + ministral3-8b | DxD, diff. vendor | -1.3 pp | -1.2 pp | 0.09837 | 0.26618 | 0.8533 | 0.9958 |
| muse-glimmer-30b + qwen3.5-35b-a3b | DxM, diff. vendor | -1.4 pp | -2.6 pp | 0.12492 | 0.2928 | 0.8712 | 0.999 |
| llama3.1-8b + muse-glimmer-30b | DxD, same vendor | -1.4 pp (*) | -1.1 pp | 0.03355 | 0.11024 | 0.6624 | 0.9995 |
| gemma4-31b + muse-glimmer-30b | DxD, diff. vendor | -1.6 pp (*) | -0.8 pp | 0.04022 | 0.12334 | 0.8569 | 0.999 |
| qwen3.5-27b + muse-glimmer-30b | DxD, diff. vendor | -1.7 pp (*) | -2.5 pp | 0.02409 | 0.09488 | 0.861 | 0.999 |
| qwen3.5-27b + qwen3.8-27b | DxD, same vendor | -1.7 pp (*) | -1.0 pp | 0.0498 | 0.14317 | 0.8596 | 0.9985 |
| gemma4-e4b + gemma4-31b | DxD, same vendor | -1.8 pp (*) | -1.2 pp | 0.01135 | 0.05801 | 0.8872 | 0.9955 |
| qwen3.5-397b-a17b + qwen3.8-flash-next | MxM, same vendor | -1.8 pp (*) | -1.7 pp | 0.02475 | 0.09488 | 0.8009 | 1.0 |
| qwen3.8-27b + qwen3.8-flash-next | DxM, same vendor | -2.0 pp (*) | -1.0 pp | 0.01686 | 0.07756 | 0.8513 | 1.0 |
| qwen3.5-27b + gemma4-31b | DxD, diff. vendor | -2.4 pp * | -2.7 pp | 0.00184 | 0.01209 | 0.8748 | 0.997 |
| ministral3-8b + mistral-small4-119b | DxM, same vendor | -2.7 pp * | -1.6 pp | 0.0036 | 0.0207 | 0.8202 | 0.9968 |
| qwen3.5-4b + nemotron3-nano-4b | DxD, diff. vendor | -2.8 pp * | -2.7 pp | 0.00031 | 0.00475 | 0.7645 | 0.998 |
| qwen3.5-35b-a3b + gemma4-26b-a4b | MxM, diff. vendor | -2.9 pp * | -3.4 pp | 0.00077 | 0.00708 | 0.8248 | 0.9945 |
| qwen3.5-27b + gemma4-26b-a4b | DxM, diff. vendor | -2.9 pp * | -3.2 pp | 0.00015 | 0.00345 | 0.8215 | 0.997 |
| gemma4-31b + qwen3.5-35b-a3b | DxM, diff. vendor | -3.1 pp * | -3.3 pp | 0.00054 | 0.00621 | 0.8766 | 0.9965 |

## 4. Where do differences come from?

| Grouping | Group | Pairs | mean ΔEM | mean ΔF1 | better / nominal / BH-sig. better / BH-sig. worse | mean Jaccard r1 |
|---|---|---|---|---|---|---|
| pair_type | DxD | 19 | -0.4 pp | -0.2 pp | 32 % / 2 / 2 / 2 | 0.812 |
| pair_type | DxM | 21 | -0.9 pp | -0.9 pp | 29 % / 0 / 0 / 3 | 0.836 |
| pair_type | MxM | 6 | -0.9 pp | -1.3 pp | 0 % / 0 / 0 / 1 | 0.862 |
| same_vendor | False | 16 | -1.1 pp | -1.2 pp | 19 % / 1 / 1 / 5 | 0.795 |
| same_vendor | True | 30 | -0.4 pp | -0.4 pp | 30 % / 1 / 1 / 1 | 0.848 |
| pair_type+same_vendor | DxD / False | 8 | -0.7 pp | -0.6 pp | 25 % / 1 / 1 / 2 | 0.826 |
| pair_type+same_vendor | DxD / True | 11 | -0.1 pp | +0.2 pp | 36 % / 1 / 1 / 0 | 0.802 |
| pair_type+same_vendor | DxM / False | 6 | -1.5 pp | -1.7 pp | 17 % / 0 / 0 / 2 | 0.742 |
| pair_type+same_vendor | DxM / True | 15 | -0.7 pp | -0.6 pp | 33 % / 0 / 0 / 1 | 0.873 |
| pair_type+same_vendor | MxM / False | 2 | -1.7 pp | -2.1 pp | 0 % / 0 / 0 / 1 | 0.826 |
| pair_type+same_vendor | MxM / True | 4 | -0.6 pp | -0.9 pp | 0 % / 0 / 0 / 0 | 0.88 |

Least-squares fit over all mixed pairs (descriptive, small n):

| Outcome | Term | Coefficient | Std. error |
|---|---|---|---|
| delta_em | intercept | 0.0983 | 0.0257 |
| delta_em | different_vendor | -0.0092 | 0.0041 |
| delta_em | dense_x_moe | -0.0011 | 0.0047 |
| delta_em | moe_x_moe | 0.0034 | 0.0066 |
| delta_em | log2_active_param_ratio | 0.002 | 0.0018 |
| delta_em | mean_homog_em | -0.1537 | 0.0386 |
| jaccard_r1 | intercept | 0.1253 | 0.146 |
| jaccard_r1 | different_vendor | -0.0392 | 0.023 |
| jaccard_r1 | dense_x_moe | -0.018 | 0.0268 |
| jaccard_r1 | moe_x_moe | -0.0266 | 0.0374 |
| jaccard_r1 | log2_active_param_ratio | -0.0182 | 0.0103 |
| jaccard_r1 | mean_homog_em | 1.1114 | 0.2187 |
