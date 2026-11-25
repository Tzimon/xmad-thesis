# Smoke test results

Updated 2026-10-24 14:06. Details per model in `<model>.json`, vLLM log in `<model>_server.log`.
Only models with **PASS** are used by `phase4.py run`.

| Model | Result | Tool parser | Parallel calls | Final answer line | Reasoning in output | Load (s) | Tested |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b | PASS | hermes | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are explicitly identified a | 0 chars | 205.1 | 2026-10-23T09:06 |
| qwen3.5-4b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that both Scott Derrickso | 0 chars | 190.1 | 2026-10-24T08:46 |
| qwen3.5-9b | PASS | qwen3_coder | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 0 chars | 115.0 | 2026-10-24T08:48 |
| qwen3.5-27b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 190.0 | 2026-10-24T08:52 |
| qwen3.5-35b-a3b | PASS | qwen3_coder | yes | Reasoning summary: The search results confirm that Scott Derrickson is an Americ | 0 chars | 230.1 | 2026-10-24T08:56 |
| qwen3.5-122b-a10b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 851.5 | 2026-10-24T09:46 |
| qwen3.5-397b-a17b | PASS | qwen3_xml | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 2195.3 | 2026-10-24T14:06 |
| qwen3.8-27b | PASS | qwen3_xml | yes | Reasoning summary: The knowledge base states Scott Derrickson is an American dir | 0 chars | 205.1 | 2026-10-24T08:59 |
| qwen3.8-flash-next | PASS | qwen3_xml | yes | Reasoning summary: The knowledge base states Scott Derrickson is an American dir | 0 chars | 1340.5 | 2026-10-24T11:54 |
| gemma4-e4b | PASS | gemma4 | yes | Reasoning summary: The search results indicate that Scott Derrickson is an Ameri | 428 chars | 330.1 | 2026-10-24T09:05 |
| gemma4-12b | PASS | gemma4 | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are identified as American  | 7 chars | 295.1 | 2026-10-24T09:10 |
| gemma4-31b | PASS | gemma4 | yes | Reasoning summary: Scott Derrickson is an American director, screenwriter, and p | 0 chars | 385.1 | 2026-10-24T09:16 |
| gemma4-26b-a4b | PASS | gemma4 | yes |  / Final answer: yes | 0 chars | 285.1 | 2026-10-24T09:21 |
| ministral3-8b | PASS | mistral | yes |  / Final answer: no | 0 chars | 90.0 | 2026-10-24T09:23 |
| ministral3-14b | PASS | mistral | yes |  / Final answer: Yes | 0 chars | 140.1 | 2026-10-24T09:25 |
| mistral-small4-119b | PASS | mistral | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are explicitly described as | 0 chars | 685.1 | 2026-10-24T09:58 |
| nemotron3-nano-4b | PASS | qwen3_coder | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are identified as American  | 0 chars | 90.0 | 2026-10-24T09:27 |
| nemotron3-nano-30b-a3b | not tested | | | | | | |
| nemotron3-super-120b-a12b | not tested | | | | | | |
| llama3.1-8b | PASS | llama3_json | no | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 0 chars | 100.0 | 2026-10-24T09:28 |
| muse-glimmer-30b | PASS | muse_glimmer | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 200 chars | 200.0 | 2026-10-24T09:32 |
| llama3.3-70b | PASS | llama3_json | no | Reasoning summary: The search results provide information about Scott Derrickson | 0 chars | 795.1 | 2026-10-24T10:11 |
| deepseek-v4-flash | PASS | deepseek_v4 | no | yes |  | 730.1 | 2026-10-23T09:18 |
