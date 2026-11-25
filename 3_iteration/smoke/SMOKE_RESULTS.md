# Smoke test results

Updated 2026-10-09 06:21. Details per model in `<model>.json`, vLLM log in `<model>_server.log`.
Only models with **PASS** are used by `phase3.py run`.

| Model | Result | Tool parser | Parallel calls | Final answer line | Reasoning in output | Load (s) | Tested |
|---|---|---|---|---|---|---|---|
| qwen2.5-7b | PASS | hermes | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are explicitly identified a | 0 chars | 210.1 | 2026-10-06T02:47 |
| qwen3.5-4b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that both Scott Derrickso | 0 chars | 270.1 | 2026-10-05T15:22 |
| qwen3.5-9b | PASS | qwen3_coder | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 0 chars | 75.0 | 2026-10-05T15:28 |
| qwen3.5-27b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 110.0 | 2026-10-05T15:33 |
| qwen3.5-35b-a3b | PASS | qwen3_coder | yes | Reasoning summary: The search results confirm that Scott Derrickson is an Americ | 0 chars | 130.0 | 2026-10-05T15:45 |
| qwen3.5-122b-a10b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 535.0 | 2026-10-05T15:54 |
| qwen3.5-397b-a17b | PASS | qwen3_coder | yes | Reasoning summary: The search results explicitly state that Scott Derrickson is  | 0 chars | 850.1 | 2026-10-06T03:08 |
| qwen3.8-27b | PASS | qwen3_xml | yes | Reasoning summary: The knowledge base states Scott Derrickson is an American dir | 0 chars | 105.0 | 2026-10-05T15:34 |
| qwen3.8-flash-next | PASS | qwen3_xml | yes | Reasoning summary: The knowledge base states Scott Derrickson is an American dir | 0 chars | 420.1 | 2026-10-06T03:16 |
| gemma4-e4b | PASS | gemma4 | yes | Reasoning summary: The search results indicate that Scott Derrickson is an Ameri | 428 chars | 320.1 | 2026-10-06T16:54 |
| gemma4-12b | PASS | gemma4 | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are identified as American  | 7 chars | 180.0 | 2026-10-06T16:57 |
| gemma4-31b | PASS | gemma4 | yes | Reasoning summary: Scott Derrickson is an American director, screenwriter, and p | 0 chars | 175.0 | 2026-10-05T15:40 |
| gemma4-26b-a4b | PASS | gemma4 | yes |  / Final answer: yes | 0 chars | 175.0 | 2026-10-05T15:31 |
| ministral3-8b | PASS | mistral | yes |  / Final answer: no | 0 chars | 70.0 | 2026-10-06T16:59 |
| ministral3-14b | PASS | mistral | yes |  / Final answer: Yes | 0 chars | 85.0 | 2026-10-05T15:25 |
| mistral-small4-119b | PASS | mistral | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are explicitly described as | 0 chars | 185.0 | 2026-10-06T02:54 |
| nemotron3-nano-4b | PASS | qwen3_coder | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are identified as American  | 0 chars | 105.0 | 2026-10-06T02:49 |
| nemotron3-nano-30b-a3b | **FAIL** | qwen3_xml | yes | [answer forced after 6 searches]  | 0 chars | 65.0 | 2026-10-06T17:01 |
| nemotron3-super-120b-a12b | PASS | qwen3_xml | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 0 chars | 410.0 | 2026-10-06T17:08 |
| llama3.1-8b | PASS | llama3_json | no | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 0 chars | 50.0 | 2026-10-06T02:51 |
| muse-glimmer-30b | PASS | muse_glimmer | yes | Reasoning summary: Both Scott Derrickson and Ed Wood are described as American i | 200 chars | 125.0 | 2026-10-05T15:37 |
| llama3.3-70b | PASS | llama3_json | no | Reasoning summary: The search results provide information about Scott Derrickson | 0 chars | 135.0 | 2026-10-06T17:10 |
| deepseek-v4-flash | PASS | deepseek_v4 | no | yes |  | 1280.1 | 2026-10-06T03:37 |
