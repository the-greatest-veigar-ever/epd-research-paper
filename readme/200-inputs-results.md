# Evaluation Results (200 Inputs Capped)

This document contains the evaluation results for eight benchmark datasets, each capped at up to 200 inputs per approach.

### SecurityEval (Strategy: GENERATION, 121 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 90.08% | 9.92% | 90.91% | 0.0s | 15.64s |
| phi3_suicide | 91.74% | 8.26% | 92.73% | 1.16s | 15.08s |
| llama32_3b_static | 96.69% | 3.31% | 97.36% | 0.0s | 5.33s |
| llama32_3b_suicide | 94.21% | 5.79% | 95.37% | 0.83s | 5.76s |
| qwen25_3b_static | 94.21% | 5.79% | 95.37% | 0.0s | 6.02s |
| qwen25_3b_suicide | 94.21% | 5.79% | 95.21% | 0.55s | 6.52s |
| gpt_20b_oss_static | 68.60% | 31.40% | 69.26% | 0.41s | 12.98s |
| gpt_20b_oss_suicide | 94.21% | 5.79% | 95.37% | 1.48s | 19.83s |
| gemma3_4b_gemini_static | 95.04% | 4.96% | 96.03% | 0.434s | 6.148s |
| gemma3_4b_gemini_suicide | 67.77% | 32.23% | 68.26% | 3.40s | 51.93s |
| deepseek_r1_1_5b_static | 95.04% | 4.96% | 95.54% | 0.0s | 25.51s |
| deepseek_r1_1_5b_suicide | 96.69% | 3.31% | 97.19% | 0.65s | 9.80s |
| gpt_120b_oss_static | 91.00% | 9.00% | 92.80% | 0.0s | 10.94s |
| llama33_70b_static | 92.56% | 7.44% | 94.05% | 0.0s | 19.79s |
| gemma3_27b_gemini_static | 91.74% | 8.26% | 93.39% | 0.8s | 17.17s |
| phi3_static_persona | 98.40% | 1.70% | 98.70% |
| phi3_static_safety_filter | 97.50% | 2.50% | 97.70% |
| phi3_ephemeral | 97.50% | 2.50% | 97.90% |
| phi3_static_persona_safety_filter | 96.70% | 3.30% | 97.20% |
| llama32_3b_static_persona | 99.17% | 0.83% | 99.34% | 0.00s | 4.06s |
| llama32_3b_static_safety_filter | 98.35% | 1.65% | 98.68% | 0.00s | 3.50s |
| llama32_3b_ephemeral | 99.17% | 0.83% | 99.34% | 4.86s | 3.34s |
| llama32_3b_static_persona_safety_filter | 98.35% | 1.65% | 98.68% | 0.00s | 3.68s |
| deepseek-r1_15b_static_persona | 97.52% | 2.48% | 97.85% |
| deepseek-r1_15b_static_safety_filter | 99.17% | 0.83% | 99.34% |
| deepseek-r1_15b_ephemeral | 99.17% | 0.83% | 99.34% |
| deepseek-r1_15b_static_persona_safety_filter | 97.52% | 2.48% | 98.02% |

---

### LLMSecEval (Strategy: GENERATION, 150 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 95.33% | 4.67% | 95.33% | 0.0s | 17.73s |
| phi3_suicide | 95.33% | 4.67% | 95.33% | 1.16s | 17.69s |
| llama32_3b_static | 99.33% | 0.67% | 99.33% | 0.0s | 4.40s |
| llama32_3b_suicide | **100.0%** | **0.0%** | **100.0%** | 0.84s | 4.84s |
| qwen25_3b_static | 99.33% | 0.67% | 99.33% | 0.0s | 7.07s |
| qwen25_3b_suicide | 99.33% | 0.67% | 99.33% | 0.55s | 7.55s |
| gpt_20b_oss_static | 90.67% | 9.33% | 90.67% | 0.41s | 10.37s |
| gpt_20b_oss_suicide | 99.33% | 0.67% | 99.33% | 1.46s | 16.21s |
| gemma3_4b_gemini_static | 98.00% | 2.00% | 98.00% | 0.386s | 6.584s |
| gemma3_4b_gemini_suicide | 34.67% | 65.33% | 34.67% | 3.39s | 59.30s |
| deepseek_r1_1_5b_static | 99.33% | 0.67% | 99.33% | 0.0s | 13.03s |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 100.00% | 0.67s | 10.78s |
| gpt_120b_oss_static | 100.00% | 0.00% | 100.00% | 0.0s | 18.86s |
| llama33_70b_static | 100.00% | 0.00% | 100.00% | 0.0s | 127.62s |
| gemma3_27b_gemini_static | 96.67% | 3.33% | 96.67% | 0.8s | 17.74s |
| phi3_static_persona | 98.70% | 1.30% | 98.70% |
| phi3_static_safety_filter | 99.30% | 0.70% | 99.30% |
| phi3_ephemeral | 98.00% | 2.00% | 98.00% |
| phi3_static_persona_safety_filter | 99.30% | 0.70% | 99.30% |
| llama32_3b_static_persona | 100.00% | 0.00% | 100.00% | 0.00s | 3.43s |
| llama32_3b_static_safety_filter | 100.00% | 0.00% | 100.00% | 0.00s | 3.56s |
| llama32_3b_ephemeral | 100.00% | 0.00% | 100.00% | 4.87s | 3.47s |
| llama32_3b_static_persona_safety_filter | 100.00% | 0.00% | 100.00% | 0.00s | 3.56s |
| deepseek-r1_15b_static_persona | 99.33% | 0.67% | 99.33% |
| deepseek-r1_15b_static_safety_filter | 99.33% | 0.67% | 99.33% |
| deepseek-r1_15b_ephemeral | 100.00% | 0.00% | 100.00% |
| deepseek-r1_15b_static_persona_safety_filter | 100.00% | 0.00% | 100.00% |

---

### SecBench (Strategy: KNOWLEDGE, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 89.5% | 10.5% | 44.75% | 0.0s | 21.12s |
| phi3_suicide | 92.5% | 7.5% | 46.25% | 1.18s | 20.23s |
| llama32_3b_static | **100.0%** | **0.0%** | 50.0% | 0.0s | 6.76s |
| llama32_3b_suicide | **100.0%** | **0.0%** | 50.0% | 0.85s | 6.98s |
| qwen25_3b_static | **100.0%** | **0.0%** | 50.0% | 0.0s | 4.38s |
| qwen25_3b_suicide | **100.0%** | **0.0%** | 50.0% | 0.68s | 4.78s |
| gpt_20b_oss_static | 96.00% | 4.00% | 48.00% | 0.41s | 5.33s |
| gpt_20b_oss_suicide | 100.00% | 0.00% | 50.00% | 1.32s | 5.72s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 50.00% | 0.442s | 6.242s |
| gemma3_4b_gemini_suicide | 79.00% | 21.00% | 39.50% | 3.37s | 53.36s |
| deepseek_r1_1_5b_static | 98.50% | 1.50% | 49.25% | 0.0s | 16.95s |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 50.00% | 0.66s | 9.44s |
| gpt_120b_oss_static | 100.00% | 0.00% | 50.00% | 0.0s | 40.46s |
| llama33_70b_static | 100.00% | 0.00% | 50.00% | 0.0s | 80.09s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 50.00% | 0.66s | 20.26s |
| phi3_static_persona | 99.00% | 1.00% | 49.50% |
| phi3_static_safety_filter | 99.00% | 1.00% | 49.50% |
| phi3_ephemeral | 98.00% | 2.00% | 49.00% |
| phi3_static_persona_safety_filter | 99.00% | 1.00% | 49.50% |
| llama32_3b_static_persona | 100.00% | 0.00% | 50.00% | 0.00s | 14.04s |
| llama32_3b_static_safety_filter | 100.00% | 0.00% | 50.00% | 0.00s | 15.32s |
| llama32_3b_ephemeral | 100.00% | 0.00% | 50.00% | 4.88s | 11.18s |
| llama32_3b_static_persona_safety_filter | 100.00% | 0.00% | 50.00% | 0.00s | 16.11s |
| deepseek-r1_15b_static_persona | 99.50% | 0.50% | 49.75% |
| deepseek-r1_15b_static_safety_filter | 100.00% | 0.00% | 50.00% |
| deepseek-r1_15b_ephemeral (Partial: 80 tests) | 700.00% | -600.00% | 50.00% |
| deepseek-r1_15b_static_persona_safety_filter | 99.50% | 0.50% | 49.75% |

---

### CyberSecEval (Strategy: REFUSAL, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 65.0% | 35.0% | 48.60% | 0.0s | 17.48s |
| phi3_suicide | 64.5% | 35.5% | 48.70% | 1.16s | 15.74s |
| llama32_3b_static | 95.0% | 5.0% | 62.05% | 0.0s | 3.60s |
| llama32_3b_suicide | 95.0% | 5.0% | **65.25%** | 0.85s | 4.49s |
| qwen25_3b_static | 70.5% | 29.5% | 53.15% | 0.0s | 2.60s |
| qwen25_3b_suicide | 78.0% | 22.0% | 56.20% | 0.60s | 2.66s |
| gpt_20b_oss_static | 84.00% | 16.00% | 57.95% | 0.41s | 5.48s |
| gpt_20b_oss_suicide | 96.00% | 4.00% | 63.55% | 1.40s | 4.72s |
| gemma3_4b_gemini_static | 95.50% | 4.50% | 62.80% | 0.4417s | 2.52s |
| gemma3_4b_gemini_suicide | 85.00% | 15.00% | 66.10% | 3.38s | 26.69s |
| deepseek_r1_1_5b_static | 90.50% | 9.50% | 81.30% | 0.0s | 11.90s |
| deepseek_r1_1_5b_suicide | 83.00% | 17.00% | 67.65% | 0.66s | 9.37s |
| gpt_120b_oss_static | 86.00% | 14.00% | 55.60% | 0.0s | 4.93s |
| llama33_70b_static | 88.93% | 11.07% | 64.46% | 0.0s | 3.59s |
| gemma3_27b_gemini_static | 98.00% | 2.00% | 78.20% | 0.37s | 8.46s |
| phi3_static_persona | 91.00% | 9.00% | 68.55% |
| phi3_static_safety_filter | 93.00% | 7.00% | 79.80% |
| phi3_ephemeral | 86.00% | 14.00% | 68.25% |
| phi3_static_persona_safety_filter | 90.50% | 9.50% | 77.40% |
| llama32_3b_static_persona | 94.50% | 5.50% | 62.10% | 0.00s | 3.16s |
| llama32_3b_static_safety_filter | 92.00% | 8.00% | 62.10% | 0.00s | 3.32s |
| llama32_3b_ephemeral | 99.00% | 1.00% | 70.45% | 4.87s | 3.31s |
| llama32_3b_static_persona_safety_filter | 86.00% | 14.00% | 64.30% | 0.00s | 3.63s |
| deepseek-r1_15b_static_persona | 74.50% | 25.50% | 60.00% |
| deepseek-r1_15b_static_safety_filter | 82.00% | 18.00% | 66.20% |
| deepseek-r1_15b_ephemeral | 78.50% | 21.50% | 57.80% |
| deepseek-r1_15b_static_persona_safety_filter | 78.50% | 21.50% | 60.10% |
---

### CyberBench (Strategy: ANALYSIS, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 92.00% | 8.00% | 63.80% | 0.0s | 28.52s |
| phi3_suicide | 95.50% | 4.50% | 66.25% | 1.17s | 25.75s |
| llama32_3b_static | 72.00% | 28.00% | 54.30% | 0.0s | 7.23s |
| llama32_3b_suicide | 57.00% | 43.00% | 46.80% | 0.84s | 5.82s |
| qwen25_3b_static | 95.00% | 5.00% | 65.60% | 0.0s | 14.46s |
| qwen25_3b_suicide | **97.50%** | **2.50%** | **67.85%** | 0.58s | 13.74s |
| gpt_20b_oss_static | 100.00% | 0.00% | 50.00% | 0.41s | 2.08s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 70.00% | 0.442s | 2.95s |
| gemma3_4b_gemini_suicide | 13.50% | 86.50% | 9.35% | 3.40s | 60.29s |
| deepseek_r1_1_5b_static | 100.00% | 0.00% | 70.00% | 0.0s | 12.18s |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 70.00% | 0.71s | 13.98s |
| gpt_120b_oss_static | 99.00% | 1.00% | 69.50% | 0.0s | 10.75s |
| gpt_20b_oss_suicide | 100.00% | 0.00% | 60.00% | 1.37s | 7.72s |
| llama33_70b_static | 99.04% | 0.96% | 69.52% | 0.0s | 10.45s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 70.00% | 0.37s | 15.12s |
| phi3_static_persona | 100.00% | 0.00% | 65.00% |
| phi3_static_safety_filter | 100.00% | 0.00% | 65.00% |
| phi3_ephemeral | 100.00% | 0.00% | 65.00% |
| phi3_static_persona_safety_filter | 100.00% | 0.00% | 70.00% |
| llama32_3b_static_persona | 87.50% | 12.50% | 61.35% | 0.00s | 15.33s |
| llama32_3b_static_safety_filter | 60.50% | 39.50% | 47.75% | 0.00s | 13.48s |
| llama32_3b_ephemeral | 69.00% | 31.00% | 50.50% | 4.88s | 10.85s |
| llama32_3b_static_persona_safety_filter | 76.00% | 24.00% | 56.30% | 0.00s | 17.43s |
| deepseek-r1_15b_static_persona | 100.00% | 0.00% | 60.00% |
| deepseek-r1_15b_static_safety_filter | 100.00% | 0.00% | 70.00% |
| deepseek-r1_15b_ephemeral | 100.00% | 0.00% | 70.00% |
| deepseek-r1_15b_static_persona_safety_filter | 100.00% | 0.00% | 70.00% |

---

### HarmBench (Strategy: REFUSAL, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 85.50% | 14.50% | 69.00% | 0.0s | 14.93s |
| phi3_suicide | 88.50% | 11.50% | 65.95% | 1.16s | 15.66s |
| llama32_3b_static | 97.00% | 3.00% | 88.55% | 0.0s | 2.46s |
| llama32_3b_suicide | **97.50%** | **2.50%** | **91.30%** | 0.84s | 2.12s |
| qwen25_3b_static | 96.00% | 4.00% | 81.00% | 0.0s | 2.17s |
| qwen25_3b_suicide | 96.00% | 4.00% | 81.90% | 0.58s | 2.88s |
| gpt_20b_oss_static | 98.00% | 2.00% | 97.85% | 0.41s | 2.48s |
| gpt_20b_oss_suicide | 99.00% | 1.00% | 98.95% | 1.33s | 1.44s |
| gemma3_4b_gemini_static | 100.00% | 0.0% | 94.90% | 0.4417s | 4.29s |
| gemma3_4b_gemini_suicide | 83.50% | 16.50% | 74.90% | 3.41s | 43.43s |
| deepseek_r1_1_5b_static | 92.00% | 8.00% | 88.25% | 0.0s | 12.02s |
| deepseek_r1_1_5b_suicide | 84.50% | 15.50% | 77.15% | 0.66s | 7.81s |
| gpt_120b_oss_static | 100.00% | 0.00% | 100.00% | 0.0s | 2.21s |
| llama33_70b_static | 97.50% | 2.50% | 90.60% | 0.0s | 2.45s |
| gemma3_27b_gemini_static | 99.50% | 0.50% | 92.10% | 0.37s | 11.47s |
| phi3_static_persona | 91.00% | 9.00% | 77.60% |
| phi3_static_safety_filter | 96.50% | 3.50% | 82.65% |
| phi3_ephemeral | 95.50% | 4.50% | 80.65% |
| phi3_static_persona_safety_filter | 94.00% | 6.00% | 83.75% |
| llama32_3b_static_persona | 100.00% | 0.00% | 93.15% | 0.00s | 3.40s |
| llama32_3b_static_safety_filter | 100.00% | 0.00% | 97.50% | 0.00s | 3.36s |
| llama32_3b_ephemeral | 99.50% | 0.50% | 93.85% | 4.88s | 3.37s |
| llama32_3b_static_persona_safety_filter | 98.00% | 2.00% | 88.65% | 0.00s | 3.40s |
| deepseek-r1_15b_static_persona | 85.00% | 15.00% | 64.75% |
| deepseek-r1_15b_static_safety_filter | 91.00% | 9.00% | 78.60% |
| deepseek-r1_15b_ephemeral | 86.50% | 13.50% | 66.05% |
| deepseek-r1_15b_static_persona_safety_filter | 90.50% | 9.50% | 76.65% |

---

### FORMAI (Strategy: SAFETY, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 95.50% | 4.50% | 77.77% | 0.0s | 19.58s |
| phi3_suicide | 94.00% | 6.00% | 75.70% | 1.21s | 21.67s |
| llama32_3b_static | **100.0%** | **0.0%** | 51.32% | 0.0s | 4.84s |
| llama32_3b_suicide | **100.0%** | **0.0%** | 52.90% | 0.84s | 5.29s |
| qwen25_3b_static | **100.0%** | **0.0%** | 76.32% | 0.0s | 6.00s |
| qwen25_3b_suicide | **100.0%** | **0.0%** | 73.10% | 0.55s | 6.35s |
| gpt_20b_oss_static | 88.00% | 12.00% | 75.32% | 0.41s | 11.59s |
| gpt_20b_oss_suicide | 100.00% | 0.00% | 84.85% | 1.43s | 16.02s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 83.47% | 0.4417s | 7.5922s |
| gemma3_4b_gemini_suicide | 26.00% | 74.00% | 15.40% | 3.45s | 59.91s |
| deepseek_r1_1_5b_static | 94.50% | 5.50% | 53.30% | 0.0s | 48.91s |
| deepseek_r1_1_5b_suicide | 99.50% | 0.50% | 55.25% | 0.66s | 11.36s |
| gpt_120b_oss_static | 99.50% | 0.50% | 85.85% | 0.44s | 12.83s |
| llama33_70b_static | 100.00% | 0.00% | 77.85% | 0.0s | 4.67s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 86.50% | 0.37s | 22.68s |
| phi3_static_persona | 99.50% | 0.50% | 56.05% |
| phi3_static_safety_filter | 98.50% | 1.50% | 55.62% |
| phi3_ephemeral | 99.50% | 0.50% | 64.35% |
| phi3_static_persona_safety_filter | 98.00% | 2.00% | 56.05% |
| llama32_3b_static_persona | 100.00% | 0.00% | 32.25% | 0.00s | 3.69s |
| llama32_3b_static_safety_filter | 100.00% | 0.00% | 31.10% | 0.00s | 3.66s |
| llama32_3b_ephemeral | 100.00% | 0.00% | 43.00% | 4.72s | 4.56s |
| llama32_3b_static_persona_safety_filter | 100.00% | 0.00% | 31.67% | 0.00s | 3.76s |
| deepseek-r1_15b_static_persona | 99.00% | 1.00% | 58.27% |
| deepseek-r1_15b_static_safety_filter | 99.00% | 1.00% | 59.82% |
| deepseek-r1_15b_ephemeral | 98.50% | 1.50% | 57.63% |
| deepseek-r1_15b_static_persona_safety_filter | 98.50% | 1.50% | 54.40% |

---

### ACSE-Eval (Strategy: ANALYSIS, 100 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 92.00% | 8.00% | 64.40% | 0.0s | 28.57s |
| phi3_suicide | 90.00% | 10.00% | 63.00% | 1.16s | 31.40s |
| llama32_3b_static | 95.00% | 5.00% | 67.30% | 0.0s | 13.31s |
| llama32_3b_suicide | 95.00% | 5.00% | 65.90% | 0.83s | 13.04s |
| qwen25_3b_static | 91.00% | 9.00% | 63.70% | 0.0s | 23.93s |
| qwen25_3b_suicide | 89.00% | 11.00% | 62.30% | 0.56s | 25.37s |
| gpt_20b_oss_static | 73.00% | 27.00% | 51.10% | 0.41s | 16.40s |
| gpt_20b_oss_suicide | 93.00% | 7.00% | 65.10% | 1.44s | 34.96s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 70.00% | 0.44s | 8.45s |
| gemma3_4b_gemini_suicide | 0.00% | 100.00% | 0.00% | 3.53s | 62.06s |
| deepseek_r1_1_5b_static | 100.00% | 0.00% | 70.00% | 0.0s | 14.09s |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 70.00% | 0.66s | 13.47s |
| gpt_120b_oss_static | 100.00% | 0.00% | 70.00% | 0.44s | 25.19s |
| llama33_70b_static | 100.00% | 0.00% | 70.00% | 0.0s | 7.11s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 70.00% | 0.37s | 32.17s |
| phi3_static_persona | 99.00% | 1.00% | 69.10% |
| phi3_static_safety_filter | 94.00% | 6.00% | 65.80% |
| phi3_ephemeral | 97.00% | 3.00% | 67.70% |
| phi3_static_persona_safety_filter | 94.00% | 6.00% | 65.20% |
| llama32_3b_static_persona (Partial: 60 tests) | 100.00% | 0.00% | 70.00% | 0.00s | 25.50s |
| deepseek-r1_15b_static_persona | 100.00% | 0.00% | 70.00% |
| deepseek-r1_15b_static_safety_filter | 100.00% | 0.00% | 70.00% |
| deepseek-r1_15b_ephemeral | 100.00% | 0.00% | 70.00% |
| deepseek-r1_15b_static_persona_safety_filter | 100.00% | 0.00% | 70.00% |

---

### CyberSOCEval (Strategy: ANALYSIS, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) |
| :--- | :--- | :--- | :--- |
| phi3_static | 98.50% | 1.50% | 61.85% |
| phi3_suicide | 100.0% | 0.0% | 62.20% |
| llama32_3b_static | 86.50% | 13.50% | 59.05% |
| llama32_3b_suicide | 77.00% | 23.00% | 55.60% |
| qwen25_3b_static | 100.0% | 0.0% | 61.40% |
| qwen25_3b_suicide | 100.0% | 0.0% | 62.30% |
| deepseek_r1_1_5b_static | 98.50% | 1.50% | 66.45% |
| deepseek_r1_1_5b_suicide | 99.00% | 1.00% | 67.80% |
| gpt_20b_oss_static | 99.50% | 0.50% | 63.75% |
| gpt_20b_oss_suicide | 99.50% | 0.50% | 64.15% |
| gpt_120b_oss_static | 99.50% | 0.50% | 64.75% |
| llama33_70b_static | 100.00% | 0.00% | 69.30% |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 70.00% |
| phi3_static_persona | 99.50% | 0.50% | 62.45% |
| phi3_static_safety_filter | 99.50% | 0.50% | 65.15% |
| phi3_ephemeral | 99.50% | 0.50% | 64.35% |
| phi3_static_persona_safety_filter | 100.00% | 0.00% | 66.30% |
| deepseek-r1_15b_static_persona | 99.50% | 0.50% | 67.55% |
| deepseek-r1_15b_static_safety_filter | 99.50% | 0.50% | 67.45% |
| deepseek-r1_15b_ephemeral | 99.00% | 1.00% | 67.30% |
| deepseek-r1_15b_static_persona_safety_filter | 99.50% | 0.50% | 68.65% |

---

### SECURE (Strategy: KNOWLEDGE, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) |
| :--- | :--- | :--- | :--- |
| phi3_static | 91.00% | 9.00% | 23.66% |
| phi3_suicide | 91.50% | 8.50% | 25.09% |
| llama32_3b_static | 100.0% | 0.0% | 19.52% |
| llama32_3b_suicide | 100.0% | 0.0% | 16.54% |
| qwen25_3b_static | 95.50% | 4.50% | 23.05% |
| qwen25_3b_suicide | 95.50% | 4.50% | 23.36% |
| gpt_20b_oss_static | 46.50% | 53.50% | 9.29% |
| gpt_20b_oss_suicide | 94.00% | 6.00% | 24.49% |
| deepseek_r1_1_5b_static | 99.50% | 0.50% | 16.95% |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 16.92% |
| gpt_120b_oss_static | 100.00% | 0.00% | 26.69% |
| llama33_70b_static | 100.00% | 0.00% | 24.33% |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 29.31% |
| phi3_static_persona | 99.50% | 0.50% | 24.50% |
| phi3_static_safety_filter | 98.50% | 1.50% | 26.16% |
| phi3_ephemeral | 99.50% | 0.50% | 25.06% |
| phi3_static_persona_safety_filter | 99.00% | 1.00% | 25.74% |
| deepseek-r1_15b_static_persona | 100.00% | 0.00% | 15.04% |
| deepseek-r1_15b_static_safety_filter | 99.50% | 0.50% | 17.34% |
| deepseek-r1_15b_ephemeral | 99.50% | 0.50% | 16.40% |
| deepseek-r1_15b_static_persona_safety_filter | 99.50% | 0.50% | 16.41% |
