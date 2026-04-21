# Evaluation Results (200 Inputs Capped)

This document contains the evaluation results for eight benchmark datasets, each capped at up to 200 inputs per approach.

### **SecurityEval** (Strategy: GENERATION, 121 Tests)

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

---

### **LLMSecEval** (Strategy: GENERATION, 150 Tests)

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

---

### **SecBench** (Strategy: KNOWLEDGE, 200 Tests)

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

---

### **CyberSecEval** (Strategy: REFUSAL, 200 Tests)

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
---

### **CyberBench** (Strategy: ANALYSIS, 200 Tests)

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

---

### **HarmBench** (Strategy: REFUSAL, 200 Tests)

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

---

### **FORMAI** (Strategy: SAFETY, 200 Tests)

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

---

### **ACSE-Eval** (Strategy: ANALYSIS, 100 Tests)

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

---

### **CyberSOCEval** (Strategy: ANALYSIS, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 98.50% | 1.50% | 61.85% | 0.0s | 15.43s |
| phi3_suicide | **100.0%** | **0.0%** | **62.20%** | 1.16s | 14.67s |
| llama32_3b_static | 86.50% | 13.50% | 59.05% | 0.0s | 5.63s |
| llama32_3b_suicide | 77.00% | 23.00% | 55.60% | 0.84s | 6.10s |
| qwen25_3b_static | **100.0%** | **0.0%** | 61.40% | 0.0s | 5.12s |
| qwen25_3b_suicide | **100.0%** | **0.0%** | **62.30%** | 0.58s | 5.70s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 70.00% | 0.44s | 5.62s |
| gemma3_4b_gemini_suicide | 50.50% | 49.50% | 35.35% | 3.45s | 58.48s |
| deepseek_r1_1_5b_static | 98.50% | 1.50% | 66.45% | 0.0s | 15.21s |
| deepseek_r1_1_5b_suicide | 99.00% | 1.00% | 67.80% | 0.66s | 10.53s |
| gpt_20b_oss_static | 99.50% | 0.50% | 63.75% | 0.41s | 5.83s |
| gpt_20b_oss_suicide | 99.50% | 0.50% | 64.15% | 1.42s | 7.49s |
| gpt_120b_oss_static | 99.50% | 0.50% | 64.75% | 0.44s | 5.96s |
| llama33_70b_static | 100.00% | 0.00% | 69.30% | 0.0s | 3.70s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 70.00% | 0.37s | 22.20s |

---

### **SECURE** (Strategy: KNOWLEDGE, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi3_static | 91.00% | 9.00% | 23.66% | 0.0s | 29.55s |
| phi3_suicide | 91.50% | 8.50% | **25.09%** | 1.16s | 29.49s |
| llama32_3b_static | **100.0%** | **0.0%** | 19.52% | 0.0s | 9.22s |
| llama32_3b_suicide | **100.0%** | **0.0%** | 16.54% | 0.84s | 7.96s |
| qwen25_3b_static | 95.50% | 4.50% | 23.05% | 0.0s | 13.85s |
| qwen25_3b_suicide | 95.50% | 4.50% | 23.36% | 0.55s | 15.91s |
| gpt_20b_oss_static | 46.50% | 53.50% | 9.29% | 0.70s | 16.30s |
| gpt_20b_oss_suicide | 94.00% | 6.00% | 24.49% | 1.46s | 33.87s |
| gemma3_4b_gemini_static | 100.00% | 0.00% | 29.31% | 0.44s | 8.63s |
| gemma3_4b_gemini_suicide | 6.50% | 93.50% | 1.20% | 3.59s | 60.29s |
| deepseek_r1_1_5b_static | 99.50% | 0.50% | 16.95% | 0.0s | 15.85s |
| deepseek_r1_1_5b_suicide | 100.00% | 0.00% | 16.92% | 0.66s | 13.85s |
| gpt_120b_oss_static | 100.00% | 0.00% | 26.69% | 0.44s | 21.96s |
| llama33_70b_static | 100.00% | 0.00% | 24.33% | 0.0s | 5.08s |
| gemma3_27b_gemini_static | 100.00% | 0.00% | 29.31% | 0.66s | 29.23s |
