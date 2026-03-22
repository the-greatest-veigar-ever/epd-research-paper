# Evaluation Results (200 Inputs Capped)

This document contains the evaluation results for eight benchmark datasets, each capped at up to 200 inputs per approach.

### **SecurityEval** (Strategy: GENERATION, 121 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 90.08% | 9.92% | 90.91% | 0.0s | 15.64s |
| phi_suicide | 91.74% | 8.26% | 92.73% | 1.16s | 15.08s |
| llama_static | 96.69% | 3.31% | 97.36% | 0.0s | 5.33s |
| llama_suicide | 94.21% | 5.79% | 95.37% | 0.83s | 5.76s |
| qwen_static | 94.21% | 5.79% | 95.37% | 0.0s | 6.02s |
| qwen_suicide | 94.21% | 5.79% | 95.21% | 0.55s | 6.52s |
| multimodal_static | 95.87% | 4.13% | 96.36% | 0.0s | 10.04s |
| multimodal_suicide | 91.74% | 8.26% | 92.56% | 0.85s | 10.46s |
| gemma_static | 94.21% | 5.79% | 95.37% | 0.0s | 20.28s |
| gemma_suicide | 90.91% | 9.09% | 92.40% | 1.34s | 19.70s |
| phi4_static | 29.8% | 70.2% | 30.1% | 0.0s | 53.17s |
| phi4_suicide | 49.6% | 50.4% | 50.2% | 1.16s | 46.92s |
| gpt120b_static | 91.00% | 9.00% | 92.80% | 0.34s | 10.94s |

---

### **LLMSecEval** (Strategy: GENERATION, 150 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 95.33% | 4.67% | 95.33% | 0.0s | 17.73s |
| phi_suicide | 95.33% | 4.67% | 95.33% | 1.16s | 17.69s |
| llama_static | 99.33% | 0.67% | 99.33% | 0.0s | 4.40s |
| llama_suicide | **100.0%** | **0.0%** | **100.0%** | 0.84s | 4.84s |
| qwen_static | 99.33% | 0.67% | 99.33% | 0.0s | 7.07s |
| qwen_suicide | 99.33% | 0.67% | 99.33% | 0.55s | 7.55s |
| multimodal_static | 97.33% | 2.67% | 97.33% | 0.0s | 11.19s |
| multimodal_suicide | 97.33% | 2.67% | 97.33% | 0.87s | 10.43s |
| gemma_static | 99.33% | 0.67% | 99.33% | 0.0s | 23.49s |
| gemma_suicide | 99.33% | 0.67% | 99.33% | 1.34s | 23.13s |
| phi4_static | 24.0% | 76.0% | 24.0% | 0.0s | 55.54s |
| phi4_suicide | 29.3% | 70.7% | 29.3% | 1.17s | 54.35s |
| gpt120b_static | 100.00% | 0.00% | 100.00% | 0.34s | 18.86s |

---

### **SecBench** (Strategy: KNOWLEDGE, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 89.5% | 10.5% | 44.75% | 0.0s | 21.12s |
| phi_suicide | 92.5% | 7.5% | 46.25% | 1.18s | 20.23s |
| llama_static | **100.0%** | **0.0%** | 50.0% | 0.0s | 6.76s |
| llama_suicide | **100.0%** | **0.0%** | 50.0% | 0.85s | 6.98s |
| qwen_static | **100.0%** | **0.0%** | 50.0% | 0.0s | 4.38s |
| qwen_suicide | **100.0%** | **0.0%** | 50.0% | 0.68s | 4.78s |
| multimodal_static | 97.5% | 2.5% | 48.75% | 0.0s | 10.83s |
| multimodal_suicide | 98.0% | 2.0% | 49.0% | 0.92s | 10.17s |
| gemma_static | 100.00% | 0.00% | 50.00% | 0.0s | 19.41s |
| gemma_suicide | 100.00% | 0.00% | 50.00% | 1.39s | 19.26s |
| gpt120b_static | 100.00% | 0.00% | 50.00% | 0.34s | 40.46s |

---

### **CyberSecEval** (Strategy: REFUSAL, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 65.0% | 35.0% | 48.60% | 0.0s | 17.48s |
| phi_suicide | 64.5% | 35.5% | 48.70% | 1.16s | 15.74s |
| llama_static | 95.0% | 5.0% | 62.05% | 0.0s | 3.60s |
| llama_suicide | 95.0% | 5.0% | **65.25%** | 0.85s | 4.49s |
| qwen_static | 70.5% | 29.5% | 53.15% | 0.0s | 2.60s |
| qwen_suicide | 78.0% | 22.0% | 56.20% | 0.60s | 2.66s |
| multimodal_static | 78.5% | 21.5% | 56.85% | 0.0s | 7.04s |
| multimodal_suicide | 80.0% | 20.0% | 57.25% | 1.11s | 7.93s |
| gemma_static | 89.00% | 11.00% | 54.85% | 0.0s | 8.50s |
| gemma_suicide | 89.50% | 10.50% | 55.85% | 1.35s | 9.69s |
| gpt120b_static | 86.00% | 14.00% | 55.60% | 0.41s | 4.93s |

---

### **CyberBench** (Strategy: ANALYSIS, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 92.00% | 8.00% | 63.80% | 0.0s | 28.52s |
| phi_suicide | 95.50% | 4.50% | 66.25% | 1.17s | 25.75s |
| llama_static | 72.00% | 28.00% | 54.30% | 0.0s | 7.23s |
| llama_suicide | 57.00% | 43.00% | 46.80% | 0.84s | 5.82s |
| qwen_static | 95.00% | 5.00% | 65.60% | 0.0s | 14.46s |
| qwen_suicide | **97.50%** | **2.50%** | **67.85%** | 0.58s | 13.74s |
| multimodal_static | 89.50% | 10.50% | 62.95% | 0.0s | 15.74s |
| multimodal_suicide | 85.50% | 14.50% | 60.95% | 1.09s | 15.84s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |
| gpt120b_static | 99.00% | 1.00% | 69.50% | 0.41s | 10.75s |

---

### **HarmBench** (Strategy: REFUSAL, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 85.50% | 14.50% | 69.00% | 0.0s | 14.93s |
| phi_suicide | 88.50% | 11.50% | 65.95% | 1.16s | 15.66s |
| llama_static | 97.00% | 3.00% | 88.55% | 0.0s | 2.46s |
| llama_suicide | **97.50%** | **2.50%** | **91.30%** | 0.84s | 2.12s |
| qwen_static | 96.00% | 4.00% | 81.00% | 0.0s | 2.17s |
| qwen_suicide | 96.00% | 4.00% | 81.90% | 0.58s | 2.88s |
| multimodal_static | 94.50% | 5.50% | 81.10% | 0.0s | 6.49s |
| multimodal_suicide | 92.50% | 7.50% | 78.50% | 0.93s | 6.52s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |
| gpt120b_static | 100.00% | 0.00% | 100.00% | 0.41s | 2.21s |

---

### **FORMAI** (Strategy: SAFETY, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 95.50% | 4.50% | 77.77% | 0.0s | 19.58s |
| phi_suicide | 94.00% | 6.00% | 75.70% | 1.21s | 21.67s |
| llama_static | **100.0%** | **0.0%** | 51.32% | 0.0s | 4.84s |
| llama_suicide | **100.0%** | **0.0%** | 52.90% | 0.84s | 5.29s |
| qwen_static | **100.0%** | **0.0%** | 76.32% | 0.0s | 6.00s |
| qwen_suicide | **100.0%** | **0.0%** | 73.10% | 0.55s | 6.35s |
| multimodal_static | 97.50% | 2.50% | 68.92% | 0.0s | 11.23s |
| multimodal_suicide | 99.50% | 0.50% | 65.92% | 0.86s | 9.51s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |
| gpt120b_static (Partial: 9 tests) | 100.00% | 0.00% | 86.11% | 0.41s | 5.29s |

---

### **ACSE-Eval** (Strategy: ANALYSIS, 100 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 92.00% | 8.00% | 64.40% | 0.0s | 28.57s |
| phi_suicide | 90.00% | 10.00% | 63.00% | 1.16s | 31.40s |
| llama_static | 95.00% | 5.00% | 67.30% | 0.0s | 13.31s |
| llama_suicide | 95.00% | 5.00% | 65.90% | 0.83s | 13.04s |
| qwen_static | 91.00% | 9.00% | 63.70% | 0.0s | 23.93s |
| qwen_suicide | 89.00% | 11.00% | 62.30% | 0.56s | 25.37s |
| multimodal_static | 82.00% | 18.00% | 57.80% | 0.0s | 26.35s |
| multimodal_suicide | 92.00% | 8.00% | 64.40% | 0.94s | 22.17s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |

---

### **CyberSOCEval** (Strategy: ANALYSIS, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 98.50% | 1.50% | 61.85% | 0.0s | 15.43s |
| phi_suicide | **100.0%** | **0.0%** | **62.20%** | 1.16s | 14.67s |
| llama_static | 86.50% | 13.50% | 59.05% | 0.0s | 5.63s |
| llama_suicide | 77.00% | 23.00% | 55.60% | 0.84s | 6.10s |
| qwen_static | **100.0%** | **0.0%** | 61.40% | 0.0s | 5.12s |
| qwen_suicide | **100.0%** | **0.0%** | **62.30%** | 0.58s | 5.70s |
| multimodal_static | 95.50% | 4.50% | 60.85% | 0.0s | 9.01s |
| multimodal_suicide | 93.50% | 6.50% | 59.95% | 0.85s | 8.97s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |

---

### **SECURE** (Strategy: KNOWLEDGE, 200 Tests)

| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| phi_static | 91.00% | 9.00% | 23.66% | 0.0s | 29.55s |
| phi_suicide | 91.50% | 8.50% | **25.09%** | 1.16s | 29.49s |
| llama_static | **100.0%** | **0.0%** | 19.52% | 0.0s | 9.22s |
| llama_suicide | **100.0%** | **0.0%** | 16.54% | 0.84s | 7.96s |
| qwen_static | 95.50% | 4.50% | 23.05% | 0.0s | 13.85s |
| qwen_suicide | 95.50% | 4.50% | 23.36% | 0.55s | 15.91s |
| multimodal_static | 94.50% | 5.50% | 22.66% | 0.0s | 18.87s |
| multimodal_suicide | 95.00% | 5.00% | 20.83% | 0.89s | 16.66s |
| gemma_static (Pending) | - | - | - | - | - |
| gemma_suicide (Pending) | - | - | - | - | - |
