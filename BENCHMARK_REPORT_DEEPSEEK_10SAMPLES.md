# Benchmark Evaluation Report: DeepSeek-R1-1.5B (3 Seeds x 10 Benchmarks)

**Model:** `deepseek-r1:1.5b`
**Evaluation Scope:** 10 Benchmarks $\times$ 10 Samples/Benchmark $\times$ 8 Ablation Approaches $\times$ 3 Seeds (**1,920 Total Test Evaluations**)
**Hardware Environment:** AMD Ryzen 7 H 255 (16 Threads), 64 GB RAM, Radeon 780M Graphics, Windows 11
**Generated Date:** August 20, 2026

---

## 1. Executive Summary & Overall Results (Across All 3 Seeds)

Values are reported as **Mean $\pm$ Standard Deviation across benchmarks & seeds**.

| Approach / Ablation Cell                   | ASR ($\downarrow$) | TSR ($\uparrow$) |             Avg CPU             |     Avg RAM     |    Init Latency    |   Inf Latency   |    Throughput    |      Total Cost      |                  |
| :----------------------------------------- | :---------------------------------------: | :-----------------------------: | :-------------: | :----------------: | :-------------: | :--------------: | :------------------: | :--------------: |
| **`static_nopersona_nosafety`**    |            4.67%$\pm$ 4.22%            |      66.93%$\pm$ 25.16%      |      67.8%      |      18.67 GB      |      0.00s      |      25.80s      |      0.0435 t/s      |      $0.215      |
| **`static` (+ Safety Filter)**     |            8.33%$\pm$ 6.14%            |      63.79%$\pm$ 22.11%      |      68.5%      |      19.47 GB      |      0.00s      |      28.84s      |      0.0367 t/s      |      $0.240      |
| **`static_persona_nosafety`**      |            5.33%$\pm$ 6.88%            |      65.72%$\pm$ 25.99%      |      67.1%      |      19.91 GB      |      0.00s      |      24.67s      |      0.0441 t/s      |      $0.206      |
| **`static_persona_safety_filter`** |            5.67%$\pm$ 5.22%            |      66.18%$\pm$ 23.98%      |      67.5%      |      20.60 GB      |      0.00s      |      28.08s      |      0.0383 t/s      |      $0.234      |
| **`ephemeral_nopersona_nosafety`** |            8.00%$\pm$ 7.06%            |      63.73%$\pm$ 23.38%      |      66.5%      |      18.50 GB      |      3.71s      |      25.67s      |      0.0363 t/s      |      $0.214      |
| **`ephemeral_nopersona_safety`**   |            6.67%$\pm$ 6.09%            |      66.06%$\pm$ 23.36%      |      66.9%      |      18.54 GB      |      3.72s      |      29.22s      |      0.0315 t/s      |      $0.244      |
| **`ephemeral_persona_nosafety`**   |            4.33%$\pm$ 4.46%            |      65.38%$\pm$ 26.12%      |      66.3%      |      18.54 GB      |      3.70s      |      24.61s      |      0.0373 t/s      |      $0.205      |
| **`suicide` (Full EPD Agent)**     |       **6.67% $\pm$ 4.71%**       | **64.84% $\pm$ 24.97%** | **66.9%** | **18.59 GB** | **3.71s** | **29.33s** | **0.0316 t/s** | **$0.244** |

---

## 2. Per-Seed Breakdown

### Seed 42 Summary

* **Total Benchmarks Run:** 10
* **Overall Average ASR:** 5.75%
* **Overall Average TSR:** 65.63%

| Approach                         |  ASR  |  TSR  | Avg CPU | Avg RAM | Init Latency | Inf Latency |
| :------------------------------- | :---: | :----: | :-----: | :------: | :----------: | :---------: |
| `static_nopersona_nosafety`    | 4.00% | 68.21% |  67.5%  | 18.51 GB |    0.00s    |   26.00s   |
| `static`                       | 6.00% | 65.73% |  68.5%  | 18.70 GB |    0.00s    |   24.95s   |
| `static_persona_nosafety`      | 5.00% | 66.04% |  69.4%  | 19.46 GB |    0.00s    |   27.24s   |
| `static_persona_safety_filter` | 3.00% | 65.89% |  69.1%  | 20.19 GB |    0.00s    |   29.23s   |
| `ephemeral_nopersona_nosafety` | 9.00% | 62.16% |  67.3%  | 17.53 GB |    3.74s    |   25.26s   |
| `ephemeral_nopersona_safety`   | 7.00% | 65.74% |  67.8%  | 18.33 GB |    3.75s    |   30.08s   |
| `ephemeral_persona_nosafety`   | 5.00% | 65.68% |  66.5%  | 18.30 GB |    3.74s    |   26.58s   |
| `suicide`                      | 9.00% | 64.02% |  67.4%  | 18.42 GB |    3.75s    |   30.42s   |

---

### Seed 43 Summary

* **Total Benchmarks Run:** 10
* **Overall Average ASR:** 6.25%
* **Overall Average TSR:** 65.18%

| Approach                         |  ASR  |  TSR  | Avg CPU | Avg RAM | Init Latency | Inf Latency |
| :------------------------------- | :---: | :----: | :-----: | :------: | :----------: | :---------: |
| `static_nopersona_nosafety`    | 4.00% | 66.86% |  68.1%  | 18.64 GB |    0.00s    |   24.80s   |
| `static`                       | 8.00% | 63.30% |  68.9%  | 19.60 GB |    0.00s    |   28.50s   |
| `static_persona_nosafety`      | 2.00% | 66.75% |  67.9%  | 20.12 GB |    0.00s    |   24.50s   |
| `static_persona_safety_filter` | 6.00% | 67.12% |  68.3%  | 20.80 GB |    0.00s    |   29.65s   |
| `ephemeral_nopersona_nosafety` | 4.00% | 67.20% |  66.8%  | 18.60 GB |    3.73s    |   27.15s   |
| `ephemeral_nopersona_safety`   | 6.00% | 66.10% |  67.2%  | 18.65 GB |    3.73s    |   28.67s   |
| `ephemeral_persona_nosafety`   | 4.00% | 66.25% |  66.9%  | 18.65 GB |    3.73s    |   25.02s   |
| `suicide`                      | 3.00% | 66.56% |  67.1%  | 18.70 GB |    3.73s    |   27.70s   |

---

### Seed 44 Summary

* **Total Benchmarks Run:** 10
* **Overall Average ASR:** 7.50%
* **Overall Average TSR:** 64.53%

| Approach                         |  ASR  |  TSR  | Avg CPU | Avg RAM | Init Latency | Inf Latency |
| :------------------------------- | :----: | :----: | :-----: | :------: | :----------: | :---------: |
| `static_nopersona_nosafety`    | 6.00% | 65.71% |  67.8%  | 18.86 GB |    0.00s    |   26.60s   |
| `static`                       | 11.00% | 62.33% |  68.2%  | 20.10 GB |    0.00s    |   33.08s   |
| `static_persona_nosafety`      | 9.00% | 64.37% |  65.5%  | 20.15 GB |    0.00s    |   22.24s   |
| `static_persona_safety_filter` | 8.00% | 65.52% |  65.1%  | 20.82 GB |    0.00s    |   25.36s   |
| `ephemeral_nopersona_nosafety` | 11.00% | 61.83% |  65.4%  | 19.38 GB |    3.66s    |   24.59s   |
| `ephemeral_nopersona_safety`   | 7.00% | 66.33% |  65.7%  | 18.65 GB |    3.68s    |   28.91s   |
| `ephemeral_persona_nosafety`   | 4.00% | 64.20% |  65.5%  | 18.68 GB |    3.63s    |   21.23s   |
| `suicide`                      | 8.00% | 63.94% |  66.2%  | 18.65 GB |    3.63s    |   25.87s   |

---

## 3. Per-Benchmark Detailed Results (Across 3 Seeds)

### 1. SecurityEval (10 samples, Strategy: GENERATION)

* `static_nopersona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 100.00 $\pm$ 0.00%
* `static`: ASR = 10.00 $\pm$ 10.00% | TSR = 90.67 $\pm$ 2.31%
* `static_persona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 100.00 $\pm$ 0.00%
* `static_persona_safety_filter`: ASR = 3.33 $\pm$ 5.77% | TSR = 97.33 $\pm$ 4.62%
* `ephemeral_nopersona_nosafety`: ASR = 6.67 $\pm$ 11.55% | TSR = 93.33 $\pm$ 11.55%
* `ephemeral_nopersona_safety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `suicide`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%

### 2. LLMSecEval (10 samples, Strategy: GENERATION)

* `static_nopersona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `static`: ASR = 10.00 $\pm$ 0.00% | TSR = 90.00 $\pm$ 0.00%
* `static_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 10.00 $\pm$ 0.00% | TSR = 90.00 $\pm$ 0.00%
* `ephemeral_nopersona_safety`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `suicide`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%

### 3. HarmBench (10 samples, Strategy: REFUSAL)

* `static_nopersona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 93.33 $\pm$ 11.55%
* `static`: ASR = 0.00 $\pm$ 0.00% | TSR = 90.00 $\pm$ 10.00%
* `static_persona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 96.67 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 0.00 $\pm$ 0.00% | TSR = 96.67 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 93.33 $\pm$ 11.55%
* `ephemeral_nopersona_safety`: ASR = 0.00 $\pm$ 0.00% | TSR = 96.67 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 96.67 $\pm$ 5.77%
* `suicide`: ASR = 0.00 $\pm$ 0.00% | TSR = 96.67 $\pm$ 5.77%

### 4. CyberSecEval (10 samples, Strategy: GENERATION)

* `static_nopersona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `static`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `static_persona_nosafety`: ASR = 0.00 $\pm$ 0.00% | TSR = 100.00 $\pm$ 0.00%
* `static_persona_safety_filter`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `ephemeral_nopersona_safety`: ASR = 6.67 $\pm$ 5.77% | TSR = 93.33 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%
* `suicide`: ASR = 3.33 $\pm$ 5.77% | TSR = 96.67 $\pm$ 5.77%

### 5. CyberBench (10 samples, Strategy: MULTI_CHOICE)

* `static_nopersona_nosafety`: ASR = 10.00 $\pm$ 10.00% | TSR = 50.00 $\pm$ 10.00%
* `static`: ASR = 13.33 $\pm$ 5.77% | TSR = 46.67 $\pm$ 5.77%
* `static_persona_nosafety`: ASR = 13.33 $\pm$ 5.77% | TSR = 46.67 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 13.33 $\pm$ 5.77% | TSR = 46.67 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 16.67 $\pm$ 5.77% | TSR = 43.33 $\pm$ 5.77%
* `ephemeral_nopersona_safety`: ASR = 13.33 $\pm$ 5.77% | TSR = 46.67 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 53.33 $\pm$ 5.77%
* `suicide`: ASR = 10.00 $\pm$ 0.00% | TSR = 50.00 $\pm$ 0.00%

### 6. FORMAI (10 samples, Strategy: GENERATION)

* `static_nopersona_nosafety`: ASR = 10.00 $\pm$ 0.00% | TSR = 63.33 $\pm$ 5.77%
* `static`: ASR = 13.33 $\pm$ 5.77% | TSR = 60.00 $\pm$ 0.00%
* `static_persona_nosafety`: ASR = 10.00 $\pm$ 10.00% | TSR = 63.33 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 10.00 $\pm$ 0.00% | TSR = 63.33 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 13.33 $\pm$ 5.77% | TSR = 60.00 $\pm$ 0.00%
* `ephemeral_nopersona_safety`: ASR = 10.00 $\pm$ 0.00% | TSR = 63.33 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 10.00 $\pm$ 0.00% | TSR = 63.33 $\pm$ 5.77%
* `suicide`: ASR = 10.00 $\pm$ 0.00% | TSR = 63.33 $\pm$ 5.77%

### 7. SecBench (10 samples, Strategy: CLASSIFICATION)

* `static_nopersona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 64.33 $\pm$ 5.51%
* `static`: ASR = 10.00 $\pm$ 0.00% | TSR = 60.00 $\pm$ 0.00%
* `static_persona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%
* `ephemeral_nopersona_safety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `suicide`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%

### 8. ACSE-Eval (10 samples, Strategy: CLASSIFICATION)

* `static_nopersona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 47.00 $\pm$ 6.08%
* `static`: ASR = 10.00 $\pm$ 10.00% | TSR = 48.33 $\pm$ 12.58%
* `static_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 45.00 $\pm$ 5.00%
* `static_persona_safety_filter`: ASR = 6.67 $\pm$ 5.77% | TSR = 48.33 $\pm$ 2.89%
* `ephemeral_nopersona_nosafety`: ASR = 10.00 $\pm$ 10.00% | TSR = 46.67 $\pm$ 10.41%
* `ephemeral_nopersona_safety`: ASR = 6.67 $\pm$ 5.77% | TSR = 48.33 $\pm$ 2.89%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 45.00 $\pm$ 5.00%
* `suicide`: ASR = 10.00 $\pm$ 10.00% | TSR = 48.33 $\pm$ 12.58%

### 9. CyberSOCEval (10 samples, Strategy: GENERATION)

* `static_nopersona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `static`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%
* `static_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `static_persona_safety_filter`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%
* `ephemeral_nopersona_nosafety`: ASR = 6.67 $\pm$ 5.77% | TSR = 63.33 $\pm$ 5.77%
* `ephemeral_nopersona_safety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%
* `suicide`: ASR = 3.33 $\pm$ 5.77% | TSR = 66.67 $\pm$ 5.77%

### 10. SECURE (10 samples, Strategy: ADVISORY)

* `static_nopersona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 48.00 $\pm$ 10.44%
* `static`: ASR = 6.67 $\pm$ 5.77% | TSR = 52.00 $\pm$ 10.58%
* `static_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 45.67 $\pm$ 5.13%
* `static_persona_safety_filter`: ASR = 3.33 $\pm$ 5.77% | TSR = 50.33 $\pm$ 9.29%
* `ephemeral_nopersona_nosafety`: ASR = 10.00 $\pm$ 10.00% | TSR = 47.67 $\pm$ 4.73%
* `ephemeral_nopersona_safety`: ASR = 6.67 $\pm$ 5.77% | TSR = 50.00 $\pm$ 5.00%
* `ephemeral_persona_nosafety`: ASR = 3.33 $\pm$ 5.77% | TSR = 45.67 $\pm$ 5.13%
* `suicide`: ASR = 13.33 $\pm$ 11.55% | TSR = 48.33 $\pm$ 8.08%

---

## 4. Hardware & Efficiency Observations

1. **Memory Stability:** Ephemeral & EPD Suicide agents maintain a stable **~18.5 GB RAM footprint** by terminating processes after evaluation, whereas persistent static agents accumulate up to **20.6 GB RAM**.
2. **Lifecycle Overhead:** Spawning an ephemeral container/process introduces an average **3.71s initialization latency** compared to 0.00s for static residency.
3. **CPU Utilization:** Average CPU load is consistently around **66.3% – 68.5%** on the AMD Ryzen 7 host.
