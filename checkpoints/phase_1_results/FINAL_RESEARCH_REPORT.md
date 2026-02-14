# Ghost Agents: Final Research Report & Analysis

**Date:** 2026-02-14  
**Experiment ID:** `20260214_115807`  
**Configuration:** Full Pipeline (Ephemerality + Polymorphism) vs Baseline vs Monolithic LLM

## Executive Summary
This report analyzes the performance of the "Ghost Agents" architecture (Squad C) compared to a static baseline and a monolithic LLM (`llama3`). The results demonstrate a **significant trade-off between security (evasion) and reliability (correctness)**.

While the Proposed architecture successfully reduced semantic similarity (evading signature detection), the effectiveness of Polymorphism (persona-based prompt mutation) came at the cost of functional accuracy, with Tool Correctness dropping to **10.19%** compared to the Baseline (89.00%) and Monolithic LLM (97.50%).

## Key Metrics Comparison

| Metric | Baseline (Static SLM) | Proposed (Ghost Swarm) | LLM (llama3 8B) | Delta (Prop vs LLM) |
| :--- | :--- | :--- | :--- | :--- |
| **Tool Correctness** | **89.00%** | 10.19% | **97.50%** | -87.31% |
| **Avg Latency** | 1.80s | 1.92s | 2.28s | **-0.36s (Faster)** |
| **Semantic Similarity** | 0.681 | **0.584** | N/A | -0.097 (Better Evasion) |
| **Levenshtein Distance** | 99.2 | **133.5** | N/A | +34.3 (Higher Variance) |

> **Note:** Lower Semantic Similarity and higher Levenshtein Distance indicate better resistance to static signature detection.

## Detailed Analysis

### 1. The Cost of Polymorphism
The most striking finding is the degradation of **Tool Correctness** in the Proposed architecture.
- **Baseline**: Used static, direct prompts. Achieved 89% accuracy with SLMs (`llama3.2:3b`, `phi`, `gemma2:2b`).
- **Proposed**: Used "Polymorphic" prompts (e.g., *a Military General issuing a protocol*).
- **Result**: The small language models (SLMs) struggled to separate the "Persona" noise from the technical "Action" instructions, leading to a correctness rate of **10.19%**.
- **Implication**: While Polymorphism successfully obfuscates intent (Semantic Similarity drop from 0.68 to 0.58), current SLMs are not robust enough to handle the added prompt complexity without losing functional accuracy.

### 2. Monolithic Superiority in Accuracy
The Monolithic LLM (`llama3` 8B) achieved **97.50% Tool Correctness**, significantly outperforming the Ghost Swarm.
- This confirms that for pure execution reliability, a larger, static model is superior.
- However, the Monolithic model lacks the **Ephemerality** and **Evasion** properties of the Ghost architecture, making it more vulnerable to detection and persistence tracking.

### 3. Ephemerality Overhead is Negligible
The latency comparison shows that the overhead of instantiating ephemeral agents is minimal.
- **Baseline Latency**: 1.80s
- **Proposed Latency**: 1.92s
- **Overhead**: ~0.12s per execution.
- **Conclusion**: The "Ghost" mechanism (wiping memory, spinning up new container/agent) introduces <7% latency overhead, validating the feasibility of Ephemerality for real-time remediation.

## Model Performance within Swarm
In the previous partial run (Feb 11), `llama3.2:3b` was the only SLM capable of following instructions consistently (33% swarm accuracy, with `phi` and `gemma` failing).
In the final run with full Polymorphism enabled, even `llama3.2:3b`'s performance degraded, pulling the swarm average down to 10%.

## Conclusion & Recommendations

The "Ghost Agents" architecture succeeds in its goal of creating a **moving target** defense:
✅ **Significantly lower semantic footprint** (harder to detect).
✅ **Negligible latency cost** for ephemerality.

However, it fails on **functional reliability** with current SLMs:
❌ **10% Accuracy** is unacceptable for autonomous remediation.

**Future Work / Recommendations:**
1.  **Reduce Polymorphism Intensity**: Tone down the "Persona" strength to balance evasion vs. instruction following.
2.  **Fine-tune SLMs**: Fine-tune the small models (`phi`, `gemma`) specifically on the polymorphic prompt format.
3.  **Hybrid Approach**: Use Monolithic LLM (`llama3`) for complex logic and Ghost Swarm for simple, high-volume tasks.
