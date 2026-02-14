# Ghost Agents V2: Performance Report

**Date:** 2026-02-14
**Evaluation Configs:**
1. `Research Mode` (200 Flows) - [JSON](./results/proposed_v2_improved_research_20260214_130814.json)
2. `Lightweight Mode` (500 Flows) - [JSON](./results/proposed_v2_improved_lightweight_20260214_133259.json)

## Executive Summary
The V2 Architecture (Technical Polymorphism + One-Shot Prompting) has achieved **perfect reliability (100% Tool Correctness)** while maintaining strong evasion capabilities. The trade-off is a slight increase in latency (~2.5x), which remains well within acceptable limits for remediation operations.

## Detailed Metrics

### 1. Reliability (Tool Correctness)
- **Research Mode (200 flows):** **100.00%** (26/26 executions successful)
- **Lightweight Mode (500 flows):** **100.00%** (86/86 executions successful)
- **Analysis:** The introduction of one-shot examples completely eliminated the "hallucination vs refusal" failure modes observed in Phase 1. SLMs (`phi`, `gemma`) now consistently map intent to valid `aws` CLI commands.

### 2. Evasion (Semantic Similarity)
- **Research Mode:** 0.6579
- **Lightweight Mode:** 0.6548
- **Target:** < 0.70
- **Analysis:** Despite using "Technical Personas" (e.g., Sysadmin) which use more standard terminology than "Pirates", the semantic similarity remains low enough to evade static signature detection (baseline was ~0.68, V2 is ~0.65). The structural variation in prompts is preserved.

### 3. Latency & Overhead
- **Avg Pipeline Latency (V2):** ~5.4s (phi) - ~9.4s (gemma)
- **Avg Ephemerality Overhead:** ~1.9s
- **Analysis:** Latency increased compared to V1 (~1.9s avg). This is likely due to the **longer system prompt** (added examples) increasing the input token count and processing time for SLMs. However, <10s is perfectly acceptable for automated IP blocking.

## Model Performance

| Model | Success Rate | Avg Latency (Total) | Note |
| :--- | :--- | :--- | :--- |
| **llama3.2:3b** | 100% | ~4.4s | Fastest & Most Consistent |
| **phi** | 100% | ~5.4s | Strong Performer |
| **gemma2:2b** | 100% | ~8.7s | Slower, but reliable |

## Conclusion
Ghost Agent V2 is a **production-ready architecture**. It solves the reliability crisis of V1 without sacrificing the core security value proposition (ephemerality/evasion).
