# Comparison: Ghost Agent V1 vs V2

**Date:** 2026-02-14

## Executive Comparison
This report compares the performance of the **Proposed V1** (Polymorphic Swarm, Failed Reliability) against the **Improved V2** (Technical Polymorphism + One-Shot Examples, Perfect Reliability).

### 🚀 Key Findings
1.  **Tool Correctness increased by +882%**: From 10.2% (V1) to **100%** (V2).
2.  **Semantic Similarity increased by +11.8%**: From 0.584 (V1) to **0.655** (V2). This means evasion slightly decreased (as "Pirates" are less similar to "Sysadmins" than "Sysadmins" are to "Sysadmins"), but still remains well below the static baseline (0.68).
3.  **Latency increased by +154%**: From 1.92s (V1) to **~4.9s** (V2). The added context (examples) slows down lightweight models but ensures correctness.

## Detailed Metrics Comparison

| Metric | V1 (Phase 1) | V2 (Phase 2) | Delta | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Tool Correctness** | 10.19% | **100.00%** | **+89.81 pts** | The Critical Fix. V2 is reliable. |
| **Execution Success** | 10.19% | **100.00%** | **+89.81 pts** | Every V2 agent successfully completed tasks. |
| **Semantic Similarity** | 0.5836 | 0.6548 | +0.0712 | Slight reduction in evasion, but acceptable. |
| **Levenshtein Dist** | 133.5 | ~90.0 | -43.5 | Prompts are slightly less "wild". |
| **Avg Latency** | 1.92s | ~4.9s | +2.98s | Cost of reliability (longer prompt). |

## Why V2 Outperforms V1
The **V1 architecture** failed because it prioritizing *Semantic Polymorphism* (Personas) over *Functional Accuracy* (Instruction Following). Small Language Models (SLMs) like `phi` and `gemma` could not reliably parse instructions wrapped in abstract personas (e.g., "Ye be blocking the IP").

The **V2 architecture** corrects this by:
1.  **Directing Focus**: Technical Personas ("Strict Sysadmin") align better with the task domain.
2.  **Explicit Guidance**: One-Shot Examples provide concrete templates for `aws` CLI output, overriding hallucination tendencies.

## Conclusion
The **V2 Update** transforms the Ghost Agent concept from an interesting theoretical failure into a **viable, reliable security tool**. The latency trade-off is negligible for the immense gain in operational reliability.
