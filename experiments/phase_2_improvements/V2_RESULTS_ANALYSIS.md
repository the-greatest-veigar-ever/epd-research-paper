# Ghost Agent V2: Analysis of Improvements

**Experiment ID:** Phase 2 - Agent V2 Evaluation
**Date:** 2026-02-14

## Summary
We implemented and evaluated **Ghost Agent V2**, incorporating two key improvements:
1.  **Technical Polymorphism**: Replaced abstract personas (e.g., "Pirate") with technical roles (e.g., "Strict Sysadmin").
2.  **One-Shot Prompting**: Injected specific AWS CLI examples into the system prompt.

## Results Comparison

| Metric | Phase 1 (Proposed) | Phase 2 (Agent V2) | Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tool Correctness** | 10.19% | **95.00%** | >80% | ✅ **PASSED** |
| **Semantic Similarity** | 0.5836 | **0.5841** | <0.70 | ✅ **MAINTAINED** |
| **Avg Latency** | 1.92s | ~4.89s | <5s | ⚠️ Acceptable |

## Key Findings

### 1. Accuracy Restored
The introduction of **One-Shot Examples** was the decisive factor.
- SLMs (`phi`, `gemma2:2b`) previously struggled to generate valid `aws` commands from abstract instructions.
- With examples, they consistently mapped even out-of-distribution intent (e.g., "Disruptive Attack") to valid remediation tools (`aws ec2 create-network-acl-entry`), effectively failing safe.

### 2. Evasion Preserved
Despite moving to "Technical Personas", the Semantic Similarity score remained low (0.58).
- This indicates that the variations in phrasing (e.g., "Execute Protocol 7-Alpha-12") are sufficient to evade static signature detection (which relies on high similarity to known malicious prompts).

## Conclusion
The **Ghost Agent V2** architecture successfully resolves the reliability issues identified in Phase 1 without compromising security.

## Next Steps
- Integrate `AgentV2` into the main `full_pipeline_evaluation.py`.
- Proceed to **Item 2: Hybrid Architecture** (Routing) if further robustness is required for complex logic.
