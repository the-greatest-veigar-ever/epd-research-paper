# VeriGuard: Enhancing LLM Agent Safety via Verified Code Generation

**Source**: Under review as a conference paper at ICLR 2026

## Summary
**VeriGuard** is a framework that provides formal mathematical guarantees for the safety of LLM-based agents. It shifts from "reactive" safety (filtering bad outputs) to "proactive" safety by generating code that is mathematically proven to be correct before it runs.

## Key Problem
Autonomous agents are increasingly powerful but unpredictable. They can:
*   Delete critical files.
*   Leak sensitive data.
*   Execute destructive commands.
*   **Current Guardrails** (input filters, sandboxes) are insufficient because they treat the agent as a "black box" and can be bypassed by novel jailbreaks.

## Proposed Solution: Correct-by-Construction
VeriGuard introduces a dual-stage safety pipeline:

1.  **Offline Policy Generation (The rigorous part)**:
    *   The system generates a **Behavioral Policy** (code) AND **Formal Constraints** (mathematical rules).
    *   It uses a **Formal Verifier** (Nagini/Viper) to prove that the code satisfies the constraints.
    *   If verification fails, the system auto-corrects the code until it passes.

2.  **Online Policy Enforcement (The fast part)**:
    *   At runtime, the agent's actions are intercepted.
    *   The action is checked against the pre-verified policy.
    *   Unsafe actions are blocked immediately, with mathematical certainty.

## Key Results
*   **Provable Safety**: Unlike statistical defenses (which might catch 99% of attacks), VeriGuard offers formal verification guarantees for the policies it generates.
*   **Iterative Refinement**: The framework successfully auto-repairs unsafe code during the generation phase using verifier feedback.
*   **Trustworthiness**: Enables deployment of agents in high-stakes environments (healthcare, finance) where "mostly safe" is not good enough.
