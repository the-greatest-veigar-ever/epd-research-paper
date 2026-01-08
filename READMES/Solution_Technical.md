# Technical Architecture: Ephemeral Polymorphic Defense (EPD)

**EPD** is a Multi-Agent System (MAS) architecture designed to mitigate **Static Cognitive Attack Surfaces** and **Excessive Agency** risks in cloud security automation. It employs **Moving Target Defense (MTD)** principles at the cognitive (LLM) layer.

## Core Architectural Concepts

### 1. Ephemerality (Just-in-Time Instantiation)
Unlike persistent daemon-based agents, EPD agents are stateless and event-driven.
*   **Lifecycle**: `Spawn -> Contextualize -> Execute -> Terminate`.
*   **Benefit**: This eliminates **Context Accumulation Attacks**, where adversaries fill an agent's context window with manipulative tokens to induce drift or "hypnotize" the model over time.
*   **Mechanism**: The `GhostAgent` class initializes only upon a verified alert trigger and calls `cleanup()` (self-destruct) immediately post-execution.

### 2. Cognitive Polymorphism (Model & Prompt Rotation)
To prevent **Model Inversion** and **Prompt Injection** attacks, the system dynamically alters the cognitive substrate of the defense agents.
*   **Model Rotation**: The system randomly selects the inference engine for each task from a pool of heterogeneous LLMs (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro). An adversarial prompt optimized for GPT-4's token space will likely fail against Claude.
*   **Prompt Mutation**: The system prompt is not a static string. It is generated procedurally using a mutation engine that alters syntax, tone, and directive structure while preserving semantic intent (e.g., commanding "Terminate Instance" vs. "Execute protocol Alpha-1 to halt CPU cycles").

### 3. Distributed Consensus (The "Brain" Layer)
Before any remediation action is executed, it passes through a validation layer inspired by the **SentinelNet** and **VeriGuard** frameworks.
*   **Consensus Check**: Alerts are analyzed by a simulated ensemble. If confidence is below threshold $\tau$ (e.g., 0.8), the action is aborted.
*   **Policy Verification**: Proposed actions (e.g., IAM Policy Detach) are checked against a deterministic safety constraint logic (VeriGuard simulation) to prevent self-denial of service (e.g., deleting root credentials).

## System Components

*   **`monitor.py` (The Watchers)**: Lightweight, non-LLM scripts that poll CloudTrail/GuardDuty logs. They act as the "nervous system," triggering the brain only when necessary.
*   **`intelligence.py` (The Brain)**: Implements the Consensus and VeriGuard logic. It acts as the orchestration layer, holding the state of the OODA loop (Observe-Orient-Decide-Act).
*   **`epd_core.py` (The Ghosts)**: The execution layer. Contains the `GhostAgentFactory` which handles the polymorphism logic (random seed selection for Model/Prompt).
