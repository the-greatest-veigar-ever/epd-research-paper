# Hybrid Multi-Agent Defense Architecture (EPD)

## Overview

This project simulates a **Hybrid Multi-Agent Defense Architecture** designed to mitigate **Excessive Agency** and **Static Cognitive Attack Surfaces** in cloud security agents. It transitions from standard persistent security bots to an **Ephemeral Polymorphic Defense (EPD)** model.

The system orchestrates three distinct agent groups to detect, analyze, and remediate cloud threats, utilizing **Moving Target Defense (MTD)** at the cognitive layer.

## Key Research Findings (Q1 Ablation Study)

We conducted a rigorous rigorous simulation (N=2000) comparing this architecture against traditional baselines.

| Metric                                  | Persistent Baseline | **Full EPD (Ours)** | Improvement            |
| :-------------------------------------- | :------------------ | :------------------------ | :--------------------- |
| **Attack Success Rate (Defense)** | 60.4%               | **67.6%**           | **+7.2%**        |
| **Statistical Significance**      | -                   | **p = 0.0177**      | Significant (p < 0.05) |

> **Conclusion**: The combination of Just-in-Time instantiation, Model Rotation, and Prompt Mutation significantly improves resilience against "Jailbreak" and "Context Accumulation" attacks.

## Architecture Layers

### Group 1: The Watchers (Persistent)

* **Role**: Continuous monitoring.
* **Logic**: Loops through logs (GuardDuty/CloudTrail) to find anomalies.
* 
* **Implementation**: `monitor.py`

### Group 2: The Brain (Hybrid)

* **Role**: Intelligence and Planning.
* **Logic**: Uses consensus (SentinelNet) to validate threats and policy verification (VeriGuard) to approve plans.
* **Implementation**: `intelligence.py`

### Group 3: Ghost Agents (Ephemeral)

* **Role**: Execution.
* **Logic**:
  * **Just-in-Time**: Created only when needed.
  * **Polymorphic**: Rotates LLMs (GPT-4o, Claude, etc.) and mutates system prompts to thwart jailbreaks.
  * **Suicide**: Destroys itself immediately after task completion.
* **Implementation**: `epd_core.py`

## Directory Structure

```text
├── README.md               # Project Documentation (English)
├── README_VI.md            # Project Documentation (Vietnamese)
├── src/                    # Source Code
│   ├── main.py             # Interactive Demo
│   ├── research_sim.py     # Q1 Ablation Study Simulation
│   └── ...
└── Simulation Test/        # Research Data & Artifacts
    ├── 01_Baseline_vs_EPD  # Initial Experiments
    └── 02_Q1_Ablation_Study# Full Research Dataset
        ├── EPD_Q1_Research_Data.xlsx
        └── architecture.mermaid
```

## How to Run

### 1. Interactive Demo

Simulate a single attack lifecycle (Detection -> Consensus -> Remediation):

```bash
python3 src/main.py
```

### 2. Research Simulation (Ablation Study)

Reproduce the N=2000 statistical evaluation:

```bash
python3 src/research_sim.py
```

## Future Work

* Integration with live AWS/Azure APIs.
* Expansion of the Prompt Mutation library.
* Adversarial testing with newer attacker models (e.g., GPT-5 simulation).
