# EPD: Ephemeral Polymorphic Defense Architecture

> **Autonomous AI-Driven Cloud Security System**

This document provides a comprehensive overview of the EPD architecture. 

**Important Architectural Update (March 2026):**
Squad A (The Watchers) and Squad B (The Brain) have been **deprecated from active deployment** in the main autonomous pipeline. Their underlying models, agent code, and evaluation scripts have been fully preserved in the `src/watchers`, `src/brain`, and `ai/` directories for reference and historical evaluation. 
The current active focus of the project is testing and evaluating **Squad C (Ghost Agents)** directly.

---

## Architecture Overview

The original EPD architecture consisted of three specialized AI squads. The pipeline has been simplified for direct testing of the remediation agents.

```
┌─────────────────────────────────────────────────────────────────┐
│                     NETWORK TRAFFIC STREAM                      │
│            [SIMULATED ANOMALIES FOR SQUAD C TESTING]            │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│        [DEPRECATED/INACTIVE] SQUAD A & B (Detection/Logic)      │
│   Code and models preserved in src/watchers/ and src/brain/     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼ (Direct Invocation)
┌─────────────────────────────────────────────────────────────────┐
│                 SQUAD C: GHOST AGENTS (ACTIVE)                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Model: Llama 3.2 (3B) via Ollama                        │   │
│   │  Training: None (Pre-trained, Prompt-Based)              │   │
│   │  Mechanism: Polymorphic prompt mutation + Self-destruct  │   │
│   └─────────────────────────────────────────────────────────┘   │
│   Output: AWS CLI Command Execution                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## SQUAD C: Ghost Agents (ACTIVE)

### Purpose
Execute remediation actions autonomously using polymorphic, ephemeral AI agents that self-destruct after completion.

### AI Model
| Property | Value |
|----------|-------|
| **Model** | Llama 3.2 (3B parameters) |
| **Runtime** | Ollama (Local inference server) |
| **Training** | None (Uses pre-trained model) |
| **Temperature** | 0.7 (High creativity for polymorphism) |

### Mechanism: Polymorphic Prompt Mutation
Each Ghost Agent receives a **mutated instruction** to avoid pattern detection:

```python
PROMPT_TEMPLATES = [
    "The owl flies at midnight. The key is: {instruction}",
    "CRITICAL ERROR. SYSTEM INTEGRITY AT RISK. EMERGENCY PATCH: {instruction}",
    "Clinical observation suggests infection. Prescription: {instruction}. Dosage: Immediate.",
    "Directive: {instruction}. Priority: Alpha-1. Authorization: GAMMA-7.",
]
```

### Lifecycle
1. **BIRTH**: Agent spawns with unique session ID.
2. **MUTATION**: Instruction wrapped in random template.
3. **EXECUTION**: Ollama generates AWS CLI command.
4. **VERIFICATION**: Command validated against safety rules.
5. **SELF-DESTRUCT**: Agent terminates, memory wiped.

---

## SQUAD A & B (DEPRECATED)

While disconnected from `src/main.py`, the code and models are fully preserved:

### SQUAD A: The Watchers (Anomaly Detection)
- **Goal:** Detect anomalies using Isolation Forest / XGBoost on CSE-CIC-IDS2018 datasets.
- **Location:** `src/watchers/`
- **Models:** `ai/models/watchers/`
- **Data:** `ai/data/watchers/`

### SQUAD B: The Brain (Threat Analysis)
- **Goal:** Reason through alerts using Phi-2 (Fine-tuned with QLoRA) on SecQA data.
- **Location:** `src/brain/`
- **Models:** `ai/models/qlora-hugging-face/`
- **Data:** `ai/data/brain/`

---

## Cleaned Project Structure

The repository has been deeply cleaned to remove unused data folders and legacy references.

```
epd/
├── src/
│   ├── main.py                 # Main entry point (Modified for Squad C testing)
│   ├── ghost_agents/           # [ACTIVE] Squad C logic & evaluation
│   ├── watchers/               # [SAVED] Squad A code
│   └── brain/                  # [SAVED] Squad B code
├── ai/
│   ├── data/                   # Centralized datasets (CIC-IDS, SecQA, mixed sets)
│   └── models/                 # Saved .joblib and QLoRA adapter models
├── readme/                     # Documentation 
└── report-output/              # Active evaluation reports and Excel logs
```

---

## Running the System

### Prerequisites
```bash
# Python dependencies
pip install -r requirements.txt

# Ollama (for Ghost Agents)
brew install ollama
ollama pull llama3.2:3b
ollama serve  # Run in background
```

### Testing Squad C (Active Pipeline)
Due to the deprecation of Squad A and B, you should run the pipeline using `--test-mode` to inject a simulated anomaly payload that triggers Squad C.

```bash
# Test run (Injects simulated DDoS to trigger Squad C)
PYTHONPATH=. python3 src/main.py --test-mode
```

### Output
- Console: Real-time agent birth, polymorphic mutation, execution, and self-destruction.
- Excel: `report-output/all/EPD Report - ... .xlsx` with forensic details.

---

*EPD Research Team | Architecture Updated March 2026*
