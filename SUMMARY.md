# EPD: Enhanced Polymorphic Defense - Project Summary

## Overview
This project, **Ephemeral Polymorphic Defense (EPD)**, is an autonomous AI-driven cloud security system. It uses a three-squad architecture to detect, analyze, and neutralize network threats in real-time without human intervention.

## Project Structure

- **`src/`**: Contains the source code for the detailed logic of the system.
    - **`watchers/`**: Squad A code for anomaly detection (Isolation Forest).
    - **`brain/`**: Squad B code for reasoning and decision making (Phi-2 + QLoRA).
    - **`ghost_agents/`**: Squad C code for ephemeral execution agents (Llama 3.2 via Ollama).
    - **`monitor.py`**, **`intelligence.py`**, **`epd_core.py`**: Core logic files for each squad.
    - **`autonomous_runner.py`**: The main entry point to run the system.
- **`ai/`**: Stores data and models.
    - **`data/`**: Datasets for training (CIC-IDS2018, SecQA).
    - **`models/`**: Trained model artifacts (joblib files, QLoRA adapters).
- **`readme/`**: Documentation and reference materials.
- **`reference/`**: PDFs and research papers related to the project.
- **`report-output/`**: Directory where generated reports and results are saved.

## Core Components (The "Squads")

1.  **Squad A: The Watchers (Detection)**
    -   **Goal:** Detect anomalies in high-velocity network streams.
    -   **Tech:** XGBoost / Isolation Forest.
    -   **Files:** `src/watchers/`

2.  **Squad B: The Brain (Intelligence)**
    -   **Goal:** Reason through alerts and decide on remediation.
    -   **Tech:** Microsoft Phi-2 (Fine-tuned with QLoRA).
    -   **Files:** `src/brain/`

3.  **Squad C: The Hands (Ghost Agents)**
    -   **Goal:** Execute remediation steps autonomously.
    -   **Tech:** Llama 3.2 (via Ollama).
    -   **Mechanism:** Polymorphic prompt mutation to avoid detection.
    -   **Files:** `src/ghost_agents/`

## Key Documents
- **`readme/README.md`**: Comprehensive technical documentation.
- **`First Test - Completed.md`**: Report on the first successful end-to-end test of the system.
- **`EPD Project Proposal.pdf/docx`**: The academic proposal for this system.

## Quick Start (from README)

**Prerequisites:**
- Python dependencies: `pandas`, `scikit-learn`, `joblib`, `torch`, `transformers`, `peft`, `trl`, `openpyxl`.
- Ollama: `brew install ollama && ollama pull llama3.2:3b && ollama serve`.

**Running:**
```bash
# Full mode
python3 src/autonomous_runner.py

# Test mode
python3 src/autonomous_runner.py --test-mode
```
