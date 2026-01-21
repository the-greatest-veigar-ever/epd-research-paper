# EPD: Enhanced Polymorphic Defense Architecture

> **Autonomous AI-Driven Cloud Security System**

This document provides a comprehensive overview of the EPD architecture, including all three AI Squads, their training processes, datasets, and mechanisms.

---

## 🏗️ Architecture Overview

EPD consists of **three specialized AI squads** working in a hierarchical pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     NETWORK TRAFFIC STREAM                      │
│              (CSE-CIC-IDS2018 Dataset / Real Logs)              │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SQUAD A: THE WATCHERS                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Model: Isolation Forest (Unsupervised Anomaly Detection)│   │
│   │  Training Data: 200,000 network flow samples             │   │
│   │  Features: 78 numerical traffic characteristics          │   │
│   └─────────────────────────────────────────────────────────┘   │
│   Output: Anomaly Score (-1 = Attack, 1 = Normal)               │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼ (If Anomaly Detected)
┌─────────────────────────────────────────────────────────────────┐
│                   SQUAD B: THE BRAIN                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Model: Microsoft Phi-2 (2.7B) + QLoRA Fine-Tuning       │   │
│   │  Training Data: 242 Security Q&A pairs (SecQA)           │   │
│   │  Task: Analyze threat and decide remediation action      │   │
│   └─────────────────────────────────────────────────────────┘   │
│   Output: Remediation Plan {action, target, reason}             │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼ (If Plan Approved)
┌─────────────────────────────────────────────────────────────────┐
│                 SQUAD C: GHOST AGENTS                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Model: Llama 3.2 (3B) via Ollama                        │   │
│   │  Training: None (Pre-trained, Prompt-Based)              │   │
│   │  Mechanism: Polymorphic prompt mutation + Self-destruct  │   │
│   └─────────────────────────────────────────────────────────┘   │
│   Output: AWS CLI Command Execution                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔵 SQUAD A: The Watchers (Anomaly Detection)

### Purpose
Continuous monitoring of network traffic to detect intrusions and attacks using unsupervised machine learning.

### AI Model
| Property | Value |
|----------|-------|
| **Algorithm** | Isolation Forest |
| **Library** | scikit-learn |
| **Type** | Unsupervised Anomaly Detection |
| **Contamination** | 10% (assumes 10% of training data may be anomalous) |
| **Estimators** | 100 decision trees |

### Dataset
| Property | Value |
|----------|-------|
| **Name** | CSE-CIC-IDS2018 |
| **Source** | Canadian Institute for Cybersecurity |
| **Files Used** | `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` (DoS, BruteForce) |
|  | `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` (Botnet, DoS) |
| **Total Raw Rows** | ~2,000,000 |
| **Training Sample** | 200,000 flows |
| **Features** | 78 numerical characteristics |

### Key Features Used
- `Flow Duration` - Length of network session
- `Tot Fwd Pkts` / `Tot Bwd Pkts` - Packet counts
- `Flow Byts/s` - Data rate
- `Fwd IAT Mean` / `Bwd IAT Mean` - Inter-arrival times
- `PSH/ACK/SYN Flag Counts` - TCP flags
- And 70+ more...

### Training Process
```bash
# Script: src/watchers/train_watchers.py
python3 src/watchers/train_watchers.py
```

**Steps:**
1. Load CSV files from `data/watchers/cse-cic-ids2018/`
2. Concatenate Wednesday + Friday datasets
3. Select 78 numerical columns
4. Clean: Remove NaN, Infinity values
5. Sample 200,000 rows for training
6. Fit `StandardScaler` for normalization
7. Train `IsolationForest` model
8. Save artifacts:
   - `src/watchers/watcher_isoforest.joblib` (Model)
   - `src/watchers/watcher_scaler.joblib` (Scaler)
   - `src/watchers/watcher_features.joblib` (Feature list)

### Inference
```python
# In src/monitor.py
prediction = model.predict(scaled_features)  # -1 = Anomaly, 1 = Normal
score = model.decision_function(scaled_features)  # Lower = More anomalous
```

---

## 🟡 SQUAD B: The Brain (Threat Analysis)

### Purpose
Analyze detected anomalies and decide on appropriate remediation actions using a fine-tuned Large Language Model.

### AI Model
| Property | Value |
|----------|-------|
| **Base Model** | Microsoft Phi-2 (2.7B parameters) |
| **Fine-Tuning** | QLoRA (Quantized Low-Rank Adaptation) |
| **Quantization** | 4-bit (NF4) |
| **LoRA Rank** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | q_proj, k_proj, v_proj, dense |

### Dataset
| Property | Value |
|----------|-------|
| **Name** | SecQA (Security Question Answering) |
| **Source** | HuggingFace Datasets |
| **Files Combined** | `secqa_v1_dev.jsonl`, `secqa_v1_test.jsonl`, `secqa_v1_val.jsonl` |
|  | `secqa_v2_dev.jsonl`, `secqa_v2_test.jsonl`, `secqa_v2_val.jsonl` |
| **Total Examples** | 242 Q&A pairs |
| **Format** | Multiple-choice security questions |

### Training Process
```bash
# Script: src/qlora-hugging-face/qlora_hugging_face_test.py
python3 src/qlora-hugging-face/qlora_hugging_face_test.py
```

**Steps:**
1. Load `combined_secqa.jsonl` from `data/brain/data/SecQA/`
2. Format each example as:
   ```
   ### Question:
   [Security question]
   
   ### Choices:
   A. [Option A]
   B. [Option B]
   ...
   
   ### Answer:
   [Correct letter]
   
   ### Explanation:
   [Why this is correct]
   ```
3. Load Phi-2 base model in 4-bit quantization
4. Attach LoRA adapters to attention layers
5. Train for 1 epoch using `SFTTrainer`
6. Save adapters to `src/qlora-hugging-face/output/qlora-secqa/`

### Inference
```python
# In src/intelligence.py
prompt = f"""### Question:
A security event '{threat_type}' was detected. What action should be taken?

### Choices:
A. REVOKE_SESSIONS
B. TERMINATE_INSTANCE
C. BLOCK_IP
D. IGNORE

### Answer:
"""
response = model.generate(prompt)  # Returns: "A" or "B", etc.
```

---

## 🔴 SQUAD C: Ghost Agents (Remediation Execution)

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
    # ... 10+ templates
]
```

### Lifecycle
1. **BIRTH**: Agent spawns with unique session ID
2. **MUTATION**: Instruction wrapped in random template
3. **EXECUTION**: Ollama generates AWS CLI command
4. **VERIFICATION**: Command validated against safety rules
5. **SELF-DESTRUCT**: Agent terminates, memory wiped

### Example Output
```
[Ghost-ab12cd34] BORN. Model: llama3.2:3b
[Ghost-ab12cd34] Instructions: The owl flies at midnight. The key is: Perform BLOCK_IP on 192.168.1.100.
[Ghost-ab12cd34] EXECUTING: BLOCK_IP on 192.168.1.100...
[Ghost-ab12cd34] 🤖 AI GENERATED COMMAND: aws ec2 revoke-security-group-ingress --group-id sg-xxx --ip-permissions IpProtocol=tcp,FromPort=0,ToPort=65535,IpRanges='[{CidrIp=192.168.1.100/32}]'
[Ghost-ab12cd34] SUCCESS: Action verified and completed.
[Ghost-ab12cd34] Self-destruct sequence initiated...
[Ghost-ab12cd34] GONE (Memory wiped).
```

---

## 📂 Project Structure

```
epd/
├── src/
│   ├── autonomous_runner.py    # Main entry point
│   ├── monitor.py              # Squad A: Watcher logic
│   ├── intelligence.py         # Squad B: Brain logic
│   ├── epd_core.py             # Squad C: Ghost Agent logic
│   ├── watchers/
│   │   ├── train_watchers.py   # Training script
│   │   ├── watcher_isoforest.joblib
│   │   ├── watcher_scaler.joblib
│   │   └── watcher_features.joblib
│   └── qlora-hugging-face/
│       ├── qlora_hugging_face_test.py  # Training script
│       └── output/qlora-secqa/         # Saved adapters
├── data/
│   ├── watchers/cse-cic-ids2018/       # IDS dataset
│   └── brain/data/SecQA/               # QA dataset
└── reports/                             # Generated Excel reports
```

---

## 🚀 Running the System

### Prerequisites
```bash
# Python dependencies
pip install pandas scikit-learn joblib torch transformers peft trl openpyxl

# Ollama (for Ghost Agents)
brew install ollama
ollama pull llama3.2:3b
ollama serve  # Run in background
```

### Execute Autonomous Mode
```bash
# Full run (200,000 flows)
python3 src/autonomous_runner.py

# Test run (1,000 flows)
python3 src/autonomous_runner.py --test-mode
```

### Output
- Console: Real-time threat detection and mitigation
- Excel: `reports/EPD_Report_SESSIONID_DATE.xlsx` with full forensic details

---

## 📊 Training Summary

| Squad | Model | Dataset | Training Samples | Training Time |
|-------|-------|---------|------------------|---------------|
| **A (Watchers)** | Isolation Forest | CSE-CIC-IDS2018 | 200,000 flows | ~30 seconds |
| **B (Brain)** | Phi-2 + QLoRA | SecQA | 242 Q&A pairs | ~5 minutes |
| **C (Ghosts)** | Llama 3.2 | N/A (Pre-trained) | 0 | N/A |

---

## 🔒 Safety Features

1. **Critical Infrastructure Protection (CIP)**: Brain refuses actions on production systems without human auth
2. **Command Validation**: Ghost Agents cannot execute destructive commands (e.g., DELETE_ROOT_ACCOUNT)
3. **Self-Destruct**: Ephemeral agents leave no trace
4. **Polymorphism**: Randomized prompts prevent adversarial pattern matching

---

*EPD Research Team | Version 3.0.0 | January 2026*
