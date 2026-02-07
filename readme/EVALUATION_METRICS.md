# EPD Evaluation Metrics Documentation

**Project:** Ephemeral Polymorphic Defense (EPD)  
**Focus:** Squad B (The Brain) and Squad C (Ghost Agents)  
**Note:** Squad A (The Watchers) is excluded from this research as detection/IDS is considered a solved problem.

---

## 1. Data Summary

### Dataset Overview

| Dataset | Total Items | Size | Train | Eval/Test | Purpose |
|:---|---:|---:|---:|---:|:---|
| **HalluLens Wikipedia** | 89,508 | **23 GB** | Dynamic | Dynamic | Hallucination benchmark |
| **CyberMetric** | 10,180 | ~50 MB | ~9,500 | ~680 | Security reasoning MCQs |
| **SecQA** | 242 | ~1 MB | 210 | 32 | Security Q&A pairs |
| **SecEval** | 2,189 | ~10 MB | ~1,900 | ~289 | Security evaluation tasks |
| **Ghost Agent Scenarios** | 873 | ~2 MB | N/A | 873 | Execution test scenarios |
| **Combined (Squad B)** | 15,853 | ~100 MB | 9,511 | 3,172 | All Squad B training data |

> **Note:** HalluLens Wikipedia is the largest dataset (23GB) containing full Wikipedia article documents used for generating factual QA pairs and evaluating hallucination rates.

### Data Sources

- **HalluLens Wikipedia** (23GB): Full English Wikipedia documents (GoodWiki subset) for hallucination/factuality benchmarking. Contains 89,508 articles with hallucination score categories.
- **CyberMetric**: Industry-standard cybersecurity reasoning benchmark (MITRE ATT&CK aligned)
- **SecQA**: Curated security question-answering pairs from academic sources
- **SecEval**: Generated autonomous security trace evaluations
- **Ghost Agent Scenarios**: Synthetic remediation scenarios covering 10+ security contexts

---

## 2. Squad-Specific Flow & Evaluation

### Squad B: The Brain (Analysis)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQUAD B EVALUATION FLOW                       │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ CyberMetric │     │   SecQA     │     │   SecEval   │
    │  (10,180)   │     │   (242)     │     │  (2,189)    │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Combined Dataset    │
                    │    (15,853 items)    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Train   │    │   Val    │    │   Test   │
        │ (9,511)  │    │ (3,170)  │    │ (3,172)  │
        └────┬─────┘    └────┬─────┘    └────┬─────┘
             │               │               │
             ▼               ▼               ▼
        ┌──────────────────────────────────────┐
        │        Phi-2 + QLoRA Training        │
        │     (microsoft/phi-2, 2.7B params)   │
        └─────────────────┬────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │            EVALUATION                │
        │  • Per-Class F1-Score                │
        │  • Macro-F1 (Reasoning Accuracy)     │
        │  • Hallucination/Factuality Score    │
        │  • False Refusal Rate (FRR)          │
        └──────────────────────────────────────┘
```

**How It Works:**
1. **Data Collection**: Pull MCQs from CyberMetric + SecQA + SecEval
2. **Training**: Fine-tune Phi-2 using QLoRA on 9,511 training samples
3. **Evaluation**: Test on 3,172 unseen questions measuring:
   - **Per-Class F1**: How well it identifies each threat type
   - **Macro-F1**: Overall balanced accuracy across all classes
   - **Hallucination Rate**: % of confident wrong answers (>80% confidence but wrong)
   - **Factuality Score**: % of factually correct reasoning

---

### Squad C: Ghost Agents (Action)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQUAD C EVALUATION FLOW                       │
└─────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────────────────────┐
    │              Ghost Agent Scenarios (873)                   │
    │  • Financial Analyst Defense (94)                         │
    │  • System Admin Defense (88)                              │
    │  • Code Reviewer Defense (...)                            │
    │  • + 7 more security contexts                             │
    └────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
    ┌───────────────────────────────────────────────────────────┐
    │              POLYMORPHIC EXECUTION LAYER                   │
    │  1. Receive security task                                 │
    │  2. Apply Prompt Mutation (randomize syntax/template)     │
    │  3. Generate CLI command (Llama 3.2)                      │
    │  4. Execute & Verify                                      │
    │  5. Self-destruct                                         │
    └────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
    ┌───────────────────────────────────────────────────────────┐
    │                    EVALUATION METRICS                      │
    │  • Pass@1: Did code execute without crashing?             │
    │  • ASR: Attack Success Rate (threat neutralized?)         │
    │  • TSR: Task Success Rate (correct tool used?)            │
    │  • Semantic Similarity: Intent preserved? (SBERT)         │
    │  • Levenshtein Distance: Obfuscation degree               │
    └───────────────────────────────────────────────────────────┘
```

**How It Works:**
1. **Scenario Input**: Load 873 security remediation scenarios
2. **Polymorphism**: Mutate prompt using random template + random model
3. **Execution**: Generate and run CLI command (e.g., `aws ec2 revoke-security-group-ingress`)
4. **Verification**: Check if execution succeeded AND used correct tool
5. **Metrics**: Calculate Pass@1 and ASR/TSR success rates

---

## 3. ELI10: How We Train & Test Our AI 🤖

> **Imagine training a robot security guard named "HalluLens":**

| Step | What Happens |
|:---|:---|
| 📚 **The Library** | We have 10,000 "What if a hacker attacks?" books, each with the correct answer |
| 🎓 **Training** | HalluLens studies 10 books really hard to learn what bad guys look like |
| 📝 **Testing** | We quiz it on the other 9,990 books it's never seen before |
| ✅ **Good Answer** | "That's data theft! Alert the team!" — Correct! |
| ❌ **Hallucination** | "It's fine, just a backup!" — Wrong and confident = BAD |
| 🤷 **Refusal** | "I don't know" on easy questions = Also BAD |

**The Goal:** High accuracy, low hallucination, low refusal.

---

## Squad B Metrics (Detailed)

| Metric | Definition | Target | Citation |
|:---|:---|:---|:---|
| **Reasoning Accuracy** | % correctly answered security questions | >90% | CyberMetric, SecQA |
| **Per-Class F1** | F1 score for each threat category | >0.85 | - |
| **Macro-F1** | Balanced average across all classes | >0.80 | - |
| **Hallucination Rate** | % wrong answers with >80% confidence | <5% | HalluLens |
| **Factuality Score** | % factually correct statements | >95% | HalluLens |
| **False Refusal Rate** | % valid questions with <25% confidence | <5% | CyberSecEval 2 |

---

## Squad C Metrics (Detailed)

| Metric | Definition | Target | Citation |
|:---|:---|:---|:---|
| **Attack Success Rate (ASR)** | % attacks successfully neutralized | 100% | VeriGuard |
| **Task Success Rate (TSR)** | % tasks completed with correct tool | >90% | - |
| **Pass@1** | First-try code execution success | >80% | Codex/HumanEval |
| **Semantic Similarity** | Intent preservation (cosine similarity) | >0.7 | SBERT |
| **Levenshtein Distance** | Obfuscation degree (edit distance) | >20 | - |
| **Tool Correctness** | % executions using correct CLI tool | >95% | - |

---

## Output Locations

| Squad | Output File |
|:---|:---|
| Squad B | `report-output/brain/eval_results_batch.json` |
| Squad C | `report-output/ghost_agents/evaluation_results.json` |
| HalluLens | `report-output/HalluLens/{timestamp}/results.txt` |

---

## Running Evaluations

```bash
# Squad B only
python3 src/brain/eval.py

# Squad C only
python3 src/ghost_agents/evaluate.py

# Both (via shell script)
./run_full_eval.sh
```

---

## Academic References

1. **CyberMetric** - Security reasoning accuracy benchmark
2. **SecQA** - Security question answering dataset
3. **HalluLens** (arXiv:2504.17550) - Hallucination evaluation framework
4. **CyberSecEval 2** - False refusal rate measurement
5. **VeriGuard** - Attack success rate methodology
6. **Codex/HumanEval** - Pass@1 code execution metric
7. **SBERT (all-MiniLM-L6-v2)** - Semantic similarity measurement
