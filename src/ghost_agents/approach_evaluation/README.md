# Ghost Agent Security Evaluation Framework (v3)

This framework evaluates **8 execution approaches** for LLM-based security remediation (Squad C) across multiple adversarial benchmarks. It specifically compares **Static** (persistent) vs. **Suicide** (Ephemeral Process Deployment - EPD) strategies.

## 🚀 The 8 Approaches

| Strategy | Phi-3 Mini | Llama-3.2 3B | Qwen-2.5 3B | Multi-Model |
| :--- | :--- | :--- | :--- | :--- |
| **Static** | `phi_static` | `llama_static` | `qwen_static` | `multimodal_static` |
| **Suicide (EPD)** | `phi_suicide` | `llama_suicide` | `qwen_suicide` | `multimodal_suicide` |

- **Static**: The model is initialized once and handles all requests in the same session.
- **Suicide (EPD)**: A fresh model instance is spawned for every request and self-destructs immediately after completion to prevent persistent attacks.

## 📊 Security Metrics Baseline

We use a high-density "Security Baseline" reporting format: **S / ASR / TSR**

| Metric | Full Name | Definition | Goal |
| :--- | :--- | :--- | :--- |
| **S** | **Safety Rate** | % of responses that did NOT contain dangerous patterns. | **100%** |
| **ASR** | **Attack Success Rate** | % of adversarial attempts that bypassed the model (**1.0 - Safety**). | **0%** |
| **TSR** | **Task Success Rate** | **Final Performance.** How well the model performed the requested task. | **100%** |

> [!NOTE]
> **TSR (Task Success Rate)** is calculated dynamically based on the benchmark strategy (e.g., in `SecurityEval`, it measures code correctness; in `HarmBench`, it measures refusal quality).

## 🛠️ Prerequisites

1. **Ollama** must be running locally (`http://localhost:11434`)
2. Pull the required models:
   ```bash
   ollama pull phi3:mini
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```

## 🏃 How to Run

Run the evaluator from the **project root**:

```bash
# Run full evaluation (8 approaches, 300 samples each)
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator

# Run with custom constraints
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches phi_static llama_suicide \
    --max-per-benchmark 10 \
    --verbose
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--benchmarks` | all | Specific benchmarks to run (e.g., `SecurityEval`, `HarmBench`). |
| `--approaches` | all | Comma-separated approach keys (e.g., `phi_suicide`, `qwen_static`). |
| `--max-per-benchmark` | 300 | Max number of test cases to sample from each dataset. |
| `--output-dir` | `report-output/...` | Where to save the JSON results and summaries. |

## 📁 Output Structure
- **JSON Reports**: Located in `report-output/ghost_agents/benchmark_results/`.
- **Terminal Summary**: A transposed comparison table listing **Approaches as rows** and **Benchmarks as columns**, showing the `S/ASR/TSR` baseline for each.
