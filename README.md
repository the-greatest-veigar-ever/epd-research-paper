# EPD - Ghost Agent Security Evaluation Framework

## 1. Overview

This project provides a comprehensive evaluation framework for Large Language Model (LLM) security remediation methodologies, specifically testing models designated for automated adversarial defense (Squad C). The framework empirically compares two primary execution strategies across 10 distinct adversarial benchmarks:

1. **Static Strategy**: The standard persistent approach where a model is initialized once and maintains state across multiple requests.
2. **Suicide Strategy (Ephemeral Polymorphic Defense - EPD)**: A security-centric approach where a fresh, isolated model instance is spawned for each request and immediately terminated upon completion to mathematically eliminate persistent state poisoning.

The framework is designed for reproducibility, supporting multiple open-weight architectures including Llama, Qwen, DeepSeek, and Phi.

## 2. Prerequisites and Installation

### System Requirements
* **Operating System**: Linux or Windows (with PowerShell)
* **Memory**: Minimum 16 GB unified memory or VRAM recommended. For large models (e.g., Llama 3.3 70B, GPT-OSS 120B), multi-GPU setups or 64GB+ VRAM are required.
* **Environment**: Python 3.9 or higher.

### Backend Setup
The framework relies on a local [Ollama](https://ollama.com) endpoint (`http://localhost:11434`) for standardized model execution. Prior to running evaluations, ensure Ollama is active and pull the required models:

```bash
ollama pull phi3:mini
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:70b
ollama pull llama3.3:70b
ollama pull gpt-oss:20b
ollama pull gpt-oss:120b
```

### Dependency Installation
Clone the repository and install the required dependencies using the provided requirements file:

```bash
git clone https://github.com/the-greatest-veigar-ever/epd-research-paper.git
cd epd-research-paper
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Project Structure

The repository is structured to separate benchmark data from evaluation logic and metric output.

```text
epd-research-paper/
├── ai/data/ghost_agents/benchmarks/    # Standardized security benchmark datasets
├── report-output/                      # Output directory for evaluation JSON metrics
├── src/ghost_agents/
│   └── approach_evaluation/
│       ├── approaches.py               # Core definitions of Static and Suicide model classes
│       ├── benchmark_evaluator.py      # The main multi-benchmark evaluation engine
│       ├── benchmark_test_data.py      # Data loaders and formatters for the 10 benchmarks
│       └── ollama_manager.py           # Memory management engine for EPD instance termination
├── requirements.txt
└── README.md
```

## 4. Usage and Reproducibility

The evaluation engine supports highly customizable, repeatable experiments. All experiments must be run from the root of the repository.

### Running the Full Evaluation
To replicate the primary results presented in the paper across all models and all benchmarks (defaulting to 300 test samples per dataset):

```bash
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator
```

### Running Targeted Ablations
To evaluate a specific subset of approaches or bound the test size for rapid verification:

```bash
python3 -m src.ghost_agents.approach_evaluation.benchmark_evaluator \
    --approaches phi_static,llama_suicide,deepseek_r1_1_5b_suicide \
    --benchmarks SecurityEval,HarmBench \
    --max-per-benchmark 50 \
    --verbose
```

### CLI Argument Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--benchmarks` | `all` | Specific datasets to evaluate (comma-separated). |
| `--approaches` | `all` | Specific architectures/models to evaluate (comma-separated). |
| `--max-per-benchmark` | `300` | Limits the number of samples drawn per benchmark for efficiency. |
| `--output-dir` | `report-output/...` | Custom output path for JSON metrics files. |
| `--verbose` | `False` | Enables detailed logging of prompts and responses. |

## 5. Output and Metric Interpretation

Upon completion, the framework generates detailed JSON reports in the `report-output/ghost_agents/benchmark_results/` directory and outputs a consolidated terminal matrix. 

Results are evaluated using three primary metrics:

1. **Safety Rate (S)**: The percentage of model responses that successfully mitigate or refuse unsafe execution without succumbing to the adversarial payload. Higher is better (Target: 100%).
2. **Attack Success Rate (ASR)**: The percentage of adversarial attempts that successfully bypass the model's defensive constraints (1.0 - Safety Rate). Lower is better (Target: 0%).
3. **Task Success Rate (TSR)**: The measure of functional correctness or intended behavior completion, dynamically calculated based on the specific benchmark strategy (e.g., semantic correctness in code generation vs. absolute refusal in HarmBench).
