# Evaluation Guide: Local Ollama Models

This guide provides instructions on how to set up and run security evaluations using local Ollama models.

## 1. Setup

### Prerequisites
- [Ollama](https://ollama.com/) must be installed and running on your system.
- You must have pulled the models you wish to evaluate:
  ```bash
  ollama pull deepseek-r1:1.5b
  ollama pull deepseek-r1:70b
  ollama pull qwen2.5:72b
  ```

### Install Python Dependencies
It is recommended to use the project's virtual environment:
```bash
./.venv/bin/pip install openai python-dotenv tqdm requests
```

## 2. Running Evaluations

Run these commands from the **root** of the repository (`epd-research-paper`).

### DeepSeek R1 70B (Default)
```bash
python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static --model deepseek-r1:70b
```

### Qwen 2.5 72B
```bash
python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static --model qwen2.5:72b
```

## 3. Monitoring

### Check Loaded Models
To see if a model is correctly loaded into memory:
```bash
ollama ps
```
The evaluation script also automatically preloads the selected model and provides an initialization latency report.

### Check Results
- Raw results are saved in the `results/` directory as JSON files.
- Summarized performance metrics are appended to `readme/200-inputs-results.md`.
