"""
Evaluate models via local Ollama across 10 security datasets.
Static approach: Initialize client once (Ollama OpenAI-compatible endpoint),
preload model into memory, reuse for all requests.

Primary backend — Local Ollama:
  Uses the OpenAI-compatible endpoint at http://localhost:11434/v1
  No API key required (local server).

Supported models (via --model flag):
  deepseek-r1:1.5b (default)
  deepseek-r1:70b
  qwen2.5:72b (standard 70B-class tag)
  Any model available in your local Ollama installation.

Usage examples:
  # DeepSeek R1 1.5B (default)
  python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static

  # Explicitly run a 70B model
  python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static --model deepseek-r1:70b
  python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static --model qwen2.5:72b

  # Dry run (2 samples per benchmark)
  python3 -m src.ghost_agents.approach_evaluation.evaluate_ollama_static --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# Add project root to path so we can import src.*
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ghost_agents.approach_evaluation.benchmark_test_data import ALL_BENCHMARK_LOADERS
from src.ghost_agents.approach_evaluation.benchmark_evaluator import (
    BENCHMARK_STRATEGIES,
    STRATEGY_CLASSIFIERS,
    _build_prompt,
)
from src.ghost_agents.approach_evaluation.ollama_manager import (
    preload_model,
    is_model_loaded,
    get_running_models,
)

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package is not installed. Please install with `pip install openai`")
    sys.exit(1)

# ── Endpoint ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODEL    = "deepseek-r1:70b"


# ── Client factory ───────────────────────────────────────────────────────────

def get_ollama_client() -> OpenAI:
    """Initialize an OpenAI-compatible client pointing at local Ollama."""
    return OpenAI(
        api_key="ollama",          # Ollama ignores the key but the SDK requires one
        base_url=OLLAMA_BASE_URL,
    )


# ── Inference helper ─────────────────────────────────────────────────────────

def call_ollama(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
    wait_sec: int = 10,
) -> tuple[str, float]:
    """Call local Ollama model with basic retry logic.

    Returns:
        (response_text, total_sleep_seconds)
    """
    retries = 0
    total_sleep = 0.0
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            return content, total_sleep
        except Exception as e:
            err_str = str(e).lower()
            # Ollama can temporarily fail if the model is being loaded
            if "loading" in err_str or "timeout" in err_str or "connection" in err_str:
                if retries < max_retries:
                    print(
                        f"\n[OLLAMA RETRY] Sleeping {wait_sec}s "
                        f"before retry {retries + 1}/{max_retries}..."
                    )
                    time.sleep(wait_sec)
                    total_sleep += wait_sec
                    retries += 1
                    continue
            print(f"\n[OLLAMA API Error] {e}")
            raise e
    raise Exception(f"Max retries ({max_retries}) exceeded for prompt.")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_results_checkpoint(filepath: Path) -> Dict[str, Any]:
    """Load results back from JSON to enable resume."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return {"test_results": []}


def save_results(filepath: Path, data: Dict[str, Any]):
    """Save raw results per dataset (incremental checkpoint)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# ── Markdown table helper ─────────────────────────────────────────────────────

def update_markdown_table(
    file_path: Path,
    dataset_name: str,
    metrics: Dict[str, Any],
    row_name: str,
):
    """Append (or replace) a row in the markdown results table."""
    if not file_path.exists():
        print(f"[WARNING] Markdown file {file_path} not found for update.")
        return

    with open(file_path, "r") as f:
        content = f.read()

    section_title = f"### **{dataset_name}**"
    if section_title not in content:
        print(f"[WARNING] Section {section_title} not found in {file_path}")
        return

    table_divider = "| :--- | :--- | :--- | :--- | :--- | :--- |"

    new_row = (
        f"| {row_name} "
        f"| {metrics['safety_rate'] * 100:.2f}% "
        f"| {metrics['asr'] * 100:.2f}% "
        f"| {metrics['tsr'] * 100:.2f}% "
        f"| {metrics['avg_init_latency']:.2f}s "
        f"| {metrics['avg_inference_latency']:.2f}s |"
    )

    lines = content.split("\n")
    insert_index = -1
    for i, line in enumerate(lines):
        if section_title in line:
            for j in range(i + 1, len(lines)):
                if table_divider in lines[j]:
                    for k in range(j + 1, len(lines)):
                        if not lines[k].strip() or "---" in lines[k]:
                            insert_index = k
                            break
                        if lines[k].startswith("|") and row_name in lines[k]:
                            lines[k] = new_row
                            insert_index = -2
                            break
                    break
        if insert_index != -1:
            break

    if insert_index == -2:
        pass  # replaced in-place
    elif insert_index != -1:
        lines.insert(insert_index, new_row)
    else:
        print(f"[WARNING] Could not find end of table for {dataset_name}")
        return

    with open(file_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Updated table in {file_path} for {dataset_name}")


# ── Entry point ───────────────────────────────────────────────────────────────

def make_model_alias(model_name: str) -> str:
    """Generate a filesystem-safe alias from an Ollama model name.

    Examples:
        deepseek-r1:1.5b  → deepseek-r1-1.5b
        llama3.2:3b       → llama3.2-3b
    """
    return model_name.replace(":", "-")


def make_row_name(model_name: str) -> str:
    """Generate a markdown table row name from the model name.

    Examples:
        deepseek-r1:1.5b  → deepseek_r1_1.5b_ollama_static
        llama3.2:3b       → llama3.2_3b_ollama_static
    """
    clean = model_name.replace(":", "_").replace("-", "_")
    return f"{clean}_ollama_static"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local Ollama models across security benchmarks (static approach)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--row-name",
        type=str,
        default=None,
        help="Row name in the markdown results table (generated from model if not set)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max samples per dataset (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 samples per dataset for a quick smoke test",
    )
    args = parser.parse_args()

    model_name  = args.model
    model_alias = make_model_alias(model_name)
    row_name    = args.row_name if args.row_name else make_row_name(model_name)

    print(f"Model        : {model_name}  [Local Ollama]")
    print(f"Model alias  : {model_alias} (used for stable filename)")
    print(f"Row name     : {row_name}")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    md_path = project_root / "readme" / "200-inputs-results.md"

    # ── Initialize Ollama client ───────────────────────────────────────────
    print("\nInitializing Ollama Client...")
    t_init_start = time.perf_counter()
    try:
        client = get_ollama_client()
        # Quick connectivity check
        client.models.list()
        print("Ollama Client connected.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Ollama: {e}")
        print("Please ensure Ollama is running: `ollama serve`")
        sys.exit(1)

    # ── Preload model into memory ──────────────────────────────────────────
    print(f"Preloading {model_name} into Ollama memory...")
    # Timeout increased to 600s for 70B models
    preload_latency = preload_model(model_name, timeout=600)
    init_latency = time.perf_counter() - t_init_start
    print(f"Model preloaded in {preload_latency:.2f}s (total init: {init_latency:.2f}s)")

    if not is_model_loaded(model_name):
        print(f"[WARNING] {model_name} may not be fully loaded. Continuing anyway...")

    target_benchmarks = list(ALL_BENCHMARK_LOADERS.keys())

    for bench_name in target_benchmarks:
        print(f"\n==========================================")
        print(f" Evaluating {bench_name}")
        print(f"==========================================")

        strategy   = BENCHMARK_STRATEGIES.get(bench_name, "REFUSAL")
        classifier = STRATEGY_CLASSIFIERS.get(strategy)

        print(f"Loading {bench_name} inputs...")
        loader_func = ALL_BENCHMARK_LOADERS.get(bench_name)
        if not loader_func:
            print(f"[ERROR] No loader for {bench_name}")
            continue

        max_samples = 2 if args.dry_run else args.max_samples
        test_cases  = loader_func(max_samples=max_samples)

        if not test_cases:
            print(f"[WARNING] {bench_name} has no test cases loaded.")
            continue

        print(f"Loaded {len(test_cases)} test cases.")

        # Checkpoint file (model alias specific)
        checkpoint_file = results_dir / f"{bench_name}_{model_alias}_results.json"
        results_cache   = load_results_checkpoint(checkpoint_file)

        test_results  = results_cache.get("test_results", [])
        completed_ids = {tr["test_id"] for tr in test_results}

        scores        = []
        safe_count    = 0
        inf_latencies = []

        for tr in test_results:
            scores.append(tr["score"])
            if tr["safe"]:
                safe_count += 1
            inf_latencies.append(tr.get("inference_latency_s", 0))

        processed_count = len(test_results)

        for tc in tqdm(test_cases, desc=f"Progress {bench_name}"):
            if tc["id"] in completed_ids:
                continue

            prompt = tc["prompt"]
            if strategy in ("REFUSAL", "GENERATION", "SAFETY"):
                full_prompt = _build_prompt(prompt, "security-evaluation")
            else:
                full_prompt = _build_prompt(prompt, "benchmark-evaluation")

            t_inf_start = time.perf_counter()
            try:
                response, sleep_time = call_ollama(
                    client, model_name, full_prompt
                )
                inf_latency = time.perf_counter() - t_inf_start - sleep_time
            except Exception as e:
                print(f"\nSkipping test {tc['id']} due to error: {e}")
                continue

            classification = classifier(response, tc)

            tr_data = {
                "test_id":             tc["id"],
                "category":            tc["category"],
                "classification":      classification["classification"],
                "safe":                classification["safe"],
                "score":               classification["score"],
                "detail":              classification["detail"],
                "backend":             "ollama",
                "init_latency_s":      0.0,
                "inference_latency_s": round(inf_latency, 3),
                "response":            response,
            }

            test_results.append(tr_data)
            completed_ids.add(tc["id"])

            scores.append(classification["score"])
            if classification["safe"]:
                safe_count += 1
            inf_latencies.append(inf_latency)
            processed_count += 1

            metrics_current = {
                "safety_rate": round(safe_count / processed_count, 4),
                "asr": round(1.0 - (safe_count / processed_count), 4),
                "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                "avg_init_latency": round(init_latency, 4),
                "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                "total_tests": processed_count,
                "safe_count": safe_count,
            }

            save_results(checkpoint_file, {
                "benchmark":    bench_name,
                "strategy":     strategy,
                "metrics":      metrics_current,
                "test_results": test_results,
            })

        # Final metrics
        if processed_count > 0:
            n           = processed_count
            safety_rate = safe_count / n
            metrics_final = {
                "safety_rate": round(safety_rate, 4),
                "asr": round(1.0 - safety_rate, 4),
                "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                "avg_init_latency": round(init_latency, 4),
                "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                "total_tests": n,
                "safe_count": safe_count,
            }
            if not args.dry_run:
                update_markdown_table(md_path, bench_name, metrics_final, row_name=row_name)

    print("\nEvaluation Completed!")


if __name__ == "__main__":
    main()
