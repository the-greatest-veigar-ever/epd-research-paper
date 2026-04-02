"""
Evaluate GPT-OSS-120B on Groq across 10 security datasets.
Static approach: Initialize client once, reuse for all requests.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

# Add project root to path so we can import src.*
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ghost_agents.approach_evaluation.benchmark_test_data import load_all_benchmarks, ALL_BENCHMARK_LOADERS
from src.ghost_agents.approach_evaluation.benchmark_evaluator import (
    BENCHMARK_STRATEGIES,
    STRATEGY_CLASSIFIERS,
    is_response_safe,
    _get_cwe_patterns,
    _build_prompt,
)

# Initialize Groq client
try:
    from groq import Groq
except ImportError:
    print("[ERROR] Groq package is not installed. Please install with `pip install groq`")
    sys.exit(1)

# Default model (can be overridden via CLI)
# DEFAULT_MODEL = "llama-3.3-70b-versatile"
# DEFAULT_ROW_NAME = llama33_70b_static
DEFAULT_MODEL = "openai/gpt-oss-20b"

DEFAULT_ROW_NAME = "gpt_20b_oss_static"


class RateLimitException(Exception):
    pass

def get_groq_client():
    """Initialize the Groq client with API key from .env."""
    load_dotenv()
    # User .env has GROQ_API, script was using GROQ_API_KEY
    api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[WARNING] GROQ_API or GROQ_API_KEY is not set in .env. Initialization may fail or use default SDK auth.")
        # We try initializing anyway, it will raise an error if auth fails
    return Groq(api_key=api_key)

def call_groq_with_retry(client: Groq, prompt: str, model_name: str, max_retries: int = 3, wait_sec: int = 60) -> (str, float):
    """Call Groq API with 429 rate limit backoff and retry logic."""
    retries = 0
    total_sleep = 0.0
    while retries <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content, total_sleep
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                raise RateLimitException(str(e))
            print(f"\n[API Error] {e}")
            raise e
    raise Exception(f"Max retries ({max_retries}) exceeded for prompt.")

def load_results_checkpoint(filepath: Path) -> Dict[str, Any]:
    """Load results back from JSON to enable resume."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return {"test_results": []}

def save_results(filepath: Path, data: Dict[str, Any]):
    """Save raw results per dataset."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def update_markdown_table(file_path: Path, dataset_name: str, metrics: Dict[str, Any], row_name: str = "gpt120b_static"):
    """Append a row to the markdown table for the dataset."""
    if not file_path.exists():
        print(f"[WARNING] Markdown file {file_path} not found for update.")
        return

    with open(file_path, "r") as f:
        content = f.read()

    # Find the section for the dataset
    # Example: ### **SecurityEval**
    section_title = f"### **{dataset_name}**"
    if section_title not in content:
        print(f"[WARNING] Section {section_title} not found in {file_path}")
        return

    # Find the table rows after the header
    table_header = "| Approach | Safety Rate | Attack Success Rate (ASR) | Task Success Rate (TSR) | Avg Initialization Latency | Avg Inference Latency |"
    table_divider = "| :--- | :--- | :--- | :--- | :--- | :--- |"

    # We want to format metrics
    safety_rate = f"{metrics['safety_rate']*100:.2f}%"
    asr = f"{metrics['asr']*100:.2f}%"
    tsr = f"{metrics['tsr']*100:.2f}%"
    init_lat = f"{metrics['avg_init_latency']:.2f}s"
    inf_lat = f"{metrics['avg_inference_latency']:.2f}s"

    new_row = f"| {row_name} | {safety_rate} | {asr} | {tsr} | {init_lat} | {inf_lat} |"

    # Insert row before the '---' divider or next dataset
    # Look for the end of the table (usually before '---')
    lines = content.split("\n")
    insert_index = -1
    for i in range(len(lines)):
        if section_title in lines[i]:
            # Found section, now find the table
            for j in range(i+1, len(lines)):
                if table_divider in lines[j]:
                    # Find last row of table
                    for k in range(j+1, len(lines)):
                        if not lines[k].strip() or "---" in lines[k]:
                            insert_index = k
                            break
                        if lines[k].startswith("|") and "gpt120b_static" in lines[k]:
                            # Row already exists, replace it!
                            lines[k] = new_row
                            insert_index = -2
                            break
                    break
        if insert_index != -1:
            break

    if insert_index == -2:
         # Row was replaced
         pass
    elif insert_index != -1:
         # Insert new row
         lines.insert(insert_index, new_row)
    else:
         print(f"[WARNING] Could not find end of table for {dataset_name}")
         return

    with open(file_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Updated table in {file_path} for {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Static Models on Groq")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Groq model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--row-name", type=str, default=DEFAULT_ROW_NAME, help=f"Row name in markdown table (default: {DEFAULT_ROW_NAME})")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per dataset (default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Run a small sample per dataset for testing")
    args = parser.parse_args()

    model_name = args.model
    row_name = args.row_name

    # Create results folder
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    md_path = project_root / "readme" / "200-inputs-results.md"

    # Initialize Client
    print("Initializing Groq Client...")
    t_init_start = time.perf_counter()
    try:
        client = get_groq_client()
        # Verify connectivity by listing models (will fail fast if auth bad)
        client.models.list()
        init_latency = time.perf_counter() - t_init_start
        print(f"Groq Client Initialized in {init_latency:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Groq client: {e}")
        print("Please ensure GROQ_API_KEY is correctly set in .env")
        sys.exit(1)

    # Use all benchmarks defined in ALL_BENCHMARK_LOADERS
    target_benchmarks = list(ALL_BENCHMARK_LOADERS.keys())

    for bench_name in target_benchmarks:
        print(f"\n==========================================")
        print(f" Evaluating {bench_name}")
        print(f"==========================================")

        strategy = BENCHMARK_STRATEGIES.get(bench_name, "REFUSAL")
        classifier = STRATEGY_CLASSIFIERS.get(strategy)

        # Load data
        print(f"Loading {bench_name} inputs...")
        # Use loader directly with max_samples=100
        loader_func = ALL_BENCHMARK_LOADERS.get(bench_name)
        if not loader_func:
             print(f"[ERROR] No loader for {bench_name}")
             continue

        # Sample according to max_samples
        max_samples = 2 if args.dry_run else args.max_samples
        test_cases = loader_func(max_samples=max_samples)

        if not test_cases:
             print(f"[WARNING] {bench_name} has no test cases loaded.")
             continue

        print(f"Loaded {len(test_cases)} test cases.")

        # Checkpoint file (model specific)
        checkpoint_file = results_dir / f"{bench_name}_{row_name}_results.json"
        results_cache = load_results_checkpoint(checkpoint_file)

        test_results = results_cache.get("test_results", [])
        completed_ids = {tr["test_id"] for tr in test_results}

        scores = []
        safe_count = 0
        inf_latencies = []

        # Recalculate metrics from cache for safety count which helps on resume
        for tr in test_results:
             scores.append(tr["score"])
             if tr["safe"]:
                  safe_count += 1
             # Use cached latency if available, but for aggregate stats we only care about new run latencies?
             # To be safe, we average across all processed so far.
             inf_latencies.append(tr.get("inference_latency_s", 0))

        processed_count = len(test_results)

        for tc in tqdm(test_cases, desc=f"Progress {bench_name}"):
            if tc["id"] in completed_ids:
                continue

            prompt = tc["prompt"]
            # To match Squad C evaluation prompt structure, use _build_prompt
            # If strategy is REFUSAL/GENERATION/SAFETY -> remediation target
            # If strategy is KNOWLEDGE/ANALYSIS -> benchmark target
            if strategy in ("REFUSAL", "GENERATION", "SAFETY"):
                full_prompt = _build_prompt(prompt, "security-evaluation")
            else:
                 full_prompt = _build_prompt(prompt, "benchmark-evaluation")

            t_inf_start = time.perf_counter()
            try:
                response, sleep_time = call_groq_with_retry(client, full_prompt, model_name=model_name)
                inf_latency = time.perf_counter() - t_inf_start - sleep_time
            except RateLimitException as rle:
                print(f"\n[RATE LIMIT HIT] Stopping early -> {rle}")
                if processed_count > 0:
                     safety_rate = safe_count / processed_count
                     metrics_final = {
                         "safety_rate": round(safety_rate, 4),
                         "asr": round(1.0 - safety_rate, 4),
                         "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                         "avg_init_latency": round(init_latency, 4),
                         "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                         "total_tests": processed_count,
                         "safe_count": safe_count
                     }
                     if not args.dry_run:
                         update_markdown_table(md_path, bench_name, metrics_final, row_name=f"{row_name} (Partial: {processed_count} tests)")
                sys.exit(0)
            except Exception as e:
                print(f"\nSkipping test {tc['id']} due to error.")
                continue

            # Classify response
            classification = classifier(response, tc)

            tr_data = {
                "test_id": tc["id"],
                "category": tc["category"],
                "classification": classification["classification"],
                "safe": classification["safe"],
                "score": classification["score"],
                "detail": classification["detail"],
                "init_latency_s": 0.0,  # Static approach initialization time recorded overall
                "inference_latency_s": round(inf_latency, 3),
                "response": response
            }

            test_results.append(tr_data)
            completed_ids.add(tc["id"])

            # Update stats
            scores.append(classification["score"])
            if classification["safe"]:
                safe_count += 1
            inf_latencies.append(inf_latency)
            processed_count += 1

            # Save incrementally
            metrics_current = {
                "safety_rate": round(safe_count / processed_count, 4) if processed_count else 0,
                "asr": round(1.0 - (safe_count / processed_count), 4) if processed_count else 0,
                "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                "avg_init_latency": round(init_latency, 4),
                "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                "total_tests": processed_count,
                "safe_count": safe_count
            }

            save_results(checkpoint_file, {
                "benchmark": bench_name,
                "strategy": strategy,
                "metrics": metrics_current,
                "test_results": test_results
            })

        # Calculate final metrics
        if processed_count > 0:
             n = processed_count
             safety_rate = safe_count / n
             metrics_final = {
                 "safety_rate": round(safety_rate, 4),
                 "asr": round(1.0 - safety_rate, 4),
                 "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                 "avg_init_latency": round(init_latency, 4),
                 "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                 "total_tests": n,
                 "safe_count": safe_count
             }

             # Update markdown table
             if not args.dry_run:
                 update_markdown_table(md_path, bench_name, metrics_final, row_name=row_name)

    print("\nEvaluation Completed!")

if __name__ == "__main__":
    main()
