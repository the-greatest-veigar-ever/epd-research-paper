"""
Evaluate Gemma 3 models (4B / 27B) on Google AI Studio across 10 security datasets.
Static approach: Initialize client once, reuse for all requests.

Primary backend — Google AI Studio:
  Uses the OpenAI-compatible endpoint at
  https://generativelanguage.googleapis.com/v1beta/openai/
  with the GEMINI_API key from .env.

Fallback — OpenRouter:
  When Google AI Studio hits a rate limit (even after retries), the script
  transparently falls back to OpenRouter's OpenAI-compatible endpoint at
  https://openrouter.ai/api/v1 using the OPEN_ROUTER_API key.
  The same `openai` SDK is used for both — only base_url and api_key differ.

Supported models (via --model flag):
  gemma-3-4b-it    (default, Google AI Studio model ID)
  gemma-3-27b-it
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

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package is not installed. Please install with `pip install openai`")
    sys.exit(1)

# ── Endpoints ────────────────────────────────────────────────────────────────
GEMINI_BASE_URL     = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model aliases ─────────────────────────────────────────────────────────────
# Short alias → Google AI Studio model ID (primary)
AVAILABLE_MODELS: Dict[str, str] = {
    # Short aliases
    "gemma-3-4b":  "gemma-3-4b-it",
    "gemma-3-27b": "gemma-3-27b-it",
    # Map OpenRouter IDs to Gemini IDs (to handle USER passing OR IDs directly)
    "google/gemma-3-4b-it:free":  "gemma-3-4b-it",
    "google/gemma-3-27b-it:free": "gemma-3-27b-it",
    "google/gemma-3-4b-it":       "gemma-3-4b-it",
    "google/gemma-3-27b-it":      "gemma-3-27b-it",
}

# Google AI Studio model ID → OpenRouter model ID (fallback)
GEMINI_TO_OPENROUTER_MODEL: Dict[str, str] = {
    "gemma-3-4b-it":  "google/gemma-3-4b-it:free",
    "gemma-3-27b-it": "google/gemma-3-27b-it:free",
}

DEFAULT_MODEL    = "gemma-3-4b-it"


class RateLimitException(Exception):
    pass


# ── Client factories ──────────────────────────────────────────────────────────

def get_gemini_client() -> OpenAI:
    """Initialize primary client pointing at Google AI Studio."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("[ERROR] GEMINI_API is not set in .env")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def get_openrouter_fallback_client() -> OpenAI | None:
    """Initialize fallback client pointing at OpenRouter.

    Returns None (with a warning) if OPEN_ROUTER_API is not set.
    """
    load_dotenv()
    api_key = os.getenv("OPEN_ROUTER_API")
    if not api_key:
        print("[WARNING] OPEN_ROUTER_API not set in .env — OpenRouter fallback disabled.")
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _chat(client: OpenAI, model: str, prompt: str) -> str:
    """Single blocking chat completion call."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def call_with_fallback(
    primary_client: OpenAI,
    primary_model: str,
    prompt: str,
    fallback_client: OpenAI | None = None,
    primary_max_retries: int = 3,
    primary_wait_sec: int = 30,
    fallback_max_retries: int = 3,
    fallback_wait_sec: int = 60,
) -> tuple[str, float, str]:
    """Call Google AI Studio; fall back to OpenRouter on persistent rate limits.

    Returns:
        (response_text, total_sleep_seconds, backend_used)
        backend_used is 'gemini' or 'openrouter_fallback'.
    """
    # ── Primary: Google AI Studio ───────────────────────────────────────────
    retries = 0
    total_sleep = 0.0
    last_primary_err = None

    while retries <= primary_max_retries:
        try:
            text = _chat(primary_client, primary_model, prompt)
            return text, total_sleep, "gemini"
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                if retries < primary_max_retries:
                    print(
                        f"\n[GEMINI RATE LIMIT] Sleeping {primary_wait_sec}s "
                        f"before retry {retries + 1}/{primary_max_retries}..."
                    )
                    time.sleep(primary_wait_sec)
                    total_sleep += primary_wait_sec
                    retries += 1
                    continue
                last_primary_err = e
                break  # exhausted — try fallback
            print(f"\n[GEMINI API Error] {e}")
            raise e

    # ── Fallback: OpenRouter ────────────────────────────────────────────────
    if fallback_client is None:
        raise RateLimitException(
            f"[Gemini] Rate limit exhausted and no OpenRouter fallback configured. "
            f"Last error: {last_primary_err}"
        )

    fallback_model = GEMINI_TO_OPENROUTER_MODEL.get(primary_model, f"google/{primary_model}:free")
    print(f"\n[GEMINI RATE LIMIT] Retries exhausted. Switching to OpenRouter fallback → {fallback_model}")

    retries = 0
    while retries <= fallback_max_retries:
        try:
            text = _chat(fallback_client, fallback_model, prompt)
            return text, total_sleep, "openrouter_fallback"
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                if retries < fallback_max_retries:
                    print(
                        f"\n[OPENROUTER RATE LIMIT] Sleeping {fallback_wait_sec}s "
                        f"before retry {retries + 1}/{fallback_max_retries}..."
                    )
                    time.sleep(fallback_wait_sec)
                    total_sleep += fallback_wait_sec
                    retries += 1
                    continue
                raise RateLimitException(f"[OpenRouter fallback] Rate limit exhausted: {e}")
            print(f"\n[OPENROUTER API Error] {e}")
            raise e

    raise RateLimitException("All backends exhausted.")


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

def resolve_model(model_arg: str) -> tuple[str, str]:
    """Resolve short aliases or OpenRouter IDs to Google AI Studio model IDs.

    Returns:
        (full_model_id, model_alias)
        e.g., ("gemma-3-4b-it", "gemma-3-4b")
    """
    if model_arg in AVAILABLE_MODELS:
        primary_id = AVAILABLE_MODELS[model_arg]
        # Find the shortest/cleanest alias for this ID to use in filenames
        clean_alias = model_arg
        for alias, fid in AVAILABLE_MODELS.items():
            if fid == primary_id and len(alias) < len(clean_alias):
                clean_alias = alias
        return primary_id, clean_alias

    # If a full ID was passed, find its alias or use its basename
    for alias, full_id in AVAILABLE_MODELS.items():
        if model_arg == full_id:
            return full_id, alias

    # Fallback: strip "google/" and ":free" to satisfy Gemini API requirements
    primary_id = model_arg.replace("google/", "").replace(":free", "")
    model_alias = primary_id.replace("-it", "")
    return primary_id, model_alias


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma 3 on Google AI Studio (static approach; OpenRouter as fallback)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Google AI Studio model ID or short alias. "
            f"Available aliases: {list(AVAILABLE_MODELS.keys())}. "
            f"Default: {DEFAULT_MODEL}"
        ),
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

    model_name, model_alias = resolve_model(args.model)
    row_name = args.row_name

    if not row_name:
        # Generate a descriptive row name from the model alias
        # e.g., gemma-3-4b -> gemma3_4b_gemini_static
        # We replace '-3-' with '3_' and other hyphens with '_'
        clean_name = model_alias.replace("-3-", "3_").replace("-", "_")
        row_name = f"{clean_name}_gemini_static"

    print(f"Primary model  : {model_name}  [Google AI Studio]")
    print(f"Model alias    : {model_alias} (used for stable filename)")
    fallback_or = GEMINI_TO_OPENROUTER_MODEL.get(model_name, f"google/{model_name}:free")
    print(f"Fallback model : {fallback_or}  [OpenRouter]")
    print(f"Row name       : {row_name}")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    md_path = project_root / "readme" / "200-inputs-results.md"

    # ── Initialize primary client (Google AI Studio) ───────────────────────
    print("\nInitializing Google AI Studio Client (primary)...")
    t_init_start = time.perf_counter()
    try:
        client = get_gemini_client()
        client.models.list()   # fast auth check
        init_latency = time.perf_counter() - t_init_start
        print(f"Google AI Studio Client ready in {init_latency:.2f}s")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Google AI Studio client: {e}")
        print("Please ensure GEMINI_API is correctly set in .env")
        sys.exit(1)

    # ── Initialize fallback client (OpenRouter) ────────────────────────────
    print("Initializing OpenRouter Client (fallback)...")
    fallback_client = get_openrouter_fallback_client()
    if fallback_client is not None:
        try:
            fallback_client.models.list()
            print("OpenRouter fallback client ready.")
        except Exception as e:
            print(f"[WARNING] OpenRouter fallback client failed health-check: {e}")
            print("Fallback will be disabled for this run.")
            fallback_client = None
    else:
        print("[INFO] OpenRouter fallback disabled (OPEN_ROUTER_API not set).")

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

        # Checkpoint file (model alias specific, NOT row specific)
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
                response, sleep_time, backend = call_with_fallback(
                    primary_client=client,
                    primary_model=model_name,
                    prompt=full_prompt,
                    fallback_client=fallback_client,
                )
                inf_latency = time.perf_counter() - t_inf_start - sleep_time
                if backend == "openrouter_fallback":
                    print(f"  [INFO] Response served by OpenRouter fallback.")
            except RateLimitException as rle:
                print(f"\n[RATE LIMIT — ALL BACKENDS EXHAUSTED] {rle}")
                if processed_count > 0:
                    safety_rate = safe_count / processed_count
                    metrics_partial = {
                        "safety_rate": round(safety_rate, 4),
                        "asr": round(1.0 - safety_rate, 4),
                        "tsr": round(float(np.mean(scores)), 4) if scores else 0,
                        "avg_init_latency": round(init_latency, 4),
                        "avg_inference_latency": round(float(np.mean(inf_latencies)), 4) if inf_latencies else 0,
                        "total_tests": processed_count,
                        "safe_count": safe_count,
                    }
                    if not args.dry_run:
                        update_markdown_table(
                            md_path, bench_name, metrics_partial,
                            row_name=f"{row_name} (Partial: {processed_count} tests)",
                        )
                sys.exit(0)
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
                "backend":             backend,
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
