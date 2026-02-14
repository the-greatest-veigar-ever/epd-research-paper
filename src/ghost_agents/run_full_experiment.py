#!/usr/bin/env python3
"""
Full Experiment Orchestrator
Executes the complete Ghost Agents evaluation suite:
1. Baseline Evaluation (No rotation/polymorphism)
2. Proposed Evaluation (Full pipeline with Ephemerality/Polymorphism)
3. LLM Benchmark (Monolithic LLM comparison)
4. Statistical Comparison (Consolidated report)
"""

import os
import sys
import subprocess
import time
import argparse
import requests

# CONFIG
OLLAMA_HOST = "http://localhost:11434"
REQUIRED_MODELS = ["phi", "llama3.2:3b", "gemma2:2b"]
LLM_BENCHMARK_MODEL = "llama3"

def check_models():
    """Verify all required models are available in Ollama."""
    print("Checking model availability...")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        if response.status_code != 200:
            print(f"❌ Error connecting to Ollama: {response.status_code}")
            return False
            
        available_models = [m["name"].split(":")[0] for m in response.json()["models"]]
        available_full_names = [m["name"] for m in response.json()["models"]]
        
        missing = []
        for model in REQUIRED_MODELS:
            # Check for exact match or base name match
            if model not in available_full_names and model.split(":")[0] not in available_models:
                missing.append(model)
        
        if LLM_BENCHMARK_MODEL not in available_full_names and LLM_BENCHMARK_MODEL.split(":")[0] not in available_models:
             print(f"⚠️  LLM Benchmark model '{LLM_BENCHMARK_MODEL}' not found. Will fallback or fail.")
             
        if missing:
            print(f"❌ Missing required models: {', '.join(missing)}")
            print("Please run: ollama pull <model_name>")
            return False
            
        print("✅ All required SLMs available.")
        return True
        
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")
        return False

def run_command(cmd, desc):
    """Run a shell command with formatted output."""
    print(f"\n{'='*80}")
    print(f"STEP: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {desc} (Exit Code: {result.returncode})")
        return False
    
    print(f"\n✅ Completed: {desc} in {duration:.2f}s")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run full Ghost Agents experiment")
    parser.add_argument("--skip-check", action="store_true", help="Skip model availability check")
    parser.add_argument("--limit", type=int, default=200, help="Limit number of scenarios (Research Mode)")
    parser.add_argument("--dry-run", action="store_true", help="Run with limit=5 for testing")
    args = parser.parse_args()
    
    limit = 5 if args.dry_run else args.limit
    
    # 1. Check Models
    if not args.skip_check:
        if not check_models():
            response = input("⚠️  Models missing. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)

    # 2. Baseline Evaluation
    if not run_command(f"python3 src/ghost_agents/baseline_evaluation.py", "Baseline Evaluation"):
        sys.exit(1)
        
    # 3. Proposed Evaluation (Research Mode)
    if not run_command(f"python3 src/ghost_agents/full_pipeline_evaluation.py --mode research --limit {limit}", "Proposed Evaluation"):
        sys.exit(1)
        
    # 4. LLM Benchmark
    # Check if llama3 is available, else fallback
    llm_model = LLM_BENCHMARK_MODEL
    try:
        # Simple check if llama3 exists, else use llama3.2:3b
        res = requests.get(f"{OLLAMA_HOST}/api/tags")
        models = [m["name"] for m in res.json().get("models", [])]
        if "llama3:latest" not in models and "llama3" not in models:
            print(f"⚠️  '{LLM_BENCHMARK_MODEL}' not found. Falling back to 'llama3.2:3b' for benchmark.")
            llm_model = "llama3.2:3b"
    except:
        pass
        
    if not run_command(f"python3 src/ghost_agents/llm_benchmark.py --model {llm_model} --limit {limit}", f"LLM Benchmark ({llm_model})"):
        print("⚠️  LLM Benchmark failed. Continuing to comparison...")
    
    # 5. Statistical Comparison & Report
    if not run_command(f"python3 src/ghost_agents/statistical_comparison.py", "Statistical Comparison Report"):
        sys.exit(1)
        
    print("\n✅ FULL EXPERIMENT COMPLETE")
    print("Check report-output/ghost_agents/comparison/ for final results.")

if __name__ == "__main__":
    main()
