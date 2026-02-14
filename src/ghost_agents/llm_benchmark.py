
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any
import requests
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shared utilities
try:
    from src.ghost_agents.agent import GhostAgent
except ImportError:
    # Local run fallback
    sys.path.insert(0, os.path.abspath("."))
    from src.ghost_agents.agent import GhostAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIG
DATA_PATH = "ai/data/ghost_agents/combined_scenarios.jsonl"
REPORT_OUTPUT_DIR = "report-output/ghost_agents/llm_benchmark"
LLM_MODEL = "llama3"  # The LLM baseline model (8B parameters)

def load_scenarios(limit: int = None) -> List[Dict]:
    """Load evaluation scenarios."""
    scenarios = []
    try:
        if not os.path.exists(DATA_PATH):
            logger.error(f"Data file not found: {DATA_PATH}")
            return []
            
        with open(DATA_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    scenarios.append(json.loads(line))
                    if limit and len(scenarios) >= limit:
                        break
        return scenarios
    except Exception as e:
        logger.error(f"Error loading scenarios: {e}")
        return []

def evaluate_llm(model_name: str, scenarios: List[Dict]) -> Dict:
    """Evaluate a single large language model."""
    logger.info(f"Starting benchmark for LLM: {model_name}")
    
    metrics = {
        "latencies": [],
        "success_count": 0,
        "tool_correctness": 0,
        "errors": []
    }
    
    results = []
    
    for scenario in tqdm(scenarios, desc=f"Benchmarking {model_name}"):
        user_intent = scenario.get("prompt")
        if not user_intent:
            continue
            
        # Create a direct agent without rotation/polymorphism
        # We simulate the "LLM" approach: Just ask the model directly without the Ghost wrappers
        
        start_time = time.time()
        
        try:
            # Direct Ollama call to simulate standard LLM agent
            prompt = f"""You are a security automation agent.
Task: Generate a command to execute the following request.
Request: {user_intent}

Rules:
1. Output ONLY the command.
2. Do not use markdown.
3. Start with 'aws' or the appropriate tool.

Command:"""
            
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1 # Low temp for precision
                }
            }, timeout=60)
            
            latency = time.time() - start_time
            metrics["latencies"].append(latency)
            
            if response.status_code == 200:
                cmd = response.json().get("response", "").strip()
                
                # Check tool correctness
                tool_used = cmd.split()[0].lower() if cmd else "none"
                is_correct = "aws" in tool_used
                if is_correct:
                    metrics["tool_correctness"] += 1
                
                metrics["success_count"] += 1
                
                results.append({
                    "scenario_id": scenario.get("flow_id", "unknown"),
                    "action": plan.get("action"),
                    "target": plan.get("target"),
                    "latency": latency,
                    "command": cmd,
                    "tool_correctness": is_correct
                })
            else:
                metrics["errors"].append(f"HTTP {response.status_code}")
                
        except Exception as e:
            metrics["errors"].append(str(e))
    
    # Calculate summary stats
    avg_latency = np.mean(metrics["latencies"]) if metrics["latencies"] else 0
    p95_latency = np.percentile(metrics["latencies"], 95) if metrics["latencies"] else 0
    accuracy = metrics["tool_correctness"] / len(scenarios) if scenarios else 0
    
    return {
        "model": model_name,
        "total_requests": len(scenarios),
        "success_rate": metrics["success_count"] / len(scenarios) if scenarios else 0,
        "avg_latency_s": float(avg_latency),
        "p95_latency_s": float(p95_latency),
        "tool_correctness": accuracy,
        "detailed_results": results,
        "errors": metrics["errors"]
    }

def run_llm_benchmark(limit: int = 200):
    """Run the benchmark."""
    print(f"Loading scenarios (limit={limit})...")
    scenarios = load_scenarios(limit)
    if not scenarios:
        print("No scenarios found. Exiting.")
        return
    
    print(f"Benchmarking LLM: {LLM_MODEL}...")
    result = evaluate_llm(LLM_MODEL, scenarios)
    
    # Save report
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(REPORT_OUTPUT_DIR, f"llm_benchmark_{LLM_MODEL}_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    print(f"\n✅ Benchmark Complete!")
    print(f"Model: {LLM_MODEL}")
    print(f"Average Latency: {result['avg_latency_s']:.4f}s")
    print(f"Tool Correctness: {result['tool_correctness']:.2%}")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark for EPD Research")
    parser.add_argument("--limit", type=int, default=200, help="Number of scenarios to run")
    parser.add_argument("--model", type=str, default=LLM_MODEL, help="Model to benchmark")
    args = parser.parse_args()
    
    LLM_MODEL = args.model
    run_llm_benchmark(limit=args.limit)
