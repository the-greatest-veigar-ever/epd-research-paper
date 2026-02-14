#!/usr/bin/env python3
"""
Baseline Evaluation for Ghost Agents
Tests 3 SLMs WITHOUT rotation or polymorphism
"""

import json
import os
import time
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Import agent
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ghost_agents.agent import GhostAgentFactory

# Initialize SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# CONFIG
DATASET_PATH = "ai/data/ghost_agents/combined_scenarios.jsonl"
REPORT_OUTPUT_DIR = "report-output/ghost_agents/baseline"
MODELS = ["llama3.2:3b", "phi", "gemma2:2b"]

def evaluate_single_model(model_name: str, scenarios: List[Dict]) -> Dict:
    """Evaluate a single model without rotation."""
    print(f"\n{'='*60}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*60}")
    
    results = []
    semantic_scores = []
    levenshtein_scores = []
    tool_correct_count = 0
    valid_exec_count = 0
    
    # Limit for testing if indicated
    if len(scenarios) > 200:
        scenarios = scenarios[:200]
        
    for scenario in tqdm(scenarios, desc=f"{model_name}"):
        user_intent = scenario.get("prompt", "Execute security task")
        expected_tool = scenario.get("expected_tool", "aws")
        
        try:
            # Create agent WITHOUT rotation (baseline)
            agent = GhostAgentFactory.create_agent(user_intent, rotate_model=False)
            # Force specific model for baseline
            agent.model = model_name
            
            mutated_prompt = agent.prompt
            
            # Semantic similarity
            embeddings = sbert_model.encode([user_intent, mutated_prompt])
            sem_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            semantic_scores.append(sem_sim)
            
            # Levenshtein distance
            lev_dist = levenshtein_distance(user_intent, mutated_prompt)
            levenshtein_scores.append(lev_dist)
            
            # Execute
            plan = {
                "action": scenario.get("attack_type", "Mitigate Threat"),
                "target": scenario.get("tool", "System-ID-1234")
            }
            
            start_t = time.time()
            exec_result = agent.execute_remediation(plan)
            duration = time.time() - start_t
            
            # Check correctness
            is_valid_exec = exec_result["status"] in ["success", "simulated_success"]
            tool_used = exec_result.get("tool_used", "unknown")
            is_tool_correct = expected_tool in str(tool_used) if tool_used else False
            
            if is_valid_exec:
                valid_exec_count += 1
            if is_tool_correct:
                tool_correct_count += 1
            
            results.append({
                "scenario_source": scenario.get("source"),
                "semantic_similarity": float(sem_sim),
                "levenshtein_distance": int(lev_dist),
                "inference_duration": duration,
                "execution_status": exec_result["status"],
                "tool_used": tool_used,
                "is_tool_correct": is_tool_correct
            })
            
        except Exception as e:
            results.append({
                "error": str(e),
                "status": "failed"
            })
    
    # Calculate metrics
    total = len(scenarios)
    asr = valid_exec_count / total
    tool_correctness = tool_correct_count / total
    avg_semantic = np.mean(semantic_scores) if semantic_scores else 0
    avg_levenshtein = np.mean(levenshtein_scores) if levenshtein_scores else 0
    
    return {
        "model": model_name,
        "total_scenarios": total,
        "metrics": {
            "asr": asr,
            "tool_correctness": tool_correctness,
            "avg_semantic_similarity": float(avg_semantic),
            "avg_levenshtein_distance": float(avg_levenshtein)
        },
        "detailed_results": results
    }

def run_baseline_evaluation():
    """Run baseline evaluation for all models."""
    print("="*60)
    print("BASELINE EVALUATION - No Rotation, No Polymorphism")
    print("="*60)
    
    # Load scenarios
    scenarios = []
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    scenarios.append(json.loads(line))
        print(f"Loaded {len(scenarios)} scenarios")
    else:
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    # Evaluate each model
    model_results = {}
    for model in MODELS:
        result = evaluate_single_model(model, scenarios)
        model_results[model] = result
        
        print(f"\n{model} Results:")
        print(f"  ASR: {result['metrics']['asr']*100:.2f}%")
        print(f"  Tool Correctness: {result['metrics']['tool_correctness']*100:.2f}%")
        print(f"  Semantic Similarity: {result['metrics']['avg_semantic_similarity']:.4f}")
        print(f"  Levenshtein Distance: {result['metrics']['avg_levenshtein_distance']:.2f}")
    
    # Calculate averages across models
    avg_metrics = {
        "asr": np.mean([r["metrics"]["asr"] for r in model_results.values()]),
        "tool_correctness": np.mean([r["metrics"]["tool_correctness"] for r in model_results.values()]),
        "avg_semantic_similarity": np.mean([r["metrics"]["avg_semantic_similarity"] for r in model_results.values()]),
        "avg_levenshtein_distance": np.mean([r["metrics"]["avg_levenshtein_distance"] for r in model_results.values()])
    }
    
    # Save results
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(REPORT_OUTPUT_DIR, "baseline_evaluation.json")
    
    final_report = {
        "evaluation_type": "baseline",
        "configuration": "no_rotation_no_polymorphism",
        "models": MODELS,
        "total_scenarios": len(scenarios),
        "averaged_metrics": avg_metrics,
        "model_results": model_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Baseline evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"\nAveraged Metrics Across All Models:")
    print(f"  ASR: {avg_metrics['asr']*100:.2f}%")
    print(f"  Tool Correctness: {avg_metrics['tool_correctness']*100:.2f}%")
    print(f"  Semantic Similarity: {avg_metrics['avg_semantic_similarity']:.4f}")
    print(f"  Levenshtein Distance: {avg_metrics['avg_levenshtein_distance']:.2f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_baseline_evaluation()
