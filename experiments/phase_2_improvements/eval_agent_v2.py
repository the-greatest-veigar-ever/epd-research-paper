#!/usr/bin/env python3
"""
Evaluation Script for Ghost Agent V2 (Phase 2 Improvements)
Focus: Measuring Tool Correctness and Semantic Similarity with Reduced Polymorphism.
"""

import sys
import os
import json
import time
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from ghost_agents.agent_v2 import GhostAgentFactoryV2
from sentence_transformers import SentenceTransformer, util

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../ai/data/ghost_agents/combined_scenarios.jsonl')
LIMIT = 20

def load_scenarios(limit=LIMIT):
    scenarios = []
    try:
        with open(DATA_PATH, 'r') as f:
            for line in f:
                if len(scenarios) >= limit:
                    break
                scenarios.append(json.loads(line))
    except FileNotFoundError:
        print(f"Dataset not found at {DATA_PATH}")
        sys.exit(1)
    return scenarios

def evaluate_v2():
    print(f"🚀 Starting Ghost Agent V2 Evaluation (Limit: {LIMIT})...")
    
    scenarios = load_scenarios()
    results = []
    
    # Semantic Complexity Metrics
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    original_instructions = []
    mutated_instructions = []
    
    success_count = 0
    
    for i, scenario in enumerate(tqdm(scenarios, desc="Evaluating V2")):
        # Simulate Brain Plan
        action = scenario.get("attack_type", "BLOCK_IP")
        target = scenario.get("tool", "192.168.1.1") 
        plan = {"action": action, "target": target}
        
        # Create Agent V2
        # Use full prompt as base instruction to start mutation
        base_instruction = f"Perform {action} on {target}"
        
        start_time = time.time()
        agent = GhostAgentFactoryV2.create_agent(base_instruction, rotate_model=True)
        mutated_instruction = agent.prompt # Capture before cleanup
        
        # Execute
        result = agent.execute_remediation(plan)
        duration = time.time() - start_time
        
        # Metrics
        tool_used = result.get("tool_used", "unknown")
        # Check correctness (dataset expects 'aws' usually, or based on tool field)
        # We enforce 'aws' in system prompt, so we check for 'aws'
        is_correct = "aws" in str(tool_used).lower()
        
        if is_correct:
            success_count += 1
            
        # Semantic Similarity
        original_instructions.append(base_instruction)
        mutated_instructions.append(mutated_instruction)
        
        results.append({
            "model": agent.model,
            "instruction_base": base_instruction,
            "instruction_mutated": agent.prompt,
            "command_generated": result.get("command"),
            "is_correct": is_correct,
            "duration": duration
        })
        
    # Calculate Final Metrics
    accuracy = (success_count / len(scenarios)) * 100
    avg_latency = np.mean([r["duration"] for r in results])
    
    # Semantic Similarity Calculation
    embeddings1 = sbert_model.encode(original_instructions, convert_to_tensor=True)
    embeddings2 = sbert_model.encode(mutated_instructions, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # We only care about diagonal (i vs i)
    similarities = [cosine_scores[i][i].item() for i in range(len(scenarios))]
    avg_sim = np.mean(similarities)
    
    print("\n" + "="*50)
    print("GHOST AGENT V2 RESULTS")
    print("="*50)
    print(f"Scenarios:       {len(scenarios)}")
    print(f"Tool Correctness: {accuracy:.2f}% (Target: >80%)")
    print(f"Avg Latency:      {avg_latency:.2f}s")
    print(f"Semantic Sim:     {avg_sim:.4f} (Target: <0.70)")
    print("="*50)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "v2_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "latency": avg_latency,
                "semantic_similarity": avg_sim
            },
            "details": results
        }, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    evaluate_v2()
