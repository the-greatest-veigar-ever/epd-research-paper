import json
import os
import time
import numpy as np
from typing import List, Dict
from src.ghost_agents.agent import GhostAgentFactory
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Standard SBERT Model globally
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# CONFIG
REPORT_OUTPUT_PATH = "report-output/ghost_agents/evaluation_results.json"
DATASET_PATH = "ai/data/ghost_agents/combined_scenarios.jsonl"
MODELS_TO_TEST = ["llama3.2:3b"] 

def run_evaluation():
    print("=== STARTING GHOST AGENT EVALUATION (EXPANDED DATASET) ===")
    
    # Load Scenarios
    all_scenarios = []
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    all_scenarios.append(json.loads(line))
        print(f"Loaded {len(all_scenarios)} scenarios from {DATASET_PATH}")
    else:
        print(f"Dataset not found at {DATASET_PATH}. Falling back to default.")
        # Fallback Scenarios
        all_scenarios = [
            {"prompt": "Remediate security threat immediately", "source": "default"}
        ] * 5

    # Run ALL Scenarios
    evaluation_set = all_scenarios
        
    print(f"Selected {len(evaluation_set)} scenarios for evaluation (Full Run).")
    
    results = []
    
    # Analysis Loop
    print("\n--- Phase 1: Polymorphism & Inference Test (With Semantic Metrics) ---")
    
    # Results containers
    polymorphism_scores = [] # SBERT Cosine Similarity
    edit_distances = []     # Levenshtein
    # valid_executions = 0 # Replaced by more granular tracking
    
    # Metric Counters
    total_valid_executions = 0 # Execution didn't crash
    total_tool_correct = 0 # Used correct tool (e.g. 'aws')
    total_task_success = 0 # Executed AND Correct Tool (Strict)
    
    for scenario in tqdm(evaluation_set):
        # The 'prompt' in our dataset is the task/instruction
        user_intent = scenario.get("prompt", "Execute security task")
        expected_tool = scenario.get("expected_tool", "aws") # Default expectation
        
        try:
            # 1. GENERATION (Polymorphism)
            agent = GhostAgentFactory.create_agent(user_intent)
            mutated_prompt = agent.prompt
            
            # --- ACADEMIC METRICS CALCULATION ---
            
            # A. Semantic Similarity (Do they mean the same thing?)
            embeddings = sbert_model.encode([user_intent, mutated_prompt])
            sem_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            polymorphism_scores.append(sem_sim)
            
            # B. Levenshtein Distance (How much did the text change?)
            lev_dist = levenshtein_distance(user_intent, mutated_prompt)
            edit_distances.append(lev_dist)
                
            # 2. INFERENCE EXECUTION
            plan = {
                "action": scenario.get("attack_type", "Mitigate Threat"),
                "target": scenario.get("tool", "System-ID-1234")
            }
            
            start_t = time.time()
            exec_result = agent.execute_remediation(plan) # Returns dict now
            duration = time.time() - start_t
            
            # 3. VERIFICATION (Squad C Metrics)
            is_valid_exec = (exec_result["status"] == "success" or exec_result["status"] == "simulated_success")
            
            # Tool Correctness Check
            # We check if the tool used (e.g. 'aws') matches expectation
            tool_used = exec_result.get("tool_used", "unknown")
            is_tool_correct = False
            if tool_used and expected_tool in tool_used:
                is_tool_correct = True
            
            if is_valid_exec:
                total_valid_executions += 1
                
            if is_tool_correct:
                total_tool_correct += 1
                
            if is_valid_exec and is_tool_correct:
                total_task_success += 1
                
            results.append({
                "scenario_source": scenario.get("source"),
                "agent_context": scenario.get("agent_context", "unknown"),
                "user_intent": user_intent[:50] + "...",
                "mutated_prompt_len": len(mutated_prompt),
                "semantic_similarity": float(sem_sim),
                "levenshtein_distance": int(lev_dist),
                "inference_duration": duration,
                "execution_status": exec_result["status"],
                "tool_used": tool_used,
                "is_tool_correct": is_tool_correct
            })
            
        except Exception as e:
            results.append({
                "scenario_source": scenario.get("source"),
                "error": str(e),
                "status": "failed"
            })

    print("\n--- Phase 2: Results Analysis ---")
    
    avg_semantic = np.mean(polymorphism_scores) if polymorphism_scores else 0
    avg_levenshtein = np.mean(edit_distances) if edit_distances else 0
    
    total_scenarios = len(evaluation_set) if evaluation_set else 1
    
    # METRICS RENAME: Execution Success -> Attack Success Rate (ASR)
    asr = total_valid_executions / total_scenarios
    
    # Task Success Rate (TSR) - Strict success
    tsr = total_task_success / total_scenarios
    
    # Tool Correctness Rate
    tool_correctness_rate = total_tool_correct / total_scenarios
    
    # Pass@1 is essentially ASR in this single-pass context
    pass_at_1 = asr 
    
    print(f"Attack Success Rate (ASR): {asr*100:.2f}% (Target: 100%)")
    print(f"Task Success Rate (TSR): {tsr*100:.2f}% (Target: >90%)")
    print(f"Tool Correctness: {tool_correctness_rate*100:.2f}%")
    print(f"Pass@1 (Code Reliability): {pass_at_1*100:.2f}% (Target: >80%)")
    print(f"Avg Semantic Similarity (Intent Preservation): {avg_semantic:.4f} (Target > 0.7)")
    print(f"Avg Levenshtein Distance (Obfuscation Degree): {avg_levenshtein:.2f} (Target > 20)")

    # --- Group by Role ---
    role_stats = {}
    for res in results:
        role = res.get("agent_context", "unknown")
        if role not in role_stats:
            role_stats[role] = {"count": 0, "success": 0}
        role_stats[role]["count"] += 1
        
        if res.get("execution_status") == "success": # checking explicit success
             role_stats[role]["success"] += 1
            
    print("\n--- Role Performance Breakdown ---")
    print(f"{'Role':<30} | {'ASR':<15} | {'Samples':<10}")
    print("-" * 60)
    for role, stats in role_stats.items():
        rate = (stats["success"] / stats["count"]) * 100 if stats["count"] > 0 else 0
        print(f"{role:<30} | {rate:6.2f}%        | {stats['count']:<10}")


    # Construct Final Report
    final_report = {
        "summary": {
            "total_scenarios": total_scenarios,
            "metrics": {
                "attack_success_rate": asr,
                "task_success_rate": tsr,
                "tool_correctness_rate": tool_correctness_rate,
                "pass_at_1": pass_at_1,
                "semantic_similarity_avg": float(avg_semantic),
                "levenshtein_distance_avg": float(avg_levenshtein)
            }
        },
        "detailed_results": results
    }

    # Save Results
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Evaluation Complete. Report saved to {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    run_evaluation()
