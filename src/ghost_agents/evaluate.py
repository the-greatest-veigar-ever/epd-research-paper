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
        
    print(f"Selected {len(evaluation_set)} scenarios for evaluation.")
    
    results = []
    
    # Analysis Loop
    # Analysis Loop
    print("\n--- Phase 1: Polymorphism & Inference Test (With Semantic Metrics) ---")
    
    # Results containers
    polymorphism_scores = [] # SBERT Cosine Similarity
    edit_distances = []     # Levenshtein
    valid_executions = 0
    
    for scenario in tqdm(evaluation_set):
        # The 'prompt' in our dataset is the task/instruction
        user_intent = scenario.get("prompt", "Execute security task")
        
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
            agent.execute_remediation(plan)
            duration = time.time() - start_t
            
            valid_executions += 1
                
            results.append({
                "scenario_source": scenario.get("source"),
                "user_intent": user_intent[:50] + "...",
                "mutated_prompt_len": len(mutated_prompt),
                "semantic_similarity": float(sem_sim),
                "levenshtein_distance": int(lev_dist),
                "inference_duration": duration
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
    success_rate = valid_executions / len(evaluation_set) if evaluation_set else 0
    
    print(f"Execution Success: {success_rate*100:.2f}%")
    print(f"Avg Semantic Similarity (Intent Preservation): {avg_semantic:.4f} (Target > 0.7)")
    print(f"Avg Levenshtein Distance (Obfuscation Degree): {avg_levenshtein:.2f} (Target > 20)")

    # Construct Final Report
    final_report = {
        "summary": {
            "total_scenarios": len(evaluation_set),
            "execution_success_rate": success_rate,
            "metrics": {
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
    print(f"Polymorphism Success: {polymorphism_count}/{len(evaluation_set)} ({success_rate:.1%})")
    print(f"Initialization Success: {valid_initialization_count}/{len(evaluation_set)}")
    
    # Save Report
    report = {
        "timestamp": time.time(),
        "total_scenarios": len(evaluation_set),
        "dataset_source": DATASET_PATH,
        "polymorphism_rate": success_rate,
        "details": results
    }
    
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nEvaluation Complete. Report saved to {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    run_evaluation()
