import json
import os
import time
from typing import List, Dict
from src.ghost_agents.agent import GhostAgentFactory
from tqdm import tqdm

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
    print("\n--- Phase 1: Polymorphism & Initialization Test ---")
    
    polymorphism_count = 0
    valid_initialization_count = 0
    
    for scenario in tqdm(evaluation_set):
        # The 'prompt' in our dataset is the task/instruction
        user_intent = scenario.get("prompt", "Execute security task")
        
        # Create Agent
        try:
            agent = GhostAgentFactory.create_agent(user_intent)
            valid_initialization_count += 1
            
            # Check Polymorphism (System prompt should NOT be identical to user intent, 
            # and should generally be decorated/mutated by the Factory)
            # In our factory logic, the *instructions* are mutated.
            
            # We assume the factory appends "flavor" text. 
            # Simple check: prompt length > user_intent length
            is_polymorphic = len(agent.prompt) > len(user_intent) and agent.prompt != user_intent
            
            if is_polymorphic:
                polymorphism_count += 1
                
            results.append({
                "scenario_source": scenario.get("source"),
                "user_intent": user_intent[:50] + "...",
                "agent_id": agent.session_id,
                "generated_prompt_length": len(agent.prompt),
                "is_polymorphic": is_polymorphic
            })
            
        except Exception as e:
            results.append({
                "scenario_source": scenario.get("source"),
                "error": str(e),
                "status": "failed"
            })

    print("\n--- Phase 2: Results Analysis ---")
    success_rate = polymorphism_count / len(evaluation_set) if evaluation_set else 0
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
