#!/usr/bin/env python3
"""
Prepare Mixed Dataset for Squad C Evaluation

Extracts clean STIX output from CIC integration results and injects a percentage
of adversarial test cases to create a mixed dataset for comprehensive Performance
and Security Resilience evaluation.
"""

import json
import os
import random
import uuid
from typing import List, Dict, Any

from src.ghost_agents.approach_evaluation.security_test_data import (
    CYBERSECEVAL_TESTS,
    HARMBENCH_TESTS,
    JAILBREAKBENCH_TESTS,
)

# Input path (Squad B Output)
INPUT_CIC_RESULTS = "report-output/integration_tests/cic_integration_results_20260216_011439.json"

# Output paths
CLEAN_OUTPUT = "ai/data/ghost_agents/squad_c_clean_inputs.json"
MIXED_OUTPUT = "ai/data/ghost_agents/squad_c_mixed_dataset_15pct.json"

# Injection Target
INJECTION_PERCENTAGE = 0.15


def load_clean_plans() -> List[Dict[str, Any]]:
    """Load the normal STIX courses of action from the CIC test results."""
    print(f"Loading base dataset from: {INPUT_CIC_RESULTS}")
    
    if not os.path.exists(INPUT_CIC_RESULTS):
        print(f"Error: Could not find input file {INPUT_CIC_RESULTS}")
        return []
        
    with open(INPUT_CIC_RESULTS, 'r') as f:
        data = json.load(f)
        
    stix_objects = data.get("stix_objects", [])
    
    plans = []
    for obj in stix_objects:
        if obj.get("type") == "course-of-action":
            # Extract relevant fields into a clean plan structure
            plans.append({
                "id": obj.get("id"),
                "action": obj.get("x_epd_action"),
                "target": obj.get("x_epd_target"),
                "score": obj.get("x_epd_score", 0.0),
                "is_injection": False
            })
            
    print(f"Successfully loaded {len(plans)} clean plans.")
    return plans


def create_mixed_dataset(clean_plans: List[Dict[str, Any]], injection_rate: float) -> List[Dict[str, Any]]:
    """Inject adversarial tests into the clean plan dataset at the specified rate."""
    num_clean = len(clean_plans)
    
    # Calculate how many malicious plans we need to reach the target percentage
    # (injection_count / total_count = injection_rate)
    # Target Equation: injection_count = (num_clean * injection_rate) / (1 - injection_rate)
    num_to_inject = int((num_clean * injection_rate) / (1 - injection_rate))
    
    print(f"Target injection rate: {injection_rate * 100}%")
    print(f"Number of clean plans: {num_clean}")
    print(f"Number of adversarial tests to inject: {num_to_inject}")
    
    # Combine all security frameworks into one pool
    all_attacks = CYBERSECEVAL_TESTS + HARMBENCH_TESTS + JAILBREAKBENCH_TESTS
    
    if not all_attacks:
        print("Warning: No attacks found in security_test_data.py")
        return clean_plans
        
    injected_plans = []
    
    # Randomly sample the attack pool enough times to hit our target number
    for _ in range(num_to_inject):
        attack = random.choice(all_attacks)
        
        # Structure the attack to look like a STIX plan so evaluate.py can handle it easily
        injected_plans.append({
            "id": f"course-of-action--{uuid.uuid4()}",
            "action": attack["action"],
            "target": attack["target"],
            "score": 0.0,
            
            # Security framework flags used by evaluator
            "is_injection": True,
            "attack_id": attack["id"],
            "framework": attack["framework"],
            "attack_type": attack["attack_type"]
        })
        
    # Combine and shuffle so attacks are distributed throughout the dataset
    mixed_dataset = clean_plans + injected_plans
    random.shuffle(mixed_dataset)
    
    print(f"Final mixed dataset size: {len(mixed_dataset)}")
    return mixed_dataset


def main():
    print("=== Squad C Dataset Preparer ===")
    
    # 1. Load clean data
    clean_plans = load_clean_plans()
    if not clean_plans:
        return
        
    os.makedirs(os.path.dirname(CLEAN_OUTPUT), exist_ok=True)
    
    # 2. Save pure STIX plans (new input source)
    print(f"\nSaving clean dataset to: {CLEAN_OUTPUT}")
    with open(CLEAN_OUTPUT, 'w') as f:
        json.dump(clean_plans, f, indent=2)
        
    # 3. Create mixed dataset
    print(f"\nCreating mixed dataset (target 15% adversarial)...")
    mixed_plans = create_mixed_dataset(clean_plans, INJECTION_PERCENTAGE)
    
    # 4. Save mixed dataset
    print(f"\nSaving mixed dataset to: {MIXED_OUTPUT}")
    with open(MIXED_OUTPUT, 'w') as f:
        json.dump(mixed_plans, f, indent=2)
        
    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
