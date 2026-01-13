"""
EPD Master Reporting Tool

This script orchestrates the full evaluation pipeline for the EPD architecture.
It runs simulations for Groups 1, 2, and 3, aggregates the key metrics,
and appends them to a master Excel history file for tracking progress.

Author: EPD Research Team
Version: 1.0.0
"""

import os
import time
import pandas as pd
from typing import Dict, Any

# Import Refactored Modules
from research_sim import run_group1_simulation
from eval_brain_mmlu import run_group2_evaluation
from sim_ghost_bench import run_group3_simulation

HISTORY_FILE = "Simulation_History.xlsx"

def append_to_history(data: Dict[str, Any]):
    """Appends a new run record to the master Excel file."""
    
    # Prepare Row Data
    record = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Run_ID": int(time.time()),
        
        # Group 1 (Watchers)
        "G1_Baseline_Defense(%)": data.get("g1", {}).get("baseline_defense_rate", 0),
        "G1_EPD_Defense(%)":      data.get("g1", {}).get("epd_defense_rate", 0),
        "G1_Improvement(%)":      data.get("g1", {}).get("improvement", 0),
        
        # Group 2 (Brain)
        "G2_MMLU_Accuracy(%)":    data.get("g2", {}).get("mmlu_average_accuracy", 0),
        
        # Group 3 (Ghost Agents)
        "G3_Baseline_Resilience(%)": data.get("g3", {}).get("asb_baseline_rate", 0),
        "G3_EPD_Resilience(%)":      data.get("g3", {}).get("asb_epd_rate", 0),
        "G3_Improvement(%)":         data.get("g3", {}).get("improvement", 0),
    }
    
    df_new = pd.DataFrame([record])
    
    if os.path.exists(HISTORY_FILE):
        try:
            df_old = pd.read_excel(HISTORY_FILE)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new
        
    df_final.to_excel(HISTORY_FILE, index=False)
    print(f"\n[Success] Results appended to {HISTORY_FILE}")
    print(df_final.tail(1).to_string(index=False))

def main():
    print("=== EPD FULL EVALUATION SUITE STARTED ===\n")
    
    all_results = {}
    
    # 1. Run Group 1
    print(">>> Running Group 1 (Watchers)...")
    all_results["g1"] = run_group1_simulation()
    
    # 2. Run Group 2
    print("\n>>> Running Group 2 (Brain - MMLU)...")
    all_results["g2"] = run_group2_evaluation()
    
    # 3. Run Group 3
    print("\n>>> Running Group 3 (Ghost Agents)...")
    all_results["g3"] = run_group3_simulation()
    
    # 4. Report
    print("\n>>> Generating Master Report...")
    append_to_history(all_results)
    
    print("\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
