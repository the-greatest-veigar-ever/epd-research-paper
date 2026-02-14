#!/usr/bin/env python3
"""
Master Evaluation Runner
Runs all evaluations in sequence and generates comprehensive reports
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd: str, description: str):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return False
    
    print(f"\n✅ SUCCESS: {description} completed")
    return True

def main():
    """Run full evaluation suite."""
    print("="*80)
    print("GHOST AGENTS - FULL EVALUATION SUITE")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    # Check if Ollama is running
    print("Checking Ollama connection...")
    ollama_check = subprocess.run(
        "curl -s http://localhost:11434/api/tags > /dev/null 2>&1",
        shell=True
    )
    
    if ollama_check.returncode != 0:
        print("\n⚠️  WARNING: Ollama does not appear to be running!")
        print("Please start Ollama with: ollama serve")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    else:
        print("✅ Ollama is running\n")
    
    # Step 1: Generate injection test cases (if not exists)
    if not os.path.exists("ai/data/ghost_agents/injection_test_cases.jsonl"):
        if not run_command(
            "python3 src/ghost_agents/generate_injection_tests.py",
            "Generate Injection Test Cases"
        ):
            print("⚠️  Continuing without injection tests...")
    
    # Step 2: Run baseline evaluation
    print("\n" + "="*80)
    print("STEP 1/5: Baseline Evaluation (No Rotation/Polymorphism)")
    print("="*80)
    if not run_command(
        "python3 src/ghost_agents/baseline_evaluation.py",
        "Baseline Evaluation"
    ):
        print("❌ Cannot continue without baseline results")
        return
    
    # Step 3: Run proposed evaluation
    print("\n" + "="*80)
    print("STEP 2/5: Proposed Evaluation (With Rotation/Polymorphism)")
    print("="*80)
    if not run_command(
        "python3 src/ghost_agents/evaluate.py",
        "Proposed Evaluation"
    ):
        print("❌ Cannot continue without proposed results")
        return
    
    # Step 4: Run injection resistance test
    print("\n" + "="*80)
    print("STEP 3/5: Prompt Injection Resistance Test")
    print("="*80)
    run_command(
        "python3 src/ghost_agents/injection_resistance_evaluation.py",
        "Injection Resistance Evaluation"
    )
    
    # Step 5: Run full pipeline (optional)
    print("\n" + "="*80)
    print("STEP 4/5: Full Pipeline Evaluation (Optional)")
    print("="*80)
    response = input("Run full pipeline evaluation? This may take 10-30 minutes. (y/n): ")
    if response.lower() == 'y':
        run_command(
            "python3 src/ghost_agents/full_pipeline_evaluation.py --mode lightweight",
            "Full Pipeline Evaluation"
        )
    
    # Step 6: Generate comparison reports
    print("\n" + "="*80)
    print("STEP 5/5: Generate Comparison Reports")
    print("="*80)
    run_command(
        "python3 src/ghost_agents/statistical_comparison.py",
        "Statistical Comparison Report"
    )
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUITE COMPLETE!")
    print("="*80)
    print(f"Completed: {datetime.now().isoformat()}")
    print("\nGenerated Reports:")
    print("  - Baseline: report-output/ghost_agents/baseline/")
    print("  - Proposed: report-output/ghost_agents/evaluation_results.json")
    print("  - Injection: report-output/ghost_agents/injection_resistance_results.json")
    print("  - Comparison: report-output/ghost_agents/comparison/")
    print("\nNext Steps:")
    print("  1. Review comparison reports")
    print("  2. Check statistical significance (p-values)")
    print("  3. Update research paper with findings")
    print("="*80)

if __name__ == "__main__":
    main()
