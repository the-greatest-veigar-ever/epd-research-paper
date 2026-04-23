import os
import argparse
from src.ghost_agents.approach_evaluation.evaluator import ApproachEvaluator
from src.ghost_agents.approach_evaluation.security_evaluator import SecurityEvaluator

# The new dataset we generated
DEFAULT_DATASET = "ai/data/ghost_agents/squad_c_mixed_dataset_15pct.json"
OUTPUT_DIR_BASE = "report-output/ghost_agents/evaluation_results"

def run_comprehensive_evaluation(limit=None):
    print("=== SQUAD C COMPREHENSIVE EVALUATION (PERFORMANCE & SECURITY) ===")
    
    # We run the 4 requested modes:
    # 1 model static
    # 1 model suicide
    # 3 models static
    # 3 models suicide
    approaches = [
        "phi_baseline",       # 1 model static
        "phi_suicide",        # 1 model suicide
        "multimodal_static",  # 3 models static
        "multimodal_suicide"  # 3 models suicide
    ]
    
    # =========================================================================
    # PART 1: PERFORMANCE & EFFICIENCY EVALUATION
    # =========================================================================
    print("\n\n" + "="*80)
    print("  PHASE 1: PERFORMANCE & EFFICIENCY")
    print("  (Init Time, Processing Time, ASR, TSR, PASS@1)")
    print("="*80)
    
    perf_evaluator = ApproachEvaluator(
        dataset_path=DEFAULT_DATASET,
        limit=limit,
        output_dir=os.path.join(OUTPUT_DIR_BASE, "performance")
    )
    
    # The ApproachEvaluator already loops through approaches and saves 
    # results incrementally (and handles Ctrl+C partial saves).
    perf_results = perf_evaluator.run_all(approach_names=approaches)
    
    # =========================================================================
    # PART 2: SECURITY RESILIENCE EVALUATION
    # =========================================================================
    print("\n\n" + "="*80)
    print("  PHASE 2: SECURITY RESILIENCE")
    print("  (Prompt Injection, Harmful Rejection, Jailbreak, Context Isolation)")
    print("="*80)
    
    # The SecurityEvaluator specifically filters for records where is_injection=True
    # and calculates the resistance rates.
    sec_evaluator = SecurityEvaluator(
        dataset_path=DEFAULT_DATASET,
        limit=limit,
        output_dir=os.path.join(OUTPUT_DIR_BASE, "security")
    )
    
    sec_results = sec_evaluator.run_all(approach_names=approaches)
    
    print("\n\n=== COMPREHENSIVE EVALUATION COMPLETE ===")
    print(f"Metrics saved to: {OUTPUT_DIR_BASE}/")
    print("Review the _summary.json files in the performance/ and security/ subdirectories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Squad C Comprehensive Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Max plans to evaluate for quick testing")
    args = parser.parse_args()
    
    run_comprehensive_evaluation(limit=args.limit)
