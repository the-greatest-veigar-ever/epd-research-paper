#!/usr/bin/env python3
"""
Enhanced Comparison Report for Ghost Agents V2
Phase 2: Verifying Reduced Polymorphism + One-Shot Prompting
"""

import json
import os
import numpy as np
from datetime import datetime
from scipy import stats
from typing import Dict, List

# V2 PATHS
BASELINE_PATH = "report-output/ghost_agents/baseline/baseline_evaluation.json"
V2_PROPOSED_PATH = "experiments/phase_2_improvements/results/proposed_v2_improved_research_20260214_140448.json"
LLM_BENCHMARK_PATH = "report-output/ghost_agents/llm_benchmark/llm_benchmark_llama3_20260214_115806.json"
OUTPUT_DIR = "experiments/phase_2_improvements/comparison"

def calculate_statistical_significance(baseline_values: List[float], proposed_values: List[float]) -> Dict:
    """Calculate statistical significance between two distributions."""
    if not baseline_values or not proposed_values:
        return {}
        
    # T-test (parametric)
    t_stat, t_pvalue = stats.ttest_ind(baseline_values, proposed_values)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(baseline_values, proposed_values, alternative='two-sided')
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(proposed_values) - np.mean(baseline_values)
    pooled_std = np.sqrt((np.std(baseline_values)**2 + np.std(proposed_values)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_pvalue": float(u_pvalue),
        "cohens_d": float(cohens_d),
        "is_significant_005": bool(t_pvalue < 0.05),
        "is_significant_001": bool(t_pvalue < 0.01)
    }

def generate_v2_comparison():
    """Generate comparison for V2 with statistical tests."""
    print("="*80)
    print("ENHANCED V2 COMPARISON REPORT")
    print("="*80)
    
    # Load data
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    with open(V2_PROPOSED_PATH, 'r') as f:
        v2_proposed = json.load(f)
    with open(LLM_BENCHMARK_PATH, 'r') as f:
        llm_data = json.load(f)
    
    # Extract metrics
    baseline_avg = baseline.get("averaged_metrics", {})
    v2_metrics = v2_proposed.get("summary", {}).get("rates", {})
    
    # Collect detailed results
    baseline_results = []
    for model_data in baseline.get("model_results", {}).values():
        baseline_results.extend(model_data.get("detailed_results", []))
    v2_results = v2_proposed.get("details", [])
    
    # Extract latencies
    baseline_latencies = [r.get("inference_duration", 0) for r in baseline_results if "inference_duration" in r]
    v2_latencies = [r.get("latency", 0) for r in v2_results if "latency" in r]
    
    # Extract semantic similarities
    baseline_semantic = [r.get("semantic_similarity", 0) for r in baseline_results if "semantic_similarity" in r]
    v2_semantic = [r.get("semantic_sim", 0) for r in v2_results if "semantic_sim" in r]
    
    # Statistical tests
    stats_results = {}
    stats_results["latency"] = calculate_statistical_significance(baseline_latencies, v2_latencies)
    stats_results["semantic_similarity"] = calculate_statistical_significance(baseline_semantic, v2_semantic)
    
    # Generate report structure
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "averaged_metrics": baseline_avg,
            "total_scenarios": baseline.get("total_scenarios", 0)
        },
        "v2_proposed": {
            "metrics": v2_metrics,
            "total_scenarios": v2_proposed.get("summary", {}).get("total_scenarios", 0)
        },
        "statistical_tests": stats_results,
        "deltas_vs_baseline": {
            "tool_correctness": v2_metrics.get("tool_correctness_rate", 0) - baseline_avg.get("tool_correctness", 0),
            "semantic_similarity": v2_metrics.get("avg_semantic_similarity", 0) - baseline_avg.get("avg_semantic_similarity", 0),
            "latency": np.mean(v2_latencies) - np.mean(baseline_latencies) if v2_latencies and baseline_latencies else 0
        },
        "llm_comparison": {
            "model": llm_data.get("model"),
            "metrics": {
                "avg_latency_s": llm_data.get("avg_latency_s"),
                "tool_correctness": llm_data.get("tool_correctness")
            },
            "deltas_vs_v2": {
                "latency": np.mean(v2_latencies) - llm_data.get("avg_latency_s", 0) if v2_latencies else 0,
                "tool_correctness": v2_metrics.get("tool_correctness_rate", 0) - llm_data.get("tool_correctness", 0)
            }
        }
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"statistical_comparison_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ V2 Statistical Report saved to: {output_path}")
    
    # Text summary for terminal
    deltas = output_data["deltas_vs_baseline"]
    print("\n" + "-"*40)
    print(f"Tool Correctness: {v2_metrics.get('tool_correctness_rate')*100:.2f}% (V2) vs {baseline_avg.get('tool_correctness')*100:.2f}% (Baseline)")
    print(f"Avg Semantic Sim: {v2_metrics.get('avg_semantic_similarity'):.4f} (V2) vs {baseline_avg.get('avg_semantic_similarity'):.4f} (Baseline)")
    print(f"Significant Evasion Improvement: {'✅ YES' if stats_results['semantic_similarity'].get('is_significant_005') and deltas['semantic_similarity'] < 0 else '❌ NO'}")
    print("-" * 40)

if __name__ == "__main__":
    generate_v2_comparison()
