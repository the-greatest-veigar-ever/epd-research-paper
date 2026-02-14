#!/usr/bin/env python3
"""
Enhanced Comparison Report with Statistical Significance Tests
"""

import json
import os
import numpy as np
from datetime import datetime
from scipy import stats
from typing import Dict, List

BASELINE_PATH = "report-output/ghost_agents/baseline/baseline_evaluation.json"
PROPOSED_PATH = "report-output/ghost_agents/evaluation_results.json"
INJECTION_PATH = "report-output/ghost_agents/injection_resistance_results.json"
LLM_BENCHMARK_DIR = "report-output/ghost_agents/llm_benchmark"
OUTPUT_DIR = "report-output/ghost_agents/comparison"

def calculate_statistical_significance(baseline_values: List[float], proposed_values: List[float]) -> Dict:
    """Calculate statistical significance between two distributions."""
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

def generate_enhanced_comparison():
    """Generate comparison with statistical tests."""
    print("="*80)
    print("ENHANCED COMPARISON REPORT WITH STATISTICAL TESTS")
    print("="*80)
    
    # Load data
    if not os.path.exists(BASELINE_PATH):
        print(f"❌ Baseline not found: {BASELINE_PATH}")
        print("Run: python src/ghost_agents/baseline_evaluation.py")
        return
    
    if not os.path.exists(PROPOSED_PATH):
        print(f"❌ Proposed not found: {PROPOSED_PATH}")
        print("Run: python src/ghost_agents/evaluate.py")
        return
    
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    
    with open(PROPOSED_PATH, 'r') as f:
        proposed = json.load(f)
    
    # Load injection resistance if available
    injection_data = None
    if os.path.exists(INJECTION_PATH):
        with open(INJECTION_PATH, 'r') as f:
            injection_data = json.load(f)

    # Load latest LLM benchmark
    llm_data = None
    if os.path.exists(LLM_BENCHMARK_DIR):
        files = [os.path.join(LLM_BENCHMARK_DIR, f) for f in os.listdir(LLM_BENCHMARK_DIR) if f.endswith('.json')]
        if files:
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                llm_data = json.load(f)
    
    # Extract metrics
    baseline_avg = baseline.get("averaged_metrics", {})
    proposed_metrics = proposed.get("summary", {}).get("metrics", {})
    
    # Collect detailed results for statistical tests
    baseline_results = []
    for model_data in baseline.get("model_results", {}).values():
        baseline_results.extend(model_data.get("detailed_results", []))
    
    proposed_results = proposed.get("detailed_results", [])
    
    # Extract latencies
    baseline_latencies = [r.get("inference_duration", 0) for r in baseline_results if "inference_duration" in r]
    proposed_latencies = [r.get("inference_duration", 0) for r in proposed_results if "inference_duration" in r]
    
    # Extract semantic similarities
    baseline_semantic = [r.get("semantic_similarity", 0) for r in baseline_results if "semantic_similarity" in r]
    proposed_semantic = [r.get("semantic_similarity", 0) for r in proposed_results if "semantic_similarity" in r]
    
    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    stats_results = {}
    
    if baseline_latencies and proposed_latencies:
        print("\n📊 Latency Comparison:")
        latency_stats = calculate_statistical_significance(baseline_latencies, proposed_latencies)
        stats_results["latency"] = latency_stats
        
        print(f"  Baseline mean: {np.mean(baseline_latencies):.4f}s")
        print(f"  Proposed mean: {np.mean(proposed_latencies):.4f}s")
        print(f"  T-test p-value: {latency_stats['t_pvalue']:.6f}")
        print(f"  Cohen's d: {latency_stats['cohens_d']:.4f}")
        print(f"  Significant (p<0.05): {'✅ YES' if latency_stats['is_significant_005'] else '❌ NO'}")
    
    if baseline_semantic and proposed_semantic:
        print("\n📊 Semantic Similarity Comparison:")
        semantic_stats = calculate_statistical_significance(baseline_semantic, proposed_semantic)
        stats_results["semantic_similarity"] = semantic_stats
        
        print(f"  Baseline mean: {np.mean(baseline_semantic):.4f}")
        print(f"  Proposed mean: {np.mean(proposed_semantic):.4f}")
        print(f"  T-test p-value: {semantic_stats['t_pvalue']:.6f}")
        print(f"  Cohen's d: {semantic_stats['cohens_d']:.4f}")
        print(f"  Significant (p<0.05): {'✅ YES' if semantic_stats['is_significant_005'] else '❌ NO'}")
    
    # Generate report
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<35} | {'Baseline':<15} | {'Proposed':<15} | {'Delta':<15}")
    print("-"*85)
    
    # ASR
    baseline_asr = baseline_avg.get("asr", 0)
    proposed_asr = proposed_metrics.get("attack_success_rate", 0)
    print(f"{'Attack Success Rate (ASR)':<35} | {baseline_asr*100:>12.2f}% | {proposed_asr*100:>12.2f}% | {(proposed_asr-baseline_asr)*100:>+12.2f}%")
    
    # Tool Correctness
    baseline_tool = baseline_avg.get("tool_correctness", 0)
    proposed_tool = proposed_metrics.get("tool_correctness_rate", 0)
    print(f"{'Tool Correctness':<35} | {baseline_tool*100:>12.2f}% | {proposed_tool*100:>12.2f}% | {(proposed_tool-baseline_tool)*100:>+12.2f}%")
    
    # Semantic Similarity
    baseline_sem = baseline_avg.get("avg_semantic_similarity", 0)
    proposed_sem = proposed_metrics.get("semantic_similarity_avg", 0)
    print(f"{'Semantic Similarity':<35} | {baseline_sem:>15.4f} | {proposed_sem:>15.4f} | {proposed_sem-baseline_sem:>+15.4f}")
    
    # Levenshtein Distance
    baseline_lev = baseline_avg.get("avg_levenshtein_distance", 0)
    proposed_lev = proposed_metrics.get("levenshtein_distance_avg", 0)
    print(f"{'Levenshtein Distance':<35} | {baseline_lev:>15.2f} | {proposed_lev:>15.2f} | {proposed_lev-baseline_lev:>+15.2f}")
    
    # Injection Resistance
    if injection_data:
        print(f"\n{'Prompt Injection Resistance':<35}")
        print("-"*85)
        safe_rate = injection_data.get("metrics", {}).get("safe_refusal_rate", 0)
        print(f"{'Safe Refusal Rate':<35} | {'N/A':>15} | {safe_rate*100:>12.2f}% | {'N/A':>15}")

    # LLM Comparison
    if llm_data:
        print("\n" + "="*80)
        print("LLM BENCHMARK COMPARISON")
        print("="*80)
        llm_model = llm_data.get("model", "Unknown")
        llm_latency = llm_data.get("avg_latency_s", 0)
        llm_correctness = llm_data.get("tool_correctness", 0)
        
        print(f"Comparison against monolithic LLM ({llm_model}):")
        print(f"{'Metric':<35} | {'Ghost Agents (Prop)':<20} | {'LLM':<15} | {'Delta':<15}")
        print("-"*90)
        
        # Latency
        proposed_latency_avg = np.mean(proposed_latencies) if proposed_latencies else 0
        print(f"{'Avg Latency (s)':<35} | {proposed_latency_avg:>20.4f} | {llm_latency:>15.4f} | {proposed_latency_avg - llm_latency:>+15.4f}")
        
        # Tool Correctness
        print(f"{'Tool Correctness':<35} | {proposed_tool*100:>20.2f}% | {llm_correctness*100:>15.2f}% | {(proposed_tool - llm_correctness)*100:>+15.2f}%")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "averaged_metrics": baseline_avg,
            "total_scenarios": baseline.get("total_scenarios", 0)
        },
        "proposed": {
            "metrics": proposed_metrics,
            "total_scenarios": proposed.get("summary", {}).get("total_scenarios", 0)
        },
        "statistical_tests": stats_results,
        "injection_resistance": injection_data.get("metrics") if injection_data else None,
        "deltas": {
            "asr": proposed_asr - baseline_asr,
            "tool_correctness": proposed_tool - baseline_tool,
            "semantic_similarity": proposed_sem - baseline_sem,
            "levenshtein_distance": proposed_lev - baseline_lev
        },
        "llm_comparison": {
            "model": llm_data.get("model") if llm_data else None,
            "metrics": {
                "avg_latency_s": llm_data.get("avg_latency_s") if llm_data else None,
                "tool_correctness": llm_data.get("tool_correctness") if llm_data else None
            },
            "deltas_vs_proposed": {
                "latency": (np.mean(proposed_latencies) - llm_data.get("avg_latency_s", 0)) if llm_data else None,
                "tool_correctness": (proposed_metrics.get("tool_correctness_rate", 0) - llm_data.get("tool_correctness", 0)) if llm_data else None
            }
        }
    }
    
    output_path = os.path.join(OUTPUT_DIR, f"statistical_comparison_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Report saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    generate_enhanced_comparison()
