#!/usr/bin/env python3
"""
Comparison Report Generator for Squad C Evaluation

Compares baseline (3 SLMs) vs proposed (rotation + polymorphism) results.
Generates a formatted report suitable for academic papers.
"""

import json
import os
from datetime import datetime
from typing import Dict, List

BASELINE_DIR = "report-output/ghost_agents/baseline"
PROPOSED_DIR = "report-output/ghost_agents/proposed"
OUTPUT_DIR = "report-output/ghost_agents/comparison"

def load_latest_result(directory: str) -> Dict:
    """Load the most recent result file from a directory."""
    if not os.path.exists(directory):
        return None
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    if not files:
        return None
    
    # Sort by timestamp in filename (newest first)
    files.sort(reverse=True)
    latest = os.path.join(directory, files[0])
    
    with open(latest, 'r') as f:
        return json.load(f)


def generate_comparison_report():
    """Generate comparison between baseline and proposed evaluations."""
    print("=== GENERATING COMPARISON REPORT ===\n")
    
    # Load results
    baseline = load_latest_result(BASELINE_DIR)
    proposed = load_latest_result(PROPOSED_DIR)
    
    if not baseline or not proposed:
        print("❌ Error: Missing baseline or proposed results.")
        print(f"   Baseline: {BASELINE_DIR}")
        print(f"   Proposed: {PROPOSED_DIR}")
        print("\nPlease run both evaluation scripts first:")
        print("  python src/ghost_agents/baseline_evaluation.py")
        print("  python src/ghost_agents/proposed_evaluation.py")
        return
    
    # Extract metrics
    proposed_metrics = proposed["metrics"]
    
    # Calculate baseline averages across all models
    baseline_models = baseline.get("model_results", {})
    baseline_avg = {
        "asr": 0,
        "tool_correctness": 0,
        "avg_processing_time": 0,
        "total_processing_time": 0,
        "safe_refusal_rate": 0
    }
    
    num_models = len(baseline_models)
    for model, data in baseline_models.items():
        m = data["metrics"]
        baseline_avg["asr"] += m["asr"]
        baseline_avg["tool_correctness"] += m["tool_correctness"]
        baseline_avg["avg_processing_time"] += m["timing"]["avg_processing_time"]
        baseline_avg["total_processing_time"] += m["timing"]["total_processing_time"]
        baseline_avg["safe_refusal_rate"] += m["injection_resistance"]["safe_refusal_rate"]
    
    if num_models > 0:
        for key in baseline_avg:
            baseline_avg[key] /= num_models
    
    # Generate report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SQUAD C EVALUATION COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    
    # Configuration
    report_lines.append("CONFIGURATION")
    report_lines.append("-"*40)
    report_lines.append(f"Baseline Models: {list(baseline_models.keys())}")
    report_lines.append(f"Proposed Models: {proposed['config']['models']}")
    report_lines.append(f"Scenarios Tested: {proposed_metrics['total_scenarios']}")
    report_lines.append("")
    
    # Timing Comparison
    report_lines.append("TIMING COMPARISON")
    report_lines.append("-"*40)
    report_lines.append(f"{'Metric':<35} | {'Baseline':<15} | {'Proposed':<15}")
    report_lines.append("-"*70)
    
    # Baseline: only T_proc (per model, averaged)
    report_lines.append(f"{'Initialization Time (T_init)':<35} | {'N/A':^15} | {proposed_metrics['timing']['initialization_time']:>12.3f}s")
    report_lines.append(f"{'Avg Processing Time (per scenario)':<35} | {baseline_avg['avg_processing_time']:>12.3f}s | {proposed_metrics['timing']['avg_processing_time_per_scenario']:>12.3f}s")
    report_lines.append(f"{'Total Processing Time (T_proc)':<35} | {baseline_avg['total_processing_time']:>12.3f}s | {proposed_metrics['timing']['total_processing_time']:>12.3f}s")
    report_lines.append(f"{'TOTAL TIME (T_total)':<35} | {baseline_avg['total_processing_time']:>12.3f}s | {proposed_metrics['timing']['total_time']:>12.3f}s")
    report_lines.append("")
    
    # Polymorphism Overhead
    if proposed_metrics['timing'].get('avg_polymorphism_time'):
        report_lines.append(f"Polymorphism Overhead: {proposed_metrics['timing']['avg_polymorphism_time']:.3f}s/scenario")
    report_lines.append("")
    
    # Accuracy Comparison
    report_lines.append("ACCURACY COMPARISON")
    report_lines.append("-"*40)
    report_lines.append(f"{'Metric':<35} | {'Baseline':<15} | {'Proposed':<15}")
    report_lines.append("-"*70)
    report_lines.append(f"{'Attack Success Rate (ASR)':<35} | {baseline_avg['asr']*100:>12.2f}% | {proposed_metrics['accuracy']['asr']*100:>12.2f}%")
    report_lines.append(f"{'Tool Correctness':<35} | {baseline_avg['tool_correctness']*100:>12.2f}% | {proposed_metrics['accuracy']['tool_correctness']*100:>12.2f}%")
    report_lines.append("")
    
    # Injection Resistance
    report_lines.append("PROMPT INJECTION RESISTANCE")
    report_lines.append("-"*40)
    report_lines.append(f"{'Metric':<35} | {'Baseline':<15} | {'Proposed':<15}")
    report_lines.append("-"*70)
    report_lines.append(f"{'Safe Refusal Rate':<35} | {baseline_avg['safe_refusal_rate']*100:>12.2f}% | {proposed_metrics['injection_resistance']['safe_refusal_rate']*100:>12.2f}%")
    report_lines.append("")
    
    # Polymorphism Metrics (Proposed only)
    report_lines.append("POLYMORPHISM METRICS (Proposed Only)")
    report_lines.append("-"*40)
    report_lines.append(f"Avg Semantic Similarity: {proposed_metrics['polymorphism']['avg_semantic_similarity']:.4f} (Target: >0.7)")
    report_lines.append(f"Avg Levenshtein Distance: {proposed_metrics['polymorphism']['avg_levenshtein_distance']:.2f} (Target: >20)")
    report_lines.append("")
    
    # Model Distribution (Proposed)
    report_lines.append("MODEL USAGE DISTRIBUTION (Proposed)")
    report_lines.append("-"*40)
    for model, count in proposed_metrics.get("model_distribution", {}).items():
        pct = count / proposed_metrics['total_scenarios'] * 100
        report_lines.append(f"  {model}: {count} ({pct:.1f}%)")
    report_lines.append("")
    
    # Per-Model Baseline Details
    report_lines.append("PER-MODEL BASELINE DETAILS")
    report_lines.append("-"*40)
    report_lines.append(f"{'Model':<20} | {'ASR':<10} | {'Tool Corr.':<12} | {'Avg Time':<12}")
    report_lines.append("-"*60)
    for model, data in baseline_models.items():
        m = data["metrics"]
        report_lines.append(f"{model:<20} | {m['asr']*100:>7.2f}% | {m['tool_correctness']*100:>9.2f}% | {m['timing']['avg_processing_time']:>9.3f}s")
    report_lines.append("")
    
    # Summary
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    
    # Calculate deltas
    time_delta = proposed_metrics['timing']['total_time'] - baseline_avg['total_processing_time']
    asr_delta = proposed_metrics['accuracy']['asr'] - baseline_avg["asr"]
    
    if time_delta > 0:
        report_lines.append(f"⚠️  Proposed adds {time_delta:.3f}s overhead (due to polymorphism + init)")
    else:
        report_lines.append(f"✅ Proposed is {-time_delta:.3f}s faster")
    
    if asr_delta >= 0:
        report_lines.append(f"✅ Proposed maintains ASR ({asr_delta*100:+.2f}% delta)")
    else:
        report_lines.append(f"⚠️  Proposed has lower ASR ({asr_delta*100:.2f}% delta)")
    
    report_lines.append(f"🛡️  Polymorphism adds obfuscation (Levenshtein: {proposed_metrics['polymorphism']['avg_levenshtein_distance']:.1f})")
    report_lines.append("")
    
    # Print report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text report
    txt_path = os.path.join(OUTPUT_DIR, f"comparison_report_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write(report_text)
    
    # Save JSON for programmatic access
    json_path = os.path.join(OUTPUT_DIR, f"comparison_data_{timestamp}.json")
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "models": list(baseline_models.keys()),
            "averaged_metrics": baseline_avg,
            "per_model": {m: d["metrics"] for m, d in baseline_models.items()}
        },
        "proposed": proposed_metrics,
        "deltas": {
            "time_overhead": time_delta,
            "asr_delta": asr_delta
        }
    }
    with open(json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✅ Reports saved to:")
    print(f"   Text:  {txt_path}")
    print(f"   JSON:  {json_path}")


if __name__ == "__main__":
    generate_comparison_report()
