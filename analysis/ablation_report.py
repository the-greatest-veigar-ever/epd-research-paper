"""
Ablation Component-Contribution Report.

Reviewer 2, point 1: isolate the marginal contribution of each EPD
ingredient (ephemerality, randomized persona, static safety filter) to
ASR/TSR, rather than only comparing the two endpoint configurations
(static baseline vs. full EPD).

Reads a multi_seed_summary_*.json produced by benchmark_evaluator.py
(which already covers the full 2x2x2 factorial per model) and computes,
per model and overall, the average ASR/TSR when each factor is ON minus
when it is OFF, marginalizing over the other two factors -- a standard
main-effects decomposition for a full factorial design.

Usage:
    python3 analysis/ablation_report.py \
        report-output/ghost_agents/benchmark_results/multi_seed_summary_<id>.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ghost_agents.approach_evaluation.approaches import ALL_APPROACHES, ABLATION_MODELS

FACTORS = ["ephemeral", "persona", "safety_filter"]


def _cell_for(name: str):
    """Recover the (model, ephemeral, persona, safety_filter) cell for an
    approach name from the bound keyword arguments of its factory partial."""
    factory = ALL_APPROACHES.get(name)
    if factory is None:
        return None
    return factory.keywords


def compute_main_effects(multi_seed_summary: Dict) -> List[Dict]:
    """
    For each model in ABLATION_MODELS and each factor, compute:
        mean(metric | factor=True) - mean(metric | factor=False)
    averaged across the 4 cells on each side (i.e. marginalized over the
    other two factors). Positive ASR delta = factor makes attacks more
    likely to succeed (bad); positive TSR delta = factor helps task
    completion (good).
    """
    per_approach_overall = multi_seed_summary.get("per_approach_overall", {})

    rows = []
    for model_key, model_tag in ABLATION_MODELS.items():
        cells = {
            name: kw for name, kw in ((n, _cell_for(n)) for n in ALL_APPROACHES)
            if kw and kw.get("model") == model_tag
        }
        for factor in FACTORS:
            on_asr, off_asr, on_tsr, off_tsr = [], [], [], []
            for name, kw in cells.items():
                overall = per_approach_overall.get(name)
                if not overall or overall.get("avg_asr_mean") is None:
                    continue
                (on_asr if kw[factor] else off_asr).append(overall["avg_asr_mean"])
                (on_tsr if kw[factor] else off_tsr).append(overall["avg_tsr_mean"])

            if not on_asr or not off_asr:
                continue

            asr_on, asr_off = sum(on_asr) / len(on_asr), sum(off_asr) / len(off_asr)
            tsr_on, tsr_off = sum(on_tsr) / len(on_tsr), sum(off_tsr) / len(off_tsr)

            rows.append({
                "model": model_key,
                "factor": factor,
                "asr_on_mean": round(asr_on, 4),
                "asr_off_mean": round(asr_off, 4),
                "asr_delta": round(asr_on - asr_off, 4),
                "tsr_on_mean": round(tsr_on, 4),
                "tsr_off_mean": round(tsr_off, 4),
                "tsr_delta": round(tsr_on - tsr_off, 4),
                "n_cells_on": len(on_asr),
                "n_cells_off": len(off_asr),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Component-contribution (ablation main-effects) report")
    parser.add_argument("multi_seed_summary", help="Path to multi_seed_summary_*.json")
    parser.add_argument("--out", default="report-output/ghost_agents/ablation_report.csv")
    args = parser.parse_args()

    with open(args.multi_seed_summary) as f:
        summary = json.load(f)

    rows = compute_main_effects(summary)
    if not rows:
        print("[WARNING] No ablation rows computed -- does the summary cover the full 8-cell matrix "
              "(i.e. was the evaluator run with --approaches full_ablation / default)?")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'Model':<20}{'Factor':<16}{'ASR delta':>12}{'TSR delta':>12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<20}{r['factor']:<16}{r['asr_delta']*100:>+11.2f}%{r['tsr_delta']*100:>+11.2f}%")

    print("\nOverall main effect (averaged across the 5 SLMs) -- answers "
          "'which component contributes most':")
    for factor in FACTORS:
        factor_rows = [r for r in rows if r["factor"] == factor]
        if not factor_rows:
            continue
        avg_asr_delta = sum(r["asr_delta"] for r in factor_rows) / len(factor_rows)
        avg_tsr_delta = sum(r["tsr_delta"] for r in factor_rows) / len(factor_rows)
        print(f"  {factor:<16} ASR delta: {avg_asr_delta*100:+.2f}%   TSR delta: {avg_tsr_delta*100:+.2f}%")

    print(f"\nFull per-model breakdown written to: {args.out}")


if __name__ == "__main__":
    main()
