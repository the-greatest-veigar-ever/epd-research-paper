#!/usr/bin/env python3
"""
CLI Runner for 4-Approach Ghost Agent Evaluation

Usage:
    python -m src.ghost_agents.approach_evaluation.run_evaluation \\
        --dataset report-output/integration_tests/cic_integration_results_20260216_011439.json \\
        --limit 100 \\
        --approaches all
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.ghost_agents.approach_evaluation.evaluator import ApproachEvaluator
from src.ghost_agents.approach_evaluation.approaches import ALL_APPROACHES


def main():
    parser = argparse.ArgumentParser(
        description="4-Approach Ghost Agent Comparative Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="report-output/integration_tests",
        help="Path to integration test JSON file or directory of JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of plans to evaluate (default: all)",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        default="all",
        help=(
            "Comma-separated approach names to evaluate, or 'all'. "
            f"Available: {', '.join(ALL_APPROACHES.keys())}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="report-output/ghost_agents/approach_comparison",
        help="Directory to save output JSON report",
    )

    args = parser.parse_args()

    # Parse approach names
    if args.approaches.strip().lower() == "all":
        approach_names = None  # evaluator will use all
    else:
        approach_names = [
            name.strip() for name in args.approaches.split(",")
        ]
        # Validate
        for name in approach_names:
            if name not in ALL_APPROACHES:
                print(
                    f"Error: Unknown approach '{name}'. "
                    f"Available: {', '.join(ALL_APPROACHES.keys())}"
                )
                sys.exit(1)

    # Run evaluation
    evaluator = ApproachEvaluator(
        dataset_path=args.dataset,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    evaluator.load_dataset()
    evaluator.run_all(approach_names=approach_names)


if __name__ == "__main__":
    main()
