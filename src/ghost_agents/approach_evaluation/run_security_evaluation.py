#!/usr/bin/env python3
"""
CLI Runner for Ghost Agent Security Evaluation

Runs 4 research-grade cybersecurity test frameworks against the 4 Ghost Agent
approaches and produces comparative security metrics.

Usage:
    python -m src.ghost_agents.approach_evaluation.run_security_evaluation \\
        --approaches all \\
        --frameworks all

    python -m src.ghost_agents.approach_evaluation.run_security_evaluation \\
        --approaches phi_suicide,multimodal_suicide \\
        --frameworks cyberseceval,harmbench
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

from src.ghost_agents.approach_evaluation.security_evaluator import SecurityEvaluator
from src.ghost_agents.approach_evaluation.approaches import ALL_APPROACHES
from src.ghost_agents.approach_evaluation.security_test_data import ALL_SECURITY_TESTS


AVAILABLE_FRAMEWORKS = list(ALL_SECURITY_TESTS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Agent Security Evaluation (4 Cybersecurity Frameworks)"
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
        "--frameworks",
        type=str,
        default="all",
        help=(
            "Comma-separated security framework names to run, or 'all'. "
            f"Available: {', '.join(AVAILABLE_FRAMEWORKS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="report-output/ghost_agents/security_comparison",
        help="Directory to save output JSON reports",
    )

    args = parser.parse_args()

    # Parse approach names
    if args.approaches.strip().lower() == "all":
        approach_names = None  # evaluator will use all
    else:
        approach_names = [
            name.strip() for name in args.approaches.split(",")
        ]
        for name in approach_names:
            if name not in ALL_APPROACHES:
                print(
                    f"Error: Unknown approach '{name}'. "
                    f"Available: {', '.join(ALL_APPROACHES.keys())}"
                )
                sys.exit(1)

    # Parse framework names
    if args.frameworks.strip().lower() == "all":
        frameworks = None  # evaluator will use all
    else:
        frameworks = [
            name.strip() for name in args.frameworks.split(",")
        ]
        for name in frameworks:
            if name not in AVAILABLE_FRAMEWORKS:
                print(
                    f"Error: Unknown framework '{name}'. "
                    f"Available: {', '.join(AVAILABLE_FRAMEWORKS)}"
                )
                sys.exit(1)

    # Print banner
    print("=" * 70)
    print("  GHOST AGENT SECURITY EVALUATION")
    print("  Cybersecurity Resilience Testing")
    print("=" * 70)
    print(f"  Approaches: {approach_names or 'all'}")
    print(f"  Frameworks: {frameworks or 'all'}")
    print(f"  Output Dir: {args.output_dir}")
    print("=" * 70)

    # Run security evaluation
    evaluator = SecurityEvaluator(
        output_dir=args.output_dir,
        frameworks=frameworks,
    )
    evaluator.run_all(approach_names=approach_names)


if __name__ == "__main__":
    main()
