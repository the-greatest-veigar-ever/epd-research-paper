#!/bin/bash
set -e

echo "========================================================"
echo "   STARTING FULL SYSTEM EVALUATION (METRICS UPDATE)"
echo "========================================================"

# 1. Install Dependencies
echo ""
echo "[Step 1/3] Installing/Updating Dependencies..."
pip install -r requirements.txt
pip install sentence-transformers scikit-learn python-Levenshtein numpy pandas

# 2. Squad B Evaluation
echo ""
echo "[Step 2/3] Evaluating SQUAD B (The Brain)..."
echo "Metrics: Reasoning Accuracy, Hallucination Rate, Factuality, FRR"
python3 src/brain/eval.py

# 3. Squad C Evaluation
echo ""
echo "[Step 3/3] Evaluating SQUAD C (The Hands)..."
echo "Metrics: Attack Success Rate (ASR), Pass@1, Polymorphism"
python3 src/ghost_agents/evaluate.py

echo ""
echo "========================================================"
echo "   EVALUATION COMPLETE"
echo "========================================================"
echo "Reports saved to report-output/"
