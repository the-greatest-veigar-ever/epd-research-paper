#!/bin/bash

echo "=== STARTING EPD BRAIN TRAINING CYCLE ==="

# 1. Run Training
echo "[1/2] Training on Combined Dataset (MCQ + SAQ)..."
python3 src/brain/qlora-hugging-face/qlora_hugging_face_test.py

# 2. Run Evaluation
echo "[2/2] Evaluating on SecQA Test Set (MCQ)..."
python3 src/brain/qlora-hugging-face/eval_fast.py

echo "=== CYCLE COMPLETE ==="
echo "Report saved to: src/brain/qlora-hugging-face/eval_results_batch.json"
