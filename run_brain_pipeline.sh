#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT=$(pwd)
DATA_SOURCE_DIR="$PROJECT_ROOT/data/brain/data"
COMBINED_DIR="$DATA_SOURCE_DIR/combined_datasets"
COMBINED_FILE="$COMBINED_DIR/all_training_data.jsonl"
BRAIN_SRC_DIR="$PROJECT_ROOT/src/brain/qlora-hugging-face"

echo "========================================================"
echo "   STARTING BRAIN SQUAD FULL TRAINING PIPELINE"
echo "========================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Data Source : $DATA_SOURCE_DIR"

# ==============================================================================
# STEP 1: DATA PREPARATION
# ==============================================================================
echo ""
echo "[Step 1/3] Data Preparation: Aggregating Datasets..."

# Create directory if not exists
if [ ! -d "$COMBINED_DIR" ]; then
    echo "  - Creating directory: $COMBINED_DIR"
    mkdir -p "$COMBINED_DIR"
fi

# Clean old file
if [ -f "$COMBINED_FILE" ]; then
    echo "  - Removing old dataset: $COMBINED_FILE"
    rm "$COMBINED_FILE"
fi

echo "  - Aggregating .jsonl files from sources (CyberMetric, SecEval, SecQA, Chinese)..."

# Concatenate files
# We use find to strictly locate .jsonl files to avoid errors if a folder is empty (though we verified they aren't)
# But simple cat with wildcards is more readable and robust enough for this known structure.
cat "$DATA_SOURCE_DIR"/CyberMetric/*.jsonl \
    "$DATA_SOURCE_DIR"/SecEval/*.jsonl \
    "$DATA_SOURCE_DIR"/SecQA/*.jsonl \
    "$DATA_SOURCE_DIR"/chinese/*.jsonl > "$COMBINED_FILE"

# Verify
if [ -f "$COMBINED_FILE" ]; then
    LINE_COUNT=$(wc -l < "$COMBINED_FILE")
    echo "  - SUCCESS: Combined dataset created."
    echo "  - Total Records: $LINE_COUNT"
    echo "  - Output Path: $COMBINED_FILE"
else
    echo "  - ERROR: Failed to create combined file."
    exit 1
fi

# ==============================================================================
# STEP 2: TRAINING (QLoRA)
# ==============================================================================
echo ""
echo "[Step 2/3] Training: Fine-tuning Phi-2 (QLoRA)..."
echo "  - Script: $BRAIN_SRC_DIR/qlora_hugging_face_test.py"
echo "  - Logs: Will be visible below"

# Check if training script exists
if [ ! -f "$BRAIN_SRC_DIR/qlora_hugging_face_test.py" ]; then
    echo "  - ERROR: Training script not found at $BRAIN_SRC_DIR/qlora_hugging_face_test.py"
    exit 1
fi

python3 "$BRAIN_SRC_DIR/qlora_hugging_face_test.py"

echo "  - Training phase completed."

# ==============================================================================
# STEP 3: EVALUATION
# ==============================================================================
echo ""
echo "[Step 3/3] Evaluation: Benchmarking on SecQA Test Set..."
echo "  - Script: $BRAIN_SRC_DIR/eval_fast.py"

# Check if eval script exists
if [ ! -f "$BRAIN_SRC_DIR/eval_fast.py" ]; then
    echo "  - ERROR: Eval script not found at $BRAIN_SRC_DIR/eval_fast.py"
    exit 1
fi

python3 "$BRAIN_SRC_DIR/eval_fast.py"

# ==============================================================================
# COMPLETION
# ==============================================================================
echo ""
echo "========================================================"
echo "   PIPELINE COMPLETE - BRAIN SQUAD UPGRADED"
echo "========================================================"
echo "Report saved to: src/brain/qlora-hugging-face/eval_results_batch.json"
