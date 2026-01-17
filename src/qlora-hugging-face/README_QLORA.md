# QLoRA Fine-tuning with SecQA Dataset

## Overview
This script fine-tunes the microsoft/phi-2 model (2.7B parameters, free) on the SecQA v1 dev dataset using QLoRA (Quantized Low-Rank Adaptation).

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset
The script uses the SecQA v1 dev dataset located at:
```
data/SecQA/secqa_v1_dev.jsonl
```

Dataset format:
- `question`: The security question
- `answers`: Dictionary with keys A, B, C, D
- `solution`: The correct answer (A, B, C, or D)
- `explanation`: Detailed explanation of the answer

### 3. Model
- **Model**: microsoft/phi-2 (2.7B parameters)
- **Technique**: QLoRA (4-bit quantization + LoRA adapters)
- **Free**: Yes, no API key required

## Running the Script

### Test Data Loading (Recommended First)
```bash
cd src
python3 test_secqa_loading.py
```

### Run Training
```bash
cd src
python3 qlora_hugging_face_test.py
```

## Configuration

Key parameters in `qlora_hugging_face_test.py`:
- `MODEL_NAME`: "microsoft/phi-2"
- `DATA_PATH`: "../data/SecQA/secqa_v1_dev.jsonl"
- `OUTPUT_DIR`: "./output/qlora-secqa"
- `MAX_SEQ_LENGTH`: 2048
- `BATCH_SIZE`: 2
- `GRAD_ACCUM`: 8
- `EPOCHS`: 3
- `LR`: 2e-4

## Output
The fine-tuned LoRA adapters will be saved to:
```
src/output/qlora-secqa/
```

## How to Verify (Inference)
I've created [`inference_test.py`](file:src/inference_test.py) to help you check the model's performance.

### Run Inference
```bash
python3 inference_test.py
```

### Important Note for Mac Users
If you see gibberish output (like `!!!!!!!!!!`), change `DEVICE = "mps"` to `DEVICE = "cpu"` at the top of the script. Some models have precision issues on Mac GPU (MPS) that cause numerical instability. Running on CPU is slower but much more reliable for verification.

## Hardware Requirements
- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB+ GPU memory for 4-bit quantization
- **CPU**: Will work but very slow

## Notes
- The script uses 4-bit quantization to reduce memory usage
- Only LoRA adapters are trained, not the full model
- Training on 5 examples (dev set) is mainly for testing; use larger datasets for production
