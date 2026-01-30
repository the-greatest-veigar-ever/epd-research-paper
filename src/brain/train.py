import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# =========================
# CONFIG
# =========================

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default="ai/data/brain/data/splits/train.jsonl", help="Path to training data")
parser.add_argument("--val_file", type=str, default="ai/data/brain/data/splits/val.jsonl", help="Path to validation data")
parser.add_argument("--output_dir", type=str, default="ai/models/qlora-hugging-face/qlora-secqa", help="Output directory")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
args = parser.parse_args()


# Using a free, smaller model suitable for fine-tuning
MODEL_NAME = "microsoft/phi-2"  # Free 2.7B parameter model
OUTPUT_DIR = args.output_dir

MAX_SEQ_LENGTH = 256 # Reduced to 256 for maximum memory efficiency on Mac
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3 # Final Epoch Setting
LR = 2e-4

# =========================
# LOAD TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODEL (4-bit QLoRA)
# =========================

# Load in FP16 for MPS compatibility (bitsandbytes 4-bit requires CUDA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

# =========================
# APPLY LoRA (QLoRA)
# =========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# LOAD DATASET
# =========================

print(f"Loading Training Data: {args.train_file}")
print(f"Loading Validation Data: {args.val_file}")

dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

def format_prompt(example):
    """Format data into a training prompt (Supports MCQ and SAQ)."""
    
    # CASE 1: Multiple Choice Question (MCQ) - Checks for 'answers' dict
    if 'answers' in example and example['answers']:
        answers_text = "\n".join([f"{k}. {v}" for k, v in example['answers'].items()])
        
        prompt = f"""### Question:
{example['question']}

### Choices:
{answers_text}

### Answer:
{example['solution']}"""

    # CASE 2: Short Answer Question (SAQ) - No choices
    else:
        # Fallback for datasets like 'chinese/SAQs.jsonl' which use 'answer'
        answer_text = example.get('answer') or example.get('solution') or "Unknown"
        
        prompt = f"""### Question:
{example['question']}

### Answer:
{answer_text}"""

    # Add explanation if present (Common to both)
    if 'explanation' in example and example['explanation']:
        prompt += f"\n\n### Explanation:\n{example['explanation']}"
        
    return prompt

dataset = dataset.map(
    lambda x: {"text": format_prompt(x)},
    # remove_columns=dataset["train"].column_names # removing columns might be risky if we need them for eval, but for training text field is enough
)

# =========================
# TRAINING CONFIG (SFTConfig)
# =========================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=False,         # set to False on Mac/MPS to avoid CUDA requirements
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch", # Changed from 'steps' to 'epoch' to save massive time (save ~12 hrs)
    eval_steps=None,       # Not used when strategy is epoch
    optim="adamw_torch",   # Changed from paged_adamw_8bit (which requires CUDA)
    report_to="none",
    remove_unused_columns=False,
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH
)

# =========================
# TRAINER
# =========================

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args
)

if args.resume_from_checkpoint:
    print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

# =========================
# SAVE LoRA ADAPTERS & LOGS
# =========================

import os
import json
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Save Model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ QLoRA training complete. Adapters saved to {OUTPUT_DIR}")

# 2. Save Training History (Loss Curves)
log_history = trainer.state.log_history
history_path = os.path.join(OUTPUT_DIR, "training_log.json")
with open(history_path, "w") as f:
    json.dump(log_history, f, indent=4)
print(f"✅ Training logs saved to {history_path}")

# 3. Final Evaluation on TEST SET (New)
print("Running final evaluation on Test Set...")
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np

    # Load Test Data
    test_file = args.val_file.replace("val", "test") # Assuming standard naming
    test_dataset = load_dataset("json", data_files=test_file)["train"]
    test_dataset = test_dataset.map(lambda x: {"text": format_prompt(x)})
    
    # Predict
    predictions = trainer.predict(test_dataset)
    
    # NOTE: Generation metrics for LLMs are complex. 
    # Here we save the raw loss and perplexity on the test set.
    test_metrics = predictions.metrics
    test_metrics["perplexity"] = np.exp(test_metrics["test_loss"])
    
    report_path = os.path.join(OUTPUT_DIR, "final_test_report.json")
    with open(report_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print(f"✅ Final Test Report saved to {report_path}")
    print(f"   Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"   Perplexity: {test_metrics['perplexity']:.4f}")

except Exception as e:
    print(f"⚠️ Could not run final test evaluation: {e}")

