import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from trl import SFTTrainer, SFTConfig

# =========================
# CONFIG
# =========================

MODEL_NAME = "microsoft/phi-2"
DATA_PATH = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/SecQA/secqa_v1_dev.jsonl"
OUTPUT_DIR = "./output/base-phi2-secqa"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1 # Reduced for full fine-tuning on Mac
GRAD_ACCUM = 16
EPOCHS = 3
LR = 2e-5 # Lower LR for full fine-tuning

# =========================
# LOAD TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODEL (Full Parameter)
# =========================

# Note: Full fine-tuning requires significant VRAM/Memory.
# On Mac, we use FP16/BF16 if possible, else FP32.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
)

# =========================
# LOAD DATASET
# =========================

dataset = load_dataset("json", data_files=DATA_PATH)

def format_prompt(example):
    """Format SecQA data into a training prompt."""
    answers_text = "\n".join([f"{k}. {v}" for k, v in example['answers'].items()])
    
    return f"""### Question:
{example['question']}

### Choices:
{answers_text}

### Answer:
{example['solution']}

### Explanation:
{example['explanation']}"""

dataset = dataset.map(
    lambda x: {"text": format_prompt(x)},
    remove_columns=dataset["train"].column_names
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
    fp16=False, # Use False on Mac to avoid CUDA dependency in bitsandbytes/accelerate
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_torch",
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
    args=training_args
)

print("Starting full parameter fine-tuning...")
trainer.train()

# =========================
# SAVE FULL MODEL
# =========================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Full fine-tuning script ready. Output will be saved to {OUTPUT_DIR}")
