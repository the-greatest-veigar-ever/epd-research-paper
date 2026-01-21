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

# Using a free, smaller model suitable for fine-tuning
MODEL_NAME = "microsoft/phi-2"  # Free 2.7B parameter model
DATA_PATH = "../../data/brain/data/SecQA/secqa_v1_dev.jsonl"
OUTPUT_DIR = "./output/qlora-secqa"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8
EPOCHS = 3
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

dataset = load_dataset("json", data_files=DATA_PATH)

def format_prompt(example):
    """Format SecQA data into a training prompt."""
    # Format the multiple choice answers
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
    fp16=False,         # set to False on Mac/MPS to avoid CUDA requirements
    logging_steps=10,
    save_strategy="epoch",
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
    args=training_args
)

trainer.train()

# =========================
# SAVE LoRA ADAPTERS
# =========================

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA training complete. Adapters saved.")
