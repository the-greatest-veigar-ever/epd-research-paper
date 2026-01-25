import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# =========================
# CONFIG
# =========================

# Using the same model as the QLoRA script
MODEL_NAME = "microsoft/phi-2"
DATA_PATH = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/SecQA/secqa_v1_test.jsonl"

# =========================
# LOAD TOKENIZER
# =========================

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODEL (Base)
# =========================

print(f"Loading base model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
# Force CPU for stability as MPS has known issues with Phi-2 inference quality
device = "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# =========================
# LOAD DATASET
# =========================

print(f"Loading dataset from {DATA_PATH}...")
dataset = load_dataset("json", data_files=DATA_PATH)

def format_prompt(example):
    """Format SecQA data into a test prompt."""
    answers_text = "\n".join([f"{k}. {v}" for k, v in example['answers'].items()])
    
    return f"""### Question:
{example['question']}

### Choices:
{answers_text}

### Answer:
"""

# =========================
# INFERENCE TEST
# =========================

print("\n--- Running Inference on Base Model ---\n")

# Take first 3 examples for testing
test_examples = dataset["train"].select(range(3))

for i, example in enumerate(test_examples):
    prompt = format_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Example {i+1}:")
    print(f"Question: {example['question']}")
    print(f"Correct Answer: {example['solution']}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the newly generated part after the prompt
    generated_text = response[len(prompt):].strip()
    
    print(f"Model Prediction: {generated_text}")
    print("-" * 50)

print("\n✅ Base model testing complete.")
