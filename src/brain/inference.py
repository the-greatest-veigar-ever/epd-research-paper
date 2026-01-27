import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# =========================
# CONFIG
# =========================
BASE_MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "./output/qlora-secqa"
DEVICE = "cpu" # Force CPU for sanity check

def generate_response(question, choices):
    # Format the prompt exactly like in training
    answers_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
    
    prompt = f"""### Question:
{question}

### Choices:
{answers_text}

### Answer:
"""
    
    print("--- Prompt ---")
    print(prompt)
    print("--------------")

    # Load tokenizer and model
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=DEVICE
    )
    
    # Inference with base model
    print("Generating with base model...")
    with torch.no_grad():
        base_outputs = base_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    base_response = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
    base_clean = base_response[len(prompt):].strip()

    # Load LoRA adapters
    print(f"Loading adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()

    # Inference with fine-tuned model
    print("Generating with fine-tuned model...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_clean = response[len(prompt):].strip()
    
    return base_clean, response_clean

if __name__ == "__main__":
    # Sample question from secqa_v1_dev
    sample_question = "What is the purpose of implementing a Guest Wireless Network in a corporate environment?"
    sample_choices = {
        "A": "To provide unrestricted access to company resources",
        "B": "To replace the primary corporate wireless network",
        "C": "To bypass network security protocols",
        "D": "To offer a separate, secure network for visitors"
    }

    base_res, ft_res = generate_response(sample_question, sample_choices)
    
    print("\n=== Base Model Response ===")
    print(base_res)
    print("\n=== Fine-tuned Model Response ===")
    print(ft_res)
    print("==========================")
