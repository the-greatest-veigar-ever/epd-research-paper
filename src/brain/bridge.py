import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# CONFIG (Matching eval.py)
BASE_MODEL_NAME = "microsoft/phi-2"
# Adjust path to be absolute or relative to project root
# eval.py used "ai/models/qlora-hugging-face/qlora-secqa"
# We assume this script is run with project root as cwd or we find the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "ai/models/qlora-hugging-face/qlora-secqa")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

import threading

_MODEL = None
_TOKENIZER = None
_LOCK = threading.RLock()

def load_model():
    global _MODEL, _TOKENIZER
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _TOKENIZER
            
        print(f"[Bridge] Loading base model: {BASE_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        print(f"[Bridge] Loading adapters from: {ADAPTER_PATH}...")
        if os.path.exists(ADAPTER_PATH):
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        else:
            print(f"[Bridge] WARNING: Adapter path {ADAPTER_PATH} not found. Using base model.")
            
        model.to(DEVICE)
        model.eval()
        
        _MODEL = model
        _TOKENIZER = tokenizer
    return _MODEL, _TOKENIZER

def generate_text(prompt, max_new_tokens=256, temperature=0.0):
    # Serialize generation to avoid MPS concurrency issues
    with _LOCK:
        model, tokenizer = load_model()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Strip the prompt from the response if the model echoes it (Phi-2 often does)
        if response.startswith(prompt):
            response = response[len(prompt):]
            
        return response.strip()

if __name__ == "__main__":
    print(generate_text("What is SQL Injection?"))
