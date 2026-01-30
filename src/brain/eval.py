import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
BASE_MODEL_NAME = "microsoft/phi-2"
ADAPTER_PATH = "ai/models/qlora-hugging-face/qlora-secqa"
TEST_DATA_PATH = "ai/data/brain/data/splits/test.jsonl"
DEVICE = "mps" # Use Mac Metal Performance Shaders
print(f"Using device: {DEVICE}")

def load_model():
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading adapters from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def evaluate(batch_size=10):
    model, tokenizer = load_model()
    
    # Load test data
    all_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Only keep MCQs for this fast eval
                if 'answers' in item and item['answers'] and 'solution' in item:
                   all_data.append(item)
            except:
                pass
    
    print(f"Loaded {len(all_data)} MCQ items for evaluation (Skipped non-MCQ/SAQ items).")
    print(f"Starting BATCH evaluation on {DEVICE} (Batch Size: {batch_size})...")
    
    # Token IDs for A, B, C, D
    choice_tokens = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in choice_tokens]

    correct = 0
    total = len(all_data)
    results = []

    for i in tqdm(range(0, total, batch_size)):
        batch = all_data[i:i+batch_size]
        prompts = []
        ground_truths = []
        
        valid_batch_indices = []
        
        for idx, item in enumerate(batch):
            question = item['question']
            choices = item['answers']
            ground_truths.append(item['solution'])
            
            answers_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
            prompt = f"### Question:\n{question}\n\n### Choices:\n{answers_text}\n\n### Answer:\n"
            prompts.append(prompt)
            valid_batch_indices.append(idx)
        
        if not prompts:
            continue

        # Batch Tokenization
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits # [batch, seq_len, vocab_size]
            
            attention_mask = inputs.attention_mask
            last_token_indices = attention_mask.sum(dim=1) - 1
            
            for b_idx in range(len(prompts)):
                last_logits = logits[b_idx, last_token_indices[b_idx], :]
                choice_logits = last_logits[choice_ids]
                best_choice_idx = torch.argmax(choice_logits).item()
                model_answer = choice_tokens[best_choice_idx]
                
                is_correct = (model_answer == ground_truths[b_idx])
                if is_correct:
                    correct += 1
                
                results.append({
                    "question": batch[b_idx]['question'],
                    "ground_truth": ground_truths[b_idx],
                    "model_answer": model_answer,
                    "correct": is_correct
                })
        
        if (i // batch_size) % 10 == 0:
             print(f"Batch {i//batch_size + 1} processed. Running Accuracy: {(correct / (i+len(batch))) * 100:.2f}%")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nEvaluation Complete!")
    print(f"Total Evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    output_report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results
    }
    with open("report-output/brain/eval_results_batch.json", "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"Detailed results saved to eval_results_batch.json")

if __name__ == "__main__":
    evaluate()
