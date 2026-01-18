import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
BASE_MODEL_NAME = "microsoft/phi-2"
TEST_DATA_PATH = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/SecQA/secqa_v1_test.jsonl"
DEVICE = "cpu" # Force CPU for numeric stability
print(f"Using device: {DEVICE}")

def load_model():
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in FP32 for CPU stability
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def evaluate(batch_size=10):
    model, tokenizer = load_model()
    
    # Load test data
    test_data = []
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Data path {TEST_DATA_PATH} not found.")
        return

    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Starting BATCH evaluation of {len(test_data)} questions on {DEVICE} (Batch Size: {batch_size})...")
    
    # Token IDs for A, B, C, D
    choice_tokens = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in choice_tokens]

    correct = 0
    total = len(test_data)
    results = []

    for i in tqdm(range(0, total, batch_size)):
        batch = test_data[i:i+batch_size]
        prompts = []
        ground_truths = []
        
        for item in batch:
            question = item['question']
            choices = item['answers']
            ground_truths.append(item['solution'])
            
            answers_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
            prompt = f"### Question:\n{question}\n\n### Choices:\n{answers_text}\n\n### Answer:\n"
            prompts.append(prompt)
        
        # Batch Tokenization
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits # [batch, seq_len, vocab_size]
            
            attention_mask = inputs.attention_mask
            last_token_indices = attention_mask.sum(dim=1) - 1
            
            for b_idx in range(len(batch)):
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
        
        # print(f"Batch {i//batch_size + 1} processing complete. Total Correct: {correct}")

    accuracy = (correct / total) * 100
    print(f"\nEvaluation Complete!")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    output_report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results
    }
    output_filename = "eval_results_base.json"
    with open(output_filename, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"Detailed results saved to {output_filename}")

if __name__ == "__main__":
    evaluate()
