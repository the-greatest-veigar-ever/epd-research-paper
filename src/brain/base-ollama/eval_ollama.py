import json
import requests
import os
from tqdm import tqdm

# =========================
# CONFIG
# =========================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
TEST_DATA_PATH = "/Users/thachngo/Documents/EDP Research/epd-research-paper/data/SecQA/secqa_v1_test.jsonl"
OUTPUT_FILENAME = "eval_results_ollama.json"

def get_ollama_response(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"\nError calling Ollama API: {e}")
        return None

def extract_answer(response_text):
    """
    Tries to extract A, B, C, or D from the model response.
    Ideally, the prompt asks for JUST the letter, but this adds robustness.
    """
    response_text = response_text.upper().strip()
    # Check if the first character is a choice
    if len(response_text) > 0 and response_text[0] in ["A", "B", "C", "D"]:
        # Check if it's followed by a period or space to avoid "APPLE"
        if len(response_text) == 1 or not response_text[1].isalpha():
            return response_text[0]
            
    # Search for "Answer: X" or similar
    if "ANSWER:" in response_text:
        parts = response_text.split("ANSWER:")
        if len(parts) > 1:
            ans_part = parts[1].strip()
            if len(ans_part) > 0 and ans_part[0] in ["A", "B", "C", "D"]:
                return ans_part[0]
                
    # fallback: just look for the letters in the whole string if it's short
    if len(response_text) < 10:
        for letter in ["A", "B", "C", "D"]:
            if letter in response_text:
                return letter
                
    return None

def evaluate():
    # Load test data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Data path {TEST_DATA_PATH} not found.")
        return

    test_data = []
    with open(TEST_DATA_PATH, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    print(f"Starting Ollama evaluation of {len(test_data)} questions using model '{MODEL_NAME}'...")
    
    correct = 0
    total = len(test_data)
    results = []

    for item in tqdm(test_data):
        question = item['question']
        choices = item['answers']
        ground_truth = item['solution']
        
        answers_text = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        
        # Crafting a prompt that encourages a single-letter response
        prompt = f"""### Question:
{question}

### Choices:
{answers_text}

### Instructions:
Identify the correct choice. Output ONLY the response in the format "Answer: X" where X is the letter of the choice.

### Answer:
"""
        
        raw_response = get_ollama_response(prompt)
        if raw_response is None:
            continue
            
        model_answer = extract_answer(raw_response)
        
        is_correct = (model_answer == ground_truth)
        if is_correct:
            correct += 1
            
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "raw_response": raw_response,
            "correct": is_correct
        })

    accuracy = (correct / total) * 100
    print(f"\nEvaluation Complete!")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    output_report = {
        "model": MODEL_NAME,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": results
    }
    
    with open(OUTPUT_FILENAME, "w") as f:
        json.dump(output_report, f, indent=2)
    print(f"Detailed results saved to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    evaluate()
