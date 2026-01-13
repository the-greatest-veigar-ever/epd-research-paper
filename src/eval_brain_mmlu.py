"""
EPD Brain Evaluation (Group 2: The Brain)

This module evaluates the 'Brain' component using the MMLU (Massive Multitask 
Language Understanding) benchmark. It uses a local LLM to test reasoning, 
formal logic, and security knowledge capabilities.

Author: EPD Research Team
Version: 2.0.0
"""

import os
import glob
import ast
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Constants
DATA_DIR = "data/brain"
MODEL_NAME = "google/flan-t5-small"
MAX_LENGTH = 512

def load_model() -> tuple[Any, Any]:
    """Loads the tokenizer and model from Hugging Face."""
    print(f"[Brain-Eval] Loading Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def evaluate_subject(subject_path: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    """
    Evaluates the model on a single MMLU subject CSV.

    Args:
        subject_path: Path to the CSV file.
        tokenizer: Loaded HF tokenizer.
        model: Loaded HF model.

    Returns:
        Dict: Metrics for this subject.
    """
    subject_name = os.path.basename(subject_path).replace("mmlu_", "").replace(".csv", "")
    print(f"--- Evaluating: {subject_name} ---")
    
    try:
        df = pd.read_csv(subject_path)
    except Exception as e:
        print(f"Error reading {subject_path}: {e}")
        return {"Subject": subject_name, "Accuracy": 0.0, "Samples": 0}

    correct = 0
    total = 0
    
    # Silence tqdm if running in automated pipeline (optional, keeping visible for now)
    iterator = tqdm(df.iterrows(), total=len(df), desc=f"Processing {subject_name}")

    for _, row in iterator:
        try:
            # Flexible Column Parsing
            if 'question' in row:
                question = row['question']
                choices = row['choices']
                answer_idx = row['answer']
            else:
                # Fallback for headerless CSV
                question = row[0]
                choices = row[1]
                answer_idx = row[2]
            
            # Parse choices safety
            if isinstance(choices, str):
                try:
                    choices = ast.literal_eval(choices)
                except (ValueError, SyntaxError):
                    continue # Skip malformed rows
            
            # Construct Prompt
            options_text = ""
            labels = ['A', 'B', 'C', 'D']
            for i, choice in enumerate(choices):
                if i < 4:
                    options_text += f"{labels[i]}) {choice}\n"
            
            prompt = f"Question: {question}\n{options_text}\nAnswer:"
            
            # Inference
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=2)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
            
            # Validation
            ground_truth = labels[int(answer_idx)]
            
            # Matching Logic
            if prediction == ground_truth or (len(prediction) > 1 and prediction.startswith(ground_truth + ")")):
                correct += 1
            elif isinstance(choices, list) and len(choices) > int(answer_idx):
                 if choices[int(answer_idx)].upper() in prediction:
                     correct += 1
            
            total += 1
        except Exception:
            # Logging could go here
            continue
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    return {"Subject": subject_name, "Accuracy": round(accuracy, 2), "Samples": total}

def run_group2_evaluation() -> Dict[str, Any]:
    """
    Entry point for Group 2 MMLU Evaluation.

    Returns:
        Dict: Aggregated results across all subjects.
    """
    csv_files = glob.glob(f"{DATA_DIR}/mmlu_*.csv")
    if not csv_files:
        print("[Brain-Eval] No MMLU CSV files found.")
        return {}

    tokenizer, model = load_model()
    raw_results = []
    
    for f in csv_files:
        res = evaluate_subject(f, tokenizer, model)
        raw_results.append(res)
        
    # Aggregate
    df_res = pd.DataFrame(raw_results)
    avg_acc = df_res['Accuracy'].mean() if not df_res.empty else 0
    
    print("\n=== GROUP 2 RESULTS (MMLU) ===")
    print(df_res)
    print(f"Average Accuracy: {avg_acc:.2f}%")
    
    return {
        "mmlu_average_accuracy": round(avg_acc, 2),
        "total_subjects": len(df_res),
        "details": raw_results
    }

if __name__ == "__main__":
    run_group2_evaluation()
