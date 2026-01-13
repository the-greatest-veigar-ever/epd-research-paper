import os
import glob
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import ast

DATA_DIR = "data/brain"
MODEL_NAME = "google/flan-t5-small"

def evaluate_mmlu():
    print(f"\n[+] Loading Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    csv_files = glob.glob(f"{DATA_DIR}/mmlu_*.csv")
    if not csv_files:
        print("No MMLU CSV files found.")
        return

    results = []

    for f in csv_files:
        subject = os.path.basename(f).replace("mmlu_", "").replace(".csv", "")
        print(f"\n--- Evaluating Subject: {subject} ---")
        
        df = pd.read_csv(f)
        correct = 0
        total = 0
        
        # Debug: Print first row
        if not df.empty:
            print(f"Sample Row keys: {df.iloc[0].keys()}")
            print(f"Sample Row values: {df.iloc[0].values}")

        for index, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Flexible Column Access
                if 'question' in row:
                    question = row['question']
                    choices = row['choices']
                    answer_idx = row['answer']
                else:
                    # Fallback for headerless CSV (rare with datasets library but possible)
                    question = row[0]
                    choices = row[1]
                    answer_idx = row[2]
                
                # Parse choices if string
                if isinstance(choices, str):
                    try:
                        choices = ast.literal_eval(choices)
                    except:
                        pass
                
                # Create Prompt
                options_text = ""
                labels = ['A', 'B', 'C', 'D']
                for i, choice in enumerate(choices):
                    if i < 4:
                        options_text += f"{labels[i]}) {choice}\n"
                
                prompt = f"Question: {question}\n{options_text}\nAnswer:"
                
                # Inference
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                outputs = model.generate(**inputs, max_new_tokens=2)
                prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
                
                # Ground Truth
                ground_truth = labels[int(answer_idx)]
                
                # Check for direct match (e.g. "A" or "Option A")
                # Flan-T5 is good at outputting the specific token if prompted well
                if prediction == ground_truth or (len(prediction) > 1 and prediction.startswith(ground_truth + ")")):
                    correct += 1
                elif prediction in choices: # If model outputs the full text answer
                     # This is harder to match perfectly without fuzzy matching, but let's try strict first
                     if choices[int(answer_idx)].upper() in prediction:
                         correct += 1
                
                total += 1
            except Exception as e:
                # print(f"Error: {e}") 
                pass
                
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Subject: {subject} | Accuracy: {accuracy:.2f}% ({correct}/{total})")
        results.append({"Subject": subject, "Accuracy": accuracy, "Samples": total})

    # Save Results
    res_df = pd.DataFrame(results)
    print("\n=== FINAL MMLU RESULTS ===")
    print(res_df)
    res_df.to_csv("Simulation Test/02_Q1_Ablation_Study/MMLU_Results.csv", index=False)

if __name__ == "__main__":
    evaluate_mmlu()
