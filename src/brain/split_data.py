import json
import random
import os
from math import floor

# CONFIG
DATA_PATH = "ai/data/brain/data/combined_datasets/all_training_data.jsonl"
OUTPUT_DIR = "ai/data/brain/data/splits"
SPLIT_RATIO = (0.6, 0.2, 0.2) # Train, Val, Test

def split_data():
    print(f"Loading data from {DATA_PATH}...")
    
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        
    total_samples = len(lines)
    print(f"Total samples: {total_samples}")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(lines)
    
    # Calculate indices
    train_end = floor(total_samples * SPLIT_RATIO[0])
    val_end = train_end + floor(total_samples * SPLIT_RATIO[1])
    
    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]
    
    print(f"Split results:")
    print(f"  Train: {len(train_data)} ({len(train_data)/total_samples:.1%})")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total_samples:.1%})")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total_samples:.1%})")
    
    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save Full Splits
    paths = {
        "train": os.path.join(OUTPUT_DIR, "train.jsonl"),
        "val": os.path.join(OUTPUT_DIR, "val.jsonl"),
        "test": os.path.join(OUTPUT_DIR, "test.jsonl")
    }
    
    with open(paths["train"], 'w') as f: f.writelines(train_data)
    with open(paths["val"], 'w') as f: f.writelines(val_data)
    with open(paths["test"], 'w') as f: f.writelines(test_data)
    
    print(f"Full splits saved to {OUTPUT_DIR}")
    
    # Save 10% Subsets (Smoke Test)
    subset_ratio = 0.10
    subset_train = train_data[:int(len(train_data) * subset_ratio)]
    subset_val = val_data[:int(len(val_data) * subset_ratio)]
    
    subset_paths = {
        "train": os.path.join(OUTPUT_DIR, "train_subset.jsonl"),
        "val": os.path.join(OUTPUT_DIR, "val_subset.jsonl")
    }
    
    with open(subset_paths["train"], 'w') as f: f.writelines(subset_train)
    with open(subset_paths["val"], 'w') as f: f.writelines(subset_val)
    
    print(f"10% Subset splits saved:")
    print(f"  Train Subset: {len(subset_train)}")
    print(f"  Val Subset:   {len(subset_val)}")

if __name__ == "__main__":
    split_data()
