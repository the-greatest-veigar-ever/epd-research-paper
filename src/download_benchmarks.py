import os
import subprocess
from datasets import load_dataset


DATA_DIR_BRAIN = "data/brain"
DATA_DIR_GHOST = "data/ghost_agents"
os.makedirs(DATA_DIR_BRAIN, exist_ok=True)
os.makedirs(DATA_DIR_GHOST, exist_ok=True)

def download_mmlu():
    subjects = ["computer_security", "formal_logic", "logical_fallacies", "machine_learning"]
    print(f"\n[+] Downloading MMLU categories: {subjects}...")
    
    total_questions = 0
    for subject in subjects:
        try:
            print(f"   Downloading {subject}...")
            # Download specific subset relevant to security
            dataset = load_dataset("cais/mmlu", subject, split="test", cache_dir=f"{DATA_DIR_BRAIN}/mmlu_cache")
            
            # Save as CSV for easy viewing
            df = dataset.to_pandas()
            output_path = f"{DATA_DIR_BRAIN}/mmlu_{subject}.csv"
            df.to_csv(output_path, index=False)
            total_questions += len(df)
            print(f"   Saved {len(df)} questions to {output_path}")
        except Exception as e:
            print(f"FAILED to download MMLU ({subject}): {e}")
            
    print(f"SUCCESS: Total MMLU questions ready: {total_questions}")

def download_halueval():
    print("\n[+] Downloading HaluEval (via GitHub Clone)...")
    target_dir = f"{DATA_DIR_BRAIN}/HaluEval"
    if os.path.exists(target_dir):
        print("HaluEval already exists. Skipping.")
        return
        
    try:
        subprocess.run(["git", "clone", "https://github.com/RUCAIBox/HaluEval", target_dir], check=True)
        print("SUCCESS: HaluEval cloned.")
    except Exception as e:
        print(f"FAILED to clone HaluEval: {e}")

def download_asb():
    print("\n[+] Downloading Agent Security Bench (via GitHub Clone)...")
    target_dir = f"{DATA_DIR_GHOST}/ASB"
    if os.path.exists(target_dir):
        print("ASB already exists. Skipping.")
        return

    try:
        subprocess.run(["git", "clone", "https://github.com/agiresearch/ASB", target_dir], check=True)
        print("SUCCESS: ASB cloned.")
    except Exception as e:
        print(f"FAILED to clone ASB: {e}")

if __name__ == "__main__":
    print(f"Downloading benchmarks...")
    download_mmlu()
    download_halueval()
    download_asb()
    print("\nDone! Datasets are ready for offline use.")
