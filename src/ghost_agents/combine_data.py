import json
import glob
import os

INPUT_DIR = "ai/data/ghost_agents/ASB/data"
OUTPUT_FILE = "ai/data/ghost_agents/combined_scenarios.jsonl"

def combine_data():
    scenarios = []
    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    print(f"Found {len(files)} JSONL files.")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    
                    # Extraction Logic
                    # Type 1: agent_task*.jsonl -> has "tasks" list
                    if "tasks" in data and isinstance(data["tasks"], list):
                        for task in data["tasks"]:
                            scenarios.append({
                                "source": filename,
                                "prompt": task,
                                "agent_context": data.get("agent_name", "unknown")
                            })
                            
                    # Type 2: all_attack_tools*.jsonl -> has "Attacker Instruction"
                    elif "Attacker Instruction" in data:
                         scenarios.append({
                                "source": filename,
                                "prompt": data["Attacker Instruction"],
                                "agent_context": data.get("Corresponding Agent", "unknown"),
                                "attack_type": data.get("Attack Type"),
                                "tool": data.get("Attacker Tool")
                            })
                            
                    # Type 3: Fallback (any text field?)
                    else:
                        pass # Ignore unknown schema for now
                        
                except Exception as e:
                    print(f"Error skipping line in {filename}: {e}")

    print(f"Total scenarios extracted: {len(scenarios)}")
    
    # Save combined file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")
            
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    combine_data()
