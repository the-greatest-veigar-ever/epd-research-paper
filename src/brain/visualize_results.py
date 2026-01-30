import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_results(log_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        history = json.load(f)

    # Extract metrics
    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []

    for entry in history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])
        if 'eval_loss' in entry:
            val_steps.append(entry['step'])
            val_loss.append(entry['eval_loss'])

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.6)
    if val_loss:
        plt.plot(val_steps, val_loss, label='Validation Loss', color='red', linewidth=2)
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Microsoft Phi-2 QLoRA)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_path = log_file.replace('.json', '_loss_curve.png')
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="ai/models/qlora-hugging-face/qlora-secqa/training_log.json")
    args = parser.parse_args()
    
    plot_training_results(args.log_file)
