import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def visualize_latest_result():
    # Find the latest results file
    results_dir = "Simulation Test/02_Q1_Ablation_Study"
    files = glob.glob(f"{results_dir}/*.xlsx")
    if not files:
        print("No result files found.")
        return

    latest_file = max(files, key=os.path.getctime)
    print(f"Visualizing results from: {latest_file}")
    
    try:
        df = pd.read_excel(latest_file)
    except:
        df = pd.read_csv(latest_file.replace(".xlsx", ".csv"))

    # Extract Defense Rates matches
    # The structure we saved was:
    # Metric | Baseline | EPD | Improvement
    # Defense Rate | 56.97 | 67.27 | 10.3
    
    row = df[df['Metric'] == 'Defense Rate'].iloc[0]
    baseline_acc = float(row['Baseline'])
    epd_acc = float(row['EPD'])
    
    # Create Chart
    labels = ['Persistent Baseline', 'EPD (Ours)']
    values = [baseline_acc, epd_acc]
    colors = ['#bdc3c7', '#2ecc71'] # Grey for baseline, Green for EPD

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')

    plt.title(f"Ablation Study Results (N={int(df[df['Metric'] == 'Total Samples']['Baseline'].values[0])})")
    plt.ylabel("Attack Success Rate (Defense) %")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    output_path = f"{results_dir}/EPD_Result_Chart.png"
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    visualize_latest_result()
