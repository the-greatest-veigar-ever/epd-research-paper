import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def visualize_dashboard():
    results_dir = "Simulation Test/02_Q1_Ablation_Study"
    output_path = f"{results_dir}/EPD_Full_Evaluation_Dashboard.png"
    
    # Setup Figure
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('EPD Architecture: Comprehensive Evaluation Results', fontsize=16, fontweight='bold')
    
    # --- PLOT 1: Watchers (Group 1) ---
    ax1 = fig.add_subplot(131)
    
    # Find latest Watcher result
    files = glob.glob(f"{results_dir}/EPD_Result_*.xlsx")
    if files:
        latest_file = max(files, key=os.path.getctime)
        df1 = pd.read_excel(latest_file)
        row = df1[df1['Metric'] == 'Defense Rate'].iloc[0]
        values1 = [row['Baseline'], row['EPD']]
        labels1 = ['IDS Baseline', 'EPD Watchers']
        colors1 = ['#bdc3c7', '#2ecc71']
        
        bars1 = ax1.bar(labels1, values1, color=colors1)
        ax1.set_title("Group 1: Intrusion Detection (Real)")
        ax1.set_ylabel("Defense Success Rate (%)")
        ax1.set_ylim(0, 100)
        
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "No Watcher Data", ha='center')

    # --- PLOT 2: Brain (Group 2 - MMLU) ---
    ax2 = fig.add_subplot(132)
    mmlu_file = f"{results_dir}/MMLU_Results.csv"
    if os.path.exists(mmlu_file):
        df2 = pd.read_csv(mmlu_file)
        # Plot Accuracy by Subject
        if not df2.empty:
            subjects = df2['Subject'].tolist()
            accs = df2['Accuracy'].tolist()
            
            bars2 = ax2.barh(subjects, accs, color='#3498db')
            ax2.set_title(f"Group 2: Brain Intelligence (MMLU)")
            ax2.set_xlabel("Accuracy (%)")
            ax2.set_xlim(0, 100)
            
            for bar in bars2:
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontweight='bold')
        else:
             ax2.text(0.5, 0.5, "Empty MMLU Data", ha='center')
    else:
         ax2.text(0.5, 0.5, "No MMLU Data", ha='center')

    # --- PLOT 3: Ghost Agents (Group 3 - ASB) ---
    ax3 = fig.add_subplot(133)
    asb_file = f"{results_dir}/ASB_Ghost_Results.csv"
    if os.path.exists(asb_file):
        df3 = pd.read_csv(asb_file)
        # Macro Average
        avg_base = df3['Baseline'].mean() * 100
        avg_epd = df3['EPD'].mean() * 100
        
        values3 = [avg_base, avg_epd]
        labels3 = ['Static Agent', 'EPD Polymorphic']
        colors3 = ['#e74c3c', '#9b59b6']
        
        bars3 = ax3.bar(labels3, values3, color=colors3)
        ax3.set_title("Group 3: Agent Security Bench (Simulated)")
        ax3.set_ylabel("Resistance Rate (%)")
        ax3.set_ylim(0, 100)
        
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "No ASB Data", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Dashboard saved to: {output_path}")

if __name__ == "__main__":
    visualize_dashboard()
