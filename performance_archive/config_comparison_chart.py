import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style — identical to leading_overall_chart.py
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 18

base_path = "/EDP Research/epd-research-paper/performance_archive/"
csv_path = os.path.join(base_path, "deployment_comparison.csv")

# Load data
df = pd.read_csv(csv_path)

# Melt for seaborn grouped bar plot
df_melted = df.melt(id_vars='Architecture', var_name='Metric', value_name='Count')

# Convert to percentage (10 benchmarks total)
df_melted['Percentage'] = (df_melted['Count'] / 10) * 100

plt.figure(figsize=(13, 9))
ax = sns.barplot(data=df_melted, x='Percentage', y='Architecture', hue='Metric', palette='viridis', orient='h')

# Add labels at end of bars
for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.annotate(f'{width:.0f}%',
                    (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center',
                    xytext=(6, 0),
                    textcoords='offset points',
                    fontsize=16, fontweight='bold')

plt.xlabel("Win Rate (%)", fontsize=20, labelpad=15)
plt.ylabel("Deployment Configuration", fontsize=20, labelpad=15)
plt.xlim(0, 115)

# Improve layout
sns.despine(left=True, bottom=True)
plt.legend(title='Evaluation Metric', title_fontsize=18, bbox_to_anchor=(0.7, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()

output_path = os.path.join(base_path, "charts", "deployment_configuration_comparison.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
print(f"Chart saved to {output_path}")
