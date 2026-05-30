import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40

base_path = os.path.dirname(os.path.abspath(__file__))

# --- 1. Load Data for Chart 1 (Overall) ---
df_overall = pd.read_csv(os.path.join(base_path, "leading_overall.csv"))
df_overall_melted = df_overall.melt(id_vars='Architecture', var_name='Metric', value_name='Count')
df_overall_melted['Percentage'] = (df_overall_melted['Count'] / 10) * 100

# --- 2. Load Data for Chart 2 (SLM Only) ---
df_slm = pd.read_csv(os.path.join(base_path, "leading_slm.csv"))
df_slm_melted = df_slm.melt(id_vars='Architecture', var_name='Metric', value_name='Count')
df_slm_melted['Percentage'] = (df_slm_melted['Count'] / 50) * 100

# --- 3. Load Data for Chart 3 (Memory Usage) ---
# Replicating logic from generate_charts.py
df_asr = pd.read_csv(os.path.join(base_path, 'asr1.csv'))
df_tsr = pd.read_csv(os.path.join(base_path, 'tsr1.csv'))
df_sizes = pd.read_csv(os.path.join(base_path, 'model_sizes.csv'))

size_map = dict(zip(df_sizes['Model'], df_sizes['RAM_GB']))
def get_size(approach):
    base = approach.replace('_static', '').replace('_suicide', '')
    return size_map.get(base, np.nan)

# We only need the approaches and their categories for memory normalization
llm_static_models = ['gpt_120b_oss_static', 'llama33_70b_static']
def categorize(approach):
    if approach in llm_static_models: return 'LLM Static Architecture'
    if '_suicide' in approach: return 'EPD Framework'
    return 'SLM Static Architecture'

# Collect all unique approaches from the files
all_approaches = pd.concat([df_asr['Approach'], df_tsr['Approach']]).unique()
df_mem = pd.DataFrame({'Approach': all_approaches})
df_mem['Category'] = df_mem['Approach'].apply(categorize)
df_mem['RAM_GB'] = df_mem['Approach'].apply(get_size)

# Group by category and get mean RAM_GB
mem_cats = df_mem.groupby('Category')['RAM_GB'].mean().reset_index()
mem_cats['Percentage'] = (mem_cats['RAM_GB'] / mem_cats['RAM_GB'].max()) * 100
mem_cats['Metric'] = 'Size'

# --- 4. Prepare for Plotting ---
def wrap_label(label):
    return label.replace(' Architecture', '\nArchitecture').replace(' Framework', '\nFramework')

df_overall_melted['Architecture'] = df_overall_melted['Architecture'].apply(wrap_label)
df_slm_melted['Architecture'] = df_slm_melted['Architecture'].apply(wrap_label)
mem_cats['Category'] = mem_cats['Category'].apply(wrap_label)

arch_order = [wrap_label(a) for a in ['LLM Static Architecture', 'EPD Framework', 'SLM Static Architecture']]

fig, axes = plt.subplots(1, 3, figsize=(24, 14), sharey=True)

# Subplot 1: Overall Performance
sns.barplot(data=df_overall_melted, y='Architecture', x='Percentage', hue='Metric', 
            order=arch_order, palette='viridis', ax=axes[0], width=0.6)
axes[0].set_title("Global Architecture Comparison", pad=15)
axes[0].set_xlabel("Win Rate (%)", labelpad=10)
axes[0].set_ylabel("Architecture", labelpad=10)
axes[0].set_xlim(0, 115)

# Subplot 2: SLM vs EPD
sns.barplot(data=df_slm_melted, y='Architecture', x='Percentage', hue='Metric', 
            order=arch_order, palette='viridis', ax=axes[1], legend=False, width=0.6)
axes[1].set_title("SLM Pairwise Comparison:\nStatic vs. EPD Framework", pad=15)
axes[1].set_xlabel("Win Rate (%)", labelpad=10)
axes[1].set_ylabel("") # Shared Y
axes[1].set_xlim(0, 115)

# Subplot 3: Relative Memory Usage
sns.barplot(data=mem_cats, y='Category', x='Percentage', color='#2ca02c', # Greenish
            order=arch_order, ax=axes[2], width=0.3) # Narrower since it only has 1 bar
axes[2].set_title("Relative Memory Usage\n(Normalized to Max Model)", pad=15)
axes[2].set_xlabel("Memory Usage (%)", labelpad=10)
axes[2].set_ylabel("") # Shared Y
axes[2].set_xlim(0, 115)

# Add annotations
for ax in axes:
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{width:.0f}%', 
                        (width, p.get_y() + p.get_height() / 2.), 
                        ha = 'left', va = 'center', 
                        xytext = (5, 0), 
                        textcoords = 'offset points',
                        fontsize=40)

# Global adjustments
sns.despine(left=True, bottom=True)
plt.tight_layout()

# Adjust legends
axes[0].legend(title='Metric', loc='lower left', bbox_to_anchor=(-0.45, -0.15), fontsize=40, title_fontsize=40)

# Bold and center "EPD Framework" labels on the Y-axis
for ax in axes:
    for label in ax.get_yticklabels():
        # label.set_horizontalalignment('center')
        label.set_multialignment('center')
        if 'EPD' in label.get_text():
            label.set_fontweight('bold')

output_path = os.path.join(base_path, "charts", "04_combined_performance_metrics.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
print(f"Combined chart saved to {output_path}")
