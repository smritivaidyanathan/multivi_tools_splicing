import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style and font sizes
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

# Read the data
df = pd.read_csv("/gpfs/commons/home/kisaev/multivi_tools_splicing/results/latent_space_eval/run_20250527_094736/subcluster_eval/csv_files/subcluster_metrics.csv")

# Calculate mean and standard error across cell types
summary_df = df.groupby(['rep', 'k', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()
summary_df['std_err'] = summary_df['std'] / np.sqrt(summary_df['count'])

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

# Define metrics and their positions
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Define colors
colors = {
    'joint': '#1f77b4',  # Blue
    'expression': '#ff7f0e',  # Orange
    'splicing': '#2ca02c',  # Green
    'random': '#7f7f7f'  # Grey
}

# Plot each metric
for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    
    # Plot each representation
    for rep in summary_df['rep'].unique():
        data = summary_df[(summary_df['metric'] == metric) & (summary_df['rep'] == rep)]
        ax.errorbar(data['k'], data['mean'], 
                   yerr=data['std_err'],
                   marker='o',
                   capsize=5,
                   capthick=1,
                   elinewidth=1,
                   label=rep.capitalize(),
                   color=colors.get(rep, 'grey'),
                   linewidth=2)
    
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to match k values
    ax.set_xticks(sorted(summary_df['k'].unique()))
    
    # Set y-axis limits to be consistent across plots
    ax.set_ylim(0, 1)

# Add legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, 
          title='Representation',
          loc='upper right',
          bbox_to_anchor=(0.98, 0.98))

# Adjust layout
plt.tight_layout()

# Save figure as PDF
plt.savefig('subcluster_metrics_summary.pdf', bbox_inches='tight')
plt.close()

# make plot for just k=3
df = df[df['k'] == 3]

# Create separate bar plots for each metric
for metric, title in zip(metrics, titles):
    plt.figure(figsize=(6, 6))
    
    # Create bar plot
    sns.barplot(data=df[df['metric'] == metric],
                x='cell_type',
                y='value',
                hue='rep',
                errorbar='se',
                capsize=0.1,
                palette=colors)
    
    plt.xlabel('Cell Type')
    plt.ylabel(title)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Remove legend
    plt.legend().remove()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure as PDF
    plt.savefig(f'subcluster_metrics_{metric}.pdf', bbox_inches='tight')
    plt.close()