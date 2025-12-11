"""
Visualize comprehensive baseline comparison
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results directory
results_dir = Path(__file__).parent.parent / 'results'

# Load CSV
df = pd.read_csv(results_dir / 'baseline_comparison.csv')

# Prepare data for plotting
models = [m.replace(' ', '\n') for m in df['Model'].values]
metrics = {
    'Top-1 Acc': df['Top-1 Accuracy (%)'].values,
    'Recall@3': df['Recall@3 (%)'].values,
    'Recall@5': df['Recall@5 (%)'].values,
    'MRR': df['MRR'].values * 100,  # Convert to percentage
}
avg_ranks = df['Average Rank'].values

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Colors for each model
colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']  # Green, Blue, Red, Gray

# Plot 1: Top-1 Accuracy
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(models)), metrics['Top-1 Acc'], color=colors, alpha=0.8)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Top-1 Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylim(0, 80)
ax1.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, metrics['Top-1 Acc'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Recall@K
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(models))
width = 0.25
r3 = ax2.bar(x - width, metrics['Recall@3'], width, label='Recall@3', color='#ff7f0e', alpha=0.8)
r5 = ax2.bar(x, metrics['Recall@5'], width, label='Recall@5', color='#2ca02c', alpha=0.8)
ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax2.set_title('Recall@K Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylim(0, 110)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: MRR
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(range(len(models)), metrics['MRR'], color=colors, alpha=0.8)
ax3.set_ylabel('MRR (%)', fontsize=12, fontweight='bold')
ax3.set_title('Mean Reciprocal Rank', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, fontsize=9)
ax3.set_ylim(0, 90)
ax3.grid(axis='y', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, metrics['MRR'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 4: Average Rank (lower is better)
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(range(len(models)), avg_ranks, color=colors, alpha=0.8)
ax4.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
ax4.set_title('Average Rank (Lower is Better)', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, fontsize=9)
ax4.set_ylim(0, 6)
ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Perfect (Rank 1)')
ax4.grid(axis='y', alpha=0.3)
ax4.legend(fontsize=9)
for i, (bar, val) in enumerate(zip(bars, avg_ranks)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 5: Radar chart for comprehensive comparison
ax5 = fig.add_subplot(gs[1, 1:], projection='polar')

# Metrics for radar
categories = ['Top-1\nAcc', 'Recall@3', 'Recall@5', 'MRR', 'Rank\n(inv)']
N = len(categories)

# Normalize all metrics to 0-100 scale
# For Average Rank, invert it (lower is better -> higher is better for radar)
max_rank = 10  # Assume max rank is 10
normalized_data = []
for i in range(len(models)):
    data = [
        metrics['Top-1 Acc'][i],
        metrics['Recall@3'][i],
        metrics['Recall@5'][i],
        metrics['MRR'][i],
        (max_rank - avg_ranks[i]) / max_rank * 100  # Invert and normalize
    ]
    normalized_data.append(data)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot each model
for i, (model, data) in enumerate(zip(models, normalized_data)):
    values = data + data[:1]
    ax5.plot(angles, values, 'o-', linewidth=2, label=model.replace('\n', ' '), color=colors[i], alpha=0.8)
    ax5.fill(angles, values, alpha=0.15, color=colors[i])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=10)
ax5.set_ylim(0, 100)
ax5.set_title('Comprehensive Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax5.grid(True)

# Main title
fig.suptitle('Baseline Models Comprehensive Comparison', fontsize=18, fontweight='bold', y=0.98)

# Save figure
output_path = results_dir / 'baseline_comparison_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

plt.show()

