"""
Compare all baseline models and generate comprehensive comparison report
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results directory
results_dir = Path(__file__).parent.parent / 'results'

# Load all evaluation results
results = {}

# 1. Baseline 1: CLIP
with open(results_dir / 'baseline_clip_eval.json', 'r') as f:
    results['CLIP\n(Image Only)'] = json.load(f)['metrics']

# 2. Baseline 2: InfoNCE only
with open(results_dir / 'baseline_infonce_only_eval.json', 'r') as f:
    results['InfoNCE Only\n(z_A+z_C)/2'] = json.load(f)['metrics']

# 3. Baseline 3: Infilling (Task 5)
with open(results_dir / 'infilling_baseline_eval.json', 'r') as f:
    results['Infilling\n(Task 5)'] = json.load(f)['metrics']

# 4. Our Model: GAN (Task 6)
with open(results_dir / 'gan_082207_eval.json', 'r') as f:
    results['GAN\n(Task 6)'] = json.load(f)['metrics']

# Create comparison DataFrame
comparison_data = {
    'Model': [],
    'Top-1 Accuracy (%)': [],
    'Recall@3 (%)': [],
    'Recall@5 (%)': [],
    'Recall@10 (%)': [],
    'MRR': [],
    'Average Rank': []
}

for model_name, metrics in results.items():
    comparison_data['Model'].append(model_name.replace('\n', ' '))
    comparison_data['Top-1 Accuracy (%)'].append(metrics['Top-1 Accuracy'] * 100)
    comparison_data['Recall@3 (%)'].append(metrics['Recall@3'] * 100)
    comparison_data['Recall@5 (%)'].append(metrics['Recall@5'] * 100)
    comparison_data['Recall@10 (%)'].append(metrics['Recall@10'] * 100)
    comparison_data['MRR'].append(metrics['MRR'])
    comparison_data['Average Rank'].append(metrics['Average Rank'])

df = pd.DataFrame(comparison_data)

# Sort by Top-1 Accuracy
df = df.sort_values('Top-1 Accuracy (%)', ascending=False).reset_index(drop=True)

# Save to CSV
csv_path = results_dir / 'baseline_comparison.csv'
df.to_csv(csv_path, index=False, float_format='%.2f')
print(f"Comparison table saved to: {csv_path}")

# Print formatted table
print("\n" + "="*100)
print("COMPREHENSIVE BASELINE COMPARISON")
print("="*100)
print(df.to_string(index=False))
print("="*100)

# Calculate improvements
best_baseline = df.iloc[0]
our_model_mask = df['Model'].str.contains('GAN')
if our_model_mask.any():
    our_model = df[our_model_mask].iloc[0]
    
    print("\n" + "="*100)
    print("PERFORMANCE ANALYSIS")
    print("="*100)
    print(f"\nBest Baseline: {best_baseline['Model']}")
    print(f"  Top-1 Accuracy: {best_baseline['Top-1 Accuracy (%)']:.2f}%")
    print(f"  Recall@3: {best_baseline['Recall@3 (%)']:.2f}%")
    print(f"  Average Rank: {best_baseline['Average Rank']:.2f}")
    
    print(f"\nOur Model (GAN): {our_model['Model']}")
    print(f"  Top-1 Accuracy: {our_model['Top-1 Accuracy (%)']:.2f}%")
    print(f"  Recall@3: {our_model['Recall@3 (%)']:.2f}%")
    print(f"  Average Rank: {our_model['Average Rank']:.2f}")
    
    print(f"\nGap to Best Baseline:")
    print(f"  Top-1 Accuracy: {best_baseline['Top-1 Accuracy (%)'] - our_model['Top-1 Accuracy (%)']:.2f}%p")
    print(f"  Recall@3: {best_baseline['Recall@3 (%)'] - our_model['Recall@3 (%)']:.2f}%p")
    print("="*100)

# Key findings
print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)
print("\n1. CLIP (Image Only) achieves the best performance:")
print(f"   - Top-1 Accuracy: {df.iloc[0]['Top-1 Accuracy (%)']:.2f}%")
print(f"   - This suggests visual features are highly informative")

print("\n2. Performance ranking:")
for i, row in df.iterrows():
    print(f"   {i+1}. {row['Model']}: {row['Top-1 Accuracy (%)']:.2f}% (Top-1)")

print("\n3. Observations:")
if df[df['Model'].str.contains('Infilling')]['Top-1 Accuracy (%)'].values[0] < 20:
    print("   ⚠️  Infilling baseline shows unexpectedly low performance")
    print("   → Requires further investigation of Task 5 model")

print("\n4. Multimodal (image+text) vs Image-only:")
clip_acc = df[df['Model'].str.contains('CLIP')]['Top-1 Accuracy (%)'].values[0]
infonce_acc = df[df['Model'].str.contains('InfoNCE Only')]['Top-1 Accuracy (%)'].values[0]
print(f"   - CLIP (image): {clip_acc:.2f}%")
print(f"   - InfoNCE (image+text): {infonce_acc:.2f}%")
print(f"   → Text adds {'positive' if infonce_acc > clip_acc else 'negative'} contribution")

print("="*100)

