"""
Baseline Models Evaluation
Evaluate different baseline approaches for middle panel selection
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from datasets.panel_dataset import TripletDataset, get_default_transform


def create_middle_panel_selection_task(
    triplet_dataset: TripletDataset,
    num_candidates: int = 10,
    num_samples: int = None
) -> List[Dict]:
    """
    Create middle panel selection tasks from triplet dataset
    
    Args:
        triplet_dataset: Dataset containing triplets
        num_candidates: Total number of candidates (including correct one)
        num_samples: Number of samples to evaluate (None = all)
    
    Returns:
        List of tasks with A, C, candidates, and correct_idx
    """
    tasks = []
    dataset_size = len(triplet_dataset)
    if num_samples is not None:
        dataset_size = min(num_samples, dataset_size)
    
    print(f"Creating {dataset_size} middle panel selection tasks...")
    
    for idx in tqdm(range(dataset_size), desc="Creating tasks"):
        try:
            triplet_data = triplet_dataset[idx]
            A = triplet_data['A']
            B_correct = triplet_data['B']
            C = triplet_data['C']
            
            # Sample negative candidates
            other_indices = [i for i in range(len(triplet_dataset)) if i != idx]
            if len(other_indices) < num_candidates - 1:
                continue
            
            neg_indices = random.sample(other_indices, num_candidates - 1)
            selected_negatives = [triplet_dataset[neg_idx]['B'] for neg_idx in neg_indices]
            
            # Create candidate list and shuffle
            candidates = [B_correct] + selected_negatives
            correct_idx = 0
            indices = list(range(len(candidates)))
            random.shuffle(indices)
            
            shuffled_candidates = [candidates[i] for i in indices]
            shuffled_correct_idx = indices.index(correct_idx)
            
            task = {
                'triplet_idx': idx,
                'A': A,
                'C': C,
                'candidates': shuffled_candidates,
                'correct_idx': shuffled_correct_idx
            }
            tasks.append(task)
            
        except Exception as e:
            print(f"Error creating task {idx}: {e}")
            continue
    
    print(f"Created {len(tasks)} valid tasks")
    return tasks


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics"""
    total = len(results)
    
    # Top-1 Accuracy
    correct_count = sum(1 for r in results if r['is_correct'])
    top1_acc = correct_count / total if total > 0 else 0.0
    
    # Recall@K
    recall_at_k = {}
    for k in [3, 5, 10]:
        recall_k = sum(1 for r in results if r['rank'] <= k) / total if total > 0 else 0.0
        recall_at_k[f'Recall@{k}'] = recall_k
    
    # MRR
    reciprocal_ranks = [1.0 / r['rank'] for r in results]
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Average rank
    avg_rank = np.mean([r['rank'] for r in results]) if results else 0.0
    
    metrics = {
        'Top-1 Accuracy': top1_acc,
        **recall_at_k,
        'MRR': mrr,
        'Average Rank': avg_rank,
        'Total Samples': total
    }
    
    return metrics


def print_metrics(metrics: Dict, model_name: str):
    """Print metrics in formatted table"""
    print("\n" + "=" * 70)
    print(f"{model_name} - Evaluation Results")
    print("=" * 70)
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            if 'Accuracy' in metric_name or 'Recall' in metric_name or 'MRR' in metric_name:
                print(f"{metric_name:20s}: {metric_value:.4f} ({metric_value*100:.2f}%)")
            else:
                print(f"{metric_name:20s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:20s}: {metric_value}")
    print("=" * 70)


def save_results(output_path: Path, metrics: Dict, config: Dict, results: List[Dict]):
    """Save evaluation results to JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metrics': metrics,
        'config': config,
        'detailed_results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

