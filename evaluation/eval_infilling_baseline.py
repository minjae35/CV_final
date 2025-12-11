"""
Infilling Generator Baseline Evaluation
Evaluate middle panel selection using only infilling similarity
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
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


def load_models(encoder_path: Path, generator_path: Path, device: torch.device):
    """Load pretrained encoder and generator"""
    print(f"Loading encoder from: {encoder_path}")
    encoder = MultimodalEncoder(embedding_dim=128)
    encoder_checkpoint = torch.load(encoder_path, map_location=device)
    
    if 'encoder_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
    elif 'model_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_checkpoint)
    
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"Loading generator from: {generator_path}")
    generator_checkpoint = torch.load(generator_path, map_location=device)
    
    # Check if old simple MLP structure or new attention structure
    if 'generator_state_dict' in generator_checkpoint:
        state_dict = generator_checkpoint['generator_state_dict']
    elif 'model_state_dict' in generator_checkpoint:
        state_dict = generator_checkpoint['model_state_dict']
    else:
        state_dict = generator_checkpoint
    
    # Check if old model (has 'network.0.weight') or new model
    is_old_model = 'network.0.weight' in state_dict
    
    if is_old_model:
        # Old simple MLP structure: concat z_A and z_C, then 3-layer MLP
        from models.generator import InfillingGenerator
        # Create a simple wrapper for old model
        class OldInfillingGenerator(nn.Module):
            def __init__(self, embedding_dim=128, hidden_dim=256):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(embedding_dim * 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, embedding_dim)
                )
            
            def forward(self, z_a, z_c):
                z_concat = torch.cat([z_a, z_c], dim=1)
                return self.network(z_concat)
        
        generator = OldInfillingGenerator(embedding_dim=128, hidden_dim=256)
    else:
        # New attention-based structure
        generator = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4)
    
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()
    
    return encoder, generator


def create_middle_panel_selection_task(
    triplet_dataset: TripletDataset,
    num_candidates: int = 10,
    num_samples: int = None
) -> List[Dict]:
    """
    Create middle panel selection tasks from triplet dataset
    
    For each triplet (A, B, C):
    - Given: A and C
    - Task: Select correct B from {B_correct, B_neg1, B_neg2, ...}
    - Negatives: randomly sampled from other triplets in dataset
    
    Args:
        triplet_dataset: Dataset containing triplets
        num_candidates: Total number of candidates (including correct one)
        num_samples: Number of samples to evaluate (None = all)
    
    Returns:
        List of tasks, each containing:
        {
            'triplet_idx': int,
            'A': panel data,
            'C': panel data,
            'candidates': [B_correct, B_neg1, ...],
            'correct_idx': int (index of correct B in candidates)
        }
    """
    tasks = []
    
    dataset_size = len(triplet_dataset)
    if num_samples is not None:
        dataset_size = min(num_samples, dataset_size)
    
    print(f"Creating {dataset_size} middle panel selection tasks...")
    print(f"  - Candidates per task: {num_candidates}")
    print(f"  - Negatives sampled from other triplets")
    
    for idx in tqdm(range(dataset_size), desc="Creating tasks"):
        try:
            # Get triplet (A, B, C)
            triplet_data = triplet_dataset[idx]
            
            A = triplet_data['A']
            B_correct = triplet_data['B']
            C = triplet_data['C']
            
            # Sample negative candidates from other triplets
            # Avoid sampling from the same triplet
            other_indices = [i for i in range(len(triplet_dataset)) if i != idx]
            if len(other_indices) < num_candidates - 1:
                # Not enough other triplets
                continue
            
            neg_indices = random.sample(other_indices, num_candidates - 1)
            selected_negatives = []
            
            for neg_idx in neg_indices:
                neg_triplet = triplet_dataset[neg_idx]
                selected_negatives.append(neg_triplet['B'])
            
            # Create candidate list: [B_correct, B_neg1, B_neg2, ...]
            candidates = [B_correct] + selected_negatives
            
            # Shuffle candidates and record correct index
            correct_idx = 0  # Before shuffling
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


def evaluate_task(
    task: Dict,
    encoder: MultimodalEncoder,
    generator: InfillingGenerator,
    device: torch.device
) -> Dict:
    """
    Evaluate single middle panel selection task
    
    Returns:
        {
            'predicted_idx': int,
            'correct_idx': int,
            'is_correct': bool,
            'rank': int (1-based rank of correct answer),
            'similarities': List[float]
        }
    """
    A = task['A']
    C = task['C']
    candidates = task['candidates']
    correct_idx = task['correct_idx']
    
    # Encode A and C
    with torch.no_grad():
        img_A = A['image'].unsqueeze(0).to(device)
        text_A = [A['ocr_text']]
        z_A = encoder(img_A, text_A)  # [1, 128]
        
        img_C = C['image'].unsqueeze(0).to(device)
        text_C = [C['ocr_text']]
        z_C = encoder(img_C, text_C)  # [1, 128]
        
        # Generate predicted z_B
        z_B_hat = generator(z_A, z_C)  # [1, 128]
        z_B_hat = F.normalize(z_B_hat, dim=1)  # Normalize for similarity computation
        
        # Encode all candidates
        similarities = []
        for candidate in candidates:
            img_B = candidate['image'].unsqueeze(0).to(device)
            text_B = [candidate['ocr_text']]
            z_B = encoder(img_B, text_B)  # [1, 128]
            z_B = F.normalize(z_B, dim=1)
            
            # Compute similarity: cosine similarity (already normalized)
            sim = torch.sum(z_B_hat * z_B).item()
            similarities.append(sim)
    
    # Find predicted index (highest similarity)
    predicted_idx = int(np.argmax(similarities))
    is_correct = (predicted_idx == correct_idx)
    
    # Compute rank of correct answer (1-based)
    sorted_indices = np.argsort(similarities)[::-1]  # Descending order
    rank = int(np.where(sorted_indices == correct_idx)[0][0]) + 1
    
    return {
        'predicted_idx': predicted_idx,
        'correct_idx': correct_idx,
        'is_correct': is_correct,
        'rank': rank,
        'similarities': similarities
    }


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics
    
    Metrics:
    - Top-1 Accuracy: Correct prediction rate
    - Recall@K: Correct answer in top K predictions
    - MRR (Mean Reciprocal Rank): Average of 1/rank
    """
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Infilling Generator Baseline')
    parser.add_argument('--encoder', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--generator', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--data_dir', type=str, default=str(Path.home() / 'data' / 'raw_panel_images_small'),
                        help='Directory containing panel images')
    parser.add_argument('--triplets', type=str, default=str(Path.home() / 'data' / 'triplets_small.json'),
                        help='Path to triplets JSON file')
    parser.add_argument('--num_candidates', type=int, default=10, help='Number of candidate panels')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (None = all validation set)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (currently only 1 supported)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/infilling_baseline_eval.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    encoder, generator = load_models(
        Path(args.encoder),
        Path(args.generator),
        device
    )
    
    # Load dataset
    print(f"\nLoading dataset from: {args.triplets}")
    full_dataset = TripletDataset(
        triplets_json_path=Path(args.triplets),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=False)
    )
    
    # Use validation split (10% of data)
    from torch.utils.data import random_split
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Using validation set: {val_size} triplets")
    
    # Create wrapper dataset to access validation indices
    class SubsetWrapper:
        def __init__(self, subset):
            self.subset = subset
            self.dataset = subset.dataset
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            # Get actual index from subset
            actual_idx = self.subset.indices[idx]
            return self.dataset[actual_idx]
    
    val_wrapper = SubsetWrapper(val_dataset)
    
    # Evaluate with streaming (create and evaluate tasks on-the-fly to save memory)
    dataset_size = len(val_wrapper)
    if args.num_samples is not None:
        dataset_size = min(args.num_samples, dataset_size)
    
    print(f"\nEvaluating {dataset_size} tasks (streaming mode to save memory)...")
    print(f"  - Candidates per task: {args.num_candidates}")
    
    results = []
    valid_tasks = 0
    correct_count = 0
    
    # Create progress bar with detailed information
    pbar = tqdm(
        range(dataset_size), 
        desc="Evaluating",
        ncols=120,
        leave=True,
        unit="task"
    )
    
    for idx in pbar:
        try:
            # Get triplet (A, B, C)
            triplet_data = val_wrapper[idx]
            
            A = triplet_data['A']
            B_correct = triplet_data['B']
            C = triplet_data['C']
            
            # Sample negative candidates from other triplets
            other_indices = [i for i in range(len(val_wrapper)) if i != idx]
            if len(other_indices) < args.num_candidates - 1:
                continue
            
            neg_indices = random.sample(other_indices, args.num_candidates - 1)
            selected_negatives = []
            
            for neg_idx in neg_indices:
                neg_triplet = val_wrapper[neg_idx]
                selected_negatives.append(neg_triplet['B'])
            
            # Create candidate list: [B_correct, B_neg1, B_neg2, ...]
            candidates = [B_correct] + selected_negatives
            
            # Shuffle candidates and record correct index
            correct_idx = 0  # Before shuffling
            indices = list(range(len(candidates)))
            random.shuffle(indices)
            
            shuffled_candidates = [candidates[i] for i in indices]
            shuffled_correct_idx = indices.index(correct_idx)
            
            # Create task and evaluate immediately (don't store in memory)
            task = {
                'triplet_idx': idx,
                'A': A,
                'C': C,
                'candidates': shuffled_candidates,
                'correct_idx': shuffled_correct_idx
            }
            
            # Evaluate immediately
            result = evaluate_task(task, encoder, generator, device)
            results.append(result)
            valid_tasks += 1
            
            if result['is_correct']:
                correct_count += 1
            
            # Update progress bar with current statistics
            current_acc = (correct_count / valid_tasks * 100) if valid_tasks > 0 else 0.0
            pbar.set_postfix({
                'Valid': valid_tasks,
                'Correct': correct_count,
                'Acc': f'{current_acc:.1f}%',
                'Rank': f'{result["rank"]:.1f}'
            })
            
            # Clear task from memory (Python GC will handle it)
            del task
            
        except Exception as e:
            print(f"\nError evaluating task {idx}: {e}")
            continue
    
    pbar.close()
    print(f"\nEvaluated {valid_tasks} valid tasks out of {dataset_size} total tasks")
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    metrics = compute_metrics(results)
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            if 'Accuracy' in metric_name or 'Recall' in metric_name or 'MRR' in metric_name:
                print(f"{metric_name:20s}: {metric_value:.4f} ({metric_value*100:.2f}%)")
            else:
                print(f"{metric_name:20s}: {metric_value:.4f}")
        else:
            print(f"{metric_name:20s}: {metric_value}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metrics': metrics,
        'config': {
            'encoder': args.encoder,
            'generator': args.generator,
            'num_candidates': args.num_candidates,
            'num_samples': valid_tasks,
            'seed': args.seed
        },
        'detailed_results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

