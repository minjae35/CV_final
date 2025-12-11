"""
Student Model (Score Distillation) Evaluation
Evaluate middle panel selection using Generator + CoherenceHead
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
from models.coherence_head import CoherenceHead
from datasets.panel_dataset import TripletDataset, get_default_transform


def load_student_models(
    encoder_path: Path,
    checkpoint_path: Path,
    device: torch.device
):
    """Load pretrained encoder and student model (Generator + CoherenceHead)"""
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
    
    print(f"Loading student model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load Generator
    if 'generator_state_dict' in checkpoint:
        gen_state_dict = checkpoint['generator_state_dict']
    elif 'model_state_dict' in checkpoint:
        gen_state_dict = checkpoint['model_state_dict']
    else:
        gen_state_dict = checkpoint
    
    # Check if old simple MLP structure or new attention structure
    is_old_model = 'network.0.weight' in gen_state_dict
    
    if is_old_model:
        # Old simple MLP structure
        class SimpleMLPGenerator(nn.Module):
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
        
        generator = SimpleMLPGenerator(embedding_dim=128, hidden_dim=256)
    else:
        # New attention-based structure
        generator = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4)
    
    generator.load_state_dict(gen_state_dict)
    generator = generator.to(device)
    generator.eval()
    
    # Load CoherenceHead
    coherence_head = CoherenceHead(embedding_dim=128, hidden_dim=256)
    if 'coherence_head_state_dict' in checkpoint:
        coherence_head.load_state_dict(checkpoint['coherence_head_state_dict'])
    else:
        # Try to find coherence head in checkpoint
        coh_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('coherence_head.') or 'coherence' in key.lower():
                new_key = key.replace('coherence_head.', '')
                coh_state_dict[new_key] = value
        if coh_state_dict:
            coherence_head.load_state_dict(coh_state_dict)
        else:
            print("Warning: CoherenceHead state_dict not found, using random initialization")
    
    coherence_head = coherence_head.to(device)
    coherence_head.eval()
    
    return encoder, generator, coherence_head


def create_middle_panel_selection_task(
    triplet_dataset: TripletDataset,
    num_candidates: int = 10,
    num_samples: int = None
) -> List[Dict]:
    """Create middle panel selection tasks from triplet dataset"""
    tasks = []
    dataset_size = len(triplet_dataset)
    if num_samples is not None:
        dataset_size = min(num_samples, dataset_size)
    
    print(f"Creating {dataset_size} middle panel selection tasks...")
    
    for idx in tqdm(range(dataset_size), desc="Creating tasks"):
        try:
            triplet = triplet_dataset[idx]
            A = triplet['A']
            B_correct = triplet['B']
            C = triplet['C']
            
            # Get all other B panels as negatives
            other_indices = [i for i in range(len(triplet_dataset)) if i != idx]
            if len(other_indices) < num_candidates - 1:
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
    generator: nn.Module,
    coherence_head: CoherenceHead,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict:
    """
    Evaluate single middle panel selection task using Generator + CoherenceHead
    
    Args:
        alpha: Weight for infilling similarity
        beta: Weight for coherence score
    
    Returns:
        {
            'predicted_idx': int,
            'correct_idx': int,
            'is_correct': bool,
            'rank': int (1-based rank of correct answer),
            'similarities': List[float],
            'coherence_scores': List[float],
            'combined_scores': List[float]
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
        
        # Generate z_B from A and C
        z_B_generated = generator(z_A, z_C)  # [1, 128]
        
        # Evaluate each candidate
        similarities = []
        coherence_scores = []
        
        for candidate_B in candidates:
            # Encode candidate B
            img_B = candidate_B['image'].unsqueeze(0).to(device)
            text_B = [candidate_B['ocr_text']]
            z_B_candidate = encoder(img_B, text_B)  # [1, 128]
            
            # Infilling similarity (cosine similarity between generated and candidate)
            similarity = F.cosine_similarity(z_B_generated, z_B_candidate, dim=1).item()
            similarities.append(similarity)
            
            # Coherence score for (A, candidate_B, C)
            coherence_score = coherence_head(z_A, z_B_candidate, z_C).squeeze().item()
            coherence_scores.append(coherence_score)
        
        # Combine scores: alpha * similarity + beta * coherence_score
        combined_scores = [
            alpha * sim + beta * coh
            for sim, coh in zip(similarities, coherence_scores)
        ]
        
        # Find predicted index
        predicted_idx = np.argmax(combined_scores)
        
        # Calculate rank (1-based)
        sorted_indices = np.argsort(combined_scores)[::-1]  # Descending order
        rank = np.where(sorted_indices == correct_idx)[0][0] + 1
        
        return {
            'predicted_idx': int(predicted_idx),
            'correct_idx': int(correct_idx),
            'is_correct': bool(predicted_idx == correct_idx),
            'rank': int(rank),
            'similarities': similarities,
            'coherence_scores': coherence_scores,
            'combined_scores': combined_scores
        }


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    # Top-1 Accuracy
    top1_correct = sum(1 for r in results if r['is_correct'])
    top1_accuracy = top1_correct / total
    
    # Recall@K
    recall_at_3 = sum(1 for r in results if r['rank'] <= 3) / total
    recall_at_5 = sum(1 for r in results if r['rank'] <= 5) / total
    
    # MRR (Mean Reciprocal Rank)
    mrr = np.mean([1.0 / r['rank'] for r in results])
    
    # Average Rank
    avg_rank = np.mean([r['rank'] for r in results])
    
    return {
        'top1_accuracy': top1_accuracy,
        'recall_at_3': recall_at_3,
        'recall_at_5': recall_at_5,
        'mrr': mrr,
        'avg_rank': avg_rank,
        'total_samples': total
    }


def main():
    parser = argparse.ArgumentParser(description='Student Model Evaluation')
    
    parser.add_argument('--encoder', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to student model checkpoint')
    parser.add_argument('--triplets', type=str, required=True, help='Path to triplets JSON file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing panel images')
    
    parser.add_argument('--num_candidates', type=int, default=10, help='Number of candidate B panels')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate (None = all)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for infilling similarity')
    parser.add_argument('--beta', type=float, default=0.5, help='Weight for coherence score')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/student_model_eval.json', help='Output file path')
    
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
    encoder, generator, coherence_head = load_student_models(
        Path(args.encoder),
        Path(args.checkpoint),
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
            actual_idx = self.subset.indices[idx]
            return self.dataset[actual_idx]
    
    val_wrapper = SubsetWrapper(val_dataset)
    
    # Evaluate with streaming (create and evaluate tasks on-the-fly to save memory)
    dataset_size = len(val_wrapper)
    if args.num_samples is not None:
        dataset_size = min(args.num_samples, dataset_size)
    
    print(f"\nEvaluating {dataset_size} tasks...")
    results = []
    
    for idx in tqdm(range(dataset_size), desc="Evaluating"):
        try:
            # Create task for this triplet
            triplet = val_wrapper[idx]
            A = triplet['A']
            B_correct = triplet['B']
            C = triplet['C']
            
            # Get candidates
            other_indices = [i for i in range(len(val_wrapper)) if i != idx]
            if len(other_indices) < args.num_candidates - 1:
                continue
            
            neg_indices = random.sample(other_indices, args.num_candidates - 1)
            selected_negatives = []
            
            for neg_idx in neg_indices:
                neg_triplet = val_wrapper[neg_idx]
                selected_negatives.append(neg_triplet['B'])
            
            candidates = [B_correct] + selected_negatives
            
            # Shuffle candidates
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
            
            # Evaluate task
            result = evaluate_task(
                task, encoder, generator, coherence_head, device,
                alpha=args.alpha, beta=args.beta
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error evaluating task {idx}: {e}")
            continue
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']*100:.2f}%")
    print(f"Recall@3: {metrics['recall_at_3']*100:.2f}%")
    print(f"Recall@5: {metrics['recall_at_5']*100:.2f}%")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Average Rank: {metrics['avg_rank']:.2f}")
    print("="*60)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'metrics': metrics,
        'config': {
            'encoder': str(args.encoder),
            'checkpoint': str(args.checkpoint),
            'num_candidates': args.num_candidates,
            'num_samples': args.num_samples,
            'alpha': args.alpha,
            'beta': args.beta,
            'seed': args.seed
        },
        'results': results[:100]  # Save first 100 results for analysis
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

