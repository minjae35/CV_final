"""
Local Task Evaluation: Middle Panel Selection
Unified evaluation script for all model types (InfoNCE, Infilling, GAN, Student)
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from models.discriminator import Discriminator
from models.coherence_head import CoherenceHead
from datasets.panel_dataset import TripletDataset, get_default_transform


def load_models(
    model_type: str,
    encoder_path: Path,
    checkpoint_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple:
    """
    Load models based on model type
    
    Args:
        model_type: 'infonce', 'infilling', 'gan', 'student'
        encoder_path: Path to encoder checkpoint
        checkpoint_path: Path to model checkpoint (required for infilling, gan, student)
        device: Device to load models on
    
    Returns:
        Tuple of loaded models (encoder, generator, discriminator, coherence_head)
        - InfoNCE: (encoder, None, None, None)
        - Infilling: (encoder, generator, None, None)
        - GAN: (encoder, generator, discriminator, None)
        - Student: (encoder, generator, discriminator, coherence_head)
    """
    print(f"Loading encoder from: {encoder_path}")
    encoder = MultimodalEncoder(embedding_dim=128)
    encoder_checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
    
    if 'encoder_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
    elif 'model_state_dict' in encoder_checkpoint:
        encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_checkpoint)
    
    encoder = encoder.to(device)
    encoder.eval()
    
    generator = None
    discriminator = None
    coherence_head = None
    
    if model_type == 'infonce':
        return encoder, None, None, None
    
    if checkpoint_path is None:
        raise ValueError(f"checkpoint_path is required for model_type: {model_type}")
    
    print(f"Loading {model_type} model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load Generator (for infilling, gan, student)
    if model_type in ['infilling', 'gan', 'student']:
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
    
    # Load Discriminator (for gan, student)
    if model_type in ['gan', 'student']:
        if 'discriminator_state_dict' in checkpoint:
            disc_state_dict = checkpoint['discriminator_state_dict']
        else:
            disc_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('discriminator.') or 'discriminator' in key.lower():
                    new_key = key.replace('discriminator.', '')
                    disc_state_dict[new_key] = value
        
        if disc_state_dict:
            discriminator = Discriminator(embedding_dim=128, hidden_dim=256)
            discriminator.load_state_dict(disc_state_dict)
            discriminator = discriminator.to(device)
            discriminator.eval()
        else:
            print("Warning: Discriminator state_dict not found, skipping discriminator")
    
    # Load CoherenceHead (for student)
    if model_type == 'student':
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
    
    return encoder, generator, discriminator, coherence_head


def create_middle_panel_selection_task(
    triplet_dataset: TripletDataset,
    num_candidates: int = 10,
    num_samples: int = None
) -> List[Dict]:
    """
    Create middle panel selection tasks from triplet dataset
    
    Args:
        triplet_dataset: TripletDataset instance
        num_candidates: Number of candidate B panels (including correct one)
        num_samples: Number of samples to create (None = all)
    
    Returns:
        List of task dictionaries, each containing:
        - 'triplet_idx': Index in dataset
        - 'A': Panel A dict
        - 'C': Panel C dict
        - 'candidates': List of candidate B panels (shuffled)
        - 'correct_idx': Index of correct B in shuffled candidates
    """
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


def evaluate_task_infonce(
    task: Dict,
    encoder: MultimodalEncoder,
    device: torch.device
) -> Dict:
    """
    Evaluate task using InfoNCE encoder only
    Strategy: z_B_pred = (z_A + z_C) / 2
    """
    A = task['A']
    C = task['C']
    candidates = task['candidates']
    correct_idx = task['correct_idx']
    
    with torch.no_grad():
        # Encode A and C
        img_A = A['image'].unsqueeze(0).to(device)
        text_A = [A['ocr_text']]
        z_A = encoder(img_A, text_A)  # [1, 128]
        
        img_C = C['image'].unsqueeze(0).to(device)
        text_C = [C['ocr_text']]
        z_C = encoder(img_C, text_C)  # [1, 128]
        
        # Predict middle embedding: average of A and C
        z_B_pred = (z_A + z_C) / 2.0
        z_B_pred = F.normalize(z_B_pred, dim=1)
        
        # Compute similarity scores for each candidate
        similarities = []
        for candidate in candidates:
            img_B = candidate['image'].unsqueeze(0).to(device)
            text_B = [candidate['ocr_text']]
            z_B = encoder(img_B, text_B)  # [1, 128]
            z_B = F.normalize(z_B, dim=1)
            
            # Cosine similarity
            sim = torch.sum(z_B_pred * z_B).item()
            similarities.append(sim)
    
    # Find predicted index
    predicted_idx = int(np.argmax(similarities))
    is_correct = (predicted_idx == correct_idx)
    
    # Compute rank
    sorted_indices = np.argsort(similarities)[::-1]
    rank = int(np.where(sorted_indices == correct_idx)[0][0]) + 1
    
    return {
        'predicted_idx': predicted_idx,
        'correct_idx': correct_idx,
        'is_correct': is_correct,
        'rank': rank,
        'similarities': similarities
    }


def evaluate_task_infilling(
    task: Dict,
    encoder: MultimodalEncoder,
    generator: nn.Module,
    device: torch.device
) -> Dict:
    """
    Evaluate task using Infilling Generator
    Strategy: Generate z_B_hat = G(z_A, z_C), then compute similarity with candidates
    """
    A = task['A']
    C = task['C']
    candidates = task['candidates']
    correct_idx = task['correct_idx']
    
    with torch.no_grad():
        # Encode A and C
        img_A = A['image'].unsqueeze(0).to(device)
        text_A = [A['ocr_text']]
        z_A = encoder(img_A, text_A)  # [1, 128]
        
        img_C = C['image'].unsqueeze(0).to(device)
        text_C = [C['ocr_text']]
        z_C = encoder(img_C, text_C)  # [1, 128]
        
        # Generate predicted z_B
        z_B_hat = generator(z_A, z_C)  # [1, 128]
        z_B_hat = F.normalize(z_B_hat, dim=1)
        
        # Compute similarity scores for each candidate
        similarities = []
        for candidate in candidates:
            img_B = candidate['image'].unsqueeze(0).to(device)
            text_B = [candidate['ocr_text']]
            z_B = encoder(img_B, text_B)  # [1, 128]
            z_B = F.normalize(z_B, dim=1)
            
            # Cosine similarity
            sim = torch.sum(z_B_hat * z_B).item()
            similarities.append(sim)
    
    # Find predicted index
    predicted_idx = int(np.argmax(similarities))
    is_correct = (predicted_idx == correct_idx)
    
    # Compute rank
    sorted_indices = np.argsort(similarities)[::-1]
    rank = int(np.where(sorted_indices == correct_idx)[0][0]) + 1
    
    return {
        'predicted_idx': predicted_idx,
        'correct_idx': correct_idx,
        'is_correct': is_correct,
        'rank': rank,
        'similarities': similarities
    }


def evaluate_task_student(
    task: Dict,
    encoder: MultimodalEncoder,
    generator: nn.Module,
    coherence_head: CoherenceHead,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict:
    """
    Evaluate task using Student Model (Generator + CoherenceHead)
    Strategy: Combine infilling similarity and coherence score
    Final score: s = alpha * s_infilling + beta * s_coh
    """
    A = task['A']
    C = task['C']
    candidates = task['candidates']
    correct_idx = task['correct_idx']
    
    with torch.no_grad():
        # Encode A and C
        img_A = A['image'].unsqueeze(0).to(device)
        text_A = [A['ocr_text']]
        z_A = encoder(img_A, text_A)  # [1, 128]
        
        img_C = C['image'].unsqueeze(0).to(device)
        text_C = [C['ocr_text']]
        z_C = encoder(img_C, text_C)  # [1, 128]
        
        # Generate predicted z_B
        z_B_hat = generator(z_A, z_C)  # [1, 128]
        z_B_hat = F.normalize(z_B_hat, dim=1)
        
        # Compute scores for each candidate
        infilling_similarities = []
        coherence_scores = []
        combined_scores = []
        
        for candidate in candidates:
            img_B = candidate['image'].unsqueeze(0).to(device)
            text_B = [candidate['ocr_text']]
            z_B = encoder(img_B, text_B)  # [1, 128]
            z_B = F.normalize(z_B, dim=1)
            
            # Infilling similarity
            infilling_sim = torch.sum(z_B_hat * z_B).item()
            infilling_similarities.append(infilling_sim)
            
            # Coherence score
            coh_score = coherence_head(z_A, z_B, z_C).item()
            coherence_scores.append(coh_score)
            
            # Combined score
            combined_score = alpha * infilling_sim + beta * coh_score
            combined_scores.append(combined_score)
    
    # Find predicted index (highest combined score)
    predicted_idx = int(np.argmax(combined_scores))
    is_correct = (predicted_idx == correct_idx)
    
    # Compute rank
    sorted_indices = np.argsort(combined_scores)[::-1]
    rank = int(np.where(sorted_indices == correct_idx)[0][0]) + 1
    
    return {
        'predicted_idx': predicted_idx,
        'correct_idx': correct_idx,
        'is_correct': is_correct,
        'rank': rank,
        'infilling_similarities': infilling_similarities,
        'coherence_scores': coherence_scores,
        'combined_scores': combined_scores
    }


def evaluate_task(
    task: Dict,
    model_type: str,
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    discriminator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    device: torch.device = torch.device('cpu'),
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict:
    """
    Unified task evaluation function
    
    Args:
        task: Task dictionary
        model_type: 'infonce', 'infilling', 'gan', 'student'
        encoder: MultimodalEncoder instance
        generator: Generator instance (for infilling, gan, student)
        discriminator: Discriminator instance (for gan, student, not used in evaluation)
        coherence_head: CoherenceHead instance (for student)
        device: Device to run on
        alpha: Weight for infilling similarity (for student)
        beta: Weight for coherence score (for student)
    
    Returns:
        Evaluation result dictionary
    """
    if model_type == 'infonce':
        return evaluate_task_infonce(task, encoder, device)
    elif model_type in ['infilling', 'gan']:
        if generator is None:
            raise ValueError(f"generator is required for model_type: {model_type}")
        return evaluate_task_infilling(task, encoder, generator, device)
    elif model_type == 'student':
        if generator is None or coherence_head is None:
            raise ValueError(f"generator and coherence_head are required for model_type: student")
        return evaluate_task_student(task, encoder, generator, coherence_head, device, alpha, beta)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics from results
    
    Metrics:
    - Top-1 Accuracy: Percentage of correct predictions at rank 1
    - Recall@K: Percentage of correct answers in top K predictions
    - MRR (Mean Reciprocal Rank): Average of 1/rank for correct answers
    - Average Rank: Average rank of correct answers
    
    Args:
        results: List of evaluation result dictionaries
    
    Returns:
        Dictionary containing all metrics
    """
    total = len(results)
    if total == 0:
        return {
            'top1_accuracy': 0.0,
            'recall_at_3': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
            'mrr': 0.0,
            'avg_rank': 0.0,
            'total_samples': 0
        }
    
    # Top-1 Accuracy
    top1_correct = sum(1 for r in results if r['is_correct'])
    top1_accuracy = top1_correct / total
    
    # Recall@K
    recall_at_3 = sum(1 for r in results if r['rank'] <= 3) / total
    recall_at_5 = sum(1 for r in results if r['rank'] <= 5) / total
    recall_at_10 = sum(1 for r in results if r['rank'] <= 10) / total
    
    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0 / r['rank'] for r in results]
    mrr = np.mean(reciprocal_ranks)
    
    # Average Rank
    avg_rank = np.mean([r['rank'] for r in results])
    
    return {
        'top1_accuracy': top1_accuracy,
        'recall_at_3': recall_at_3,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10,
        'mrr': mrr,
        'avg_rank': avg_rank,
        'total_samples': total
    }


def save_results(
    output_path: Path,
    metrics: Dict,
    config: Dict,
    results: List[Dict]
):
    """
    Save evaluation results to JSON file
    
    Args:
        output_path: Path to output JSON file
        metrics: Computed metrics dictionary
        config: Configuration dictionary
        results: List of individual task results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'config': config,
        'metrics': metrics,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """Print metrics in formatted table"""
    print("\n" + "=" * 70)
    print(f"{model_name} - Evaluation Results")
    print("=" * 70)
    
    print(f"{'Top-1 Accuracy':20s}: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
    print(f"{'Recall@3':20s}: {metrics['recall_at_3']:.4f} ({metrics['recall_at_3']*100:.2f}%)")
    print(f"{'Recall@5':20s}: {metrics['recall_at_5']:.4f} ({metrics['recall_at_5']*100:.2f}%)")
    print(f"{'Recall@10':20s}: {metrics['recall_at_10']:.4f} ({metrics['recall_at_10']*100:.2f}%)")
    print(f"{'MRR':20s}: {metrics['mrr']:.4f}")
    print(f"{'Average Rank':20s}: {metrics['avg_rank']:.4f}")
    print(f"{'Total Samples':20s}: {metrics['total_samples']}")
    print("=" * 70)


def grid_search_hyperparameters(
    tasks: List[Dict],
    encoder: MultimodalEncoder,
    generator: nn.Module,
    coherence_head: CoherenceHead,
    device: torch.device,
    alpha_range: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
    beta_range: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
    constraint: str = 'sum_to_one'  # 'sum_to_one' or 'independent'
) -> Dict:
    """
    Grid search for optimal alpha and beta hyperparameters (Student model only)
    
    Args:
        tasks: List of evaluation tasks
        encoder: MultimodalEncoder instance
        generator: Generator instance
        coherence_head: CoherenceHead instance
        device: Device to run on
        alpha_range: List of alpha values to try
        beta_range: List of beta values to try
        constraint: 'sum_to_one' (alpha + beta = 1) or 'independent' (all combinations)
    
    Returns:
        Dictionary with best hyperparameters and results
    """
    print("\n" + "=" * 70)
    print("Grid Search for Hyperparameters (alpha, beta)")
    print("=" * 70)
    
    best_metrics = None
    best_alpha = None
    best_beta = None
    all_results = []
    
    if constraint == 'sum_to_one':
        # alpha + beta = 1, so beta = 1 - alpha
        param_combinations = [(alpha, 1.0 - alpha) for alpha in alpha_range if 0 <= alpha <= 1.0]
    else:
        # All combinations
        param_combinations = [(alpha, beta) for alpha in alpha_range for beta in beta_range]
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations...")
    
    for alpha, beta in tqdm(param_combinations, desc="Grid search"):
        # Evaluate with current hyperparameters
        results = []
        for task in tasks:
            result = evaluate_task_student(
                task=task,
                encoder=encoder,
                generator=generator,
                coherence_head=coherence_head,
                device=device,
                alpha=alpha,
                beta=beta
            )
            results.append(result)
        
        # Compute metrics
        metrics = compute_metrics(results)
        
        # Store results
        all_results.append({
            'alpha': alpha,
            'beta': beta,
            'metrics': metrics
        })
        
        # Update best if better (using Top-1 Accuracy as primary metric)
        if best_metrics is None or metrics['top1_accuracy'] > best_metrics['top1_accuracy']:
            best_metrics = metrics
            best_alpha = alpha
            best_beta = beta
    
    # Print best results
    print("\n" + "=" * 70)
    print("Best Hyperparameters:")
    print("=" * 70)
    print(f"Alpha: {best_alpha:.3f}")
    print(f"Beta: {best_beta:.3f}")
    print_metrics(best_metrics, model_name="Best Configuration")
    
    # Print top 5 configurations
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['top1_accuracy'], reverse=True)
    print("\n" + "=" * 70)
    print("Top 5 Configurations:")
    print("=" * 70)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Alpha={result['alpha']:.3f}, Beta={result['beta']:.3f}")
        print(f"   Top-1: {result['metrics']['top1_accuracy']*100:.2f}%, "
              f"Recall@5: {result['metrics']['recall_at_5']*100:.2f}%, "
              f"MRR: {result['metrics']['mrr']:.4f}")
    
    return {
        'best_alpha': best_alpha,
        'best_beta': best_beta,
        'best_metrics': best_metrics,
        'all_results': all_results
    }


def main():
    parser = argparse.ArgumentParser(description='Local Task Evaluation: Middle Panel Selection')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['infonce', 'infilling', 'gan', 'student'],
                        help='Model type to evaluate')
    parser.add_argument('--encoder', type=str, required=True,
                        help='Path to encoder checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (required for infilling, gan, student)')
    
    # Data configuration
    parser.add_argument('--triplets', type=str, required=True,
                        help='Path to triplets JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing panel images')
    
    # Evaluation configuration
    parser.add_argument('--num_candidates', type=int, default=10,
                        help='Number of candidate B panels')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--use_val_split', action='store_true',
                        help='Use validation split (90/10 split)')
    
    # Student model specific
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for infilling similarity (student model)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for coherence score (student model)')
    parser.add_argument('--grid_search', action='store_true',
                        help='Perform grid search for hyperparameters (student model only)')
    parser.add_argument('--alpha_range', type=str, default='0.0,0.3,0.5,0.7,1.0',
                        help='Comma-separated alpha values for grid search')
    parser.add_argument('--beta_range', type=str, default='0.0,0.3,0.5,0.7,1.0',
                        help='Comma-separated beta values for grid search')
    parser.add_argument('--constraint', type=str, default='sum_to_one',
                        choices=['sum_to_one', 'independent'],
                        help='Constraint for grid search: sum_to_one (alpha+beta=1) or independent')
    
    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: results/eval_local_{model_type}.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
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
    print(f"\nLoading {args.model_type} model...")
    encoder, generator, discriminator, coherence_head = load_models(
        model_type=args.model_type,
        encoder_path=Path(args.encoder),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        device=device
    )
    
    # Load dataset
    print(f"\nLoading dataset from: {args.triplets}")
    full_dataset = TripletDataset(
        triplets_json_path=Path(args.triplets),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=False)
    )
    
    # Split dataset if needed
    if args.use_val_split:
        total_size = len(full_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        
        _, eval_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"Using validation split: {len(eval_dataset)} samples")
    else:
        eval_dataset = full_dataset
        print(f"Using full dataset: {len(eval_dataset)} samples")
    
    # Create tasks
    tasks = create_middle_panel_selection_task(
        triplet_dataset=eval_dataset,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples
    )
    
    # Grid search for hyperparameters (Student model only)
    if args.grid_search and args.model_type == 'student':
        if generator is None or coherence_head is None:
            raise ValueError("Generator and CoherenceHead are required for grid search")
        
        # Parse alpha and beta ranges
        alpha_range = [float(x) for x in args.alpha_range.split(',')]
        beta_range = [float(x) for x in args.beta_range.split(',')]
        
        grid_search_results = grid_search_hyperparameters(
            tasks=tasks,
            encoder=encoder,
            generator=generator,
            coherence_head=coherence_head,
            device=device,
            alpha_range=alpha_range,
            beta_range=beta_range,
            constraint=args.constraint
        )
        
        # Use best hyperparameters for final evaluation
        args.alpha = grid_search_results['best_alpha']
        args.beta = grid_search_results['best_beta']
        print(f"\nUsing best hyperparameters: alpha={args.alpha:.3f}, beta={args.beta:.3f}")
        
        # Save grid search results
        grid_output_path = Path(f"results/grid_search_{args.model_type}.json")
        grid_config = {
            'model_type': args.model_type,
            'encoder_path': str(args.encoder),
            'checkpoint_path': str(args.checkpoint) if args.checkpoint else None,
            'grid_search': True,
            'alpha_range': args.alpha_range,
            'beta_range': args.beta_range,
            'constraint': args.constraint,
            'all_results': grid_search_results['all_results']
        }
        save_results(grid_output_path, grid_search_results['best_metrics'], 
                    grid_config, [])  # Individual results not saved for grid search
    
    # Evaluate tasks
    print(f"\nEvaluating {len(tasks)} tasks...")
    results = []
    
    for task in tqdm(tasks, desc="Evaluating"):
        result = evaluate_task(
            task=task,
            model_type=args.model_type,
            encoder=encoder,
            generator=generator,
            discriminator=discriminator,
            coherence_head=coherence_head,
            device=device,
            alpha=args.alpha,
            beta=args.beta
        )
        results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print results
    print_metrics(metrics, model_name=f"{args.model_type.upper()} Model")
    
    # Save results
    if args.output is None:
        output_path = Path(f"results/eval_local_{args.model_type}.json")
    else:
        output_path = Path(args.output)
    
    config = {
        'model_type': args.model_type,
        'encoder_path': str(args.encoder),
        'checkpoint_path': str(args.checkpoint) if args.checkpoint else None,
        'triplets_path': str(args.triplets),
        'data_dir': str(args.data_dir),
        'num_candidates': args.num_candidates,
        'num_samples': args.num_samples,
        'use_val_split': args.use_val_split,
        'alpha': args.alpha if args.model_type == 'student' else None,
        'beta': args.beta if args.model_type == 'student' else None,
        'grid_search': args.grid_search if args.model_type == 'student' else False,
        'seed': args.seed
    }
    
    save_results(output_path, metrics, config, results)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

