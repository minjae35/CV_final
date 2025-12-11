"""
Global Task Evaluation: Panel Order Restoration
Evaluate full panel sequence ordering using different search algorithms
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import kendalltau

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from models.discriminator import Discriminator
from models.coherence_head import CoherenceHead
from datasets.panel_dataset import TripletDataset, get_default_transform


def compute_triplet_score(
    z_A: torch.Tensor,
    z_B: torch.Tensor,
    z_C: torch.Tensor,
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Compute score for a triplet (A, B, C)
    
    Args:
        z_A, z_B, z_C: Embeddings of panels A, B, C
        encoder: MultimodalEncoder instance
        generator: Generator instance (for infilling, gan, student)
        coherence_head: CoherenceHead instance (for student)
        model_type: 'infonce', 'infilling', 'gan', 'student'
        alpha: Weight for infilling similarity (student)
        beta: Weight for coherence score (student)
        device: Device to run on
    
    Returns:
        Score for the triplet (higher is better)
    """
    with torch.no_grad():
        if model_type == 'infonce':
            # Use average of A and C as prediction, compare with B
            z_AC = (z_A + z_C) / 2.0
            z_AC = F.normalize(z_AC, dim=1)
            z_B = F.normalize(z_B, dim=1)
            score = torch.sum(z_AC * z_B).item()
        
        elif model_type in ['infilling', 'gan']:
            # Generate z_B_hat and compare with z_B
            z_B_hat = generator(z_A, z_C)
            z_B_hat = F.normalize(z_B_hat, dim=1)
            z_B = F.normalize(z_B, dim=1)
            score = torch.sum(z_B_hat * z_B).item()
        
        elif model_type == 'student':
            # Combine infilling similarity and coherence score
            z_B_hat = generator(z_A, z_C)
            z_B_hat = F.normalize(z_B_hat, dim=1)
            z_B = F.normalize(z_B, dim=1)
            infilling_sim = torch.sum(z_B_hat * z_B).item()
            
            coh_score = coherence_head(z_A, z_B, z_C).item()
            
            score = alpha * infilling_sim + beta * coh_score
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    return score


def compute_sequence_score(
    panel_embeddings: List[torch.Tensor],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Compute total score for a panel sequence
    
    Args:
        panel_embeddings: List of panel embeddings in order
        encoder: MultimodalEncoder instance
        generator: Generator instance
        coherence_head: CoherenceHead instance
        model_type: Model type
        alpha: Weight for infilling similarity
        beta: Weight for coherence score
        device: Device to run on
    
    Returns:
        Total score for the sequence (sum of all adjacent triplets)
    """
    if len(panel_embeddings) < 3:
        return 0.0
    
    total_score = 0.0
    for i in range(len(panel_embeddings) - 2):
        z_A = panel_embeddings[i]
        z_B = panel_embeddings[i + 1]
        z_C = panel_embeddings[i + 2]
        
        score = compute_triplet_score(
            z_A, z_B, z_C,
            encoder, generator, coherence_head,
            model_type, alpha, beta, device
        )
        total_score += score
    
    return total_score


def greedy_order_restoration(
    panel_embeddings: List[torch.Tensor],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Greedy algorithm for panel order restoration
    
    Strategy:
    1. Start with first panel (random or highest score)
    2. At each step, select next panel that maximizes triplet score with last two panels
    3. Continue until all panels are placed
    """
    n = len(panel_embeddings)
    if n <= 2:
        return list(range(n))
    
    # Start with first two panels (can be optimized)
    ordered = [0, 1]
    remaining = set(range(2, n))
    
    while remaining:
        best_idx = None
        best_score = float('-inf')
        best_position = len(ordered)
        
        # Try inserting each remaining panel at each position
        for panel_idx in remaining:
            for pos in range(len(ordered) - 1, len(ordered) + 1):
                # Create candidate sequence
                candidate = ordered[:pos] + [panel_idx] + ordered[pos:]
                
                if len(candidate) >= 3:
                    # Compute score for affected triplets
                    score = 0.0
                    start_idx = max(0, pos - 2)
                    end_idx = min(len(candidate), pos + 3)
                    
                    for i in range(start_idx, end_idx - 2):
                        if i + 2 < len(candidate):
                            z_A = panel_embeddings[candidate[i]]
                            z_B = panel_embeddings[candidate[i + 1]]
                            z_C = panel_embeddings[candidate[i + 2]]
                            
                            triplet_score = compute_triplet_score(
                                z_A, z_B, z_C,
                                encoder, generator, coherence_head,
                                model_type, alpha, beta, device
                            )
                            score += triplet_score
                    
                    if score > best_score:
                        best_score = score
                        best_idx = panel_idx
                        best_position = pos
        
        if best_idx is not None:
            ordered.insert(best_position, best_idx)
            remaining.remove(best_idx)
        else:
            # Fallback: add first remaining panel at end
            ordered.append(remaining.pop())
    
    return ordered


def beam_search_order_restoration(
    panel_embeddings: List[torch.Tensor],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    beam_width: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Beam Search algorithm for panel order restoration
    
    Strategy:
    1. Maintain top-k (beam_width) partial sequences
    2. At each step, extend each partial sequence with all remaining panels
    3. Keep top-k sequences based on score
    4. Continue until all panels are placed
    """
    n = len(panel_embeddings)
    if n <= 2:
        return list(range(n))
    
    # Initialize beam with empty sequence
    beam = [([], set(range(n)))]  # (sequence, remaining)
    
    while beam[0][1]:  # While there are remaining panels
        new_beam = []
        
        for sequence, remaining in beam:
            for panel_idx in remaining:
                new_sequence = sequence + [panel_idx]
                new_remaining = remaining - {panel_idx}
                
                # Compute score for this partial sequence
                if len(new_sequence) >= 3:
                    embeddings = [panel_embeddings[i] for i in new_sequence]
                    score = compute_sequence_score(
                        embeddings, encoder, generator, coherence_head,
                        model_type, alpha, beta, device
                    )
                else:
                    score = 0.0
                
                new_beam.append((new_sequence, new_remaining, score))
        
        # Keep top beam_width sequences
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = [(seq, rem) for seq, rem, _ in new_beam[:beam_width]]
    
    # Return best sequence
    return beam[0][0]


def local_search_order_restoration(
    panel_embeddings: List[torch.Tensor],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    max_iterations: int = 100,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> List[int]:
    """
    Local Search algorithm for panel order restoration
    
    Strategy:
    1. Start with initial ordering (greedy or random)
    2. Iteratively try swapping adjacent panels
    3. Accept swap if it improves total score
    4. Continue until no improvement or max iterations
    """
    n = len(panel_embeddings)
    if n <= 2:
        return list(range(n))
    
    # Initialize with greedy solution
    current_order = greedy_order_restoration(
        panel_embeddings, encoder, generator, coherence_head,
        model_type, alpha, beta, device
    )
    
    # Compute initial score
    current_embeddings = [panel_embeddings[i] for i in current_order]
    current_score = compute_sequence_score(
        current_embeddings, encoder, generator, coherence_head,
        model_type, alpha, beta, device
    )
    
    # Local search
    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping adjacent pairs
        for i in range(len(current_order) - 1):
            # Create candidate by swapping i and i+1
            candidate_order = current_order.copy()
            candidate_order[i], candidate_order[i + 1] = candidate_order[i + 1], candidate_order[i]
            
            # Compute score
            candidate_embeddings = [panel_embeddings[j] for j in candidate_order]
            candidate_score = compute_sequence_score(
                candidate_embeddings, encoder, generator, coherence_head,
                model_type, alpha, beta, device
            )
            
            # Accept if better
            if candidate_score > current_score:
                current_order = candidate_order
                current_score = candidate_score
                improved = True
                break  # Greedy: accept first improvement
        
        if not improved:
            break  # No improvement, stop
    
    return current_order


def compute_global_metrics(
    predicted_order: List[int],
    correct_order: List[int]
) -> Dict:
    """
    Compute global task evaluation metrics
    
    Metrics:
    - Perfect Match Accuracy: Percentage of sequences with exact match
    - Adjacent Pair Accuracy: Percentage of adjacent pairs that are correct
    - Kendall's Tau: Rank correlation coefficient
    
    Args:
        predicted_order: Predicted panel order (list of indices)
        correct_order: Correct panel order (list of indices)
    
    Returns:
        Dictionary containing all metrics
    """
    # Perfect Match Accuracy
    perfect_match = (predicted_order == correct_order)
    perfect_match_acc = 1.0 if perfect_match else 0.0
    
    # Adjacent Pair Accuracy
    n = len(correct_order)
    if n < 2:
        adjacent_pair_acc = 1.0 if perfect_match else 0.0
    else:
        correct_pairs = 0
        total_pairs = n - 1
        
        for i in range(n - 1):
            correct_pair = (correct_order[i], correct_order[i + 1])
            predicted_pair = (predicted_order[i], predicted_order[i + 1])
            if correct_pair == predicted_pair:
                correct_pairs += 1
        
        adjacent_pair_acc = correct_pairs / total_pairs
    
    # Kendall's Tau
    # Convert to rank-based representation
    if len(predicted_order) == len(correct_order) and len(predicted_order) > 1:
        # Create rank mappings
        pred_ranks = {panel: rank for rank, panel in enumerate(predicted_order)}
        correct_ranks = {panel: rank for rank, panel in enumerate(correct_order)}
        
        # Get ranks for all panels
        pred_rank_list = [pred_ranks[panel] for panel in correct_order]
        correct_rank_list = list(range(len(correct_order)))
        
        # Compute Kendall's Tau
        tau, p_value = kendalltau(pred_rank_list, correct_rank_list)
        if np.isnan(tau):
            tau = 0.0
    else:
        tau = 0.0
        p_value = 1.0
    
    return {
        'perfect_match': perfect_match,
        'perfect_match_accuracy': perfect_match_acc,
        'adjacent_pair_accuracy': adjacent_pair_acc,
        'kendalls_tau': tau,
        'kendalls_tau_pvalue': p_value
    }


def evaluate_sequence_restoration(
    panel_list: List[Dict],
    correct_order: List[int],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[CoherenceHead] = None,
    model_type: str = 'infonce',
    algorithm: str = 'greedy',
    beam_width: int = 5,
    max_iterations: int = 100,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Evaluate panel sequence restoration for a single page
    
    Args:
        panel_list: List of panel dictionaries (image, ocr_text, etc.)
        correct_order: Correct order of panels (list of indices)
        encoder: MultimodalEncoder instance
        generator: Generator instance
        coherence_head: CoherenceHead instance
        model_type: Model type
        algorithm: 'greedy', 'beam_search', 'local_search'
        beam_width: Beam width for beam search
        max_iterations: Max iterations for local search
        alpha: Weight for infilling similarity
        beta: Weight for coherence score
        device: Device to run on
    
    Returns:
        Dictionary with predicted order and metrics
    """
    # Encode all panels
    panel_embeddings = []
    with torch.no_grad():
        for panel in panel_list:
            img = panel['image'].unsqueeze(0).to(device)
            text = [panel['ocr_text']]
            z = encoder(img, text)
            panel_embeddings.append(z)
    
    # Restore order using selected algorithm
    if algorithm == 'greedy':
        predicted_order = greedy_order_restoration(
            panel_embeddings, encoder, generator, coherence_head,
            model_type, alpha, beta, device
        )
    elif algorithm == 'beam_search':
        predicted_order = beam_search_order_restoration(
            panel_embeddings, encoder, generator, coherence_head,
            model_type, beam_width, alpha, beta, device
        )
    elif algorithm == 'local_search':
        predicted_order = local_search_order_restoration(
            panel_embeddings, encoder, generator, coherence_head,
            model_type, max_iterations, alpha, beta, device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Compute metrics
    metrics = compute_global_metrics(predicted_order, correct_order)
    
    return {
        'predicted_order': predicted_order,
        'correct_order': correct_order,
        'metrics': metrics
    }


def load_models_from_local(
    model_type: str,
    encoder_path: Path,
    checkpoint_path: Optional[Path] = None,
    device: torch.device = torch.device('cpu')
) -> Tuple:
    """Load models using eval_local's load_models function"""
    from evaluation.eval_local import load_models
    return load_models(model_type, encoder_path, checkpoint_path, device)


def group_panels_by_page(triplets_json_path: Path, panels_dir: Path) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Group panels by (comic_no, page_no) for global task evaluation
    
    Returns:
        Dictionary mapping (comic_no, page_no) to list of panels in correct order
    """
    with open(triplets_json_path, 'r', encoding='utf-8') as f:
        triplets = json.load(f)
    
    # Group panels by page
    pages = defaultdict(lambda: {'panels': [], 'panel_indices': set()})
    
    for triplet in triplets:
        comic_no = triplet['A']['comic_no']
        page_no = triplet['A']['page_no']
        key = (comic_no, page_no)
        
        # Add panels A, B, C
        for panel_key in ['A', 'B', 'C']:
            panel = triplet[panel_key]
            panel_idx = panel['panel_index']
            
            # Avoid duplicates
            if panel_idx not in pages[key]['panel_indices']:
                pages[key]['panels'].append({
                    'panel': panel,
                    'panel_index': panel_idx
                })
                pages[key]['panel_indices'].add(panel_idx)
    
    # Sort panels by panel_index and create correct order
    page_data = {}
    for (comic_no, page_no), page_info in pages.items():
        sorted_panels = sorted(page_info['panels'], key=lambda x: x['panel_index'])
        panel_list = [item['panel'] for item in sorted_panels]
        correct_order = list(range(len(panel_list)))  # Already sorted by panel_index
        
        page_data[(comic_no, page_no)] = {
            'panels': panel_list,
            'correct_order': correct_order
        }
    
    return page_data


def main():
    parser = argparse.ArgumentParser(description='Global Task Evaluation: Panel Order Restoration')
    
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
    
    # Algorithm configuration
    parser.add_argument('--algorithms', type=str, nargs='+',
                        default=['greedy', 'beam_search', 'local_search'],
                        choices=['greedy', 'beam_search', 'local_search'],
                        help='Algorithms to evaluate')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Beam width for beam search')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Max iterations for local search')
    
    # Student model specific
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for infilling similarity (student model)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for coherence score (student model)')
    
    # Evaluation configuration
    parser.add_argument('--num_pages', type=int, default=None,
                        help='Number of pages to evaluate (None = all)')
    parser.add_argument('--min_panels', type=int, default=3,
                        help='Minimum number of panels per page to evaluate')
    
    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
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
    encoder, generator, discriminator, coherence_head = load_models_from_local(
        model_type=args.model_type,
        encoder_path=Path(args.encoder),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        device=device
    )
    
    # Load and group panels by page
    print(f"\nLoading panels from: {args.triplets}")
    page_data = group_panels_by_page(Path(args.triplets), Path(args.data_dir))
    
    # Filter pages
    valid_pages = {
        k: v for k, v in page_data.items()
        if len(v['panels']) >= args.min_panels
    }
    
    if args.num_pages is not None:
        page_keys = list(valid_pages.keys())[:args.num_pages]
        valid_pages = {k: valid_pages[k] for k in page_keys}
    
    print(f"Evaluating {len(valid_pages)} pages...")
    
    # Evaluate each algorithm
    all_results = {}
    
    for algorithm in args.algorithms:
        print(f"\n{'='*70}")
        print(f"Evaluating with {algorithm} algorithm...")
        print(f"{'='*70}")
        
        results = []
        
        for (comic_no, page_no), page_info in tqdm(valid_pages.items(), desc=f"{algorithm}"):
            panels = page_info['panels']
            correct_order = page_info['correct_order']
            
            # Load panel images
            panel_list = []
            for panel_info in panels:
                # Load panel using TripletDataset's _load_panel logic
                from datasets.panel_dataset import TripletDataset
                dataset = TripletDataset(
                    triplets_json_path=Path(args.triplets),
                    panels_dir=Path(args.data_dir),
                    transform=get_default_transform(is_train=False)
                )
                panel = dataset._load_panel(panel_info)
                panel_list.append(panel)
            
            # Evaluate
            result = evaluate_sequence_restoration(
                panel_list=panel_list,
                correct_order=correct_order,
                encoder=encoder,
                generator=generator,
                coherence_head=coherence_head,
                model_type=args.model_type,
                algorithm=algorithm,
                beam_width=args.beam_width,
                max_iterations=args.max_iterations,
                alpha=args.alpha,
                beta=args.beta,
                device=device
            )
            
            result['page_key'] = (comic_no, page_no)
            results.append(result)
        
        # Aggregate metrics
        perfect_matches = sum(1 for r in results if r['metrics']['perfect_match'])
        perfect_match_acc = perfect_matches / len(results) if results else 0.0
        
        adjacent_pair_acc = np.mean([r['metrics']['adjacent_pair_accuracy'] for r in results])
        kendalls_tau = np.mean([r['metrics']['kendalls_tau'] for r in results])
        
        all_results[algorithm] = {
            'perfect_match_accuracy': perfect_match_acc,
            'adjacent_pair_accuracy': adjacent_pair_acc,
            'kendalls_tau': kendalls_tau,
            'num_pages': len(results),
            'results': results
        }
        
        # Print results
        print(f"\n{algorithm.upper()} Results:")
        print(f"  Perfect Match Accuracy: {perfect_match_acc:.4f} ({perfect_match_acc*100:.2f}%)")
        print(f"  Adjacent Pair Accuracy: {adjacent_pair_acc:.4f} ({adjacent_pair_acc*100:.2f}%)")
        print(f"  Kendall's Tau: {kendalls_tau:.4f}")
    
    # Save results
    if args.output is None:
        output_path = Path(f"results/eval_global_{args.model_type}.json")
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_type': args.model_type,
        'encoder_path': str(args.encoder),
        'checkpoint_path': str(args.checkpoint) if args.checkpoint else None,
        'triplets_path': str(args.triplets),
        'data_dir': str(args.data_dir),
        'algorithms': args.algorithms,
        'beam_width': args.beam_width,
        'max_iterations': args.max_iterations,
        'alpha': args.alpha if args.model_type == 'student' else None,
        'beta': args.beta if args.model_type == 'student' else None,
        'num_pages': len(valid_pages),
        'min_panels': args.min_panels,
        'seed': args.seed
    }
    
    output_data = {
        'config': config,
        'algorithm_results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

