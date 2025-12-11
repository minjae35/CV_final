"""
Ablation Study: Component-wise Performance Analysis
Evaluate model performance with different components removed
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.eval_local import (
    load_models, create_middle_panel_selection_task, evaluate_task,
    compute_metrics, save_results, print_metrics
)
from datasets.panel_dataset import TripletDataset, get_default_transform
from models.encoder import MultimodalEncoder


def create_image_only_encoder(encoder: MultimodalEncoder) -> MultimodalEncoder:
    """
    Create image-only encoder (text encoder disabled)
    This is a simplified version that only uses image features
    """
    # For ablation, we'll use empty text for all panels
    # The encoder will still process text but with empty strings
    return encoder  # Return same encoder, but we'll pass empty text during evaluation


def evaluate_ablation_variant(
    variant_name: str,
    tasks: List[Dict],
    encoder: MultimodalEncoder,
    generator: Optional[nn.Module] = None,
    coherence_head: Optional[nn.Module] = None,
    model_type: str = 'student',
    variant_type: str = 'full',
    alpha: float = 0.5,
    beta: float = 0.5,
    device: torch.device = torch.device('cpu'),
    infilling_checkpoint: Optional[Path] = None,
    gan_checkpoint: Optional[Path] = None
) -> Dict:
    """
    Evaluate a specific ablation variant
    
    Variant types:
    - 'full': Full model (baseline)
    - 'w/o_text': Image only (empty OCR text)
    - 'w/o_gan': Infilling only (no GAN, no Student)
    - 'w/o_student': GAN only (no Student/CoherenceHead)
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {variant_name}")
    print(f"{'='*70}")
    
    results = []
    
    for task in tqdm(tasks, desc=variant_name):
        # Modify task based on variant
        if variant_type == 'w/o_text':
            # Use empty text for all panels
            task_modified = task.copy()
            task_modified['A'] = {**task['A'], 'ocr_text': '[EMPTY]'}
            task_modified['C'] = {**task['C'], 'ocr_text': '[EMPTY]'}
            for i, candidate in enumerate(task_modified['candidates']):
                task_modified['candidates'][i] = {**candidate, 'ocr_text': '[EMPTY]'}
            task = task_modified
        
        # Determine model_type and components based on variant
        if variant_type == 'w/o_gan':
            # Use infilling model (no GAN, no Student)
            # Load Infilling generator from Infilling checkpoint
            if infilling_checkpoint is not None and infilling_checkpoint.exists():
                # Load only generator from Infilling checkpoint
                import torch
                infilling_ckpt = torch.load(infilling_checkpoint, map_location=device, weights_only=False)
                if 'generator_state_dict' in infilling_ckpt:
                    gen_state_dict = infilling_ckpt['generator_state_dict']
                elif 'model_state_dict' in infilling_ckpt:
                    gen_state_dict = infilling_ckpt['model_state_dict']
                else:
                    gen_state_dict = infilling_ckpt
                
                # Check if old or new structure
                from models.generator import InfillingGenerator
                is_old_model = 'network.0.weight' in gen_state_dict
                
                if is_old_model:
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
                    
                    infilling_gen = SimpleMLPGenerator(embedding_dim=128, hidden_dim=256)
                else:
                    infilling_gen = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4)
                
                infilling_gen.load_state_dict(gen_state_dict)
                infilling_gen = infilling_gen.to(device)
                infilling_gen.eval()
                eval_generator = infilling_gen
            else:
                eval_generator = generator  # Fallback to provided generator
            eval_model_type = 'infilling'
            eval_coherence_head = None
            eval_alpha = 0.5
            eval_beta = 0.5
        elif variant_type == 'w/o_student':
            # Use GAN model (no Student)
            # Load GAN generator from GAN checkpoint
            if gan_checkpoint is not None and gan_checkpoint.exists():
                # Load only generator from GAN checkpoint
                import torch
                gan_ckpt = torch.load(gan_checkpoint, map_location=device, weights_only=False)
                if 'generator_state_dict' in gan_ckpt:
                    gen_state_dict = gan_ckpt['generator_state_dict']
                elif 'model_state_dict' in gan_ckpt:
                    gen_state_dict = gan_ckpt['model_state_dict']
                else:
                    gen_state_dict = gan_ckpt
                
                # Check if old or new structure
                from models.generator import InfillingGenerator
                is_old_model = 'network.0.weight' in gen_state_dict
                
                if is_old_model:
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
                    
                    gan_gen = SimpleMLPGenerator(embedding_dim=128, hidden_dim=256)
                else:
                    gan_gen = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4)
                
                gan_gen.load_state_dict(gen_state_dict)
                gan_gen = gan_gen.to(device)
                gan_gen.eval()
                eval_generator = gan_gen
            else:
                eval_generator = generator  # Fallback to provided generator
            eval_model_type = 'gan'
            eval_coherence_head = None
            eval_alpha = 0.5
            eval_beta = 0.5
        else:
            # Full model or w/o_text (uses full model but with empty text)
            eval_model_type = model_type
            eval_generator = generator
            eval_coherence_head = coherence_head
            eval_alpha = alpha
            eval_beta = beta
        
        result = evaluate_task(
            task=task,
            model_type=eval_model_type,
            encoder=encoder,
            generator=eval_generator,
            discriminator=None,  # Not used in evaluation
            coherence_head=eval_coherence_head,
            device=device,
            alpha=eval_alpha,
            beta=eval_beta
        )
        results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(results)
    print_metrics(metrics, model_name=variant_name)
    
    return {
        'variant_name': variant_name,
        'variant_type': variant_type,
        'metrics': metrics,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Component-wise Analysis')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='student',
                        choices=['infilling', 'gan', 'student'],
                        help='Base model type for ablation study')
    parser.add_argument('--encoder', type=str, required=True,
                        help='Path to encoder checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (Student checkpoint for full model)')
    parser.add_argument('--infilling_checkpoint', type=str, default=None,
                        help='Path to Infilling checkpoint (for w/o_gan variant)')
    parser.add_argument('--gan_checkpoint', type=str, default=None,
                        help='Path to GAN checkpoint (for w/o_student variant)')
    
    # Data configuration
    parser.add_argument('--triplets', type=str, required=True,
                        help='Path to triplets JSON file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing panel images')
    
    # Evaluation configuration
    parser.add_argument('--num_candidates', type=int, default=10,
                        help='Number of candidate B panels')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--use_val_split', action='store_true',
                        help='Use validation split')
    
    # Student model specific
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for infilling similarity')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for coherence score')
    
    # Ablation variants
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['full', 'w/o_text', 'w/o_gan', 'w/o_student'],
                        choices=['full', 'w/o_text', 'w/o_gan', 'w/o_student'],
                        help='Ablation variants to evaluate')
    
    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    import random
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
        checkpoint_path=Path(args.checkpoint),
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
        from torch.utils.data import random_split
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
    
    # Create tasks (same tasks for all variants)
    tasks = create_middle_panel_selection_task(
        triplet_dataset=eval_dataset,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples
    )
    
    print(f"\nCreated {len(tasks)} tasks for ablation study")
    
    # Evaluate each variant
    all_results = {}
    
    for variant in args.variants:
        variant_name_map = {
            'full': 'Full Model',
            'w/o_text': 'w/o Text Encoder',
            'w/o_gan': 'w/o GAN',
            'w/o_student': 'w/o Score Distillation'
        }
        variant_name = variant_name_map.get(variant, variant)
        
        result = evaluate_ablation_variant(
            variant_name=variant_name,
            tasks=tasks,
            encoder=encoder,
            generator=generator,
            coherence_head=coherence_head,
            model_type=args.model_type,
            variant_type=variant,
            alpha=args.alpha,
            beta=args.beta,
            device=device,
            infilling_checkpoint=Path(args.infilling_checkpoint) if args.infilling_checkpoint else None,
            gan_checkpoint=Path(args.gan_checkpoint) if args.gan_checkpoint else None
        )
        
        all_results[variant] = result
    
    # Create comparison table
    print("\n" + "="*70)
    print("Ablation Study Results Summary")
    print("="*70)
    
    comparison_data = []
    for variant, result in all_results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Variant': result['variant_name'],
            'Top-1 Acc (%)': metrics['top1_accuracy'] * 100,
            'Recall@3 (%)': metrics['recall_at_3'] * 100,
            'Recall@5 (%)': metrics['recall_at_5'] * 100,
            'MRR': metrics['mrr'],
            'Avg Rank': metrics['avg_rank']
        })
    
    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))
    
    # Save results
    if args.output is None:
        output_path = Path(f"results/ablation_study_{args.model_type}.json")
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_type': args.model_type,
        'encoder_path': str(args.encoder),
        'checkpoint_path': str(args.checkpoint),
        'triplets_path': str(args.triplets),
        'data_dir': str(args.data_dir),
        'variants': args.variants,
        'num_candidates': args.num_candidates,
        'num_samples': args.num_samples,
        'use_val_split': args.use_val_split,
        'alpha': args.alpha,
        'beta': args.beta,
        'seed': args.seed
    }
    
    output_data = {
        'config': config,
        'comparison_table': comparison_data,
        'detailed_results': {k: {
            'variant_name': v['variant_name'],
            'variant_type': v['variant_type'],
            'metrics': v['metrics']
        } for k, v in all_results.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Comparison table saved to: {csv_path}")
    print("\nAblation study completed!")


if __name__ == "__main__":
    main()




