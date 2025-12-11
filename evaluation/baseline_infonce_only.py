"""
Baseline 2: InfoNCE Only (without Infilling Generator)
Use encoder only, predict middle panel by averaging z_A and z_C
"""
import sys
from pathlib import Path
import argparse
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from datasets.panel_dataset import TripletDataset, get_default_transform
from evaluation.baselines import (
    create_middle_panel_selection_task,
    compute_metrics,
    print_metrics,
    save_results
)


def load_encoder(checkpoint_path: Path, device: torch.device):
    """Load pretrained InfoNCE encoder"""
    print(f"Loading encoder from: {checkpoint_path}")
    encoder = MultimodalEncoder(embedding_dim=128)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    elif 'model_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
    
    encoder = encoder.to(device)
    encoder.eval()
    print("Encoder loaded successfully")
    return encoder


def evaluate_infonce_only(
    task: dict,
    encoder: MultimodalEncoder,
    device: torch.device
) -> dict:
    """
    Evaluate single task using InfoNCE encoder only
    
    Strategy:
    - Encode A and C with multimodal encoder (image + text)
    - Predict middle embedding: z_B_pred = (z_A + z_C) / 2
    - For each candidate, compute similarity with z_B_pred
    - Select B that maximizes similarity
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


def main():
    parser = argparse.ArgumentParser(description='Baseline 2: InfoNCE Only')
    parser.add_argument('--encoder', type=str, required=True,
                       help='Path to encoder checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default=str(Path.home() / 'data' / 'raw_panel_images_small'),
                       help='Directory containing panel images')
    parser.add_argument('--triplets', type=str, 
                       default=str(Path.home() / 'data' / 'triplets_small.json'),
                       help='Path to triplets JSON file')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of candidate panels')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, 
                       default='results/baseline_infonce_only_eval.json',
                       help='Output file path')
    
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
    
    # Load encoder
    encoder = load_encoder(Path(args.encoder), device)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.triplets}")
    full_dataset = TripletDataset(
        triplets_json_path=Path(args.triplets),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=False)
    )
    
    # Use validation split
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
    
    # Create wrapper for validation subset
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
    
    # Create evaluation tasks
    tasks = create_middle_panel_selection_task(
        val_wrapper,
        num_candidates=args.num_candidates,
        num_samples=args.num_samples
    )
    
    if len(tasks) == 0:
        print("No valid tasks created. Exiting.")
        return
    
    # Evaluate
    print(f"\nEvaluating {len(tasks)} tasks with InfoNCE-only...")
    results = []
    
    for task in tqdm(tasks, desc="Evaluating InfoNCE-only"):
        result = evaluate_infonce_only(task, encoder, device)
        results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(results)
    print_metrics(metrics, "Baseline 2: InfoNCE Only (z_A + z_C)/2")
    
    # Save results
    config = {
        'model': 'InfoNCE Only (without Generator)',
        'strategy': 'Average of z_A and z_C',
        'encoder': args.encoder,
        'num_candidates': args.num_candidates,
        'num_samples': len(tasks),
        'seed': args.seed
    }
    
    save_results(Path(args.output), metrics, config, results)


if __name__ == '__main__':
    main()

