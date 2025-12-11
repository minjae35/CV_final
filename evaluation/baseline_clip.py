"""
Baseline 1: CLIP Image Similarity Only
Use CLIP model for image-only similarity without text
"""
import sys
from pathlib import Path
import argparse
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets.panel_dataset import TripletDataset, get_default_transform
from evaluation.baselines import (
    create_middle_panel_selection_task,
    compute_metrics,
    print_metrics,
    save_results
)

# Import CLIP
try:
    import clip
except ImportError:
    print("ERROR: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)


def load_clip_model(device: torch.device, model_name: str = "ViT-B/32"):
    """Load CLIP model"""
    print(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    print(f"CLIP model loaded successfully")
    return model, preprocess


def evaluate_clip_baseline(
    task: dict,
    clip_model,
    clip_preprocess,
    device: torch.device
) -> dict:
    """
    Evaluate single task using CLIP image similarity
    
    Strategy:
    - Encode A and C with CLIP (image only)
    - For each candidate B, compute similarity with A and C
    - Select B that maximizes combined similarity: sim(A,B) + sim(B,C)
    """
    A = task['A']
    C = task['C']
    candidates = task['candidates']
    correct_idx = task['correct_idx']
    
    # Encode A and C images with CLIP
    with torch.no_grad():
        # Preprocess images
        img_A = clip_preprocess(A['pil_image']).unsqueeze(0).to(device)
        img_C = clip_preprocess(C['pil_image']).unsqueeze(0).to(device)
        
        # Get CLIP image embeddings
        z_A_clip = clip_model.encode_image(img_A)
        z_C_clip = clip_model.encode_image(img_C)
        z_A_clip = F.normalize(z_A_clip, dim=1)
        z_C_clip = F.normalize(z_C_clip, dim=1)
        
        # Compute similarity scores for each candidate
        scores = []
        for candidate in candidates:
            img_B = clip_preprocess(candidate['pil_image']).unsqueeze(0).to(device)
            z_B_clip = clip_model.encode_image(img_B)
            z_B_clip = F.normalize(z_B_clip, dim=1)
            
            # Compute similarity: sim(A,B) + sim(B,C)
            sim_AB = torch.sum(z_A_clip * z_B_clip).item()
            sim_BC = torch.sum(z_B_clip * z_C_clip).item()
            score = sim_AB + sim_BC
            scores.append(score)
    
    # Find predicted index
    predicted_idx = int(np.argmax(scores))
    is_correct = (predicted_idx == correct_idx)
    
    # Compute rank
    sorted_indices = np.argsort(scores)[::-1]
    rank = int(np.where(sorted_indices == correct_idx)[0][0]) + 1
    
    return {
        'predicted_idx': predicted_idx,
        'correct_idx': correct_idx,
        'is_correct': is_correct,
        'rank': rank,
        'scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline 1: CLIP Image Similarity')
    parser.add_argument('--data_dir', type=str, 
                       default=str(Path.home() / 'data' / 'raw_panel_images_small'),
                       help='Directory containing panel images')
    parser.add_argument('--triplets', type=str, 
                       default=str(Path.home() / 'data' / 'triplets_small.json'),
                       help='Path to triplets JSON file')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       choices=['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'],
                       help='CLIP model architecture')
    parser.add_argument('--num_candidates', type=int, default=10,
                       help='Number of candidate panels')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, 
                       default='results/baseline_clip_eval.json',
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
    
    # Load CLIP model
    clip_model, clip_preprocess = load_clip_model(device, args.clip_model)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.triplets}")
    full_dataset = TripletDataset(
        triplets_json_path=Path(args.triplets),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=False),
        return_pil=True  # Need PIL images for CLIP preprocessing
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
    print(f"\nEvaluating {len(tasks)} tasks with CLIP...")
    results = []
    
    for task in tqdm(tasks, desc="Evaluating CLIP"):
        result = evaluate_clip_baseline(task, clip_model, clip_preprocess, device)
        results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(results)
    print_metrics(metrics, "Baseline 1: CLIP Image Similarity")
    
    # Save results
    config = {
        'model': 'CLIP Image Similarity Only',
        'clip_model': args.clip_model,
        'num_candidates': args.num_candidates,
        'num_samples': len(tasks),
        'seed': args.seed
    }
    
    save_results(Path(args.output), metrics, config, results)


if __name__ == '__main__':
    main()

