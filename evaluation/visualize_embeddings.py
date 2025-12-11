"""
Visualize Generated vs Real Embeddings
Use t-SNE or PCA to visualize the distribution of z_B vs z_B_hat
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from datasets.panel_dataset import TripletDataset, collate_triplets, get_default_transform


def collect_embeddings(
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect real and generated embeddings
    
    Returns:
        (z_b_real, z_b_fake): Each of shape [num_samples, embedding_dim]
    """
    encoder.eval()
    generator.eval()
    
    z_b_real_list = []
    z_b_fake_list = []
    
    collected = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Collecting embeddings", total=min(len(dataloader), num_samples // dataloader.batch_size + 1))
        for batch in pbar:
            if collected >= num_samples:
                break
            
            # Move to device
            img_a = batch['A']['image'].to(device)
            text_a = batch['A']['ocr_text']
            img_b = batch['B']['image'].to(device)
            text_b = batch['B']['ocr_text']
            img_c = batch['C']['image'].to(device)
            text_c = batch['C']['ocr_text']
            
            # Encode
            z_a = encoder(img_a, text_a)
            z_b = encoder(img_b, text_b)
            z_c = encoder(img_c, text_c)
            
            # Generate
            z_b_fake = generator(z_a, z_c)
            
            # Collect
            z_b_real_list.append(z_b.cpu().numpy())
            z_b_fake_list.append(z_b_fake.cpu().numpy())
            
            collected += z_b.size(0)
            pbar.set_postfix({'collected': collected})
    
    z_b_real = np.concatenate(z_b_real_list, axis=0)[:num_samples]
    z_b_fake = np.concatenate(z_b_fake_list, axis=0)[:num_samples]
    
    print(f"Collected {z_b_real.shape[0]} real and {z_b_fake.shape[0]} fake embeddings")
    
    return z_b_real, z_b_fake


def visualize_tsne(
    z_real: np.ndarray,
    z_fake: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """Visualize embeddings using t-SNE"""
    print("Running t-SNE...")
    
    # Combine embeddings
    z_combined = np.concatenate([z_real, z_fake], axis=0)
    labels = np.concatenate([
        np.zeros(len(z_real)),  # 0 for real
        np.ones(len(z_fake))    # 1 for fake
    ])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    z_embedded = tsne.fit_transform(z_combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot real embeddings
    real_mask = labels == 0
    plt.scatter(
        z_embedded[real_mask, 0],
        z_embedded[real_mask, 1],
        c='blue',
        label='Real z_B',
        alpha=0.6,
        s=20
    )
    
    # Plot fake embeddings
    fake_mask = labels == 1
    plt.scatter(
        z_embedded[fake_mask, 0],
        z_embedded[fake_mask, 1],
        c='red',
        label='Generated z_B_hat',
        alpha=0.6,
        s=20
    )
    
    plt.title('t-SNE Visualization of Real vs Generated Embeddings', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE visualization to: {output_path}")
    plt.close()


def visualize_pca(
    z_real: np.ndarray,
    z_fake: np.ndarray,
    output_path: Path
):
    """Visualize embeddings using PCA"""
    print("Running PCA...")
    
    # Combine embeddings
    z_combined = np.concatenate([z_real, z_fake], axis=0)
    labels = np.concatenate([
        np.zeros(len(z_real)),  # 0 for real
        np.ones(len(z_fake))    # 1 for fake
    ])
    
    # Apply PCA
    pca = PCA(n_components=2)
    z_embedded = pca.fit_transform(z_combined)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot real embeddings
    real_mask = labels == 0
    plt.scatter(
        z_embedded[real_mask, 0],
        z_embedded[real_mask, 1],
        c='blue',
        label='Real z_B',
        alpha=0.6,
        s=20
    )
    
    # Plot fake embeddings
    fake_mask = labels == 1
    plt.scatter(
        z_embedded[fake_mask, 0],
        z_embedded[fake_mask, 1],
        c='red',
        label='Generated z_B_hat',
        alpha=0.6,
        s=20
    )
    
    plt.title('PCA Visualization of Real vs Generated Embeddings', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PCA visualization to: {output_path}")
    plt.close()


def compute_statistics(z_real: np.ndarray, z_fake: np.ndarray):
    """Compute statistics comparing real and fake embeddings"""
    print("\n" + "=" * 70)
    print("Embedding Statistics")
    print("=" * 70)
    
    # Mean and std
    real_mean = np.mean(z_real, axis=0)
    fake_mean = np.mean(z_fake, axis=0)
    real_std = np.std(z_real, axis=0)
    fake_std = np.std(z_fake, axis=0)
    
    print(f"Real embeddings:")
    print(f"  Mean: {np.mean(real_mean):.4f} ± {np.std(real_mean):.4f}")
    print(f"  Std:  {np.mean(real_std):.4f} ± {np.std(real_std):.4f}")
    
    print(f"Generated embeddings:")
    print(f"  Mean: {np.mean(fake_mean):.4f} ± {np.std(fake_mean):.4f}")
    print(f"  Std:  {np.mean(fake_std):.4f} ± {np.std(fake_std):.4f}")
    
    # Cosine similarity between means
    cos_sim = np.dot(real_mean, fake_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(fake_mean))
    print(f"Cosine similarity (mean vectors): {cos_sim:.4f}")
    
    # L2 distance between means
    l2_dist = np.linalg.norm(real_mean - fake_mean)
    print(f"L2 distance (mean vectors): {l2_dist:.4f}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Visualize Generated vs Real Embeddings')
    parser.add_argument('--encoder', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--generator', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--data_dir', type=str, default=str(Path.home() / 'data' / 'raw_panel_images_small'),
                        help='Directory containing panel images')
    parser.add_argument('--triplets', type=str, default=str(Path.home() / 'data' / 'triplets_small.json'),
                        help='Path to triplets JSON file')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--method', type=str, default='both', choices=['tsne', 'pca', 'both'],
                        help='Visualization method')
    parser.add_argument('--output_dir', type=str, default='results/embeddings_visualization',
                        help='Output directory for visualizations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print(f"\nLoading encoder from: {args.encoder}")
    encoder = MultimodalEncoder(embedding_dim=128)
    encoder_ckpt = torch.load(args.encoder, map_location=device)
    if 'model_state_dict' in encoder_ckpt:
        encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_ckpt)
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"Loading generator from: {args.generator}")
    generator = InfillingGenerator(embedding_dim=128, hidden_dim=256)
    gen_ckpt = torch.load(args.generator, map_location=device)
    if 'generator_state_dict' in gen_ckpt:
        generator.load_state_dict(gen_ckpt['generator_state_dict'])
    elif 'model_state_dict' in gen_ckpt:
        generator.load_state_dict(gen_ckpt['model_state_dict'])
    else:
        generator.load_state_dict(gen_ckpt)
    generator = generator.to(device)
    generator.eval()
    
    # Load dataset
    print(f"\nLoading dataset from: {args.triplets}")
    dataset = TripletDataset(
        triplets_json_path=Path(args.triplets),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=False)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_triplets
    )
    
    # Collect embeddings
    print(f"\nCollecting {args.num_samples} embeddings...")
    z_real, z_fake = collect_embeddings(encoder, generator, dataloader, device, args.num_samples)
    
    # Compute statistics
    compute_statistics(z_real, z_fake)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    if args.method in ['tsne', 'both']:
        visualize_tsne(z_real, z_fake, output_dir / 'embeddings_tsne.png')
    
    if args.method in ['pca', 'both']:
        visualize_pca(z_real, z_fake, output_dir / 'embeddings_pca.png')
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()


