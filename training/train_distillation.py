"""
Score Distillation Training
Train CoherenceHead with Teacher VLM scores
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from models.discriminator import Discriminator
from models.coherence_head import CoherenceHead
from datasets.panel_dataset import get_default_transform
import pandas as pd
from PIL import Image
from torchvision import transforms


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


class TeacherScoreDataset(torch.utils.data.Dataset):
    """Dataset that loads triplets directly from teacher scores JSON (includes both positive and negative)"""
    def __init__(
        self,
        teacher_scores_path: Path,
        panels_dir: Path,
        transform: Optional[transforms.Compose] = None,
        metadata_csv_path: Optional[Path] = None
    ):
        """
        Args:
            teacher_scores_path: Path to teacher_scores_adjusted.json
            panels_dir: Directory containing panel images
            transform: Optional image transformation
            metadata_csv_path: Optional path to metadata CSV
        """
        self.teacher_scores_path = Path(teacher_scores_path)
        self.panels_dir = Path(panels_dir)
        self.transform = transform
        self.return_pil = False
        
        # Load teacher scores
        print(f"Loading teacher scores from {self.teacher_scores_path}...")
        with open(self.teacher_scores_path, 'r') as f:
            teacher_data = json.load(f)
        
        self.results = teacher_data['results']
        pos_count = sum(1 for r in self.results if r.get('is_positive', True))
        neg_count = len(self.results) - pos_count
        print(f"Loaded {len(self.results)} samples (positive: {pos_count}, negative: {neg_count})")
        
        # Load metadata if provided
        self.metadata_df = None
        if metadata_csv_path and Path(metadata_csv_path).exists():
            self.metadata_df = pd.read_csv(metadata_csv_path)
            print(f"Loaded metadata from {metadata_csv_path}")
    
    def __len__(self):
        return len(self.results)
    
    def __getitem__(self, idx):
        result = self.results[idx]
        triplet = result['triplet']
        is_positive = result.get('is_positive', True)
        teacher_score = result['teacher_score']
        
        # Load panels
        panel_a = self._load_panel(triplet['A'])
        panel_b = self._load_panel(triplet['B'])
        panel_c = self._load_panel(triplet['C'])
        
        return {
            'A': panel_a,
            'B': panel_b,
            'C': panel_c,
            'teacher_score': torch.tensor(teacher_score, dtype=torch.float32),
            'is_positive': is_positive
        }
    
    def _load_panel(self, panel_info: Dict) -> Dict:
        """Load panel image and text"""
        from PIL import Image
        
        # Get image path
        panel_filename = panel_info.get('panel_filename')
        if panel_filename:
            img_path = self.panels_dir / panel_filename
        else:
            comic_no = panel_info['comic_no']
            page_no = panel_info['page_no']
            panel_index = panel_info['panel_index']
            img_path = self.panels_dir / f"{comic_no}_{page_no}_panel_{panel_index:02d}.jpg"
        
        # Load image
        if img_path.exists():
            if self.return_pil:
                image = Image.open(img_path).convert('RGB')
            else:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
        else:
            # Create dummy image if not found
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
        
        # Get OCR text
        ocr_text = panel_info.get('ocr_text', '')
        
        # Get metadata if available
        metadata = {
            'comic_no': panel_info.get('comic_no'),
            'page_no': panel_info.get('page_no'),
            'panel_index': panel_info.get('panel_index')
        }
        
        return {
            'image': image,
            'ocr_text': ocr_text,
            'metadata': metadata
        }


def collate_teacher_scores(batch: List[Dict]) -> Dict:
    """Collate function for TeacherScoreDataset"""
    # Extract A, B, C images and texts
    images_a = torch.stack([item['A']['image'] for item in batch])
    images_b = torch.stack([item['B']['image'] for item in batch])
    images_c = torch.stack([item['C']['image'] for item in batch])
    
    ocr_texts_a = [item['A']['ocr_text'] for item in batch]
    ocr_texts_b = [item['B']['ocr_text'] for item in batch]
    ocr_texts_c = [item['C']['ocr_text'] for item in batch]
    
    # Metadata
    metadata_a = [item['A']['metadata'] for item in batch]
    metadata_b = [item['B']['metadata'] for item in batch]
    metadata_c = [item['C']['metadata'] for item in batch]
    
    # Teacher scores
    teacher_scores = torch.stack([item['teacher_score'] for item in batch])
    
    # is_positive flags
    is_positive = [item['is_positive'] for item in batch]
    
    return {
        'A': {
            'image': images_a,
            'ocr_text': ocr_texts_a,
            'metadata': metadata_a
        },
        'B': {
            'image': images_b,
            'ocr_text': ocr_texts_b,
            'metadata': metadata_b
        },
        'C': {
            'image': images_c,
            'ocr_text': ocr_texts_c,
            'metadata': metadata_c
        },
        'teacher_score': teacher_scores,
        'is_positive': is_positive
    }


def train_epoch(
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    coherence_head: nn.Module,
    train_loader: DataLoader,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    optimizer_coh: torch.optim.Optimizer,
    device: torch.device,
    lambda_recon: float,
    lambda_gan: float,
    lambda_score: float,
    epoch: int,
    logger: logging.Logger,
    grad_clip: float = 1.0
) -> Dict:
    """Train for one epoch"""
    
    encoder.eval() if hasattr(encoder, 'eval') else None  # Frozen encoder
    generator.train()
    discriminator.train()
    coherence_head.train()
    
    metrics = {
        'g_loss': [],
        'd_loss': [],
        'recon_loss': [],
        'gan_loss': [],
        'score_loss': [],
        'score_correlation': []
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", ncols=100, leave=False, dynamic_ncols=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        img_a = batch['A']['image'].to(device)
        img_b = batch['B']['image'].to(device)
        img_c = batch['C']['image'].to(device)
        text_a = batch['A']['ocr_text']
        text_b = batch['B']['ocr_text']
        text_c = batch['C']['ocr_text']
        teacher_scores = batch['teacher_score'].to(device)
        
        batch_size = img_a.size(0)
        
        # Encode panels
        with torch.no_grad():
            z_a = encoder(img_a, text_a)
            z_b_real = encoder(img_b, text_b)
            z_c = encoder(img_c, text_c)
        
        # Generate fake z_b
        z_b_fake = generator(z_a, z_c)
        
        # ==================
        # Train Discriminator
        # ==================
        optimizer_D.zero_grad()
        
        # Real and fake predictions
        d_real = discriminator(z_b_real.detach())
        d_fake = discriminator(z_b_fake.detach())
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        d_loss_fake = F.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake)
        )
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        # ==================
        # Train Generator + CoherenceHead
        # ==================
        optimizer_G.zero_grad()
        optimizer_coh.zero_grad()
        
        # Regenerate fake z_b for generator update
        z_b_fake = generator(z_a, z_c)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(z_b_fake, z_b_real.detach())
        
        # GAN loss
        d_fake = discriminator(z_b_fake)
        gan_loss = F.binary_cross_entropy_with_logits(
            d_fake, torch.ones_like(d_fake)
        )
        
        # Coherence score prediction
        student_scores = coherence_head(z_a, z_b_fake, z_c).squeeze()
        
        # Score distillation loss
        score_loss = F.mse_loss(student_scores, teacher_scores)
        
        # Total generator loss
        g_loss = lambda_recon * recon_loss + lambda_gan * gan_loss + lambda_score * score_loss
        
        g_loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(coherence_head.parameters(), grad_clip)
        
        optimizer_G.step()
        optimizer_coh.step()
        
        # Metrics
        metrics['g_loss'].append(g_loss.item())
        metrics['d_loss'].append(d_loss.item())
        metrics['recon_loss'].append(recon_loss.item())
        metrics['gan_loss'].append(gan_loss.item())
        metrics['score_loss'].append(score_loss.item())
        
        # Score correlation
        with torch.no_grad():
            corr = torch.corrcoef(torch.stack([student_scores, teacher_scores]))[0, 1]
            if not torch.isnan(corr):
                metrics['score_correlation'].append(corr.item())
        
        # Update progress bar
        pbar.set_postfix({
            'G_loss': f"{g_loss.item():.4f}",
            'D_loss': f"{d_loss.item():.4f}",
            'Score_loss': f"{score_loss.item():.4f}",
            'Corr': f"{corr.item():.3f}" if not torch.isnan(corr) else "nan"
        })
    
    # Average metrics
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def validate_epoch(
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    coherence_head: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    lambda_recon: float,
    lambda_gan: float,
    lambda_score: float,
    epoch: int,
    logger: logging.Logger
) -> Dict:
    """Validate for one epoch"""
    
    encoder.eval()
    generator.eval()
    discriminator.eval()
    coherence_head.eval()
    
    metrics = {
        'g_loss': [],
        'd_loss': [],
        'recon_loss': [],
        'gan_loss': [],
        'score_loss': [],
        'score_correlation': []
    }
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", ncols=100, leave=False, dynamic_ncols=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    with torch.no_grad():
        for batch in pbar:
            img_a = batch['A']['image'].to(device)
            img_b = batch['B']['image'].to(device)
            img_c = batch['C']['image'].to(device)
            text_a = batch['A']['ocr_text']
            text_b = batch['B']['ocr_text']
            text_c = batch['C']['ocr_text']
            teacher_scores = batch['teacher_score'].to(device)
            
            # Encode
            z_a = encoder(img_a, text_a)
            z_b_real = encoder(img_b, text_b)
            z_c = encoder(img_c, text_c)
            
            # Generate
            z_b_fake = generator(z_a, z_c)
            
            # Losses
            d_real = discriminator(z_b_real)
            d_fake = discriminator(z_b_fake)
            
            d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
            d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            recon_loss = F.mse_loss(z_b_fake, z_b_real)
            gan_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            
            student_scores = coherence_head(z_a, z_b_fake, z_c).squeeze()
            score_loss = F.mse_loss(student_scores, teacher_scores)
            
            g_loss = lambda_recon * recon_loss + lambda_gan * gan_loss + lambda_score * score_loss
            
            # Metrics
            metrics['g_loss'].append(g_loss.item())
            metrics['d_loss'].append(d_loss.item())
            metrics['recon_loss'].append(recon_loss.item())
            metrics['gan_loss'].append(gan_loss.item())
            metrics['score_loss'].append(score_loss.item())
            
            # Correlation
            corr = torch.corrcoef(torch.stack([student_scores, teacher_scores]))[0, 1]
            if not torch.isnan(corr):
                metrics['score_correlation'].append(corr.item())
            
            pbar.set_postfix({
                'G_loss': f"{g_loss.item():.4f}",
                'Score_loss': f"{score_loss.item():.4f}",
                'Corr': f"{corr.item():.3f}" if not torch.isnan(corr) else "nan"
            })
    
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description='Score Distillation Training')
    
    # Model checkpoints
    parser.add_argument('--encoder_checkpoint', type=str, required=True)
    parser.add_argument('--generator_checkpoint', type=str, required=True)
    parser.add_argument('--discriminator_checkpoint', type=str, required=True)
    parser.add_argument('--teacher_scores', type=str, required=True)
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/processed_444_pages/cropped_panels')
    parser.add_argument('--triplets', type=str, default='data/processed_444_pages/triplets_444.json')
    
    # Training
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_coh', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=1e-5)
    parser.add_argument('--lr_d', type=float, default=1e-5)
    
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=7.0)
    parser.add_argument('--lambda_gan', type=float, default=0.1)
    parser.add_argument('--lambda_score', type=float, default=4.0)
    
    # Other
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%H%M%S')
    experiment_name = f"distill_{timestamp}"
    checkpoint_dir = Path('checkpoints') / experiment_name
    log_dir = Path('logs') / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir, f"train_{experiment_name}")
    logger.info("="*70)
    logger.info("Score Distillation Training")
    logger.info("="*70)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Device: {device}")
    
    # Load models
    logger.info("\nLoading models...")
    
    # Encoder
    encoder = MultimodalEncoder(embedding_dim=128).to(device)
    encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    if 'encoder_state_dict' in encoder_ckpt:
        encoder.load_state_dict(encoder_ckpt['encoder_state_dict'])
    elif 'model_state_dict' in encoder_ckpt:
        encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_ckpt)
    logger.info(f"✅ Encoder loaded from: {args.encoder_checkpoint}")
    
    # Generator - dynamically detect architecture
    logger.info(f"Loading generator from: {args.generator_checkpoint}")
    gen_ckpt = torch.load(args.generator_checkpoint, map_location=device)
    
    # Check if old simple MLP structure or new attention structure
    if 'generator_state_dict' in gen_ckpt:
        state_dict = gen_ckpt['generator_state_dict']
    elif 'model_state_dict' in gen_ckpt:
        state_dict = gen_ckpt['model_state_dict']
    else:
        state_dict = gen_ckpt
    
    # Check if old model (has 'network.0.weight') or new model
    is_old_model = 'network.0.weight' in state_dict
    
    if is_old_model:
        # Old simple MLP structure: concat z_A and z_C, then 3-layer MLP
        logger.info("  - Detected simple MLP InfillingGenerator")
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
        
        generator = SimpleMLPGenerator(embedding_dim=128, hidden_dim=256).to(device)
    else:
        # New attention-based structure
        logger.info("  - Detected attention-based InfillingGenerator")
        generator = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4).to(device)
    
    generator.load_state_dict(state_dict)
    logger.info(f"✅ Generator loaded from: {args.generator_checkpoint}")
    
    # Discriminator
    discriminator = Discriminator(embedding_dim=128, hidden_dim=256).to(device)
    disc_ckpt = torch.load(args.discriminator_checkpoint, map_location=device)
    if 'discriminator_state_dict' in disc_ckpt:
        discriminator.load_state_dict(disc_ckpt['discriminator_state_dict'])
    else:
        discriminator.load_state_dict(disc_ckpt)
    logger.info(f"✅ Discriminator loaded from: {args.discriminator_checkpoint}")
    
    # CoherenceHead (new)
    coherence_head = CoherenceHead(embedding_dim=128, hidden_dim=256).to(device)
    logger.info(f"✅ CoherenceHead initialized")
    
    # Freeze encoder
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("🔒 Encoder frozen")
    
    # Load dataset directly from teacher scores (includes both positive and negative)
    logger.info(f"\nLoading dataset from teacher scores: {args.teacher_scores}")
    full_dataset_with_scores = TeacherScoreDataset(
        teacher_scores_path=Path(args.teacher_scores),
        panels_dir=Path(args.data_dir),
        transform=get_default_transform(is_train=True)
    )
    
    # Train/Val split
    total_size = len(full_dataset_with_scores)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset_with_scores,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Dataset split: Train={train_size}, Val={val_size}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_teacher_scores
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_teacher_scores
    )
    
    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=args.lr_g)
    optimizer_D = Adam(discriminator.parameters(), lr=args.lr_d)
    optimizer_coh = Adam(coherence_head.parameters(), lr=args.lr_coh)
    
    # Schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=args.patience)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=args.patience)
    scheduler_coh = ReduceLROnPlateau(optimizer_coh, mode='min', factor=0.5, patience=args.patience)
    
    # Training loop
    best_val_score_loss = float('inf')
    patience_counter = 0
    
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            encoder, generator, discriminator, coherence_head,
            train_loader, optimizer_G, optimizer_D, optimizer_coh,
            device, args.lambda_recon, args.lambda_gan, args.lambda_score,
            epoch, logger, grad_clip=args.grad_clip
        )
        
        # Validate
        val_metrics = validate_epoch(
            encoder, generator, discriminator, coherence_head,
            val_loader, device, args.lambda_recon, args.lambda_gan, args.lambda_score,
            epoch, logger
        )
        
        # Log metrics
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info(f"Train - G_loss: {train_metrics['g_loss']:.4f}, "
                   f"D_loss: {train_metrics['d_loss']:.4f}, "
                   f"Score_loss: {train_metrics['score_loss']:.4f}, "
                   f"Corr: {train_metrics['score_correlation']:.4f}")
        logger.info(f"Val   - G_loss: {val_metrics['g_loss']:.4f}, "
                   f"D_loss: {val_metrics['d_loss']:.4f}, "
                   f"Score_loss: {val_metrics['score_loss']:.4f}, "
                   f"Corr: {val_metrics['score_correlation']:.4f}")
        
        # Update schedulers
        scheduler_G.step(val_metrics['score_loss'])
        scheduler_D.step(val_metrics['d_loss'])
        scheduler_coh.step(val_metrics['score_loss'])
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'coherence_head_state_dict': coherence_head.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'optimizer_coh_state_dict': optimizer_coh.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'scheduler_coh_state_dict': scheduler_coh.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        torch.save(checkpoint, checkpoint_dir / f'distill_epoch_{epoch}.pth')
        
        # Save best model
        if val_metrics['score_loss'] < best_val_score_loss:
            best_val_score_loss = val_metrics['score_loss']
            torch.save(checkpoint, checkpoint_dir / 'distill_best.pth')
            logger.info(f"✅ Saved best model (val_score_loss: {best_val_score_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info("="*70)
    logger.info(f"Best val_score_loss: {best_val_score_loss:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
