"""
GAN Training for Visual Narrative Understanding
Train Generator G and Discriminator D with adversarial loss
Combined with reconstruction loss and contrastive loss
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.encoder import MultimodalEncoder
from models.generator import InfillingGenerator
from models.discriminator import Discriminator
from datasets.panel_dataset import TripletDataset, collate_triplets, get_default_transform


def setup_logging(log_dir: Path, exp_folder: str, exp_name: str) -> logging.Logger:
    """Setup logging to file and console"""
    exp_log_dir = log_dir / exp_folder
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = exp_log_dir / f"{exp_name}.log"
    
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


def send_slack_notification(message: str):
    """Send notification to Slack"""
    load_dotenv()
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    if webhook_url:
        try:
            requests.post(webhook_url, json={"text": message}, timeout=5)
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")


class GANLoss(nn.Module):
    """
    GAN Loss for Discriminator and Generator
    Uses BCEWithLogitsLoss for numerical stability with mixed precision
    """
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Discriminator loss: maximize log(D(z_real)) + log(1 - D(z_fake))
        Equivalently: minimize -log(D(z_real)) - log(1 - D(z_fake))
        Using BCEWithLogits: BCE(D(z_real), 1) + BCE(D(z_fake), 0)
        
        Args:
            real_logits: [batch_size, 1] - D(z_real) logits
            fake_logits: [batch_size, 1] - D(z_fake) logits
        
        Returns:
            loss: scalar
        """
        batch_size = real_logits.size(0)
        real_labels = torch.ones(batch_size, 1, device=real_logits.device)
        fake_labels = torch.zeros(batch_size, 1, device=fake_logits.device)
        
        real_loss = self.bce_loss(real_logits, real_labels)
        fake_loss = self.bce_loss(fake_logits, fake_labels)
        
        d_loss = real_loss + fake_loss
        return d_loss
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Generator loss: maximize log(D(z_fake))
        Equivalently: minimize -log(D(z_fake))
        Using BCEWithLogits: BCE(D(z_fake), 1)
        
        Args:
            fake_logits: [batch_size, 1] - D(z_fake) logits
        
        Returns:
            loss: scalar
        """
        batch_size = fake_logits.size(0)
        real_labels = torch.ones(batch_size, 1, device=fake_logits.device)
        
        g_loss = self.bce_loss(fake_logits, real_labels)
        return g_loss


def train_one_epoch(
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    recon_criterion: nn.Module,
    gan_criterion: GANLoss,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    lambda_recon: float = 1.0,
    lambda_gan: float = 0.5,
    scaler: GradScaler = None,
    gradient_accumulation_steps: int = 1,
    total_epochs: int = 1,
    d_steps: int = 1,
    g_steps: int = 1
) -> Tuple[float, float, float, int]:
    """
    Train for one epoch with GAN
    
    Returns:
        (avg_g_loss, avg_d_loss, avg_recon_loss, global_step)
    """
    generator.train()
    discriminator.train()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_recon_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        ncols=80,
        mininterval=1.0,
        maxinterval=2.0,
        leave=False,
        position=0,
        dynamic_ncols=False
    )
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        img_a, text_a = batch['A']['image'].to(device), batch['A']['ocr_text']
        img_b, text_b = batch['B']['image'].to(device), batch['B']['ocr_text']
        img_c, text_c = batch['C']['image'].to(device), batch['C']['ocr_text']
        
        # --- Discriminator Training ---
        d_loss_accum = 0.0
        for d_step in range(d_steps):
            d_optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                # Encode panels
                with torch.no_grad():
                    z_a = encoder(img_a, text_a)
                    z_b = encoder(img_b, text_b)
                    z_c = encoder(img_c, text_c)
                    
                    # Generate fake z_b
                    z_b_fake = generator(z_a, z_c).detach()  # Detach to not affect generator
                
                # Discriminator forward
                real_prob = discriminator(z_b)
                fake_prob = discriminator(z_b_fake)
                
                # Discriminator loss
                d_loss = gan_criterion.discriminator_loss(real_prob, fake_prob)
            
            # Backward
            if use_amp:
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                d_loss.backward()
                d_optimizer.step()
            
            d_loss_accum += d_loss.item()
        
        # --- Generator Training ---
        for _ in range(g_steps):
            g_optimizer.zero_grad()
            
            with autocast(enabled=use_amp):
                # Encode panels (allow gradients for encoder fine-tuning)
                z_a = encoder(img_a, text_a)
                z_b = encoder(img_b, text_b)
                z_c = encoder(img_c, text_c)
                
                # Generate fake z_b
                z_b_fake = generator(z_a, z_c)
                
                # Reconstruction loss
                recon_loss = recon_criterion(z_b_fake, z_b)
                
                # GAN loss (fool discriminator)
                fake_prob = discriminator(z_b_fake)
                gan_loss = gan_criterion.generator_loss(fake_prob)
                
                # Total generator loss
                g_loss = lambda_recon * recon_loss + lambda_gan * gan_loss
            
            # Backward
            if use_amp:
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                g_loss.backward()
                g_optimizer.step()
        
        # Accumulate losses
        total_g_loss += g_loss.item()
        total_d_loss += d_loss_accum / d_steps  # Average over d_steps
        total_recon_loss += recon_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'G_loss': f'{g_loss.item():.4f}',
            'D_loss': f'{d_loss.item():.4f}',
            'Recon': f'{recon_loss.item():.4f}'
        })
        
        # Log to TensorBoard
        writer.add_scalar('Train/G_Loss', g_loss.item(), global_step)
        writer.add_scalar('Train/D_Loss', d_loss.item(), global_step)
        writer.add_scalar('Train/Recon_Loss', recon_loss.item(), global_step)
        
        global_step += 1
    
    avg_g_loss = total_g_loss / num_batches
    avg_d_loss = total_d_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    
    return avg_g_loss, avg_d_loss, avg_recon_loss, global_step


def validate(
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    recon_criterion: nn.Module,
    gan_criterion: GANLoss,
    device: torch.device,
    epoch: int,
    lambda_recon: float = 1.0,
    lambda_gan: float = 0.5,
    total_epochs: int = 1
) -> Tuple[float, float, float]:
    """
    Validate the model
    
    Returns:
        (avg_g_loss, avg_d_loss, avg_recon_loss)
    """
    generator.eval()
    discriminator.eval()
    encoder.eval()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_recon_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [Val]",
        ncols=80,
        mininterval=1.0,
        maxinterval=2.0,
        leave=False,
        position=0,
        dynamic_ncols=False
    )
    
    with torch.no_grad():
        for batch in pbar:
            # Move data to device
            img_a, text_a = batch['A']['image'].to(device), batch['A']['ocr_text']
            img_b, text_b = batch['B']['image'].to(device), batch['B']['ocr_text']
            img_c, text_c = batch['C']['image'].to(device), batch['C']['ocr_text']
            
            # Encode panels
            z_a = encoder(img_a, text_a)
            z_b = encoder(img_b, text_b)
            z_c = encoder(img_c, text_c)
            
            # Generate fake z_b
            z_b_fake = generator(z_a, z_c)
            
            # Reconstruction loss
            recon_loss = recon_criterion(z_b_fake, z_b)
            
            # Discriminator evaluation
            real_prob = discriminator(z_b)
            fake_prob = discriminator(z_b_fake)
            d_loss = gan_criterion.discriminator_loss(real_prob, fake_prob)
            
            # Generator GAN loss
            gan_loss = gan_criterion.generator_loss(fake_prob)
            g_loss = lambda_recon * recon_loss + lambda_gan * gan_loss
            
            # Accumulate losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_recon_loss += recon_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}'
            })
    
    avg_g_loss = total_g_loss / num_batches
    avg_d_loss = total_d_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    
    return avg_g_loss, avg_d_loss, avg_recon_loss


def save_checkpoint(
    epoch: int,
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    g_scheduler: torch.optim.lr_scheduler._LRScheduler,
    d_scheduler: torch.optim.lr_scheduler._LRScheduler,
    val_g_loss: float,
    checkpoint_path: Path
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_scheduler_state_dict': g_scheduler.state_dict(),
        'd_scheduler_state_dict': d_scheduler.state_dict(),
        'val_g_loss': val_g_loss,
    }, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    encoder: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    g_scheduler: torch.optim.lr_scheduler._LRScheduler,
    d_scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> Tuple[int, float]:
    """
    Load checkpoint
    
    Returns:
        (start_epoch, best_val_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
    d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('val_g_loss', float('inf'))
    
    return start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='GAN Training for Visual Narrative Understanding')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=str(Path.home() / 'data' / 'raw_panel_images_small'),
                        help='Directory containing panel images')
    parser.add_argument('--triplets', type=str, default=str(Path.home() / 'data' / 'triplets_small.json'),
                        help='Path to triplets JSON file')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for G and D')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-4, help='Discriminator learning rate')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--lambda_gan', type=float, default=0.5, help='GAN loss weight')
    parser.add_argument('--d_steps', type=int, default=1, help='D update steps per iteration')
    parser.add_argument('--g_steps', type=int, default=1, help='G update steps per iteration')
    
    # Pretrained models
    parser.add_argument('--encoder_checkpoint', type=str, required=True, help='Path to pretrained encoder')
    parser.add_argument('--generator_checkpoint', type=str, required=True, help='Path to pretrained generator')
    
    # Optimization
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    log_dir = project_root / 'logs'
    
    # Determine experiment folder name
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            exp_folder = resume_path.parent.name
        else:
            timestamp = datetime.now().strftime("%H%M%S")
            exp_folder = f'gan_{timestamp}'
    else:
        timestamp = datetime.now().strftime("%H%M%S")
        exp_folder = f'gan_{timestamp}'
    
    # Create checkpoint directory
    checkpoint_dir = project_root / 'checkpoints' / exp_folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = project_root / 'runs'
    
    # Setup logging
    logger = setup_logging(log_dir, exp_folder, 'train_gan')
    logger.info("=" * 70)
    logger.info("GAN Training for Visual Narrative Understanding")
    logger.info("=" * 70)
    logger.info(f"Arguments: {args}")
    
    # Device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    
    device = torch.device('cuda')
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    logger.info("\nLoading dataset...")
    triplets_path = Path(args.triplets)
    panels_dir = Path(args.data_dir)
    
    full_dataset = TripletDataset(
        triplets_json_path=triplets_path,
        panels_dir=panels_dir,
        transform=get_default_transform(is_train=True)
    )
    
    # Split train/val (90/10)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Total: {total_size} triplets")
    logger.info(f"Train: {train_size} triplets (90%)")
    logger.info(f"Val: {val_size} triplets (10%)")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=True
    )
    
    # Load models
    logger.info("\nLoading models...")
    
    # Load pretrained encoder
    logger.info(f"Loading encoder from: {args.encoder_checkpoint}")
    encoder = MultimodalEncoder(embedding_dim=args.embedding_dim)
    encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    if 'model_state_dict' in encoder_ckpt:
        encoder.load_state_dict(encoder_ckpt['model_state_dict'])
    else:
        encoder.load_state_dict(encoder_ckpt)
    encoder = encoder.to(device)
    
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        logger.info("Encoder frozen")
    else:
        encoder.train()
        logger.info("Encoder fine-tuning enabled")
    
    # Load pretrained generator
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
        
        generator = SimpleMLPGenerator(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim)
    else:
        # New attention-based structure
        logger.info("  - Detected improved InfillingGenerator (Cross-Attention + Residual)")
        generator = InfillingGenerator(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, num_heads=4)
    
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    
    # Create discriminator
    discriminator = Discriminator(embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim)
    discriminator = discriminator.to(device)
    logger.info("Discriminator created")
    
    # Count parameters
    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in generator.parameters()) +
        sum(p.numel() for p in discriminator.parameters())
    )
    trainable_params = (
        sum(p.numel() for p in encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in generator.parameters() if p.requires_grad) +
        sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    gan_criterion = GANLoss()
    
    # Optimizers
    g_params = list(generator.parameters())
    if not args.freeze_encoder:
        g_params += list(encoder.parameters())
    
    g_optimizer = torch.optim.Adam(g_params, lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # Schedulers
    g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        g_optimizer, mode='min', factor=0.5, patience=5
    )
    d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        d_optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        logger.info("Using Mixed Precision (FP16) training")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=runs_dir / 'gan')
    
    # Resume or start from scratch
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"\nResuming from checkpoint: {resume_path}")
            start_epoch, best_val_loss = load_checkpoint(
                resume_path, encoder, generator, discriminator,
                g_optimizer, d_optimizer, g_scheduler, d_scheduler, device
            )
            logger.info(f"Resumed from epoch {start_epoch - 1}, best_val_loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting GAN training...")
    logger.info("=" * 70)
    logger.info(f"Early stopping patience: {args.patience} epochs")
    logger.info(f"Lambda Recon: {args.lambda_recon}, Lambda GAN: {args.lambda_gan}")
    logger.info(f"D steps: {args.d_steps}, G steps: {args.g_steps}")
    logger.info("")
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.info(f"Epoch {epoch}/{args.epochs}")
            
            # Train
            train_g_loss, train_d_loss, train_recon_loss, global_step = train_one_epoch(
                encoder, generator, discriminator,
                train_loader,
                recon_criterion, gan_criterion,
                g_optimizer, d_optimizer,
                device, epoch, writer, global_step,
                lambda_recon=args.lambda_recon,
                lambda_gan=args.lambda_gan,
                scaler=scaler,
                total_epochs=args.epochs,
                d_steps=args.d_steps,
                g_steps=args.g_steps
            )
            
            # Validate
            val_g_loss, val_d_loss, val_recon_loss = validate(
                encoder, generator, discriminator,
                val_loader,
                recon_criterion, gan_criterion,
                device, epoch,
                lambda_recon=args.lambda_recon,
                lambda_gan=args.lambda_gan,
                total_epochs=args.epochs
            )
            
            # Log results
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Train G Loss: {train_g_loss:.4f}, D Loss: {train_d_loss:.4f}, Recon: {train_recon_loss:.4f}")
            logger.info(f"  Val G Loss: {val_g_loss:.4f}, D Loss: {val_d_loss:.4f}, Recon: {val_recon_loss:.4f}")
            logger.info(f"  G LR: {g_optimizer.param_groups[0]['lr']:.6f}, D LR: {d_optimizer.param_groups[0]['lr']:.6f}")
            
            # TensorBoard logging
            writer.add_scalar('Val/G_Loss', val_g_loss, epoch)
            writer.add_scalar('Val/D_Loss', val_d_loss, epoch)
            writer.add_scalar('Val/Recon_Loss', val_recon_loss, epoch)
            writer.add_scalar('Learning_Rate/G', g_optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Learning_Rate/D', d_optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint every epoch
            save_checkpoint(
                epoch, encoder, generator, discriminator,
                g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                val_g_loss,
                checkpoint_dir / f'gan_epoch_{epoch}.pth'
            )
            
            # Check for best model
            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                patience_counter = 0
                logger.info(f"  ✅ New best validation loss: {best_val_loss:.4f}")
                
                # Save best model
                save_checkpoint(
                    epoch, encoder, generator, discriminator,
                    g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                    val_g_loss,
                    checkpoint_dir / 'gan_best.pth'
                )
            else:
                patience_counter += 1
                logger.info(f"  ⚠️  No improvement ({patience_counter}/{args.patience})")
                
                if patience_counter >= args.patience:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            # Update learning rate
            g_scheduler.step(val_g_loss)
            d_scheduler.step(val_d_loss)
            
            logger.info("")
    
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    
    # Final results
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {checkpoint_dir / 'gan_best.pth'}")
    
    # Send Slack notification
    send_slack_notification(
        f"🎉 GAN Training Complete!\n"
        f"Experiment: {exp_folder}\n"
        f"Epochs: {epoch}/{args.epochs}\n"
        f"Best Val Loss: {best_val_loss:.4f}"
    )
    
    writer.close()


if __name__ == '__main__':
    main()

