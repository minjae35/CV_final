"""
Infilling Generator Training for Visual Narrative Understanding
Train generator G to predict middle panel embedding from A and C
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
from datasets.panel_dataset import TripletDataset, collate_triplets, get_default_transform


def setup_logging(log_dir: Path, exp_folder: str, exp_name: str) -> logging.Logger:
    """Setup logging to file and console"""
    # Create experiment folder in logs directory
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


def train_one_epoch(
    encoder: nn.Module,
    generator: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    scaler: GradScaler = None,
    gradient_accumulation_steps: int = 1,
    total_epochs: int = 1
) -> Tuple[float, float, int]:
    """Train for one epoch"""
    generator.train()
    # Encoder can be frozen or in train mode (fine-tuning)
    
    total_loss = 0.0
    total_similarity = 0.0
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
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images_a = batch['A']['image'].to(device)
        images_b = batch['B']['image'].to(device)
        images_c = batch['C']['image'].to(device)
        texts_a = batch['A']['ocr_text']
        texts_b = batch['B']['ocr_text']
        texts_c = batch['C']['ocr_text']
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            # Encode panels
            z_a = encoder.encode_panels(images_a, texts_a)
            z_b = encoder.encode_panels(images_b, texts_b)
            z_c = encoder.encode_panels(images_c, texts_c)
            
            # Generate middle panel embedding
            z_b_hat = generator(z_a, z_c)
            
            # Compute reconstruction loss (MSE)
            loss = criterion(z_b_hat, z_b)
            
            # Compute similarity for monitoring
            with torch.no_grad():
                z_b_norm = F.normalize(z_b, dim=1)
                z_b_hat_norm = F.normalize(z_b_hat, dim=1)
                similarity = torch.mean(torch.sum(z_b_norm * z_b_hat_norm, dim=1))
            
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Log (denormalize loss for logging)
        total_loss += loss.item() * gradient_accumulation_steps
        total_similarity += similarity.item()
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'sim': f'{similarity.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'avg_sim': f'{total_similarity / num_batches:.4f}'
        })
        
        # Log to TensorBoard
        if global_step % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/Similarity', similarity.item(), global_step)
    
    avg_loss = total_loss / num_batches
    avg_similarity = total_similarity / num_batches
    
    return avg_loss, avg_similarity, global_step


@torch.no_grad()
def validate(
    encoder: nn.Module,
    generator: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Validate model"""
    encoder.eval()
    generator.eval()
    
    total_loss = 0.0
    total_similarity = 0.0
    num_batches = 0
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch} [Val]",
        ncols=80,
        mininterval=1.0,
        maxinterval=2.0,
        leave=False,
        position=0,
        dynamic_ncols=False
    )
    
    for batch in pbar:
        # Move to device
        images_a = batch['A']['image'].to(device)
        images_b = batch['B']['image'].to(device)
        images_c = batch['C']['image'].to(device)
        texts_a = batch['A']['ocr_text']
        texts_b = batch['B']['ocr_text']
        texts_c = batch['C']['ocr_text']
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            # Encode panels
            z_a = encoder.encode_panels(images_a, texts_a)
            z_b = encoder.encode_panels(images_b, texts_b)
            z_c = encoder.encode_panels(images_c, texts_c)
            
            # Generate middle panel embedding
            z_b_hat = generator(z_a, z_c)
            
            # Compute reconstruction loss (MSE)
            loss = criterion(z_b_hat, z_b)
            
            # Compute similarity
            z_b_norm = F.normalize(z_b, dim=1)
            z_b_hat_norm = F.normalize(z_b_hat, dim=1)
            similarity = torch.mean(torch.sum(z_b_norm * z_b_hat_norm, dim=1))
        
        total_loss += loss.item()
        total_similarity += similarity.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'sim': f'{similarity.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'avg_sim': f'{total_similarity / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_similarity = total_similarity / num_batches
    
    return avg_loss, avg_similarity


def save_checkpoint(
    encoder: nn.Module,
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    val_similarity: float,
    checkpoint_path: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_similarity': val_similarity
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'generator_best.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: Path,
    encoder: nn.Module,
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    return start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train Infilling Generator')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for generator')
    parser.add_argument('--encoder_lr', type=float, default=1e-5, help='Learning rate for encoder (fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Use mixed precision (FP16)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--val_freq', type=int, default=1, help='Validate every N epochs')
    
    # Encoder fine-tuning options
    parser.add_argument('--freeze_encoder', action='store_true', default=False, help='Freeze encoder (no fine-tuning)')
    parser.add_argument('--infonce_checkpoint', type=str, required=True, help='Path to InfoNCE pretrained encoder checkpoint')
    parser.add_argument('--text_max_length', type=int, default=64, help='Max text sequence length')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    log_dir = project_root / 'logs'
    
    # Determine experiment folder name
    if args.resume:
        # Extract experiment folder from resume checkpoint path
        resume_path = Path(args.resume)
        if resume_path.exists():
            # e.g., checkpoints/infilling_043214/infilling_epoch_10.pth -> infilling_043214
            exp_folder = resume_path.parent.name
        else:
            # If resume path doesn't exist, create new folder
            timestamp = datetime.now().strftime("%H%M%S")
            exp_folder = f'infilling_{timestamp}'
    else:
        # Create new timestamped experiment folder
        timestamp = datetime.now().strftime("%H%M%S")
        exp_folder = f'infilling_{timestamp}'
    
    # Create checkpoint directory
    checkpoint_dir = project_root / 'checkpoints' / exp_folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = project_root / 'runs'
    
    # Setup logging (save to same folder structure as checkpoints)
    logger = setup_logging(log_dir, exp_folder, 'train_infilling')
    logger.info("=" * 70)
    logger.info("Infilling Generator Training")
    logger.info("=" * 70)
    logger.info(f"Arguments: {args}")
    
    # Device (GPU required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    
    device = torch.device('cuda')
    logger.info(f"Device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    logger.info("\nLoading dataset...")
    triplets_path = Path.home() / 'data' / 'triplets_small.json'
    panels_dir = Path.home() / 'data' / 'raw_panel_images_small'
    
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
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_triplets,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Load pretrained encoder
    logger.info("\nLoading pretrained encoder...")
    encoder = MultimodalEncoder(
        image_pretrained=True,
        embedding_dim=128,
        text_max_length=args.text_max_length
    ).to(device)
    
    # Load InfoNCE checkpoint
    infonce_checkpoint = torch.load(args.infonce_checkpoint, map_location=device)
    encoder.load_state_dict(infonce_checkpoint['model_state_dict'])
    logger.info(f"Loaded InfoNCE checkpoint from: {args.infonce_checkpoint}")
    logger.info(f"InfoNCE epoch: {infonce_checkpoint['epoch']}, val_loss: {infonce_checkpoint.get('val_loss', 'N/A')}")
    
    # Freeze or fine-tune encoder
    if args.freeze_encoder:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen (no fine-tuning)")
    else:
        encoder.train()
        logger.info("Encoder will be fine-tuned")
    
    # Initialize generator
    logger.info("\nInitializing generator...")
    generator = InfillingGenerator(embedding_dim=128, hidden_dim=256, num_heads=4).to(device)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    encoder_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    generator_params = sum(p.numel() for p in generator.parameters())
    generator_trainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    logger.info(f"Encoder parameters: {encoder_params:,} (trainable: {encoder_trainable:,})")
    logger.info(f"Generator parameters: {generator_params:,} (trainable: {generator_trainable:,})")
    logger.info(f"Total trainable: {encoder_trainable + generator_trainable:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    
    # Setup optimizer with different learning rates
    if args.freeze_encoder:
        optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': args.encoder_lr},
            {'params': generator.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=args.min_lr
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        logger.info("Using Mixed Precision (FP16) training")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=runs_dir / 'infilling')
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"\nResuming from checkpoint: {resume_path}")
            start_epoch, best_val_loss = load_checkpoint(
                resume_path, encoder, generator, optimizer, scheduler, device
            )
            logger.info(f"Resumed from epoch {start_epoch - 1}, best_val_loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    logger.info(f"Early stopping patience: {args.patience} epochs")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_similarity, global_step = train_one_epoch(
            encoder, generator, train_loader, criterion, optimizer, device, epoch, writer, global_step, scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_epochs=args.epochs
        )
        
        # Validate (only every val_freq epochs or on last epoch)
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_loss, val_similarity = validate(encoder, generator, val_loader, criterion, device, epoch, args.mixed_precision)
            
            # Learning rate schedule (based on validation loss)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch results
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Similarity: {train_similarity:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Similarity: {val_similarity:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # TensorBoard
            writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            writer.add_scalar('Epoch/Train_Similarity', train_similarity, epoch)
            writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            writer.add_scalar('Epoch/Val_Similarity', val_similarity, epoch)
            writer.add_scalar('Epoch/LR', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"  ✅ New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  ⚠️  No improvement ({patience_counter}/{args.patience})")
            
            checkpoint_path = checkpoint_dir / f'infilling_epoch_{epoch}.pth'
            save_checkpoint(encoder, generator, optimizer, scheduler, epoch, val_loss, val_similarity, checkpoint_path, is_best)
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
                break
        else:
            # No validation this epoch, just log training results
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Similarity: {train_similarity:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  (Validation skipped - next at epoch {epoch + args.val_freq})")
            
            # TensorBoard (training only)
            writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            writer.add_scalar('Epoch/Train_Similarity', train_similarity, epoch)
            writer.add_scalar('Epoch/LR', current_lr, epoch)
    
    # Training complete
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {checkpoint_dir / 'generator_best.pth'}")
    
    # Close writer
    writer.close()
    
    # Slack notification
    message = f"✅ Infilling Generator Training Complete!\nEpochs: {args.epochs}\nBest Val Loss: {best_val_loss:.4f}"
    send_slack_notification(message)
    logger.info("\nSlack notification sent!")


if __name__ == '__main__':
    main()

