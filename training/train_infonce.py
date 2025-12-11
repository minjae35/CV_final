"""
InfoNCE Contrastive Learning for Visual Narrative Understanding
Train multimodal encoder with contrastive loss
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

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
from datasets.panel_dataset import TripletDataset, collate_triplets, get_default_transform


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for learning panel embeddings
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        neg_batch_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            anchor: [batch_size, embedding_dim] - z_AC (average of A and C)
            positive: [batch_size, embedding_dim] - z_B (correct middle panel)
            negatives: [num_total_negatives, embedding_dim] - z_neg (multiple hard negatives)
            neg_batch_indices: [num_total_negatives] - which batch item each negative belongs to
        
        Returns:
            loss: Scalar loss
            accuracy: Contrastive accuracy (0-1)
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [batch_size]
        
        # Compute negative similarities for each batch item
        # anchor: [batch_size, dim], negatives: [num_negatives, dim]
        # Compute all pairwise similarities
        anchor_neg_sim = torch.matmul(anchor, negatives.t()) / self.temperature  # [batch_size, num_negatives]
        
        # For each anchor, we want to contrast with its positive and all its negatives
        # Create mask to select only the negatives belonging to each batch item
        if neg_batch_indices is not None:
            # Create a mask: [batch_size, num_negatives]
            batch_mask = (neg_batch_indices.unsqueeze(0) == torch.arange(batch_size, device=anchor.device).unsqueeze(1))
            
            # For each sample, compute InfoNCE loss with its positive and its negatives
            loss_total = 0.0
            correct_count = 0
            
            for i in range(batch_size):
                # Get negatives for this sample
                neg_mask_i = batch_mask[i]  # [num_negatives]
                if neg_mask_i.sum() == 0:
                    # No negatives for this sample, skip
                    continue
                
                neg_sims_i = anchor_neg_sim[i][neg_mask_i]  # [num_negatives_i]
                pos_sim_i = pos_sim[i]  # scalar
                
                # Combine positive and negative similarities
                logits_i = torch.cat([pos_sim_i.unsqueeze(0), neg_sims_i])  # [1 + num_negatives_i]
                
                # InfoNCE: positive should have highest similarity
                loss_i = -pos_sim_i + torch.logsumexp(logits_i, dim=0)
                loss_total += loss_i
                
                # Accuracy: positive > all negatives
                if (pos_sim_i > neg_sims_i).all():
                    correct_count += 1
            
            loss = loss_total / batch_size
            accuracy = correct_count / batch_size
        else:
            # Fallback: assume each negative corresponds to each batch item (old behavior)
            # Use first negative for each sample
            neg_sim = anchor_neg_sim[:, 0]  # [batch_size]
            
            # InfoNCE loss
            logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch_size, 2]
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            
            loss = F.cross_entropy(logits, labels)
            
            # Accuracy
            with torch.no_grad():
                correct = (pos_sim > neg_sim).float().sum()
                accuracy = (correct / batch_size).item()
        
        return loss, accuracy


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
    model: nn.Module,
    dataloader: DataLoader,
    criterion: InfoNCELoss,
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
    model.train()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    use_amp = scaler is not None
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        ncols=120,
        leave=True
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
        
        # Get negative samples (multiple negatives)
        neg_images = batch['neg_candidates']['image'].to(device)
        neg_texts = batch['neg_candidates']['ocr_text']
        neg_batch_indices = torch.tensor(batch['neg_candidates']['batch_indices'], device=device)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            z_a = model.encode_panels(images_a, texts_a)
            z_b = model.encode_panels(images_b, texts_b)
            z_c = model.encode_panels(images_c, texts_c)
            z_neg = model.encode_panels(neg_images, neg_texts)
            
            # Create anchor from A and C (average)
            z_ac = (z_a + z_c) / 2
            
            # Compute loss: z_AC should be close to z_B (contrast with multiple negatives)
            loss, acc = criterion(z_ac, z_b, z_neg, neg_batch_indices)
            
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
        total_acc += acc
        num_batches += 1
        global_step += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'acc': f'{acc:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'avg_acc': f'{total_acc / num_batches:.4f}'
        })
        
        # Log to TensorBoard
        if global_step % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/Accuracy', acc, global_step)
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: InfoNCELoss,
    device: torch.device,
    epoch: int,
    use_amp: bool = False
) -> Tuple[float, float]:
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch} [Val]",
        ncols=120,
        leave=True
    )
    
    for batch in pbar:
        # Move to device
        images_a = batch['A']['image'].to(device)
        images_b = batch['B']['image'].to(device)
        images_c = batch['C']['image'].to(device)
        texts_a = batch['A']['ocr_text']
        texts_b = batch['B']['ocr_text']
        texts_c = batch['C']['ocr_text']
        
        # Get negative samples (multiple negatives)
        neg_images = batch['neg_candidates']['image'].to(device)
        neg_texts = batch['neg_candidates']['ocr_text']
        neg_batch_indices = torch.tensor(batch['neg_candidates']['batch_indices'], device=device)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            z_a = model.encode_panels(images_a, texts_a)
            z_b = model.encode_panels(images_b, texts_b)
            z_c = model.encode_panels(images_c, texts_c)
            z_neg = model.encode_panels(neg_images, neg_texts)
            
            # Create anchor from A and C (average)
            z_ac = (z_a + z_c) / 2
            
            # Compute loss: z_AC should be close to z_B (contrast with multiple negatives)
            loss, acc = criterion(z_ac, z_b, z_neg, neg_batch_indices)
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}',
            'avg_acc': f'{total_acc / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_loss: float,
    val_acc: float,
    checkpoint_path: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.parent / 'infonce_best.pth'
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    
    return start_epoch, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train InfoNCE Contrastive Learning')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for InfoNCE')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--mixed_precision', action='store_true', default=True, help='Use mixed precision (FP16)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps (1=disabled)')
    parser.add_argument('--freeze_image_layers', type=int, default=3, help='Freeze first N ResNet blocks')
    parser.add_argument('--freeze_text_layers', type=int, default=3, help='Freeze first N DistilBERT layers')
    parser.add_argument('--text_max_length', type=int, default=64, help='Max text sequence length')
    parser.add_argument('--val_freq', type=int, default=1, help='Validate every N epochs')
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
            # e.g., checkpoints/infonce_032001/infonce_epoch_3.pth -> infonce_032001
            exp_folder = resume_path.parent.name
        else:
            # If resume path doesn't exist, create new folder
            timestamp = datetime.now().strftime("%H%M%S")
            exp_folder = f'infonce_{timestamp}'
    else:
        # Create new timestamped experiment folder
        timestamp = datetime.now().strftime("%H%M%S")
        exp_folder = f'infonce_{timestamp}'
    
    # Create checkpoint directory
    checkpoint_dir = project_root / 'checkpoints' / exp_folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    runs_dir = project_root / 'runs'
    
    # Setup logging (save to same folder structure as checkpoints)
    logger = setup_logging(log_dir, exp_folder, 'train_infonce')
    logger.info("=" * 70)
    logger.info("InfoNCE Contrastive Learning")
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
    
    # Model
    logger.info("\nInitializing model...")
    model = MultimodalEncoder(
        image_pretrained=True,
        embedding_dim=128,
        text_max_length=args.text_max_length,
        freeze_image_layers=args.freeze_image_layers,
        freeze_text_layers=args.freeze_text_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {frozen_params:,}")
    logger.info(f"Freeze: ResNet {args.freeze_image_layers} blocks, DistilBERT {args.freeze_text_layers} layers")
    
    # Loss and optimizer
    criterion = InfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=args.min_lr
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    if args.mixed_precision:
        logger.info("Using Mixed Precision (FP16) training")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=runs_dir / 'infonce')
    
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
                resume_path, model, optimizer, scheduler, device
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
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, global_step, scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_epochs=args.epochs
        )
        
        # Validate (only every val_freq epochs or on last epoch)
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, args.mixed_precision)
            
            # Learning rate schedule (based on validation loss)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch results
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # TensorBoard
            writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
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
            
            checkpoint_path = checkpoint_dir / f'infonce_epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, checkpoint_path, is_best)
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
                break
        else:
            # No validation this epoch, just log training results
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  (Validation skipped - next at epoch {epoch + args.val_freq})")
            
            # TensorBoard (training only)
            writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            writer.add_scalar('Epoch/LR', current_lr, epoch)
    
    # Training complete
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved to: {checkpoint_dir / 'infonce_best.pth'}")
    
    # Close writer
    writer.close()
    
    # Slack notification
    message = f"✅ InfoNCE Training Complete!\nEpochs: {args.epochs}\nBest Val Loss: {best_val_loss:.4f}"
    send_slack_notification(message)
    logger.info("\nSlack notification sent!")


if __name__ == '__main__':
    main()

