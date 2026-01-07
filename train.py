"""
Training Script for Neuro-Symbolic LPR System.

This script implements the staged training schedule:
    Step 0: Synthetic pre-training (PARSeq recognizer)
    Step 1: Geometry warm-up (STN with corner loss)
    Step 2: Restoration & Layout (SwinIR + Classifier)
    Step 3: End-to-end fine-tuning (all modules)

Usage:
    python train.py --config config.yaml --stage all
    python train.py --stage 0  # Pre-train recognizer only
    python train.py --stage 3 --resume checkpoints/step2.pth
"""

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from config import Config, get_default_config
from models import NeuroSymbolicLPR, PatchDiscriminator
from losses import CompositeLoss, CornerLoss, GANLoss
from losses.composite_loss import StagedTrainingManager
from data import RodoSolDataset, SyntheticLPRDataset, LPRAugmentation


# =============================================================================
# Utility Classes
# =============================================================================

class EMA:
    """Exponential Moving Average of model weights for more stable final model."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_param, model_param in zip(
            self.model.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.should_stop


def log_model_info(model: nn.Module, logger: logging.Logger):
    """Log model architecture and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Per-module breakdown
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {params:,} params")


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    stage: str,
    config: Config
) -> optim.Optimizer:
    """
    Create optimizer for specific training stage.
    
    Args:
        model: The model to optimize.
        stage: Training stage name.
        config: Training configuration.
        
    Returns:
        Configured optimizer.
    """
    param_groups = model.get_trainable_params(stage)
    
    # Adjust learning rates based on lr_scale if present
    base_lr = {
        'stn': config.training.lr_stn,
        'restoration': config.training.lr_restoration,
        'full': config.training.lr_finetune
    }.get(stage, 1e-4)
    
    for group in param_groups:
        lr_scale = group.pop('lr_scale', 1.0)
        group['lr'] = base_lr * lr_scale
    
    if config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=config.training.betas,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=config.training.betas
        )
    else:
        optimizer = optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    config: Config
) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with optional warmup."""
    warmup_epochs = getattr(config.training, 'warmup_epochs', 0)
    
    if config.training.scheduler == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=config.training.min_lr
        )
        
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler
            
    elif config.training.scheduler == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, (num_epochs - warmup_epochs) // 3),
            gamma=0.1
        )
        
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    return scheduler


def train_epoch(
    model: nn.Module,
    discriminator: Optional[nn.Module],
    dataloader: DataLoader,
    criterion: CompositeLoss,
    optimizer_g: optim.Optimizer,
    optimizer_d: Optional[optim.Optimizer],
    device: torch.device,
    epoch: int,
    stage: str,
    writer: Optional[SummaryWriter] = None,
    logger: Optional[logging.Logger] = None,
    scaler: Optional[GradScaler] = None,
    ema: Optional[EMA] = None
) -> Dict[str, float]:
    """
    Train for one epoch with optional mixed precision.
    
    Args:
        model: Generator model.
        discriminator: Discriminator model (optional).
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer_g: Generator optimizer.
        optimizer_d: Discriminator optimizer (optional).
        device: Training device.
        epoch: Current epoch number.
        stage: Training stage.
        writer: TensorBoard writer.
        logger: Logger instance.
        scaler: GradScaler for mixed precision training.
        ema: EMA model for weight averaging.
        
    Returns:
        Dictionary of average losses.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    epoch_losses = {}
    num_batches = len(dataloader)
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        lr_frames = batch['lr_frames'].to(device)
        hr_image = batch['hr_image'].to(device)
        text_indices = batch['text_indices'].to(device)
        layout = batch['layout'].to(device)
        corners = batch.get('corners', None)
        if corners is not None:
            corners = corners.to(device)
        
        # Prepare targets
        targets = {
            'hr_image': hr_image,
            'text_indices': text_indices,
            'layout': layout
        }
        if corners is not None:
            targets['corners'] = corners
        
        # Forward pass with optional autocast for mixed precision
        with autocast('cuda', enabled=use_amp):
            outputs = model(lr_frames, return_intermediates=True)
            
            # Compute generator loss
            if stage == 'stn':
                loss_g, loss_dict = criterion.get_stage_loss('stn', outputs, targets)
            elif stage == 'restoration':
                loss_g, loss_dict = criterion.get_stage_loss('restoration', outputs, targets, discriminator)
            else:
                loss_g, loss_dict = criterion(outputs, targets, discriminator)
        
        # Update generator with gradient scaling for mixed precision
        optimizer_g.zero_grad()
        
        if use_amp:
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer_g)
            scaler.update()
        else:
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()
        
        # Update EMA model
        if ema is not None:
            ema.update(model)
        
        # Update discriminator if available and in appropriate stage
        if discriminator is not None and optimizer_d is not None and stage in ['restoration', 'full']:
            # Get real and fake images
            real_hr = targets['hr_image']
            fake_hr = outputs['hr_image'].detach()
            
            with autocast('cuda', enabled=use_amp):
                # Discriminator forward
                pred_real = discriminator(real_hr)
                pred_fake = discriminator(fake_hr)
                
                # Discriminator loss
                gan_loss = GANLoss('vanilla')
                loss_d = (
                    gan_loss(pred_real, target_is_real=True) +
                    gan_loss(pred_fake, target_is_real=False)
                ) / 2
            
            optimizer_d.zero_grad()
            if use_amp:
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
            else:
                loss_d.backward()
                optimizer_d.step()
            
            loss_dict['loss_d'] = loss_d.item()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            epoch_losses[key] = epoch_losses.get(key, 0) + value
        
        # Update progress bar
        pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value, global_step)
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    device: torch.device,
    epoch: int,
    stage: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Validate the model with comprehensive metrics.
    
    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.
        epoch: Current epoch.
        stage: Training stage.
        
    Returns:
        Tuple of (loss_dict, metrics_dict).
        metrics_dict includes plate_accuracy, char_accuracy, layout_accuracy.
    """
    model.eval()
    
    total_losses = {}
    correct_plates = 0
    correct_chars = 0
    correct_layout = 0
    total_plates = 0
    total_chars = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            lr_frames = batch['lr_frames'].to(device)
            hr_image = batch['hr_image'].to(device)
            text_indices = batch['text_indices'].to(device)
            layout = batch['layout'].to(device)
            corners = batch.get('corners', None)
            if corners is not None:
                corners = corners.to(device)
            
            targets = {
                'hr_image': hr_image,
                'text_indices': text_indices,
                'layout': layout
            }
            if corners is not None:
                targets['corners'] = corners
            
            outputs = model(lr_frames, return_intermediates=True)
            
            # Compute loss
            _, loss_dict = criterion(outputs, targets)
            
            for key, value in loss_dict.items():
                total_losses[key] = total_losses.get(key, 0) + value
            
            # Compute plate-level accuracy (all characters correct)
            predictions = outputs['masked_logits'].argmax(dim=-1)
            correct_plates += (predictions == text_indices).all(dim=-1).sum().item()
            total_plates += text_indices.size(0)
            
            # Compute character-level accuracy
            correct_chars += (predictions == text_indices).sum().item()
            total_chars += text_indices.numel()
            
            # Compute layout classification accuracy
            if 'layout_logits' in outputs:
                layout_preds = (torch.sigmoid(outputs['layout_logits']) > 0.5).long().squeeze(-1)
                correct_layout += (layout_preds == layout).sum().item()
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    # Compute metrics
    metrics = {
        'plate_accuracy': correct_plates / total_plates if total_plates > 0 else 0.0,
        'char_accuracy': correct_chars / total_chars if total_chars > 0 else 0.0,
        'layout_accuracy': correct_layout / total_plates if total_plates > 0 else 0.0,
    }
    
    return total_losses, metrics


def save_checkpoint(
    model: nn.Module,
    discriminator: Optional[nn.Module],
    optimizer_g: optim.Optimizer,
    optimizer_d: Optional[optim.Optimizer],
    epoch: int,
    stage: str,
    save_path: str,
    best_acc: float = 0.0
):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'epoch': epoch,
        'stage': stage,
        'best_acc': best_acc
    }
    
    if discriminator is not None:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    
    if optimizer_d is not None:
        checkpoint['optimizer_d_state_dict'] = optimizer_d.state_dict()
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    discriminator: Optional[nn.Module] = None,
    optimizer_g: Optional[optim.Optimizer] = None,
    optimizer_d: Optional[optim.Optimizer] = None
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if optimizer_g is not None and 'optimizer_g_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    return checkpoint


def train_stage(
    model: nn.Module,
    discriminator: Optional[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    stage: str,
    num_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    writer: SummaryWriter,
    logger: logging.Logger,
    start_epoch: int = 0,
    use_amp: bool = True,
    use_ema: bool = True,
    early_stopping_patience: int = 0
):
    """
    Train for a specific stage with full feature support.
    
    Args:
        model: Generator model.
        discriminator: Discriminator model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        stage: Training stage name.
        num_epochs: Number of epochs to train.
        device: Training device.
        checkpoint_dir: Directory for checkpoints.
        writer: TensorBoard writer.
        logger: Logger instance.
        start_epoch: Starting epoch (for resume).
        use_amp: Whether to use mixed precision training.
        use_ema: Whether to use EMA for model weights.
        early_stopping_patience: Patience for early stopping (0 to disable).
    """
    logger.info(f"Starting training stage: {stage}")
    logger.info(f"  Mixed precision: {use_amp}")
    logger.info(f"  EMA: {use_ema}")
    logger.info(f"  Early stopping patience: {early_stopping_patience}")
    
    # Create criterion
    criterion = CompositeLoss(
        weight_pixel=config.training.weight_pixel,
        weight_gan=config.training.weight_gan,
        weight_ocr=config.training.weight_ocr,
        weight_geometry=config.training.weight_geometry
    )
    
    # Create optimizers
    optimizer_g = create_optimizer(model, stage, config)
    optimizer_d = None
    
    if discriminator is not None and stage in ['restoration', 'full']:
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=config.training.lr_restoration,
            betas=config.training.betas
        )
    
    # Create scheduler
    scheduler = create_scheduler(optimizer_g, num_epochs, config)
    
    # Create GradScaler for mixed precision
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    
    # Create EMA model
    ema = EMA(model) if use_ema else None
    
    # Create early stopping
    early_stopper = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    
    best_acc = 0.0
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_losses = train_epoch(
            model, discriminator, train_loader, criterion,
            optimizer_g, optimizer_d, device, epoch, stage, writer, logger,
            scaler=scaler, ema=ema
        )
        
        # Log training losses
        logger.info(f"Epoch {epoch} - Train losses: {train_losses}")
        
        # Validate
        if (epoch + 1) % config.training.eval_every == 0:
            val_losses, val_metrics = validate(
                model, val_loader, criterion, device, epoch, stage
            )
            
            plate_acc = val_metrics['plate_accuracy']
            char_acc = val_metrics['char_accuracy']
            layout_acc = val_metrics['layout_accuracy']
            
            logger.info(
                f"Epoch {epoch} - Val losses: {val_losses}, "
                f"Plate Acc: {plate_acc:.4f}, Char Acc: {char_acc:.4f}, Layout Acc: {layout_acc:.4f}"
            )
            
            # Log to tensorboard
            writer.add_scalar('val/plate_accuracy', plate_acc, epoch)
            writer.add_scalar('val/char_accuracy', char_acc, epoch)
            writer.add_scalar('val/layout_accuracy', layout_acc, epoch)
            for key, value in val_losses.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save best model
            if plate_acc > best_acc:
                best_acc = plate_acc
                save_checkpoint(
                    model, discriminator, optimizer_g, optimizer_d,
                    epoch, stage,
                    os.path.join(checkpoint_dir, f'{stage}_best.pth'),
                    best_acc
                )
                
                # Also save EMA model if available
                if ema is not None:
                    torch.save(
                        {'model_state_dict': ema.state_dict()},
                        os.path.join(checkpoint_dir, f'{stage}_best_ema.pth')
                    )
            
            # Check early stopping
            if early_stopper is not None:
                if early_stopper(plate_acc):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save regular checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            save_checkpoint(
                model, discriminator, optimizer_g, optimizer_d,
                epoch, stage,
                os.path.join(checkpoint_dir, f'{stage}_epoch{epoch}.pth'),
                best_acc
            )
        
        # Update scheduler
        scheduler.step()
    
    # Save final checkpoint
    save_checkpoint(
        model, discriminator, optimizer_g, optimizer_d,
        num_epochs - 1, stage,
        os.path.join(checkpoint_dir, f'{stage}_final.pth'),
        best_acc
    )
    
    if ema is not None:
        torch.save(
            {'model_state_dict': ema.state_dict()},
            os.path.join(checkpoint_dir, f'{stage}_final_ema.pth')
        )
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Train Neuro-Symbolic LPR System')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--stage', type=str, default='all', 
                        choices=['0', '1', '2', '3', 'all'],
                        help='Training stage (0-3 or all)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--no-ema', action='store_true', help='Disable EMA model averaging')
    parser.add_argument('--early-stopping', type=int, default=0, 
                        help='Early stopping patience (0 to disable)')
    args = parser.parse_args()
    
    # Load config
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # TODO: Parse config_dict into Config object
        config = get_default_config()
    else:
        config = get_default_config()
    
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = NeuroSymbolicLPR(
        num_frames=config.model.num_frames,
        lr_size=(config.data.lr_height, config.data.lr_width),
        hr_size=(config.data.hr_height, config.data.hr_width),
    ).to(device)
    
    # Create discriminator
    discriminator = PatchDiscriminator(
        in_channels=3,
        base_channels=64,
        use_spectral_norm=True
    ).to(device)
    
    # Create datasets with style-aware augmentation
    if config.data.use_augmentation:
        augmentation = LPRAugmentation(
            style_aware=config.data.style_aware_augmentation,
            mercosur_brightness_range=config.data.mercosur_brightness_range,
            mercosur_contrast_range=config.data.mercosur_contrast_range,
            brazilian_brightness_range=config.data.brazilian_brightness_range,
            brazilian_contrast_range=config.data.brazilian_contrast_range
        )
    else:
        augmentation = None
    
    # Check if data directory exists, use synthetic data if not
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    if os.path.exists(train_dir):
        train_dataset = RodoSolDataset(
            train_dir,
            num_frames=config.model.num_frames,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            transform=augmentation
        )
        val_dataset = RodoSolDataset(
            val_dir,
            num_frames=config.model.num_frames,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width)
        )
    else:
        logger.warning("Data directory not found, using synthetic data")
        train_dataset = SyntheticLPRDataset(
            num_samples=10000,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            num_frames=config.model.num_frames
        )
        val_dataset = SyntheticLPRDataset(
            num_samples=1000,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            num_frames=config.model.num_frames
        )
    
    # Note: Data loaders will be created per stage with appropriate batch sizes
    # We'll create them in the training loop
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Log model info
    log_model_info(model, logger)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(model, args.resume, discriminator)
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resumed from {args.resume}, starting at epoch {start_epoch}")
    
    # Determine stages to train
    if args.stage == 'all':
        stages = [
            ('stn', config.training.epochs_stn),
            ('restoration', config.training.epochs_restoration),
            ('full', config.training.epochs_finetune)
        ]
    else:
        stage_map = {
            '0': ('pretrain', config.training.epochs_pretrain),
            '1': ('stn', config.training.epochs_stn),
            '2': ('restoration', config.training.epochs_restoration),
            '3': ('full', config.training.epochs_finetune)
        }
        stages = [stage_map[args.stage]]
    
    # Train each stage
    for stage_name, num_epochs in stages:
        # Prepare model for stage
        if stage_name == 'stn':
            model.freeze_recognizer()
        elif stage_name == 'restoration':
            model.freeze_stn()
            model.freeze_recognizer()
        else:  # full
            model.unfreeze_stn()
            model.unfreeze_recognizer()
        
        # Get appropriate batch size for this stage
        batch_size_map = {
            'pretrain': config.training.batch_size_pretrain,
            'stn': config.training.batch_size_stn,
            'restoration': config.training.batch_size_restoration,
            'full': config.training.batch_size_finetune
        }
        batch_size = batch_size_map.get(stage_name, config.training.batch_size_finetune)
        
        # Create data loaders with stage-specific batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Stage {stage_name}: Using batch size {batch_size}")
        
        best_acc = train_stage(
            model, discriminator, train_loader, val_loader,
            config, stage_name, num_epochs, device,
            checkpoint_dir, writer, logger, start_epoch,
            use_amp=not args.no_amp,
            use_ema=not args.no_ema,
            early_stopping_patience=args.early_stopping
        )
        
        logger.info(f"Stage {stage_name} completed with best accuracy: {best_acc:.4f}")
        start_epoch = 0  # Reset for next stage
    
    writer.close()
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
