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
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from config import Config, get_default_config
from models import NeuroSymbolicLPR, PatchDiscriminator
from losses import CompositeLoss, CornerLoss, GANLoss
from losses.composite_loss import StagedTrainingManager
from data import RodoSolDataset, SyntheticLPRDataset, LPRAugmentation, lpr_collate_fn


# =============================================================================
# Utility Classes
# =============================================================================

class EMA:
    """Exponential Moving Average of model weights for more stable final model."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay
        
        # Ensure EMA model is on the same device as the original model
        device = next(model.parameters()).device
        self.model = self.model.to(device)
        
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


def check_for_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values."""
    if tensor is None:
        return False
    if isinstance(tensor, (int, float)):
        return math.isnan(tensor) or math.isinf(tensor)
    return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()


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
    ema: Optional[EMA] = None,
    grad_clip_norm: float = 1.0
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
        grad_clip_norm: Maximum norm for gradient clipping.
        
    Returns:
        Dictionary of average losses.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    epoch_losses = {}
    num_batches = len(dataloader)
    use_amp = scaler is not None
    nan_batch_count = 0
    max_nan_batches = 10  # Stop epoch if too many NaN batches
    scaler_reset_threshold = 5  # Reset scaler if too many NaN batches in a row
    
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
        # CRITICAL: Disable AMP for STN stage to prevent FP16 precision issues
        # STN grid generation and affine transformations are highly sensitive to
        # precision loss in FP16, causing NaN/Inf gradients and training collapse
        use_amp_for_batch = use_amp and (stage != 'stn')
        
        with autocast('cuda', enabled=use_amp_for_batch):
            outputs = model(lr_frames, return_intermediates=True)
            
            # Compute generator loss
            if stage == 'stn':
                loss_g, loss_dict = criterion.get_stage_loss('stn', outputs, targets)
            elif stage == 'restoration':
                loss_g, loss_dict = criterion.get_stage_loss('restoration', outputs, targets, discriminator)
            else:
                loss_g, loss_dict = criterion(outputs, targets, discriminator)
        
        # Check for NaN loss and skip batch if detected
        if check_for_nan(loss_g, "loss_g"):
            nan_batch_count += 1
            if logger is not None:
                logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN loss detected, skipping batch ({nan_batch_count}/{max_nan_batches})")
            
            # Clear gradients and skip this batch
            optimizer_g.zero_grad()
            if optimizer_d is not None:
                optimizer_d.zero_grad()
            
            if nan_batch_count >= max_nan_batches:
                if logger is not None:
                    logger.error(f"Epoch {epoch}: Too many NaN batches ({nan_batch_count}), stopping epoch early")
                break
            continue
        
        # Update generator with gradient scaling for mixed precision
        optimizer_g.zero_grad()
        
        if use_amp:
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            
            # Check for NaN/Inf gradients after unscaling
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            # Skip optimizer step if gradients are invalid
            if check_for_nan(grad_norm, "grad_norm"):
                if logger is not None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN/Inf gradients detected, skipping optimizer step")
                scaler.update()  # Still update scaler to adjust scale factor
                nan_batch_count += 1
                
                # Reset scaler if we've hit threshold - the scale may have shrunk too much
                if nan_batch_count == scaler_reset_threshold and use_amp:
                    if logger is not None:
                        logger.warning(f"Resetting GradScaler due to {scaler_reset_threshold} consecutive NaN batches")
                    # Create new scaler with fresh state
                    scaler = GradScaler('cuda')
                
                if nan_batch_count >= max_nan_batches:
                    if logger is not None:
                        logger.error(f"Epoch {epoch}: Too many NaN batches ({nan_batch_count}), stopping epoch early")
                    break
                continue
            
            scaler.step(optimizer_g)
            scaler.update()
        else:
            loss_g.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            # Check for NaN/Inf gradients
            if check_for_nan(grad_norm, "grad_norm"):
                if logger is not None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN/Inf gradients detected, skipping optimizer step")
                nan_batch_count += 1
                # Zero out any NaN gradients to prevent accumulation
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                
                if nan_batch_count >= max_nan_batches:
                    if logger is not None:
                        logger.error(f"Epoch {epoch}: Too many NaN batches ({nan_batch_count}), stopping epoch early")
                    break
                continue
            
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
            
            # Check for NaN discriminator loss
            if not check_for_nan(loss_d, "loss_d"):
                optimizer_d.zero_grad()
                if use_amp:
                    scaler.scale(loss_d).backward()
                    scaler.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer_d)
                    scaler.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                    optimizer_d.step()
                
                loss_dict['loss_d'] = loss_d.item()
            else:
                if logger is not None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN discriminator loss, skipping D update")
        
        # Accumulate losses (only for valid batches)
        for key, value in loss_dict.items():
            if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                epoch_losses[key] = epoch_losses.get(key, 0) + value
        
        # Update progress bar
        pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_dict.items()})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * num_batches + batch_idx
            for key, value in loss_dict.items():
                if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    writer.add_scalar(f'train/{key}', value, global_step)
    
    # Average losses (accounting for skipped batches)
    valid_batches = num_batches - nan_batch_count
    if valid_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= valid_batches
    
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
        
    Note:
        Character accuracy is computed EXCLUDING padding tokens (PAD_IDX=0),
        BOS tokens (BOS_IDX=1), and EOS tokens (EOS_IDX=2) to give accurate
        performance metrics on actual plate characters.
        
        Loss computation is stage-aware: during STN stage, only geometry-related
        losses are computed to avoid NaN from OCR loss (pretrained PARSeq may
        produce invalid logits for unmapped characters).
    """
    from config import PAD_IDX, BOS_IDX, EOS_IDX
    
    model.eval()
    
    total_losses = {}
    correct_plates = 0
    correct_chars = 0
    correct_layout = 0
    total_plates = 0
    total_chars = 0
    valid_layout_samples = 0  # Count samples with valid layout labels
    
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
            
            # Compute loss - use stage-specific loss to avoid NaN from OCR
            # during early training stages (STN stage especially)
            if stage == 'stn':
                _, loss_dict = criterion.get_stage_loss('stn', outputs, targets)
            elif stage == 'restoration':
                _, loss_dict = criterion.get_stage_loss('restoration', outputs, targets)
            else:
                _, loss_dict = criterion(outputs, targets)
            
            for key, value in loss_dict.items():
                # Skip NaN values when accumulating
                if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    total_losses[key] = total_losses.get(key, 0) + value
            
            # Compute predictions
            predictions = outputs['masked_logits'].argmax(dim=-1)  # (B, PLATE_LENGTH)
            
            # Extract target characters (exclude BOS at position 0 and EOS at end)
            # text_indices shape: (B, PLATE_LENGTH + 2) with [BOS, char1, ..., char7, EOS]
            # We need positions 1 to PLATE_LENGTH+1 (i.e., indices 1:8 for 7-char plates)
            plate_length = predictions.size(1)
            target_chars = text_indices[:, 1:plate_length + 1]  # (B, PLATE_LENGTH)
            
            # Create mask for non-special tokens (actual characters only)
            # Exclude PAD (0), BOS (1), EOS (2) from accuracy calculation
            char_mask = (target_chars != PAD_IDX) & (target_chars != BOS_IDX) & (target_chars != EOS_IDX)
            
            # Compute plate-level accuracy (all NON-PADDING characters correct)
            # For each sample, check if all masked positions match
            batch_size = target_chars.size(0)
            for i in range(batch_size):
                sample_mask = char_mask[i]
                if sample_mask.any():  # Only count if there are actual characters
                    sample_correct = (predictions[i][sample_mask] == target_chars[i][sample_mask]).all().item()
                    correct_plates += int(sample_correct)
                    total_plates += 1
            
            # Compute character-level accuracy (excluding special tokens)
            if char_mask.any():
                correct_chars += (predictions[char_mask] == target_chars[char_mask]).sum().item()
                total_chars += char_mask.sum().item()
            
            # Compute layout classification accuracy (only for valid layout labels >= 0)
            if 'layout_logits' in outputs:
                valid_layout_mask = layout >= 0  # Filter out invalid layout samples (-1)
                if valid_layout_mask.any():
                    layout_preds = (torch.sigmoid(outputs['layout_logits']) > 0.5).long().squeeze(-1)
                    correct_layout += (layout_preds[valid_layout_mask] == layout[valid_layout_mask]).sum().item()
                    valid_layout_samples += valid_layout_mask.sum().item()
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses:
        total_losses[key] /= num_batches
    
    # Compute metrics
    metrics = {
        'plate_accuracy': correct_plates / total_plates if total_plates > 0 else 0.0,
        'char_accuracy': correct_chars / total_chars if total_chars > 0 else 0.0,
        'layout_accuracy': correct_layout / valid_layout_samples if valid_layout_samples > 0 else 0.0,
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


def check_model_for_nan(model: nn.Module, logger: logging.Logger) -> bool:
    """
    Check if model has any NaN or Inf parameters.
    
    Args:
        model: The model to check.
        logger: Logger for warnings.
        
    Returns:
        True if NaN/Inf found, False otherwise.
    """
    has_nan = False
    for name, param in model.named_parameters():
        if param is not None and param.data is not None:
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                logger.warning(f"NaN/Inf found in parameter: {name}")
                has_nan = True
    return has_nan


def reset_nan_parameters(model: nn.Module, logger: logging.Logger):
    """
    Reset any NaN/Inf parameters to reasonable values.
    
    Args:
        model: The model to fix.
        logger: Logger for info messages.
    """
    for name, param in model.named_parameters():
        if param is not None and param.data is not None:
            nan_mask = torch.isnan(param.data) | torch.isinf(param.data)
            if nan_mask.any():
                # Reset NaN values to small random values
                num_nan = nan_mask.sum().item()
                logger.warning(f"Resetting {num_nan} NaN/Inf values in {name}")
                param.data[nan_mask] = torch.randn(num_nan, device=param.device, dtype=param.dtype) * 0.01


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
    # Check for NaN in model parameters before starting
    if check_model_for_nan(model, logger):
        logger.warning(f"Model has NaN parameters at start of stage {stage}, attempting to reset...")
        reset_nan_parameters(model, logger)
    
    # Get stage-specific gradient clipping norm
    grad_clip_norm = config.training.grad_clip_norm
    if stage == 'stn' and hasattr(config.training, 'grad_clip_norm_stn'):
        grad_clip_norm = config.training.grad_clip_norm_stn
    
    logger.info(f"Starting training stage: {stage}")
    logger.info(f"  Mixed precision: {use_amp}")
    logger.info(f"  EMA: {use_ema}")
    logger.info(f"  Early stopping patience: {early_stopping_patience}")
    logger.info(f"  Gradient clip norm: {grad_clip_norm}")
    
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
    consecutive_nan_epochs = 0
    max_consecutive_nan_epochs = 3  # Stop stage if too many consecutive NaN epochs
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_losses = train_epoch(
            model, discriminator, train_loader, criterion,
            optimizer_g, optimizer_d, device, epoch, stage, writer, logger,
            scaler=scaler, ema=ema, grad_clip_norm=grad_clip_norm
        )
        
        # Log training losses
        logger.info(f"Epoch {epoch} - Train losses: {train_losses}")
        
        # Check if this was a NaN epoch (empty losses means all batches had NaN)
        if not train_losses:
            consecutive_nan_epochs += 1
            logger.warning(f"Epoch {epoch} had no valid batches (consecutive NaN epochs: {consecutive_nan_epochs})")
            
            if consecutive_nan_epochs >= max_consecutive_nan_epochs:
                logger.error(f"Stage {stage}: {max_consecutive_nan_epochs} consecutive NaN epochs, stopping stage early")
                # Check and reset any NaN parameters before next stage
                if check_model_for_nan(model, logger):
                    logger.warning("Model has NaN parameters, attempting to load best checkpoint...")
                    best_checkpoint_path = os.path.join(checkpoint_dir, f'{stage}_best.pth')
                    if os.path.exists(best_checkpoint_path):
                        load_checkpoint(model, best_checkpoint_path, discriminator)
                        logger.info(f"Loaded best checkpoint from {best_checkpoint_path}")
                    else:
                        reset_nan_parameters(model, logger)
                break
        else:
            consecutive_nan_epochs = 0  # Reset counter on successful epoch
        
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
    
    # After stage completion, load the best checkpoint to ensure clean state for next stage
    best_checkpoint_path = os.path.join(checkpoint_dir, f'{stage}_best.pth')
    if os.path.exists(best_checkpoint_path):
        logger.info(f"Loading best checkpoint from stage {stage} for next stage...")
        load_checkpoint(model, best_checkpoint_path, discriminator)
        # Verify model is clean
        if check_model_for_nan(model, logger):
            logger.warning("Best checkpoint has NaN values, resetting...")
            reset_nan_parameters(model, logger)
    else:
        # Check current model for NaN
        if check_model_for_nan(model, logger):
            logger.warning("Final model has NaN values, resetting...")
            reset_nan_parameters(model, logger)
    
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
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio when using single data folder (default: 0.1)')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio when using single data folder (default: 0.1)')
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
    # Support both pre-split folders (data/train, data/val) and single folder (data/)
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Split ratios for validation and test (when using single data folder)
    # Default: 80% train, 10% val, 10% test
    val_split_ratio = args.val_split
    test_split_ratio = args.test_split
    train_split_ratio = 1.0 - val_split_ratio - test_split_ratio
    
    # Determine the data source directory for dynamic splitting
    # Priority: 1) Pre-split folders, 2) Single train folder, 3) Root data folder, 4) Synthetic
    data_source_dir = None
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Pre-split folder structure exists - use it directly
        logger.info("Using pre-split data folders (train/ and val/)")
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
    elif os.path.exists(train_dir) and not os.path.exists(val_dir):
        # Only train folder exists - use it for dynamic splitting
        data_source_dir = train_dir
        logger.info(f"Found train folder only, will split dynamically from: {train_dir}")
    elif os.path.exists(args.data_dir) and (
        list(Path(args.data_dir).glob('Scenario-*')) or 
        (Path(args.data_dir) / 'annotations.json').exists() or
        list(Path(args.data_dir).glob('*.jpg')) or
        list(Path(args.data_dir).glob('*.png'))
    ):
        # Root data folder has data directly
        data_source_dir = args.data_dir
        logger.info(f"Found data in root folder, will split dynamically from: {args.data_dir}")
    
    # Handle dynamic splitting if data_source_dir was set
    if data_source_dir is not None:
        logger.info(f"Dynamically splitting data: {train_split_ratio:.0%} train / {val_split_ratio:.0%} val / {test_split_ratio:.0%} test")
        
        # Load full dataset
        full_dataset = RodoSolDataset(
            data_source_dir,
            num_frames=config.model.num_frames,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            transform=None  # We'll handle transforms separately
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(total_size * val_split_ratio)
        test_size = int(total_size * test_split_ratio)
        train_size = total_size - val_size - test_size
        
        # Random split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.training.seed)
        train_indices, val_indices, test_indices = random_split(
            range(total_size), 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Save test indices for later evaluation
        test_indices_path = os.path.join(output_dir, 'test_indices.pt')
        torch.save(test_indices.indices, test_indices_path)
        logger.info(f"Saved {len(test_indices)} test indices to {test_indices_path}")
        
        # For proper augmentation, we create two dataset instances
        train_dataset = RodoSolDataset(
            data_source_dir,
            num_frames=config.model.num_frames,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            transform=augmentation
        )
        val_dataset = RodoSolDataset(
            data_source_dir,
            num_frames=config.model.num_frames,
            lr_size=(config.data.lr_height, config.data.lr_width),
            hr_size=(config.data.hr_height, config.data.hr_width),
            transform=None  # No augmentation for validation
        )
        
        # Apply the split indices (train and val only - test is saved for later)
        train_dataset.samples = [train_dataset.samples[i] for i in train_indices.indices]
        val_dataset.samples = [val_dataset.samples[i] for i in val_indices.indices]
        
        logger.info(f"Split complete: {len(train_dataset)} train, {len(val_dataset)} val samples")
    elif not os.path.exists(train_dir) or not os.path.exists(val_dir):
        # Neither pre-split folders nor data found - use synthetic data
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
        
        # Create data loaders with stage-specific batch size and custom collate
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lpr_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            collate_fn=lpr_collate_fn
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
