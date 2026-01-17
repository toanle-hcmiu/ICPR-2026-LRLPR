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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from config import Config, get_default_config
from models import NeuroSymbolicLPR, PatchDiscriminator
from losses import CompositeLoss, CornerLoss, GANLoss, ConfusionMatrixTracker
from losses.composite_loss import StagedTrainingManager
from losses.ocr_discriminator import create_ocr_discriminator, OCRDiscriminatorLoss
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
    
    # Filter out None param groups (e.g., from optional modules)
    param_groups = [g for g in param_groups if g is not None]
    
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
    scaler_g: Optional[GradScaler] = None,
    scaler_d: Optional[GradScaler] = None,
    ema: Optional[EMA] = None,
    grad_clip_norm: float = 1.0,
    total_epochs: int = 20,
    ocr_discriminator_loss: Optional['OCRDiscriminatorLoss'] = None
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
        scaler_g: GradScaler for generator mixed precision training.
        scaler_d: GradScaler for discriminator mixed precision training.
        ema: EMA model for weight averaging.
        grad_clip_norm: Maximum norm for gradient clipping.
        total_epochs: Total epochs for this stage (for noise decay).
        
    Returns:
        Dictionary of average losses.
    """
    model.train()
    if discriminator is not None:
        discriminator.train()
    
    epoch_losses = {}
    num_batches = len(dataloader)
    use_amp = scaler_g is not None
    nan_batch_count = 0
    max_nan_batches = 10  # Stop epoch if too many NaN batches
    scaler_reset_threshold = 5  # Reset scaler if too many NaN batches in a row
    
    # GAN training stabilization parameters
    label_smoothing_real = 0.9  # Smooth real labels to prevent overconfidence
    label_smoothing_fake = 0.0  # Keep fake labels at 0
    d_update_freq = 1  # Update D every N batches (will increase if D too strong)
    r1_gamma = 1.0  # R1 gradient penalty weight (reduced from 5.0 for LSGAN stability)
    # Instance noise: start at 0.1, decay to 0 over training
    instance_noise_std = max(0.0, 0.1 * (1.0 - epoch / max(total_epochs, 1)))
    
    # Discriminator stability: NaN tracking and warm-up
    d_nan_count = 0  # Track consecutive NaN batches for discriminator
    d_nan_threshold = 50  # Reset D if this many consecutive NaN batches
    d_warmup_epochs = 10  # Increased from 5 - Generator needs more time to learn basic colors via L1 before GAN kicks in
    d_lr_warmup_epochs = 5  # Ramp up D learning rate gradually after warm-up
    d_reset_count = 0  # Track how many times D has been reset this epoch
    max_d_resets_per_epoch = 3  # Fallback to pixel-only after this many resets
    d_fallback_mode = False  # When True, skip GAN loss for rest of epoch
    
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
        scaler = scaler_g  # Use generator scaler for main forward pass
        
        with autocast('cuda', enabled=use_amp_for_batch):
            outputs = model(lr_frames, return_intermediates=True)
            
            # Validate generator output isn't producing extreme values
            # This catches cases where the generator is collapsing before it causes NaN
            if 'hr_image' in outputs and stage in ['restoration', 'full']:
                hr_output = outputs['hr_image']
                max_val = hr_output.abs().max().item()
                if max_val > 2.0:  # After tanh, should be in [-1, 1], allow some margin
                    if logger is not None and batch_idx % 100 == 0:
                        logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Generator output extreme (max={max_val:.2f})")
                    # Safety clamp to prevent downstream NaN
                    outputs['hr_image'] = torch.clamp(hr_output, -1.0, 1.0)
            
            # FIX: Completely disable GAN loss during warm-up period
            # During warm-up, G learns only from Pixel Loss (L1) to produce basic colors/shapes
            # before the adversarial training begins. This prevents D from suppressing G too early.
            current_discriminator = discriminator
            if discriminator is not None and epoch < d_warmup_epochs:
                current_discriminator = None  # Pass None to skip GAN loss computation
            
            # Compute generator loss
            if stage == 'stn':
                loss_g, loss_dict = criterion.get_stage_loss('stn', outputs, targets)
            elif stage == 'restoration':
                loss_g, loss_dict = criterion.get_stage_loss('restoration', outputs, targets, current_discriminator)
            else:
                loss_g, loss_dict = criterion(outputs, targets, current_discriminator)
            
            # Add OCR-based guidance loss if configured
            # Uses recognition confidence to guide super-resolution
            if ocr_discriminator_loss is not None and stage in ['restoration', 'full']:
                # Get text targets (remove BOS/EOS tokens)
                text_targets = targets['text_indices'][:, 1:-1]  # (B, PLATE_LENGTH)
                
                # Compute OCR guidance loss on generated HR images
                ocr_loss, ocr_metrics = ocr_discriminator_loss.generator_loss(
                    fake_images=outputs['hr_image'],
                    targets=text_targets,
                    real_images=targets['hr_image']
                )
                
                # Only apply OCR guidance if the OCR model has meaningful confidence
                # If real image confidence is too low, the OCR model is untrained/unreliable
                real_conf = ocr_metrics.get('real_confidence', 0.0)
                min_ocr_confidence = 0.10  # 10% threshold
                
                if real_conf >= min_ocr_confidence:
                    # Add to total loss
                    if not check_for_nan(ocr_loss, "ocr_guidance"):
                        loss_g = loss_g + ocr_loss
                        loss_dict['ocr_guidance'] = ocr_metrics.get('total', ocr_loss.item())
                else:
                    # Skip OCR guidance - model not ready
                    loss_dict['ocr_guidance'] = 0.0
                
                # Always log confidence for monitoring
                loss_dict['ocr_fake_conf'] = ocr_metrics.get('fake_confidence', 0.0)
                if 'real_confidence' in ocr_metrics:
                    loss_dict['ocr_real_conf'] = ocr_metrics['real_confidence']
        
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
            # Skip D update if in fallback mode (too many resets this epoch)
            if d_fallback_mode:
                # Still log GAN loss as 0 for monitoring
                loss_dict['loss_d'] = 0.0
                loss_dict['d_accuracy'] = 0.0
                # Skip entire D update block
            else:
                # Skip D update based on frequency control and warm-up
                # Delay D training for first few epochs to let G produce reasonable outputs
                should_update_d = (epoch >= d_warmup_epochs) and (batch_idx % d_update_freq == 0)
                
                # Apply D learning rate warmup after warm-up epochs
                if epoch >= d_warmup_epochs and optimizer_d is not None:
                    d_lr_factor = min(1.0, (epoch - d_warmup_epochs + 1) / max(1, d_lr_warmup_epochs))
                    # Note: We don't modify LR in-place here as it's handled at epoch level
                
                # Mode collapse detection: if pixel loss is suspiciously low, G may be outputting constant
                # Skip D update to let G recover via pixel loss alone
                if 'pixel' in loss_dict and loss_dict['pixel'] < 0.01:
                    should_update_d = False
                
                # Get real and fake images
                real_hr = targets['hr_image'].clone()
                fake_hr = outputs['hr_image'].detach().clone()
                
                # Safety clamp: ensure images are in valid range before D
                # This prevents NaN when generator produces extreme values
                real_hr = torch.clamp(real_hr, -1.0, 1.0)
                fake_hr = torch.clamp(fake_hr, -1.0, 1.0)
                
                # Add instance noise for training stability (decays over epochs)
                if instance_noise_std > 0:
                    real_hr = real_hr + torch.randn_like(real_hr) * instance_noise_std
                    fake_hr = fake_hr + torch.randn_like(fake_hr) * instance_noise_std
                
                # Initialize loss_d to None to track if it was computed
                loss_d = None
                d_accuracy = 0.5  # Default to random accuracy
                
                # STABILITY FIX: Disable AMP for discriminator forward to prevent FP16 overflow
                # When D becomes confident, logits can exceed FP16 range before clamping
                with autocast('cuda', enabled=False):
                    # Discriminator forward in FP32 for stability
                    pred_real = discriminator(real_hr.float())
                    pred_fake = discriminator(fake_hr.float())
                    
                    # Check for NaN in D outputs - skip entirely if corrupted
                    if torch.isnan(pred_real).any() or torch.isnan(pred_fake).any():
                        d_nan_count += 1
                        should_update_d = False
                        if d_nan_count >= d_nan_threshold:
                            d_reset_count += 1
                            if logger is not None:
                                logger.warning(f"Resetting discriminator weights due to {d_nan_count} consecutive NaN outputs (reset #{d_reset_count} this epoch)")
                            # Reset D weights
                            for m in discriminator.modules():
                                if isinstance(m, (nn.Conv2d, nn.Linear)):
                                    nn.init.normal_(m.weight, 0, 0.02)
                                    if m.bias is not None:
                                        nn.init.zeros_(m.bias)
                            d_nan_count = 0
                            
                            if d_reset_count >= max_d_resets_per_epoch:
                                d_fallback_mode = True
                                if logger:
                                    logger.warning(f"Epoch {epoch}: Too many D resets ({d_reset_count}), falling back to pixel-only training")
                    else:
                        # Reset NaN counter on successful D forward
                        d_nan_count = 0
                        
                        # Discriminator loss with label smoothing
                        # Create smoothed target tensors
                        real_target = torch.full_like(pred_real, label_smoothing_real)
                        fake_target = torch.full_like(pred_fake, label_smoothing_fake)
                        
                        # Stricter clamping with NaN replacement for numerical stability
                        pred_real_safe = torch.where(
                            torch.isnan(pred_real) | torch.isinf(pred_real),
                            torch.zeros_like(pred_real),
                            torch.clamp(pred_real, -10.0, 10.0)
                        )
                        pred_fake_safe = torch.where(
                            torch.isnan(pred_fake) | torch.isinf(pred_fake),
                            torch.zeros_like(pred_fake),
                            torch.clamp(pred_fake, -10.0, 10.0)
                        )
                        
                        # LSGAN discriminator loss: use MSE instead of BCE
                        # Real samples should output ~1.0, fake samples should output ~0.0
                        # Use MSE loss (LSGAN) instead of BCE
                        loss_d_real = F.mse_loss(pred_real_safe, real_target)
                        loss_d_fake = F.mse_loss(pred_fake_safe, fake_target)
                        loss_d = (loss_d_real + loss_d_fake) / 2
                        
                        # For LSGAN, accuracy is based on distance from target
                        # Real is correct if pred > 0.5, Fake is correct if pred < 0.5
                        d_acc_real = (pred_real_safe > 0.5).float().mean().item()
                        d_acc_fake = (pred_fake_safe < 0.5).float().mean().item()
                        d_accuracy = (d_acc_real + d_acc_fake) / 2
                
                # Adaptive D update frequency: skip more often if D is winning
                if d_accuracy > 0.9 and d_update_freq < 5:
                    d_update_freq = min(5, d_update_freq + 1)
                elif d_accuracy < 0.7 and d_update_freq > 1:
                    d_update_freq = max(1, d_update_freq - 1)
                
                # Check for NaN discriminator loss and update if valid
                if loss_d is not None and not check_for_nan(loss_d, "loss_d") and should_update_d:
                    optimizer_d.zero_grad()
                    
                    # R1 gradient penalty on real images (every 16 batches to save compute)
                    r1_penalty = torch.tensor(0.0, device=device)
                    if batch_idx % 16 == 0:
                        try:
                            with torch.enable_grad():
                                real_hr_gp = targets['hr_image'].detach().requires_grad_(True)
                                # Disable AMP for gradient penalty computation
                                with autocast('cuda', enabled=False):
                                    pred_real_gp = discriminator(real_hr_gp)
                                    grad_real = torch.autograd.grad(
                                        outputs=pred_real_gp.sum(),
                                        inputs=real_hr_gp,
                                        create_graph=False,  # Only need 1st order gradients for R1
                                        retain_graph=False
                                    )[0]
                                    r1_penalty = grad_real.pow(2).flatten(start_dim=1).sum(1).mean() * 0.5
                            # Add with penalty (already scaled by 0.5 above)
                            if not check_for_nan(r1_penalty, "r1_penalty"):
                                loss_d = loss_d + r1_gamma * r1_penalty
                        except RuntimeError as e:
                            # Skip R1 penalty if gradient computation fails
                            if logger is not None:
                                logger.debug(f"R1 penalty computation failed: {e}")
                    
                    if use_amp and scaler_d is not None:
                        scaler_d.scale(loss_d).backward()
                        scaler_d.unscale_(optimizer_d)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                    else:
                        loss_d.backward()
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip_norm)
                        optimizer_d.step()
                    
                    loss_dict['loss_d'] = loss_d.item()
                    loss_dict['d_accuracy'] = d_accuracy
                elif loss_d is not None and check_for_nan(loss_d, "loss_d"):
                    d_nan_count += 1
                    if logger is not None and d_nan_count <= 5:  # Only log first few to reduce spam
                        logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN discriminator loss, skipping D update (count: {d_nan_count}/{d_nan_threshold})")
                    
                    # Reset discriminator if too many consecutive NaN batches
                    if d_nan_count >= d_nan_threshold:
                        d_reset_count += 1
                        if logger is not None:
                            logger.warning(f"Resetting discriminator weights due to {d_nan_count} consecutive NaN losses (reset #{d_reset_count} this epoch)")
                        
                        # Reinitialize discriminator weights
                        for m in discriminator.modules():
                            if isinstance(m, (nn.Conv2d, nn.Linear)):
                                nn.init.normal_(m.weight, 0, 0.02)
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)
                        d_nan_count = 0
                        
                        if d_reset_count >= max_d_resets_per_epoch:
                            d_fallback_mode = True
                            if logger:
                                logger.warning(f"Epoch {epoch}: Too many D resets ({d_reset_count}), falling back to pixel-only training")
        
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
    
    # Confusion matrix tracking for LCOFL weight updates
    # Check if criterion has LCOFL loss with confusion tracker
    has_lcofl = hasattr(criterion, 'lcofl_loss') and criterion.lcofl_loss is not None
    
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
            
            # Update confusion matrix for LCOFL (tracks confused character pairs)
            if has_lcofl:
                criterion.lcofl_loss.update_confusion_matrix(predictions, target_chars)
    
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
    
    # Finalize LCOFL confusion matrix weights for next epoch
    # and add confused pairs to metrics for logging
    if has_lcofl:
        # Update character weights based on confusion matrix
        criterion.lcofl_loss.finalize_epoch_weights(threshold=0.05)
        
        # Get confused character pairs for logging
        confused_pairs = criterion.lcofl_loss.get_confused_pairs(threshold=0.05)
        if confused_pairs:
            # Add top 5 confused pairs to metrics for logging
            metrics['confused_pairs'] = confused_pairs[:5]
        
        # Reset confusion matrix for next epoch
        criterion.lcofl_loss.reset_confusion_matrix()
    
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
    """
    Load training checkpoint with backward compatibility.
    
    Uses strict=False to allow loading checkpoints from older model versions
    that may not have new components (e.g., shared attention module).
    Missing weights will be randomly initialized.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Use strict=False for backward compatibility with older checkpoints
    # This allows loading checkpoints that don't have shared attention weights
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], 
        strict=False
    )
    
    # Log any missing or unexpected keys for debugging
    if missing_keys:
        # Filter to just show summary instead of all keys
        shared_attn_keys = [k for k in missing_keys if 'shared_attention' in k]
        other_keys = [k for k in missing_keys if 'shared_attention' not in k]
        
        if shared_attn_keys:
            print(f"[INFO] Checkpoint missing {len(shared_attn_keys)} shared_attention weights (will be randomly initialized)")
        if other_keys:
            print(f"[WARNING] Checkpoint missing {len(other_keys)} other weights: {other_keys[:5]}...")
    
    if unexpected_keys:
        print(f"[WARNING] Checkpoint has {len(unexpected_keys)} unexpected keys: {unexpected_keys[:5]}...")
    
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
    
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
        weight_geometry=config.training.weight_geometry,
        gan_mode='lsgan'  # Use LSGAN (MSE loss) instead of vanilla (BCE) to prevent vanishing gradients when D is too strong
    )
    
    # Create OCR Discriminator if configured
    # Uses OCR recognition confidence instead of binary real/fake classification
    ocr_discriminator = None
    ocr_discriminator_loss = None
    if config.training.use_ocr_discriminator and stage in ['restoration', 'full']:
        logger.info("Using OCR-as-Discriminator for GAN training")
        logger.info(f"  OCR frozen: {config.training.freeze_ocr_discriminator}")
        logger.info(f"  Confidence mode: {config.training.ocr_confidence_mode}")
        logger.info(f"  OCR guidance weight: {config.training.weight_ocr_guidance}")
        
        ocr_discriminator = create_ocr_discriminator(
            ocr_model=model.recognizer,
            freeze_ocr=config.training.freeze_ocr_discriminator,
            confidence_mode=config.training.ocr_confidence_mode
        )
        ocr_discriminator_loss = OCRDiscriminatorLoss(
            ocr_discriminator=ocr_discriminator,
            generator_loss_type='combined',
            lambda_conf=config.training.weight_ocr_guidance
        )
    
    # Create optimizers
    optimizer_g = create_optimizer(model, stage, config)
    optimizer_d = None
    
    if discriminator is not None and stage in ['restoration', 'full']:
        # TTUR (Two-Timescale Update Rule): Use same LR as G (reduced from 4x)
        # High D LR was causing instability and mode collapse
        base_lr = config.training.lr_restoration if stage == 'restoration' else config.training.lr_finetune
        lr_d = base_lr * 1.0  # Reduced from 4x to 1x for stability
        
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(0.0, 0.99)  # TTUR uses beta1=0 for discriminator
        )
        logger.info(f"Stage {stage}: Using TTUR for D optimizer (lr={lr_d:.2e}, betas=(0.0, 0.99))")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer_g, num_epochs, config)
    
    # Create separate GradScalers for G and D to prevent scale interference
    scaler_g = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    scaler_d = GradScaler('cuda') if use_amp and device.type == 'cuda' and discriminator is not None else None
    
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
            scaler_g=scaler_g, scaler_d=scaler_d, ema=ema, grad_clip_norm=grad_clip_norm,
            total_epochs=num_epochs, ocr_discriminator_loss=ocr_discriminator_loss
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
            
            # ============================================
            # Visualize Super-Resolution Samples
            # ============================================
            # Only visualize during stages where SR model is trained
            # Log sample images every N epochs (configurable)
            vis_interval = 5  # Visualize every 5 epochs
            num_vis_samples = 4  # Number of samples to visualize
            
            if stage in ['restoration', 'full'] and ((epoch + 1) % vis_interval == 0 or epoch == 0):
                model.eval()
                with torch.no_grad():
                    # Get a batch from validation loader
                    vis_batch = next(iter(val_loader))
                    vis_lr = vis_batch['lr_frames'][:num_vis_samples].to(device)
                    vis_hr_gt = vis_batch['hr_image'][:num_vis_samples].to(device)
                    vis_text = vis_batch['text'][:num_vis_samples]
                    vis_text_indices = vis_batch['text_indices'][:num_vis_samples].to(device)
                    
                    # Get model outputs
                    vis_outputs = model(vis_lr, vis_text_indices, return_intermediates=True)
                    vis_hr_pred = vis_outputs['hr_image']
                    
                    # Denormalize images from [-1, 1] to [0, 1] for visualization
                    vis_lr_first = (vis_lr[:, 0, :, :, :] + 1) / 2  # Take first frame
                    vis_hr_gt_norm = (vis_hr_gt + 1) / 2
                    vis_hr_pred_norm = (vis_hr_pred + 1) / 2
                    
                    # Clamp to valid range
                    vis_lr_first = torch.clamp(vis_lr_first, 0, 1)
                    vis_hr_gt_norm = torch.clamp(vis_hr_gt_norm, 0, 1)
                    vis_hr_pred_norm = torch.clamp(vis_hr_pred_norm, 0, 1)
                    
                    # Upsample LR for comparison (so all images are same size)
                    import torch.nn.functional as F_vis
                    vis_lr_upsampled = F_vis.interpolate(
                        vis_lr_first, 
                        size=vis_hr_gt_norm.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Log to TensorBoard as image grids
                    from torchvision.utils import make_grid
                    
                    # Create comparison: LR (upsampled) | Predicted HR | Ground Truth HR
                    for i in range(min(num_vis_samples, vis_hr_pred_norm.size(0))):
                        comparison = torch.stack([
                            vis_lr_upsampled[i],
                            vis_hr_pred_norm[i],
                            vis_hr_gt_norm[i]
                        ], dim=0)
                        grid = make_grid(comparison, nrow=3, padding=2, normalize=False)
                        writer.add_image(
                            f'samples/{vis_text[i]}_LR_Pred_GT', 
                            grid, 
                            epoch
                        )
                    
                    # Also create a combined grid of all samples
                    all_pred = make_grid(vis_hr_pred_norm, nrow=num_vis_samples, padding=2)
                    all_gt = make_grid(vis_hr_gt_norm, nrow=num_vis_samples, padding=2)
                    writer.add_image('samples/all_predicted_HR', all_pred, epoch)
                    writer.add_image('samples/all_ground_truth_HR', all_gt, epoch)
                    
                    logger.info(f"Logged {num_vis_samples} SR visualization samples to TensorBoard")
            
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
    parser.add_argument('--reset-epoch', action='store_true',
                        help='Reset epoch to 0 when resuming (use when loading weights from different stage)')
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
    
    # Start TensorBoard in background for JupyterLab access
    import subprocess
    tensorboard_port = 6007
    try:
        tb_process = subprocess.Popen(
            ['tensorboard', '--logdir', output_dir, '--port', str(tensorboard_port), '--bind_all'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"TensorBoard started on port {tensorboard_port}")
        logger.info(f"Access via: https://toanle.cvip.id.vn/proxy/{tensorboard_port}/")
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")
        logger.info(f"TensorBoard logs directory: {log_dir}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    # With optional shared attention and deformable convolutions
    model = NeuroSymbolicLPR(
        num_frames=config.model.num_frames,
        lr_size=(config.data.lr_height, config.data.lr_width),
        hr_size=(config.data.hr_height, config.data.hr_width),
        use_shared_attention=config.training.use_shared_attention,
        use_deformable_conv=config.training.use_deformable_conv,
    ).to(device)
    
    if config.training.use_shared_attention:
        logger.info("Model using shared attention")
    if config.training.use_deformable_conv:
        logger.info("Model using deformable convolutions")
    
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
        
        total_size = len(full_dataset)
        
        # ============================================
        # Stratified Split by Scenario and Plate Type
        # ============================================
        # Group samples by their stratification key (scenario + plate_type)
        from collections import defaultdict
        import random as py_random
        
        strata_groups = defaultdict(list)
        for idx, sample in enumerate(full_dataset.samples):
            # Extract scenario from track_dir path (e.g., "Scenario-A")
            track_dir = sample.get('track_dir', sample.get('image_path', ''))
            scenario = 'unknown'
            for part in Path(track_dir).parts:
                if part.startswith('Scenario-'):
                    scenario = part
                    break
            
            # Get plate type
            plate_type = sample.get('layout', 'unknown')
            
            # Create stratification key
            strata_key = f"{scenario}_{plate_type}"
            strata_groups[strata_key].append(idx)
        
        # Log strata distribution
        logger.info(f"Found {len(strata_groups)} stratification groups:")
        for key, indices in sorted(strata_groups.items()):
            logger.info(f"  {key}: {len(indices)} samples")
        
        # Split each stratum proportionally
        py_random.seed(config.training.seed)
        train_indices = []
        val_indices = []
        test_indices = []
        
        for strata_key, indices in strata_groups.items():
            py_random.shuffle(indices)
            n = len(indices)
            n_val = max(1, int(n * val_split_ratio)) if n > 2 else 0
            n_test = max(1, int(n * test_split_ratio)) if n > 2 else 0
            n_train = n - n_val - n_test
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        # Shuffle within each set for training
        py_random.shuffle(train_indices)
        py_random.shuffle(val_indices)
        py_random.shuffle(test_indices)
        
        logger.info(f"Stratified split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
        
        # Save test indices for later evaluation
        test_indices_path = os.path.join(output_dir, 'test_indices.pt')
        torch.save(test_indices, test_indices_path)
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
        train_dataset.samples = [train_dataset.samples[i] for i in train_indices]
        val_dataset.samples = [val_dataset.samples[i] for i in val_indices]
        
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
        if args.reset_epoch:
            start_epoch = 0
            logger.info(f"Loaded weights from {args.resume}, reset epoch to 0 (--reset-epoch)")
        else:
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
