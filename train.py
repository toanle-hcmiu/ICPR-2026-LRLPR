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
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from copy import deepcopy
import math

import numpy as np
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
# Reproducibility Helpers
# =============================================================================

def seed_everything(seed: int, strict_determinism: bool = True):
    """
    Set seeds for Python, NumPy, and PyTorch for reproducibility.
    
    By default, enables strict determinism for exact reproducibility across runs.
    This may have a performance impact (typically 5-15% slower training).
    
    Args:
        seed: Random seed to use.
        strict_determinism: If True (default), enables cuDNN deterministic mode,
            torch.use_deterministic_algorithms(), and disables TF32.
            Set to False for faster but non-reproducible training.
    """
    # Set environment variable for hash seed (must be done before other imports in practice)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed Python's random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch CPU
    torch.manual_seed(seed)
    
    # Seed PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if strict_determinism:
        # Enable cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable PyTorch's deterministic algorithms with warn_only=True
        # We use warn_only=True because some operations (F.grid_sample backward,
        # deformable convolutions) don't have deterministic implementations.
        # This allows training to proceed while warning about non-deterministic ops.
        # See: https://github.com/pytorch/pytorch/issues/for deterministic alternatives
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set CUBLAS workspace config for deterministic cuBLAS operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Disable TF32 for exact reproducibility (TF32 trades precision for speed)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def worker_init_fn(worker_id: int):
    """
    Initialize worker with a deterministic seed based on the global seed.
    
    Called by DataLoader for each worker process to ensure reproducible
    data loading across workers.
    
    Args:
        worker_id: Worker process index.
    """
    # Get the initial seed from torch and combine with worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def apply_config_overrides(config: Config, overrides: dict, logger: Optional[logging.Logger] = None) -> Config:
    """
    Apply YAML config overrides to a Config object.
    
    Supports nested keys via section.key structure (e.g., training.lr_stn).
    Unknown keys are logged as warnings but do not cause errors.
    
    Args:
        config: Base Config object to modify.
        overrides: Dictionary from YAML config file.
        logger: Optional logger for warnings about unknown keys.
        
    Returns:
        Modified Config object (same instance, mutated in-place).
    """
    if overrides is None:
        return config
    
    def _warn(msg: str):
        if logger:
            logger.warning(msg)
        else:
            print(f"Warning: {msg}")
    
    for section_name, section_values in overrides.items():
        # Check if the section exists in Config
        if not hasattr(config, section_name):
            _warn(f"Unknown config section '{section_name}', ignoring")
            continue
        
        subconfig = getattr(config, section_name)
        
        # Handle non-dict values at top level (e.g., experiment_name)
        if not isinstance(section_values, dict):
            if hasattr(config, section_name):
                setattr(config, section_name, section_values)
            else:
                _warn(f"Unknown config key '{section_name}', ignoring")
            continue
        
        # Apply nested key-value pairs
        for key, value in section_values.items():
            if hasattr(subconfig, key):
                old_value = getattr(subconfig, key)
                setattr(subconfig, key, value)
                # Log the override for debugging
                if logger:
                    logger.debug(f"Config override: {section_name}.{key} = {value} (was {old_value})")
            else:
                _warn(f"Unknown config key '{section_name}.{key}', ignoring")
    
    return config


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
    
    # Filter out empty param groups and groups with no trainable params
    # This prevents "optimizer got an empty parameter list" errors
    filtered_groups = []
    for group in param_groups:
        # Convert generator to list to check if it has any parameters
        params = list(group['params'])
        if not params:
            continue  # Skip empty groups
        # Check if at least one param requires grad
        has_trainable = any(p.requires_grad for p in params)
        if not has_trainable:
            continue  # Skip groups where all params are frozen
        # Re-create the group with the list (generator was exhausted)
        new_group = {k: v for k, v in group.items() if k != 'params'}
        new_group['params'] = params
        filtered_groups.append(new_group)
    
    param_groups = filtered_groups
    
    # Adjust learning rates based on lr_scale if present
    base_lr = {
        'pretrain': config.training.lr_pretrain,
        'stn': config.training.lr_stn,
        'parseq_warmup': config.training.lr_parseq_warmup,
        'restoration': config.training.lr_restoration,
        'full': config.training.lr_finetune
    }.get(stage, 1e-4)
    
    for group in param_groups:
        lr_scale = group.pop('lr_scale', 1.0)
        group['lr'] = base_lr * lr_scale
        # Keep 'name' for logging (AdamW ignores unknown keys in param groups)
    
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
    ocr_discriminator_loss: Optional['OCRDiscriminatorLoss'] = None,
    model_stage2: Optional[nn.Module] = None,  # Stage 2 model for anchor generation
    config: Optional[Config] = None            # For Stage 3 parameters
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
        # Also disable for parseq_warmup since pretrained PARSeq has FP32 weights
        use_amp_for_batch = use_amp and (stage not in ['stn', 'parseq_warmup'])
        scaler = scaler_g  # Use generator scaler for main forward pass
        
        with autocast('cuda', enabled=use_amp_for_batch):
            # DEBUG: Log OCR weight at first batch of each epoch to trace curriculum issues
            if batch_idx == 0 and stage == 'full' and hasattr(criterion, 'weights'):
                if logger:
                    logger.info(f"Epoch {epoch} Batch 0: criterion.weights['ocr'] = {criterion.weights.get('ocr', 'NOT SET')}")
            # SPECIAL CASE: parseq_warmup stage - bypass generator, use GT HR directly
            # This trains PARSeq on clean GT images, not noisy generated ones
            if stage == 'parseq_warmup':
                # Directly feed GT HR to recognizer
                hr_image = targets['hr_image']
                raw_logits = model.recognizer.forward_parallel(hr_image)
                
                # Get layout from first frame (simplified)
                B = hr_image.shape[0]
                is_mercosul = torch.zeros(B, device=hr_image.device)  # Placeholder
                
                # Apply syntax mask
                masked_logits = model.syntax_mask(raw_logits, is_mercosul, training=True)
                
                # Create minimal outputs dict for loss computation
                outputs = {
                    'raw_logits': raw_logits,
                    'masked_logits': masked_logits,
                    'hr_image': hr_image,  # Use GT HR for logging purposes
                }
            else:
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
            if stage == 'pretrain' or stage == 'parseq_warmup':
                loss_g, loss_dict = criterion.get_stage_loss('pretrain', outputs, targets)
            elif stage == 'stn':
                loss_g, loss_dict = criterion.get_stage_loss('stn', outputs, targets)
            elif stage == 'restoration':
                loss_g, loss_dict = criterion.get_stage_loss('restoration', outputs, targets, current_discriminator)
            else:
                # Stage 3 ('full'): Use anti-collapse loss with OCR warmup
                # Compute global_step for OCR warmup scheduling
                global_step = epoch * num_batches + batch_idx
                
                # Generate Stage 2 anchor signals if model is available
                sr_stage2 = None
                ocr_baseline = None
                
                if model_stage2 is not None:
                    with torch.no_grad():
                        # Run Stage 2 model on same input
                        # Note: We must enable mixed precision if used, but gradients are disabled
                        with autocast('cuda', enabled=use_amp_for_batch):
                            out_stage2 = model_stage2(lr_frames)
                            sr_stage2 = out_stage2['hr_image'].detach()
                            
                            # Compute OCR baseline from Stage 2 model for hinge constraint
                            # This ensures Stage 3 OCR loss only activates when SR makes OCR worse
                            if 'masked_logits' in out_stage2:
                                stage2_logits = out_stage2['masked_logits'].detach()
                                # Compute OCR loss on Stage-2 SR output as baseline
                                ocr_baseline = criterion.ocr_loss(
                                    stage2_logits, 
                                    targets['text_indices']
                                ).detach().item()  # Scalar baseline
                
                loss_g, loss_dict = criterion.get_stage3_loss(
                    outputs, targets, current_discriminator,
                    global_step=global_step,
                    sr_stage2=sr_stage2,  # Detached Stage 2 output
                    ocr_baseline=ocr_baseline,  # Validated Stage 2 OCR loss
                    config=config.training if config else None
                )
            
            # Add OCR-based guidance loss if configured
            # Uses recognition confidence to guide super-resolution
            # OPTIMIZATION: Only run OCR every N steps to reduce overhead
            ocr_every_n_steps = 4  # Run OCR every 4 iterations (4x speedup)
            ocr_batch_size = 8  # Only use subset of batch for OCR (4x speedup)
            
            run_ocr_this_step = (batch_idx % ocr_every_n_steps == 0)
            
            if ocr_discriminator_loss is not None and stage == 'restoration' and run_ocr_this_step:
                # Get text targets (remove BOS/EOS tokens)
                text_targets = targets['text_indices'][:, 1:-1]  # (B, PLATE_LENGTH)
                
                # OPTIMIZATION: Only use subset of batch for OCR (saves ~75% OCR cost)
                hr_for_ocr = outputs['hr_image'][:ocr_batch_size]
                gt_for_ocr = targets['hr_image'][:ocr_batch_size]
                text_for_ocr = text_targets[:ocr_batch_size]
                
                # Compute OCR guidance loss on generated HR images
                ocr_loss, ocr_metrics = ocr_discriminator_loss.generator_loss(
                    fake_images=hr_for_ocr,
                    targets=text_for_ocr,
                    real_images=gt_for_ocr
                )
                
                # Only apply OCR guidance if the OCR model has meaningful confidence
                # If real image confidence is too low, the OCR model is untrained/unreliable
                real_conf = ocr_metrics.get('real_confidence', 0.0)
                min_ocr_confidence = 0.10  # 10% threshold
                
                if real_conf >= min_ocr_confidence:
                    # Add to total loss (scaled by ocr_every_n_steps to compensate for frequency)
                    if not check_for_nan(ocr_loss, "ocr_guidance"):
                        loss_g = loss_g + ocr_loss * ocr_every_n_steps
                        loss_dict['ocr_guidance'] = ocr_metrics.get('total', ocr_loss.item())
                else:
                    # Skip OCR guidance - model not ready
                    loss_dict['ocr_guidance'] = 0.0
                
                # Always log confidence for monitoring
                loss_dict['ocr_fake_conf'] = ocr_metrics.get('fake_confidence', 0.0)
                if 'real_confidence' in ocr_metrics:
                    loss_dict['ocr_real_conf'] = ocr_metrics['real_confidence']
            elif ocr_discriminator_loss is not None and stage == 'restoration':
                # Not running OCR this step - use cached values for logging
                loss_dict['ocr_guidance'] = 0.0
                loss_dict['ocr_fake_conf'] = 0.0
                loss_dict['ocr_real_conf'] = 0.0
        
        # Check for NaN loss and skip batch if detected
        if check_for_nan(loss_g, "loss_g"):
            nan_batch_count += 1
            if logger is not None:
                logger.warning(f"Epoch {epoch}, Batch {batch_idx}: NaN loss detected, skipping batch ({nan_batch_count}/{max_nan_batches})")
            
            # Clear gradients and skip this batch
            optimizer_g.zero_grad(set_to_none=True)
            if optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)
            
            if nan_batch_count >= max_nan_batches:
                if logger is not None:
                    logger.error(f"Epoch {epoch}: Too many NaN batches ({nan_batch_count}), stopping epoch early")
                break
            continue
        
        # Update generator with gradient scaling for mixed precision
        optimizer_g.zero_grad(set_to_none=True)
        
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
                    optimizer_d.zero_grad(set_to_none=True)
                    
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
            
            # Per-module gradient norm logging for Stage 3 debugging
            if stage == 'full' and batch_idx % 100 == 0:
                for name, module in model.named_children():
                    try:
                        grad_norm_module = sum(
                            p.grad.data.norm(2).item() ** 2 
                            for p in module.parameters() 
                            if p.grad is not None
                        ) ** 0.5
                        writer.add_scalar(f'grad_norm/{name}', grad_norm_module, global_step)
                        
                        # Debug: Check recognizer requires_grad status
                        if name == 'recognizer' and grad_norm_module == 0:
                            trainable_count = sum(1 for p in module.parameters() if p.requires_grad)
                            total_count = sum(1 for p in module.parameters())
                            logger.warning(f"Recognizer grad_norm=0! Trainable: {trainable_count}/{total_count}")
                            
                            # Emergency re-unfreeze
                            for param in module.parameters():
                                param.requires_grad = True
                            logger.info("Emergency: Force-enabled requires_grad on all recognizer params")
                            
                    except Exception:
                        pass  # Skip if module has no gradients
                
                # Log learning rates per param group
                for i, group in enumerate(optimizer_g.param_groups):
                    group_name = group.get('name', f'group_{i}')
                    writer.add_scalar(f'lr/{group_name}', group['lr'], global_step)
                
                # Log recognizer trainability status (should be True in Stage 3)
                recognizer_trainable = any(p.requires_grad for p in model.recognizer.parameters())
                writer.add_scalar('debug/recognizer_requires_grad', float(recognizer_trainable), global_step)
                
                # Log OCR curriculum weight if present in loss_dict
                if 'ocr_weight' in loss_dict:
                    writer.add_scalar('train/ocr_curriculum_weight', loss_dict['ocr_weight'], global_step)
    
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
    
    # Stage 3 SR-vs-GT OCR diagnostics: track OCR accuracy on GT HR for comparison
    # This helps detect SR distribution shift vs PARSeq degradation
    correct_chars_gt_hr = 0  # OCR accuracy on ground truth HR images
    correct_plates_gt_hr = 0
    total_chars_gt_hr = 0
    total_plates_gt_hr = 0
    
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
            
            # SPECIAL CASE: parseq_warmup stage - bypass generator, use GT HR directly
            if stage == 'parseq_warmup':
                raw_logits = model.recognizer.forward_parallel(hr_image)
                B = hr_image.shape[0]
                is_mercosul = torch.zeros(B, device=hr_image.device)
                masked_logits = model.syntax_mask(raw_logits, is_mercosul, training=False)
                outputs = {
                    'raw_logits': raw_logits,
                    'masked_logits': masked_logits,
                    'hr_image': hr_image,
                }
            else:
                outputs = model(lr_frames, return_intermediates=True)
            
            # Compute loss - use stage-specific loss to avoid NaN from OCR
            # during early training stages (STN stage especially)
            if stage == 'pretrain' or stage == 'parseq_warmup':
                _, loss_dict = criterion.get_stage_loss('pretrain', outputs, targets)
            elif stage == 'stn':
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
            
            # Stage 3 diagnostic: OCR accuracy on GT HR images
            # This distinguishes "PARSeq degraded" from "SR distribution shift"
            if stage == 'full' and 'hr_image' in targets:
                # Run recognizer on ground truth HR (not generated SR)
                gt_hr = targets['hr_image']
                gt_logits = model.recognizer.forward_parallel(gt_hr)
                
                # Apply syntax mask
                if 'layout_logits' in outputs:
                    is_mercosul_gt = torch.sigmoid(outputs['layout_logits']).squeeze(-1)
                else:
                    is_mercosul_gt = torch.zeros(gt_hr.size(0), device=gt_hr.device)
                gt_masked_logits = model.syntax_mask(gt_logits, is_mercosul_gt, training=False)
                gt_predictions = gt_masked_logits.argmax(dim=-1)
                
                # Compute GT HR accuracy
                for i in range(batch_size):
                    sample_mask = char_mask[i]
                    if sample_mask.any():
                        gt_correct = (gt_predictions[i][sample_mask] == target_chars[i][sample_mask]).all().item()
                        correct_plates_gt_hr += int(gt_correct)
                        total_plates_gt_hr += 1
                
                if char_mask.any():
                    correct_chars_gt_hr += (gt_predictions[char_mask] == target_chars[char_mask]).sum().item()
                    total_chars_gt_hr += char_mask.sum().item()
    
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
    
    # Stage 3 diagnostic: add GT HR OCR metrics for SR distribution shift detection
    if total_chars_gt_hr > 0:
        metrics['char_accuracy_gt_hr'] = correct_chars_gt_hr / total_chars_gt_hr
        metrics['plate_accuracy_gt_hr'] = correct_plates_gt_hr / total_plates_gt_hr if total_plates_gt_hr > 0 else 0.0
        # SR distribution shift = GT HR good but SR bad
        metrics['sr_distribution_gap'] = metrics['char_accuracy_gt_hr'] - metrics['char_accuracy']
    
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
    
    Security Note:
        torch.load() uses pickle which can execute arbitrary code.
        Only load checkpoints from trusted sources.
    """
    # Security: Validate checkpoint path is a local file
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found or not a local file: {checkpoint_path}. "
            "For security reasons, only local file paths are accepted."
        )
    
    # Security warning for users
    import warnings
    warnings.warn(
        f"Loading checkpoint from '{checkpoint_path}'. "
        "torch.load() uses pickle which can execute arbitrary code. "
        "Only load checkpoints from trusted sources.",
        UserWarning
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    early_stopping_patience: int = 0,
    stage2_anchor_path: Optional[str] = None
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
    
    # Create criterion with all config-driven weights and options
    criterion = CompositeLoss(
        weight_pixel=config.training.weight_pixel,
        weight_gan=config.training.weight_gan,
        weight_ocr=config.training.weight_ocr,
        weight_geometry=config.training.weight_geometry,
        weight_perceptual=0.1,  # VGG feature matching weight
        weight_lcofl=config.training.weight_lcofl if config.training.use_lcofl else 0.0,
        weight_ssim=config.training.weight_ssim if config.training.use_lcofl else 0.0,
        weight_tv=config.training.weight_tv,  # Total Variation for wavy artifact suppression
        use_perceptual=True,  # Enable perceptual loss for sharper images
        use_lcofl=config.training.use_lcofl,
        lcofl_alpha=config.training.lcofl_alpha,
        lcofl_beta=config.training.lcofl_beta,
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
        
        # Stage 3 FIX: The OCR discriminator above freezes the recognizer when
        # freeze_ocr=True, but Stage 3 requires the recognizer to be trainable
        # for end-to-end fine-tuning. We unfreeze it here AFTER creating the
        # discriminator so the discriminator still uses the frozen reference
        # for stable discrimination, but gradients can flow through for training.
        if stage == 'full':
            logger.info("Stage 3: Re-unfreezing recognizer after OCR discriminator creation")
            model.unfreeze_recognizer()
            for param in model.recognizer.parameters():
                param.requires_grad = True
            recognizer_param_count = sum(1 for _ in model.recognizer.parameters())
            logger.info(f"  Set requires_grad=True on {recognizer_param_count} recognizer parameters")
    
    # Create optimizers
    optimizer_g = create_optimizer(model, stage, config)
    
    # Log optimizer param groups for debugging (especially Stage 3 recognizer LR)
    logger.info(f"Optimizer param groups for stage '{stage}':")
    for i, group in enumerate(optimizer_g.param_groups):
        group_name = group.get('name', f'group_{i}')
        num_params = sum(p.numel() for p in group['params'])
        logger.info(f"  {group_name}: lr={group['lr']:.2e}, params={num_params:,}")
    
    # Stage 3: Log key hyperparameters for diagnosing OCR collapse
    if stage == 'full':
        logger.info("Stage 3 Key Hyperparameters:")
        logger.info(f"  OCR warmup steps: {config.training.stage3_ocr_warmup_steps}")
        logger.info(f"  OCR ramp steps: {config.training.stage3_ocr_ramp_steps}")
        logger.info(f"  OCR max weight: {config.training.stage3_ocr_max_weight}")
        logger.info(f"  OCR hinge enabled: {config.training.stage3_use_ocr_hinge}")
        logger.info(f"  SR anchor weight: {config.training.stage3_sr_anchor_weight}")
        logger.info(f"  TV weight: {config.training.weight_tv}")
        logger.info(f"  LCOFL enabled: {config.training.use_lcofl}")
        if config.training.use_lcofl:
            logger.info(f"  LCOFL weight: {config.training.weight_lcofl}")
        
        # Verify recognizer is unfrozen
        # Ensure recognizer model is loaded (for lazy-loaded models)
        if hasattr(model.recognizer, '_load_model'):
            model.recognizer._load_model()
        
        # Convert to list to avoid iterator consumption issues
        recognizer_params_list = list(model.recognizer.parameters())
        recognizer_trainable = sum(1 for p in recognizer_params_list if p.requires_grad)
        recognizer_total = len(recognizer_params_list)
        recognizer_total_params = sum(p.numel() for p in recognizer_params_list)
        recognizer_trainable_params = sum(p.numel() for p in recognizer_params_list if p.requires_grad)
        
        logger.info(f"  Recognizer trainable: {recognizer_trainable}/{recognizer_total} param groups")
        logger.info(f"  Recognizer trainable: {recognizer_trainable_params:,}/{recognizer_total_params:,} total params")
        if recognizer_trainable == 0 or recognizer_trainable_params == 0:
            logger.warning("  WARNING: Recognizer is frozen! This may cause OCR collapse.")
            # Try to unfreeze again as a safety measure
            model.unfreeze_recognizer()
            for param in model.recognizer.parameters():
                param.requires_grad = True
            logger.info("  Attempted to unfreeze recognizer again.")
    
    optimizer_d = None
    
    # DISABLE GAN for Stage 3 (full) - GAN causes mode collapse
    # Stage 3 uses only pixel + OCR loss for stability
    if discriminator is not None and stage == 'restoration':  # Changed: removed 'full'
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
    
    # Stage 3 Anti-Collapse: Load Stage 2 model as anchor
    model_stage2 = None
    if stage == 'full':
        # Use provided anchor path or fallback to default
        stage2_checkpoint = stage2_anchor_path if stage2_anchor_path else os.path.join(checkpoint_dir, 'restoration_best.pth')
        
        if os.path.exists(stage2_checkpoint):
            logger.info(f"Stage 3 Anti-Collapse: Loading Stage 2 anchor from {stage2_checkpoint}")
            # Clone model architecture
            model_stage2 = deepcopy(model)
            # Load Stage 2 weights
            load_checkpoint(model_stage2, stage2_checkpoint)
            # Freeze and set to eval
            for param in model_stage2.parameters():
                param.requires_grad = False
            model_stage2.eval()
            model_stage2.to(device)
        else:
            logger.warning(f"Stage 3 Warning: Stage 2 checkpoint not found at {stage2_checkpoint}. Anchoring disabled!")

    best_acc = 0.0
    consecutive_nan_epochs = 0
    max_consecutive_nan_epochs = 3  # Stop stage if too many consecutive NaN epochs
    
    for epoch in range(start_epoch, num_epochs):
        # Curriculum learning for Stage 3: gradually increase OCR weight
        # CRITICAL: Start at 0 and delay OCR to let generator stabilize with pixel loss first
        # Otherwise OCR gradients cause mode collapse (characters fade to minimize OCR confusion)
        if stage == 'full' and hasattr(criterion, 'weights'):
            # OCR curriculum: 0 for first N epochs (dead zone), then ramp to target
            ocr_dead_epochs = 5        # No OCR for first 5 epochs - generator must learn colors/shapes first
            ocr_ramp_epochs = 15       # Ramp from 0 to target over next 15 epochs
            target_ocr_weight = 0.3    # Reduced target - OCR is secondary to visual quality
            
            if epoch < ocr_dead_epochs:
                # Dead zone: no OCR loss at all
                current_ocr_weight = 0.0
            elif epoch < ocr_dead_epochs + ocr_ramp_epochs:
                # Ramp zone: linear ramp from 0 to target
                progress = (epoch - ocr_dead_epochs) / ocr_ramp_epochs
                current_ocr_weight = target_ocr_weight * progress
            else:
                # Stable zone: full OCR weight
                current_ocr_weight = target_ocr_weight
            
            criterion.weights['ocr'] = current_ocr_weight
            # DEBUG: Verify the weight was set correctly
            assert criterion.weights['ocr'] == current_ocr_weight, \
                f"Curriculum weight not persisted: expected {current_ocr_weight}, got {criterion.weights['ocr']}"
            if epoch % 5 == 0 or epoch < 3:  # Log frequently early on
                logger.info(f"Curriculum: OCR weight = {current_ocr_weight:.3f} (epoch {epoch}, dead_epochs={ocr_dead_epochs})")
        
        # Train
        train_losses = train_epoch(
            model, discriminator, train_loader, criterion,
            optimizer_g, optimizer_d, device, epoch, stage, writer, logger,
            scaler_g=scaler_g, scaler_d=scaler_d, ema=ema, grad_clip_norm=grad_clip_norm,
            total_epochs=num_epochs, ocr_discriminator_loss=ocr_discriminator_loss,
            model_stage2=model_stage2, config=config
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
            
            # Log epoch-level learning rates
            for i, group in enumerate(optimizer_g.param_groups):
                group_name = group.get('name', f'group_{i}')
                writer.add_scalar(f'lr_epoch/{group_name}', group['lr'], epoch)
            
            # Stage 3: Log key training diagnostics
            if stage == 'full':
                # Recognizer trainability
                recognizer_trainable = sum(1 for p in model.recognizer.parameters() if p.requires_grad)
                recognizer_total = sum(1 for p in model.recognizer.parameters())
                writer.add_scalar('debug/recognizer_trainable_params', recognizer_trainable, epoch)
                logger.info(f"  Recognizer: {recognizer_trainable}/{recognizer_total} params trainable")
            
            # Stage 3 diagnostic: Log GT HR OCR metrics and SR distribution gap
            if 'char_accuracy_gt_hr' in val_metrics:
                gt_hr_char_acc = val_metrics['char_accuracy_gt_hr']
                gt_hr_plate_acc = val_metrics['plate_accuracy_gt_hr']
                sr_gap = val_metrics['sr_distribution_gap']
                writer.add_scalar('val/char_accuracy_gt_hr', gt_hr_char_acc, epoch)
                writer.add_scalar('val/plate_accuracy_gt_hr', gt_hr_plate_acc, epoch)
                writer.add_scalar('val/sr_distribution_gap', sr_gap, epoch)
                logger.info(
                    f"  GT HR OCR: Char={gt_hr_char_acc:.4f}, Plate={gt_hr_plate_acc:.4f}, "
                    f"SR Gap={sr_gap:+.4f} (positive = SR worse than GT)"
                )
            
            # Log confused character pairs from LCOFL (helps diagnose character collapse)
            if 'confused_pairs' in val_metrics:
                pairs = val_metrics['confused_pairs']
                pairs_str = ', '.join([f"{p[0]}->{p[1]}({p[2]:.2f})" for p in pairs])
                logger.info(f"  Most confused chars: {pairs_str}")
                # Log as text to TensorBoard
                writer.add_text('val/confused_pairs', pairs_str, epoch)
            
            # Log LCOFL confusion matrix heatmap (every 5 epochs to save space)
            if hasattr(criterion, 'lcofl_loss') and criterion.lcofl_loss is not None:
                if epoch % 5 == 0 or epoch == 0:
                    confusion_img = criterion.lcofl_loss.get_confusion_matrix_image()
                    if confusion_img is not None:
                        writer.add_image('val/confusion_matrix', confusion_img, epoch)
            
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
                    
                    # ============================================
                    # Visualize STN Rectified Output
                    # ============================================
                    # Check if model has STN intermediates
                    logger.info(f"vis_outputs keys: {list(vis_outputs.keys())}")
                    if 'rectified_features' in vis_outputs:
                        # Get rectified features from model output
                        # The rectified_features are in feature space (B, T, C, H, W)
                        # We need to convert them to image space for visualization
                        
                        rectified = vis_outputs['rectified_features'][:num_vis_samples]
                        # rectified shape: (B, T, C, H, W) - take first frame
                        rectified_first = rectified[:, 0]  # (B, C, H, W)
                        # Reduce channels to 3 for visualization
                        if rectified_first.shape[1] > 3:
                            # Take first 3 channels
                            stn_output = rectified_first[:, :3]  # (B, 3, H, W)
                        else:
                            stn_output = rectified_first
                        
                        # Normalize for visualization
                        stn_min = stn_output.min()
                        stn_max = stn_output.max()
                        if stn_max - stn_min > 1e-6:
                            stn_normalized = (stn_output - stn_min) / (stn_max - stn_min)
                        else:
                            stn_normalized = stn_output
                        stn_normalized = torch.clamp(stn_normalized, 0, 1)
                        
                        # Upsample to HR size for better visibility
                        stn_upsampled = F_vis.interpolate(
                            stn_normalized,
                            size=(64, 192),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        # Log STN output grid
                        stn_grid = make_grid(stn_upsampled, nrow=num_vis_samples, padding=2)
                        writer.add_image('samples/stn_rectified', stn_grid, epoch)
                        logger.info(f"Logged STN rectified visualization to TensorBoard")
                    
                    # Also visualize input LR frames for comparison
                    all_lr = make_grid(vis_lr_upsampled, nrow=num_vis_samples, padding=2)
                    writer.add_image('samples/all_input_LR', all_lr, epoch)
                    
                    # Visualize lr_image (after feature_to_image, before SwinIR)
                    # This is the actual input to the super-resolution
                    if 'lr_image' in vis_outputs:
                        lr_img = vis_outputs['lr_image'][:num_vis_samples]
                        # Denormalize from [-1, 1] to [0, 1]
                        lr_img_norm = (lr_img + 1) / 2
                        lr_img_norm = torch.clamp(lr_img_norm, 0, 1)
                        # Upsample for visibility
                        lr_img_upsampled = F_vis.interpolate(
                            lr_img_norm,
                            size=(64, 192),
                            mode='bilinear',
                            align_corners=False
                        )
                        lr_img_grid = make_grid(lr_img_upsampled, nrow=num_vis_samples, padding=2)
                        writer.add_image('samples/lr_image_before_SR', lr_img_grid, epoch)
                    
                    # ============================================
                    # Visualize OCR Predictions
                    # ============================================
                    # Decode predictions and compare to ground truth
                    if 'masked_logits' in vis_outputs or 'raw_logits' in vis_outputs:
                        # Get logits (prefer masked if available)
                        logits_key = 'masked_logits' if 'masked_logits' in vis_outputs else 'raw_logits'
                        vis_logits = vis_outputs[logits_key][:num_vis_samples]
                        
                        # Decode predictions
                        pred_indices = vis_logits.argmax(dim=-1)  # (B, seq_len)
                        
                        # Get charset from config
                        from config import CHARSET, CHAR_START_IDX
                        
                        # Convert indices to text
                        ocr_results = []
                        for i in range(min(num_vis_samples, pred_indices.size(0))):
                            pred_text = ""
                            for idx in pred_indices[i]:
                                char_idx = idx.item() - CHAR_START_IDX
                                if 0 <= char_idx < len(CHARSET):
                                    pred_text += CHARSET[char_idx]
                                # Skip padding/special tokens
                            ocr_results.append(f"GT: {vis_text[i]} | Pred (Generated): {pred_text}")
                        
                        # ============================================
                        # Also run OCR on Real HR images for comparison
                        # ============================================
                        # This helps diagnose if issues are from SR quality or OCR model itself
                        with torch.no_grad():
                            # Run OCR recognizer directly on ground truth HR images
                            real_hr_logits = model.recognizer(vis_hr_gt_norm * 2 - 1)  # Convert back to [-1, 1]
                            real_pred_indices = real_hr_logits.argmax(dim=-1)  # (B, seq_len)
                            
                            ocr_real_results = []
                            for i in range(min(num_vis_samples, real_pred_indices.size(0))):
                                real_pred_text = ""
                                for idx in real_pred_indices[i]:
                                    char_idx = idx.item() - CHAR_START_IDX
                                    if 0 <= char_idx < len(CHARSET):
                                        real_pred_text += CHARSET[char_idx]
                                ocr_real_results.append(f"GT: {vis_text[i]} | Pred (Real HR): {real_pred_text}")
                        
                        # Log as text to TensorBoard - both generated and real HR predictions
                        ocr_text = "=== OCR on Generated HR ===\n" + "\n".join(ocr_results)
                        ocr_text += "\n\n=== OCR on Real HR (Ground Truth) ===\n" + "\n".join(ocr_real_results)
                        writer.add_text('samples/ocr_predictions', ocr_text, epoch)
                        
                        # Also log per-sample confidence if available
                        if vis_logits.dim() == 3:
                            # Softmax to get probabilities
                            probs = torch.softmax(vis_logits, dim=-1)
                            # Max probability per position
                            max_probs = probs.max(dim=-1).values
                            # Average confidence per sample
                            avg_conf = max_probs.mean(dim=-1)
                            for i in range(min(num_vis_samples, avg_conf.size(0))):
                                writer.add_scalar(f'samples/ocr_conf_generated_sample{i}', avg_conf[i].item(), epoch)
                            
                            # Log confidence for real HR too
                            real_probs = torch.softmax(real_hr_logits[:num_vis_samples], dim=-1)
                            real_max_probs = real_probs.max(dim=-1).values
                            real_avg_conf = real_max_probs.mean(dim=-1)
                            for i in range(min(num_vis_samples, real_avg_conf.size(0))):
                                writer.add_scalar(f'samples/ocr_conf_real_sample{i}', real_avg_conf[i].item(), epoch)
                        
                        logger.info(f"OCR on Generated HR: {ocr_results}")
                        logger.info(f"OCR on Real HR: {ocr_real_results}")
            
            # Save best model - use char_accuracy as metric (more reliable than plate_acc
            # which stays 0 until all 7 chars are correct)
            if char_acc > best_acc:
                best_acc = char_acc
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
            
            # Check early stopping - use char_accuracy for better gradual improvement tracking
            if early_stopper is not None:
                if early_stopper(char_acc):
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
                        choices=['0', '1', '1.5', '2', '3', 'all'],
                        help='Training stage (0-3 or all, 1.5=parseq_warmup)')
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
    parser.add_argument('--stage2-anchor', type=str, default=None, 
                        help='Path to Stage 2 checkpoint for anchoring (Stage 3 anti-collapse)')
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Apply YAML overrides to default config (logger not yet available)
        config = apply_config_overrides(config, config_dict, logger=None)
    
    # Set random seeds for reproducibility
    seed_everything(config.training.seed)
    
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
    
    # Log config file usage
    if args.config is not None:
        logger.info(f"Loaded config overrides from: {args.config}")
    
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
            ('parseq_warmup', config.training.epochs_parseq_warmup),
            ('restoration', config.training.epochs_restoration),
            ('full', config.training.epochs_finetune)
        ]
    else:
        stage_map = {
            '0': ('pretrain', config.training.epochs_pretrain),
            '1': ('stn', config.training.epochs_stn),
            '1.5': ('parseq_warmup', config.training.epochs_parseq_warmup),
            '2': ('restoration', config.training.epochs_restoration),
            '3': ('full', config.training.epochs_finetune)
        }
        stages = [stage_map[args.stage]]
    
    # Train each stage
    for stage_name, num_epochs in stages:
        # Prepare model for stage
        if stage_name == 'pretrain':
            # Pretrain: freeze all except recognizer (train OCR only)
            model.freeze_all_except_recognizer()
        elif stage_name == 'stn':
            model.freeze_recognizer()
        elif stage_name == 'parseq_warmup':
            # PARSeq warm-up: train only recognizer on GT HR images
            model.freeze_all_except_recognizer()
        elif stage_name == 'restoration':
            model.freeze_stn()
            model.freeze_recognizer()
        else:  # full
            model.unfreeze_stn()
            # Stage 3: Train recognizer end-to-end with SR
            # PARSeq stays in eval mode (dropout off) to stabilize gradients
            # while still allowing weight updates via requires_grad=True
            model.unfreeze_recognizer()
            
            # Force recognizer to be trainable (checkpoint may have restored frozen state)
            # Ensure all recognizer parameters have requires_grad=True
            recognizer_params = list(model.recognizer.parameters())
            for param in recognizer_params:
                param.requires_grad = True
            logger.info(f"  Explicitly set requires_grad=True on {len(recognizer_params)} recognizer parameters")
        
        # Get appropriate batch size for this stage
        batch_size_map = {
            'pretrain': config.training.batch_size_pretrain,
            'stn': config.training.batch_size_stn,
            'parseq_warmup': config.training.batch_size_parseq_warmup,
            'restoration': config.training.batch_size_restoration,
            'full': config.training.batch_size_finetune
        }
        batch_size = batch_size_map.get(stage_name, config.training.batch_size_finetune)
        
        # Create data loaders with stage-specific batch size, custom collate, and deterministic seeding
        # Use a generator seeded from config for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(config.training.seed)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lpr_collate_fn,
            worker_init_fn=worker_init_fn,
            generator=g
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            collate_fn=lpr_collate_fn,
            worker_init_fn=worker_init_fn
        )
        
        logger.info(f"Stage {stage_name}: Using batch size {batch_size}")
        
        # Determine Stage 2 anchor path (for Stage 3 anti-collapse)
        # 1. Use explicit argument if provided
        # 2. Or infer from resume path if provided
        stage2_anchor_path = args.stage2_anchor
        if stage2_anchor_path is None and args.resume is not None:
            # Try to find restoration_best.pth in same directory as resume checkpoint
            resume_dir = os.path.dirname(args.resume)
            potential_path = os.path.join(resume_dir, 'restoration_best.pth')
            if os.path.exists(potential_path):
                stage2_anchor_path = potential_path
                logger.info(f"Auto-inferred Stage 2 anchor path: {stage2_anchor_path}")
            else:
                # Fallback: Use the resume checkpoint itself as anchor
                # This handles cases where we start training from a specific checkpoint
                # and want to anchor to it (e.g., when restoration_best doesn't exist)
                stage2_anchor_path = args.resume
                logger.info(f"Using resume checkpoint as Stage 2 anchor: {stage2_anchor_path}")
        
        best_acc = train_stage(
            model, discriminator, train_loader, val_loader,
            config, stage_name, num_epochs, device,
            checkpoint_dir, writer, logger, start_epoch,
            use_amp=not args.no_amp,
            use_ema=not args.no_ema,
            early_stopping_patience=args.early_stopping,
            stage2_anchor_path=stage2_anchor_path
        )
        
        logger.info(f"Stage {stage_name} completed with best accuracy: {best_acc:.4f}")
        start_epoch = 0  # Reset for next stage
    
    writer.close()
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
