"""
Configuration for Neuro-Symbolic LPR System.

This module contains all hyperparameters and constants for the ALPR system
designed for Brazilian license plates (both old Brazilian and Mercosul formats).
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================================
# License Plate Syntax Patterns
# ============================================================================

# Old Brazilian format: LLL-NNNN (3 letters, 4 numbers)
BRAZILIAN_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$')

# Mercosul format: LLLNLNN (3 letters, 1 number, 1 letter, 2 numbers)
MERCOSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')

# Character sets
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGITS = '0123456789'
CHARSET = LETTERS + DIGITS  # 36 characters total
CHARSET_SIZE = len(CHARSET)

# Special tokens for sequence recognition
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
BOS_TOKEN = '<BOS>'

# Full vocabulary including special tokens
VOCABULARY = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + list(CHARSET)
VOCAB_SIZE = len(VOCABULARY)

# Token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
CHAR_START_IDX = 3

# Plate length (without dash)
PLATE_LENGTH = 7


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input configuration
    num_frames: int = 5
    input_channels: int = 3
    input_height: int = 64
    input_width: int = 192
    
    # Shared CNN Encoder
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    encoder_kernel_size: int = 3
    
    # Spatial Transformer Network
    stn_feature_dim: int = 512
    stn_localization_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    stn_fc_dims: List[int] = field(default_factory=lambda: [256, 128])
    max_rotation_degrees: float = 35.0
    
    # Layout Classifier
    layout_hidden_dim: int = 256
    
    # Quality Scorer
    quality_hidden_dim: int = 128
    
    # SwinIR Generator
    swinir_embed_dim: int = 96
    swinir_depths: List[int] = field(default_factory=lambda: [6, 6, 6, 6])
    swinir_num_heads: List[int] = field(default_factory=lambda: [6, 6, 6, 6])
    swinir_window_size: int = 8
    swinir_mlp_ratio: float = 4.0
    upscale_factor: int = 4
    
    # Discriminator
    disc_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    disc_use_spectral_norm: bool = True
    
    # PARSeq Recognizer
    parseq_embed_dim: int = 384
    parseq_num_heads: int = 6
    parseq_depth: int = 12
    parseq_patch_size: int = 16
    parseq_max_length: int = PLATE_LENGTH + 2  # +2 for BOS and EOS
    
    # Syntax Mask
    soft_mask_value: float = -100.0  # For training stability
    

@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # General
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    # Batch sizes for each training stage
    batch_size_pretrain: int = 128
    batch_size_stn: int = 32
    batch_size_restoration: int = 16
    batch_size_finetune: int = 8
    
    # Learning rates
    lr_pretrain: float = 1e-4
    lr_stn: float = 1e-4
    lr_restoration: float = 2e-4
    lr_finetune: float = 1e-5
    lr_parseq_finetune: float = 1e-6  # Lower LR for pre-trained OCR
    
    # Training epochs for each stage
    epochs_pretrain: int = 50
    epochs_stn: int = 20
    epochs_restoration: int = 30
    epochs_finetune: int = 100
    
    # Loss weights (for L_total = L_pixel + w1*L_GAN + w2*L_OCR + w3*L_geo)
    weight_pixel: float = 1.0
    weight_gan: float = 0.1
    weight_ocr: float = 0.5
    weight_geometry: float = 0.1
    
    # Optimizer
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 1
    checkpoint_dir: str = 'checkpoints'
    
    # Logging
    log_dir: str = 'logs'
    log_every: int = 100


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Dataset paths
    train_dir: str = 'data/train'
    val_dir: str = 'data/val'
    test_dir: str = 'data/test'
    
    # Image sizes
    lr_height: int = 16
    lr_width: int = 48
    hr_height: int = 64
    hr_width: int = 192
    
    # Multi-frame settings
    num_frames: int = 5
    frame_interval: int = 1  # Frames apart in video sequence
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    blur_probability: float = 0.3
    noise_probability: float = 0.3
    
    # Synthetic data generation
    synthetic_count: int = 5_000_000
    fonts: List[str] = field(default_factory=lambda: ['Mandatory', 'FE-Schrift'])


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment name
    experiment_name: str = 'neuro_symbolic_lpr'
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.model.num_frames == self.data.num_frames, \
            "Model and data num_frames must match"
        assert self.model.input_height == self.data.hr_height, \
            "Model input height must match data HR height"
        assert self.model.input_width == self.data.hr_width, \
            "Model input width must match data HR width"


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_position_constraints(is_mercosul: bool) -> List[str]:
    """
    Get character type constraints for each position.
    
    Args:
        is_mercosul: Whether the plate is Mercosul format.
        
    Returns:
        List of 'L' (letter) or 'N' (number) for each position.
    """
    if is_mercosul:
        # Mercosul: LLLNLNN (positions 0,1,2,4 are letters; 3,5,6 are numbers)
        return ['L', 'L', 'L', 'N', 'L', 'N', 'N']
    else:
        # Brazilian: LLLNNNN (positions 0,1,2 are letters; 3,4,5,6 are numbers)
        return ['L', 'L', 'L', 'N', 'N', 'N', 'N']


def validate_plate_text(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a text string is a valid Brazilian license plate.
    
    Args:
        text: License plate text (without dash).
        
    Returns:
        Tuple of (is_valid, plate_type) where plate_type is 'brazilian' or 'mercosul'.
    """
    text = text.upper().replace('-', '').replace(' ', '')
    
    if BRAZILIAN_PATTERN.match(text):
        return True, 'brazilian'
    elif MERCOSUL_PATTERN.match(text):
        return True, 'mercosul'
    else:
        return False, None


def infer_layout_from_text(text: str) -> int:
    """
    Infer layout type from ground-truth text.
    
    Args:
        text: Ground-truth plate text.
        
    Returns:
        0 for Brazilian, 1 for Mercosul, -1 for invalid.
    """
    is_valid, plate_type = validate_plate_text(text)
    if not is_valid:
        return -1
    return 1 if plate_type == 'mercosul' else 0
