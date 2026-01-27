"""
Configuration for Neuro-Symbolic LPR System.

This module contains all hyperparameters and constants for the ALPR system
designed for Brazilian license plates (both old Brazilian and Mercosul formats).

================================================================================
HARDCODED PLATE FORMAT SPECIFICATIONS
================================================================================

These format specifications are HARDCODED throughout the codebase and cannot
be changed without updating multiple files:

1. Brazilian Format (Old):
   - Pattern: LLL-NNNN (display) / LLLNNNN (storage)
   - Regex: ^[A-Z]{3}[0-9]{4}$
   - Position constraints: [L, L, L, N, N, N, N]
   - Example: ABC-1234 → ABC1234

2. Mercosul Format (New):
   - Pattern: LLLNLNN (no dash)
   - Regex: ^[A-Z]{3}[0-9][A-Z][0-9]{2}$
   - Position constraints: [L, L, L, N, L, N, N]
   - Example: ABC1D23

3. Constants:
   - PLATE_LENGTH = 7 (both formats, without dash)
   - CHARSET = A-Z + 0-9 (36 characters)
   - VOCAB_SIZE = 39 (36 chars + 3 special tokens: PAD, BOS, EOS)

Files that contain hardcoded format references:
- config.py: Pattern definitions, get_position_constraints()
- models/syntax_mask.py: Mask generation (lines 317, 320, 349, 352)
- data/dataset.py: Text validation and layout inference

================================================================================
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================================
# License Plate Syntax Patterns (HARDCODED)
# ============================================================================
#
# IMPORTANT: These format specifications are HARDCODED throughout the system.
# Changing these requires updates to:
#   - config.py: Pattern definitions and get_position_constraints()
#   - models/syntax_mask.py: Mask generation (lines 317, 320, 349, 352)
#   - data/dataset.py: Text validation and layout inference
#
# Brazilian Format (Old):
#   - Display: LLL-NNNN (with dash)
#   - Storage: LLLNNNN (without dash, PLATE_LENGTH=7)
#   - Pattern: 3 letters followed by 4 digits
#   - Positions: [L, L, L, N, N, N, N]
#   - Example: ABC-1234 → ABC1234
#
# Mercosul Format (New):
#   - Display: LLLNLNN (no dash)
#   - Storage: LLLNLNN (PLATE_LENGTH=7)
#   - Pattern: 3 letters, 1 digit, 1 letter, 2 digits
#   - Positions: [L, L, L, N, L, N, N]
#   - Example: ABC1D23
#

# Old Brazilian format: LLL-NNNN (3 letters, 4 numbers)
# HARDCODED: Position constraints [L, L, L, N, N, N, N]
BRAZILIAN_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$')

# Mercosul format: LLLNLNN (3 letters, 1 number, 1 letter, 2 numbers)
# HARDCODED: Position constraints [L, L, L, N, L, N, N]
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
# HARDCODED: Both Brazilian and Mercosul formats use 7 characters
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

    # Text-Prior Guided Generator (2026-01-27)
    # Replaces SwinIR with text-aware architecture based on TPGSR + LP-Diff + TSRN
    use_tp_generator: bool = True  # Enable new text-prior guided generator
    tp_generator_channels: int = 64  # Feature channels in generator
    tp_num_srb: int = 4  # Number of Sequential Residual Blocks
    tp_lstm_hidden: int = 32  # Hidden size for BiLSTM in SRB
    tp_lightweight: bool = False  # Use lightweight variant for testing
    tp_text_feature_dim: int = 256  # Dimension for text prior features
    tp_num_heads: int = 4  # Number of attention heads in cross-attention

    # Multi-Frame Fusion (NEW - from LP-Diff)
    fusion_type: str = 'icam'  # 'icam' for cross-attention, '3d' for 3d conv, 'adaptive'
    fusion_num_heads: int = 4  # Number of attention heads (for ICAM)

    # Text Prior (NEW - from TPGSR)
    use_text_prior_training: bool = True  # Use text prior during training only
    text_prior_aggregation: str = 'average'  # How to aggregate multi-frame text prior

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
    batch_size_parseq_warmup: int = 64  # Larger batch for OCR-only training on GT HR
    batch_size_restoration: int = 32  # Increased from 16 to stabilize BatchNorm in Discriminator
    batch_size_finetune: int = 32  # Increased from 8 for faster training
    
    # Learning rates
    # Note: STN learning rate further reduced from 5e-5 to 1e-5 to prevent gradient explosion
    # during geometry warm-up phase. STN affine transformations are extremely sensitive
    # to learning rate - grid sampling gradients can explode with large parameter updates.
    lr_pretrain: float = 1e-4
    lr_stn: float = 1e-5  # Further reduced for stability - STN is extremely sensitive to large LR
    lr_parseq_warmup: float = 5e-5  # Higher LR for faster warm-up on GT HR
    lr_restoration: float = 5e-5  # Reduced from 2e-4 to fix GAN instability (discriminator collapse)
    lr_finetune: float = 1e-5  # Reduced from 2e-5 for Stage 3 stability
    lr_parseq_finetune: float = 1e-6  # Lower LR for pre-trained OCR
    
    # Training epochs for each stage
    epochs_pretrain: int = 50
    epochs_stn: int = 50
    epochs_parseq_warmup: int = 30  # PARSeq warm-up on GT HR before restoration
    epochs_restoration: int = 500
    epochs_finetune: int = 500  # Increased from 100 for better convergence
    
    # Loss weights - FIXED to rebalance Stage 3 training
    # INCREASED pixel loss to anchor visual quality and prevent character destruction
    # REDUCED LCOFL/SSIM/TV to prevent domination and over-smoothing
    # Original paper uses ONLY LCOFL, but staged training requires stronger pixel anchor
    weight_pixel: float = 0.50      # FIXED: Increased from 0.25 to 0.50 for better quality anchor
    weight_perceptual: float = 0.0  # DISABLED - not in original paper, causes blur
    weight_gan: float = 0.0         # DISABLED - original uses OCR-as-discriminator
    weight_ocr: float = 0.2         # Increased for stronger character guidance (0.05 was too weak)
    weight_geometry: float = 0.0    # DISABLED - simplify

    # LCOFL Loss (from Nascimento et al. "Enhancing LP Super-Resolution" paper)
    # PRIMARY LOSS - following original paper configuration exactly
    # 
    # CRITICAL SCALING FIX:
    # - LCOFL classification CE loss is ~38 (high when predictions are wrong)
    # - Pixel L1 loss is ~0.3 typical
    # - With weight_lcofl=0.25: LCOFL contributes ~9.5, Pixel contributes ~0.15
    # - This 60x imbalance caused the generator to ONLY optimize for OCR, ignoring pixel fidelity
    # - Result: mode collapse to a single character pattern "AEE-9333" that minimizes OCR loss
    # 
    # NEW WEIGHTS:
    # - weight_lcofl=0.005: LCOFL contributes ~0.19, Pixel contributes ~0.15 (balanced!)
    # - This ensures generator produces images that are BOTH visually correct AND OCR-readable
    use_lcofl: bool = True     # Enable LCOFL loss
    weight_lcofl: float = 0.005  # FIXED: Reduced from 0.25 to 0.005 to balance with pixel loss
    weight_ssim: float = 0.05   # SSIM for structural similarity
    lcofl_alpha: float = 1.0   # Penalty increment for confused character pairs
    lcofl_beta: float = 2.0    # Layout violation penalty
    use_frozen_ocr_for_lcofl: bool = False  # DISABLED: Use trainable OCR only (simplifies training, removes confusing signals)
    weight_edge: float = 0.0   # DISABLED - not in original paper
    
    # PARSeq Feature Loss - DISABLED (not in original paper)
    # This was experimental, but adds complexity
    use_parseq_feature_loss: bool = False  # DISABLED - simplify
    weight_parseq_feature: float = 0.0
    parseq_feature_warmup_steps: int = 5000
    parseq_feature_pixel_threshold: float = 0.5
    generator_grad_clip: float = 0.5
    
    # Total Variation Loss for suppressing wavy/checkerboard artifacts
    # Recommended: 1e-5 to 1e-4 for subtle smoothing without blur
    # FIXED: Reduced to prevent character oversmoothing and edge blurring
    weight_tv: float = 5e-6  # FIXED: Reduced from 1e-5 to 5e-6 for sharper characters
    
    # Shared Attention and Deformable Convolutions (from same paper)
    use_shared_attention: bool = True  # Enable shared attention module
    use_deformable_conv: bool = True  # Enable deformable convolutions
    
    # OCR-as-Discriminator (from LCOFL paper)
    # Replaces binary discriminator with OCR-based guidance for more stable training
    use_ocr_discriminator: bool = True  # Enabled: Uses OCR confidence as discriminator (more stable than binary GAN)
    weight_ocr_guidance: float = 0.0  # DISABLED - causes mode collapse (maximizes ANY confidence, not correct chars)
    freeze_ocr_discriminator: bool = True  # Keep OCR frozen during training
    ocr_confidence_mode: str = 'mean'  # 'mean', 'min', or 'product'
    
    # Stage 3 Parameters - SIMPLIFIED
    # No SR anchor (not in original paper), no OCR hinge
    stage3_sr_anchor_weight: float = 0.0      # DISABLED - not in original paper
    stage3_ocr_warmup_steps: int = 0          # No warmup - use LCOFL from start (like paper)
    stage3_ocr_ramp_steps: int = 0            # No ramp
    stage3_ocr_max_weight: float = 0.0        # Handled by LCOFL directly
    stage3_use_ocr_hinge: bool = False        # Disabled

    # Text-Prior Guided Architecture Training (2026-01-27)
    # Training stages for the new text-prior guided architecture
    use_tp_architecture: bool = True          # Use new TP-guided architecture
    tp_architecture_stages: List[str] = field(
        default_factory=lambda: ['tp_stn', 'tp_generator', 'tp_full']
    )  # Training stages for TP architecture
    tp_stn_epochs: int = 15                  # STN pre-training with TP-generator
    tp_generator_epochs: int = 50             # Generator training with text prior
    tp_full_epochs: int = 30                  # End-to-end fine-tuning

    # TP Architecture loss weights
    weight_tp_pixel: float = 1.0              # Reconstruction loss (L1)
    weight_tp_text_prior: float = 0.3         # Text prior loss (during training)
    weight_tp_perceptual: float = 0.1         # Perceptual loss (VGG)
    weight_tp_ssim: float = 0.2                # SSIM loss
    # Disable GAN loss for TP architecture (causes instability)
    weight_tp_gan: float = 0.0                 # DISABLE - GAN causes mode collapse

    # TP Architecture learning rates
    tp_stn_lr: float = 1e-4                   # STN learning rate
    tp_generator_lr: float = 1e-4               # Generator learning rate
    tp_full_lr: float = 5e-5                    # Full model learning rate

    # Optimizer
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Gradient clipping
    grad_clip_norm: float = 0.5  # Reduced from 1.0 to prevent gradient explosion during GAN training
    grad_clip_norm_stn: float = 0.5  # Tighter clipping for STN stage to prevent explosion
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 1
    checkpoint_dir: str = 'checkpoints'
    
    # Logging
    log_dir: str = 'logs'
    log_every: int = 100


@dataclass
class DataConfig:
    """
    Data configuration.
    
    Note on Plate Styles:
    - Brazilian (Old): Grey background, format LLL-NNNN
    - Mercosur (New): White background with blue band at top, format LLLNLNN
      The blue band contains country name, flag, and Mercosur logo.
      Uses FE-Schrift typeface for enhanced security.
    """
    
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
    
    # Style-aware augmentation (for preserving Mercosur blue band visibility)
    style_aware_augmentation: bool = True
    # Mercosur: More conservative brightness/contrast to preserve blue band
    mercosur_brightness_range: Tuple[float, float] = (0.85, 1.15)
    mercosur_contrast_range: Tuple[float, float] = (0.85, 1.15)
    # Brazilian: Standard ranges (grey background more tolerant)
    brazilian_brightness_range: Tuple[float, float] = (0.8, 1.2)
    brazilian_contrast_range: Tuple[float, float] = (0.8, 1.2)
    
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
    
    HARDCODED: These constraints are hardcoded and used throughout the system
    for syntax mask generation, validation, and text processing.
    
    Args:
        is_mercosul: Whether the plate is Mercosul format.
        
    Returns:
        List of 'L' (letter) or 'N' (number) for each position (length=7).
        
    Format Specifications:
        Brazilian: LLLNNNN → ['L', 'L', 'L', 'N', 'N', 'N', 'N']
        Mercosul:  LLLNLNN → ['L', 'L', 'L', 'N', 'L', 'N', 'N']
    """
    if is_mercosul:
        # HARDCODED: Mercosul format LLLNLNN
        # Positions 0,1,2,4 are letters; positions 3,5,6 are numbers
        return ['L', 'L', 'L', 'N', 'L', 'N', 'N']
    else:
        # HARDCODED: Brazilian format LLLNNNN
        # Positions 0,1,2 are letters; positions 3,4,5,6 are numbers
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
