# Neuro-Symbolic LPR System

A sophisticated Automatic License Plate Recognition (ALPR) system for Brazilian license plates, combining deep neural networks with symbolic reasoning. Designed for the ICPR 2026 conference.

## Overview

This system recognizes Brazilian license plates in two formats:
- **Brazilian (Old)**: `LLL-NNNN` (e.g., ABC-1234)
- **Mercosul (New)**: `LLLNLNN` (e.g., ABC1D23)

### Key Features

- **Multi-frame Processing**: Processes 5 LR frames for robust recognition
- **Spatial Transformer Network**: Corrects geometric distortions with self-supervised alignment
- **Layout Classification**: Automatically detects Brazilian vs Mercosul format (with optional attention mechanism)
- **GAN Super-Resolution**: Full SwinIR-based image restoration (6 RSTB blocks, 180 embed dim)
- **Pretrained PARSeq**: Uses pretrained PARSeq from [baudm/parseq](https://github.com/baudm/parseq) for state-of-the-art OCR
- **Syntax-Masked Recognition**: Enforces valid plate formats using symbolic constraints
- **LCOFL Loss**: Layout and Character Oriented Focal Loss with SSIM and confusion matrix tracking
- **Deformable Convolutions**: Adaptive spatial sampling for better character handling
- **Shared Attention Module**: PLTFAM-style attention with shared weights across blocks
- **Mixed Precision Training**: 2x faster training with automatic mixed precision (AMP)
- **EMA Model Averaging**: Exponential moving average for more stable final models
- **Stage-Aware Validation**: Prevents NaN losses during staged training
- **Soft Inference Mode**: Robust handling of damaged/non-standard plates

## Architecture

```
Input (5 LR Frames: 16×48)
       │
       ▼
┌─────────────────┐
│ Shared CNN      │  Phase 1: Feature Extraction
│ Encoder         │  (4 blocks, 64→512 channels)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spatial         │  Phase 1: Geometric Alignment
│ Transformer Net │  (Self-supervised + Corner Loss)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌────────────┐
│Layout │ │Quality     │  Phase 2: Classification & Fusion
│Classif│ │Scorer+Fuse │
└───┬───┘ └─────┬──────┘
    │           │
    │      ┌────┘
    │      ▼
    │  ┌─────────────────┐
    │  │ SwinIR          │  Phase 3: Super-Resolution (4×)
    │  │ Generator       │  (6 RSTB, 180 dim, window=8)
    │  └────────┬────────┘
    │           │
    │           ▼
    │  ┌─────────────────┐
    │  │ Pretrained      │  Phase 4: Recognition
    │  │ PARSeq          │  (from baudm/parseq)
    │  └────────┬────────┘
    │           │
    ▼           ▼
┌─────────────────────────┐
│ Syntax Mask Layer       │  Neuro-Symbolic Integration
│ (Dynamic Constraints)   │
└────────────┬────────────┘
             │
             ▼
      Plate Text Output (64×192 HR)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ICPR-2026-LRLPR.git
cd ICPR-2026-LRLPR

# Create virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- ~8GB VRAM for training with batch size 32

## Usage

### Training

The system uses a staged training schedule:

```bash
# Train all stages (with defaults: AMP enabled, EMA enabled)
python train.py --data-dir data/ --output-dir outputs/

# Train specific stage
python train.py --stage 1 --data-dir data/  # STN only
python train.py --stage 3 --resume checkpoints/step2.pth  # Fine-tune

# Training stages:
# 0: Synthetic pre-training (PARSeq) - NOT USED when using pretrained
# 1: Geometry warm-up (STN) - Self-supervised + pixel loss
# 2: Restoration + Layout (SwinIR, Classifier)
# 3: End-to-end fine-tuning (All modules)
```

#### Advanced Training Options

```bash
# Full training with all options
python train.py --data-dir data/ --output-dir outputs/ \
    --stage all \
    --early-stopping 15      # Stop if no improvement for 15 epochs

# Disable mixed precision (for debugging or CPU training)
python train.py --no-amp --data-dir data/

# Disable EMA model averaging
python train.py --no-ema --data-dir data/

# Resume from checkpoint
python train.py --stage 3 --resume checkpoints/restoration_best.pth
```

#### Training Features

| Feature | Default | Description |
|---------|---------|-------------|
| **Mixed Precision (AMP)** | Enabled | 2x faster training, lower memory usage |
| **EMA** | Enabled | Exponential moving average of weights for stable models |
| **Early Stopping** | Disabled | Stop training if validation accuracy plateaus |
| **LR Warmup** | 5 epochs | Linear warmup before main scheduler |
| **Gradient Clipping** | 1.0 (0.5 for STN) | Prevents gradient explosion |
| **Stage-Aware Validation** | Enabled | Uses stage-specific loss to prevent NaN |

#### Training Stages

| Stage | Modules Trained | Loss Functions | Batch Size |
|-------|-----------------|----------------|------------|
| STN | Encoder, STN | Self-supervised, Pixel, Corner | 32 |
| Restoration | Generator, Layout, Fusion | Pixel, GAN, Layout | 16 |
| Full | All | Pixel, GAN, OCR, Geometry, Layout | 8 |

#### Metrics Tracked

- **Plate Accuracy**: Exact match (all 7 characters correct)
- **Character Accuracy**: Per-character accuracy (excluding special tokens)
- **Layout Accuracy**: Brazilian vs Mercosul classification accuracy

### Inference

```bash
# Single image
python inference.py --model checkpoints/best.pth --input image.jpg

# Batch processing
python inference.py --model checkpoints/best.pth --input folder/

# Video processing
python inference.py --model checkpoints/best.pth --input video.mp4
```

### Python API

```python
from models import NeuroSymbolicLPR
from inference import load_model, predict_single
import torch
from PIL import Image
import numpy as np

# Load model with default settings
device = torch.device('cuda')
model = load_model('checkpoints/best.pth', device)

# Run inference
image = np.array(Image.open('plate.jpg').convert('RGB'))
text, confidence, sr_image, is_mercosul = predict_single(model, image, device)

print(f"Plate: {text}")
print(f"Format: {'Mercosul' if is_mercosul else 'Brazilian'}")
print(f"Confidence: {confidence:.2%}")
```

### Model Configuration Options

```python
from models import NeuroSymbolicLPR

# Create model with all options
model = NeuroSymbolicLPR(
    num_frames=5,
    lr_size=(16, 48),
    hr_size=(64, 192),
    
    # Use attention-enhanced layout classifier (recommended for difficult cases)
    use_attention_layout=True,
    
    # Soft inference for robustness to damaged/non-standard plates
    soft_inference=True,
    soft_inference_value=-50.0,
    
    # Use pretrained PARSeq (recommended)
    use_pretrained_parseq=True,
    parseq_model_name='parseq',  # or 'parseq_tiny' for faster inference
    
    # SwinIR configuration (full model, not lightweight)
    swinir_embed_dim=180,  # Full SwinIR uses 180
    swinir_depths=[6, 6, 6, 6, 6, 6],  # 6 RSTB blocks
    swinir_num_heads=[6, 6, 6, 6, 6, 6],
    swinir_window_size=8,
)
```

## Project Structure

```
ICPR-2026-LRLPR/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script with staged training
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
│
├── models/
│   ├── __init__.py
│   ├── neuro_symbolic_lpr.py # Main model (4-phase pipeline)
│   ├── encoder.py            # Shared CNN encoder (4 blocks)
│   ├── stn.py                # Spatial Transformer Network + MultiFrameSTN
│   ├── layout_classifier.py  # Layout classifier (+ attention variant)
│   ├── feature_fusion.py     # Quality scorer & weighted fusion
│   ├── swinir.py             # Full SwinIR generator
│   ├── discriminator.py      # PatchGAN discriminator
│   ├── parseq.py             # PARSeq wrapper (pretrained + custom fallback)
│   ├── syntax_mask.py        # Dynamic syntax mask with soft inference
│   ├── deformable_conv.py    # Deformable Convolution v2
│   └── shared_attention.py   # PLTFAM-style shared attention module
│
├── losses/
│   ├── __init__.py
│   ├── corner_loss.py        # STN corner supervision
│   ├── gan_loss.py           # Adversarial losses (vanilla, lsgan, wgan)
│   ├── composite_loss.py     # Combined loss + SelfSupervisedSTNLoss
│   ├── ocr_perceptual_loss.py # OCR-aware perceptual losses
│   └── lcofl_loss.py         # LCOFL loss with SSIM and confusion tracking
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # RodoSolDataset + SyntheticLPRDataset
│   ├── augmentation.py       # Style-aware augmentation pipeline
│   ├── train/                # Training data
│   ├── val/                  # Validation data
│   └── test/                 # Test data
│
├── scripts/
│   └── split_dataset.py      # Dataset splitting utility
│
└── utils/
    ├── __init__.py
    └── visualization.py      # Visualization tools
```

## Loss Functions

### Composite Training Loss

The total loss for end-to-end training:

```
L_total = L_pixel + 0.1 × L_GAN + 0.5 × L_OCR + 0.1 × L_geo + 0.1 × L_layout
```

| Loss | Weight | Description |
|------|--------|-------------|
| L_pixel | 1.0 | L1 pixel reconstruction loss |
| L_GAN | 0.1 | Adversarial loss for realism |
| L_OCR | 0.5 | Cross-entropy character recognition loss |
| L_geo | 0.1 | Corner loss for STN (+ self-supervised) |
| L_layout | 0.1 | Binary cross-entropy for layout classification |

### Self-Supervised STN Loss

For training without corner annotations (used in STN stage):

```python
L_stn = w_id × L_identity + w_cons × L_consistency + w_smooth × L_smoothness
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Identity | 0.1 | Prevents collapse, keeps transforms near identity |
| Consistency | 1.0 | Encourages similar transforms across frames |
| Smoothness | 0.5 | Penalizes extreme scaling/rotation/shear |

### OCR-Aware Perceptual Losses

Additional losses in `losses/ocr_perceptual_loss.py`:

| Loss Class | Description |
|------------|-------------|
| `OCRAwarePerceptualLoss` | Uses downstream OCR model to guide restoration |
| `CharacterFocusLoss` | Edge-aware loss using Sobel operators |
| `MultiScaleOCRLoss` | Evaluates OCR at multiple scales (1.0×, 0.5×, 0.25×) |

### LCOFL Loss (Layout and Character Oriented Focal Loss)

From "Enhancing License Plate Super-Resolution" (Nascimento et al.):

```python
# Enable in config
config.training.use_lcofl = True
config.training.weight_lcofl = 0.5
```

| Component | Description |
|-----------|-------------|
| Classification Loss | Weighted cross-entropy with dynamic character weights |
| Layout Penalty | Penalizes digit/letter misplacements based on format |
| SSIM Loss | Structural similarity for image quality |
| Confusion Tracking | Increases weights for frequently confused character pairs |

## Plate Format Specifications (Hardcoded)

### Brazilian Format (Old)
- **Pattern**: `LLL-NNNN` (displayed with dash) or `LLLNNNN` (stored without dash)
- **Regex**: `^[A-Z]{3}[0-9]{4}$`
- **Position Constraints**: `[L, L, L, N, N, N, N]`
- **Example**: `ABC1234` → `ABC-1234`

### Mercosul Format (New)
- **Pattern**: `LLLNLNN` (no dash)
- **Regex**: `^[A-Z]{3}[0-9][A-Z][0-9]{2}$`
- **Position Constraints**: `[L, L, L, N, L, N, N]`
- **Example**: `ABC1D23`

### Hardcoded Constants

```python
PLATE_LENGTH = 7
VOCAB_SIZE = 39  # 36 chars (A-Z, 0-9) + 3 special tokens (PAD, BOS, EOS)
CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
```

**Important**: These formats are hardcoded in:
- `config.py`: Pattern definitions and position constraints
- `models/syntax_mask.py`: Mask generation logic
- `data/dataset.py`: Text validation and layout inference

## Syntax Mask

The key neuro-symbolic innovation. The mask enforces valid formats dynamically based on layout prediction:

### Masking Modes

| Mode | Mask Value | Use Case |
|------|------------|----------|
| **Training** | `-100` | Soft mask for gradient stability |
| **Hard Inference** | `-inf` | Guarantees valid output (default) |
| **Soft Inference** | `-50` | Allows invalid chars if evidence is strong |

## Dataset

Designed for the RodoSol-ALPR dataset format:

```
data/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   └── ...
└── test/
    └── ...
```

Annotations format:
```json
{
  "img_0001.jpg": {
    "text": "ABC1234",
    "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "layout": "brazilian"
  }
}
```

## Known Issues & Solutions

### OCR Loss NaN During STN Stage
**Issue**: Validation shows `ocr: nan` during STN training stage.  
**Cause**: Pretrained PARSeq produces logits with unmapped vocabulary positions initialized to -100, causing CrossEntropyLoss to produce NaN.  
**Solution**: Stage-aware validation now uses stage-specific losses, avoiding OCR loss computation during STN stage.

### Gradient Explosion in STN
**Issue**: NaN loss values after several epochs of STN training.  
**Cause**: STN transformation parameters can explode during training.  
**Solution**: 
- Reduced STN learning rate to 5e-5 (from 1e-4)
- Tighter gradient clipping for STN stage (0.5 vs 1.0)
- Self-supervised STN loss with clamping and numerical safeguards

## Changelog

### v1.2.0 (Latest)

**Training Improvements:**
- ✅ Stage-aware validation to prevent NaN losses
- ✅ Self-supervised STN loss with numerical stability safeguards
- ✅ Improved gradient clipping per stage
- ✅ Better handling of invalid layout labels (-1)

**Model Updates:**
- ✅ Full SwinIR architecture (6 RSTB blocks, 180 embed dim)
- ✅ Pretrained PARSeq integration with charset adaptation
- ✅ Fallback to custom PARSeq if torch.hub fails

**Bug Fixes:**
- 🐛 Fixed OCR loss NaN during STN stage validation
- 🐛 Fixed loss accumulation skipping NaN values

### v1.3.0 (Current)

**LCOFL Paper Implementation:**
- ✅ LCOFL Loss with 4 components (classification, layout penalty, SSIM, confusion tracking)
- ✅ Deformable Convolution v2 module
- ✅ PLTFAM-style Shared Attention Module
- ✅ Confusion matrix tracking during validation
- ✅ Dynamic character weight updates based on confusion pairs

### v1.1.0

**Training Improvements:**
- ✅ Mixed precision training (AMP) for 2x faster training
- ✅ EMA (Exponential Moving Average) for stable model weights
- ✅ Early stopping with configurable patience
- ✅ Learning rate warmup

**Model Enhancements:**
- ✅ Attention-enhanced layout classifier
- ✅ Soft inference constraints for damaged plates

**New Loss Functions:**
- ✅ `OCRAwarePerceptualLoss`
- ✅ `CharacterFocusLoss`
- ✅ `MultiScaleOCRLoss`

### v1.0.0

- Initial implementation of the Neuro-Symbolic LPR system
- 4-phase pipeline: STN → Layout/Fusion → SwinIR → PARSeq
- Syntax-masked recognition for valid plate outputs
- Staged training schedule

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{neurosymboliclpr2026,
  title={Neuro-Symbolic License Plate Recognition for Brazilian Plates},
  author={Your Name},
  booktitle={ICPR 2026},
  year={2026}
}
```

## License

MIT License
