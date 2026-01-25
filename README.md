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
- **OCR-as-Discriminator**: Uses OCR confidence as GAN discriminator for stable training
- **Deformable Convolutions**: Adaptive spatial sampling for better character handling
- **Shared Attention Module**: PLTFAM-style attention with shared weights across blocks
- **Mixed Precision Training**: 2x faster training with automatic mixed precision (AMP)
- **EMA Model Averaging**: Exponential moving average for more stable final models
- **Stage-Aware Validation**: Prevents NaN losses during staged training
- **Soft Inference Mode**: Robust handling of damaged/non-standard plates

## Architecture

The system implements a **4-phase end-to-end pipeline** for low-resolution license plate recognition, combining deep learning with symbolic reasoning.

### High-Level Overview

```
Input: 5 LR Frames (16×48×3 each)
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Feature Extraction & Geometric Alignment           │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │ SharedCNN   │──▶──│    STN      │  Rectified Features    │
│  │ Encoder     │     │ (Affine)    │  (B,T,512,4,12)        │
│  └─────────────┘     └──────┬──────┘                        │
└──────────────────────────────┼──────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: Classification & Frame Fusion                      │
│  ┌─────────────┐     ┌─────────────────────────┐            │
│  │  Layout     │     │ Quality Scorer + Fusion │            │
│  │ Classifier  │     │ (Weighted Average)      │            │
│  └──────┬──────┘     └───────────┬─────────────┘            │
│         │                        │                          │
│    Layout Prob              Fused Feature                   │
│   (Brazilian/               (B,512,4,12)                    │
│    Mercosul)                     │                          │
└─────────────────────────────────┬┼──────────────────────────┘
                               │  │
                               ▼  ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Super-Resolution (4× Upscaling)                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              SwinIR Generator                        │    │
│  │  ┌─────────────────────────────────────────────────┐│    │
│  │  │ 6 RSTB Blocks (each with Shared Attention)      ││    │
│  │  │ • Swin Transformer Layers (window=8)            ││    │
│  │  │ • 180 embed dim, 6 heads                        ││    │
│  │  │ • Deformable Conv support                       ││    │
│  │  └─────────────────────────────────────────────────┘│    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │                                  │
│                     HR Image (64×192×3)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Recognition & Neuro-Symbolic Decoding              │
│  ┌─────────────┐     ┌─────────────────────────────────┐    │
│  │ PARSeq      │──▶──│     Syntax Mask Layer           │    │
│  │ (Pretrained)│     │ (Dynamic Position Constraints)  │    │
│  └─────────────┘     └───────────────┬─────────────────┘    │
│                                      │                       │
│                              Masked Logits (B,7,39)          │
└──────────────────────────────────────┼──────────────────────┘
                                       │
                                       ▼
                         Output: Plate Text (7 chars)
                         Example: "ABC1234" or "ABC1D23"
```

---

### Detailed Component Specifications

#### Phase 1: Feature Extraction & Geometric Alignment

| Component | Architecture | Input → Output |
|-----------|--------------|----------------|
| **SharedCNNEncoder** | 4 Conv blocks (64→128→256→512 channels), each with Conv3×3 + BN + ReLU + MaxPool2×2 | `(B,T,3,16,48)` → `(B,T,512,4,12)` |
| **SpatialTransformerNetwork** | Localization CNN + FC → 6 affine params, then `grid_sample` | `(B,T,512,4,12)` → Rectified `(B,T,512,4,12)` |
| **CornerPredictor** | GAP + FC(512→256→8) with tanh activation | `(B,T,512,4,12)` → `(B,T,4,2)` corners |

**STN Transformation Constraints (tanh-bounded):**
| Parameter | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| Scale | `1.0 + 0.5 × tanh(x)` | [0.5, 1.5] | Uniform for x and y (rectangular output) |
| Shear | `0.1 × tanh(x)` | [-0.1, 0.1] | Minimal shear to prevent parallelogram |
| Translation | `0.5 × tanh(x)` | [-0.5, 0.5] | Bounded to keep plate in frame |

---

#### Phase 2: Classification & Frame Fusion

| Component | Architecture | Input → Output |
|-----------|--------------|----------------|
| **LayoutClassifier** | GAP + FC(512→256→2) with optional attention | `(B,T,512,4,12)` → `(B,2)` logits |
| **QualityScorerFusion** | Per-frame quality MLP + softmax → weighted average | `(B,T,512,4,12)` → `(B,512,4,12)` |

**Layout Classification:**
- Predicts Brazilian (class 0) vs Mercosul (class 1)
- Optional attention mechanism for difficult cases
- Output probability determines which syntax mask to apply

**Quality-Weighted Fusion:**
```
quality_score[t] = sigmoid(MLP(GAP(features[t])))
weights = softmax(quality_scores)
fused = Σ(weights[t] × features[t])
```

---

#### Phase 3: Super-Resolution (SwinIR Generator)

**Architecture Configuration:**
```python
SwinIRGenerator(
    in_channels=512,        # From encoder (feature space SR)
    out_channels=3,         # RGB output
    embed_dim=180,          # Transformer embedding dimension
    depths=[6,6,6,6,6,6],   # 6 RSTB blocks
    num_heads=[6,6,6,6,6,6],# 6 attention heads per block
    window_size=8,          # Swin Transformer window size
    upscale=4,              # 4× upscaling (16×48 → 64×192)
    use_shared_attention=True,   # PLTFAM-style attention
    use_deformable=True          # Deformable conv in attention
)
```

**RSTB Block Structure:**
```
Input (B, L, C)
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  │
┌────────────────────────────┐         │
│ Swin Transformer Layers    │         │
│ (depth=6, window=8)        │         │
└─────────────┬──────────────┘         │
              │                        │
              ▼                        │
┌────────────────────────────┐         │
│ Conv 3×3 (residual path)   │         │
└─────────────┬──────────────┘         │
              │                        │
              ▼                        │
┌────────────────────────────┐         │
│ Shared Attention Module    │         │
│ (Channel + Positional +    │         │
│  Geometrical Attention)    │         │
└─────────────┬──────────────┘         │
              │                        │
              └────────┬───────────────┘
                       │ (Residual Add)
                       ▼
              Output (B, L, C)
```

**Shared Attention Module:**
| Component | Architecture | Purpose |
|-----------|--------------|---------|
| Channel Attention | GAP → FC → sigmoid | Channel-wise recalibration |
| Positional Attention | Conv → sigmoid | Spatial position weighting |
| Geometrical Attention | 3×3 Deformable Conv + sigmoid | Adaptive spatial sampling |

---

#### Phase 4: Recognition & Neuro-Symbolic Decoding

| Component | Architecture | Input → Output |
|-----------|--------------|----------------|
| **PretrainedPARSeq** | ViT encoder (12 layers, 384 dim) + Transformer decoder | `(B,3,64,192)` → `(B,7,39)` logits |
| **SyntaxMaskLayer** | Dynamic masking based on layout + position | Masks invalid characters per position |

**PARSeq OCR:**
- Uses pretrained weights from [baudm/parseq](https://github.com/baudm/parseq)
- Trained on scene text datasets (MJSynth, SynthText, etc.)
- Charset adapted from pretrained → our 36-char vocabulary
- Falls back to custom implementation if pytorch_lightning unavailable

**Syntax Mask (Neuro-Symbolic Integration):**
```
Position:    0   1   2   3   4   5   6
Brazilian:  [L] [L] [L] [N] [N] [N] [N]   (L=letter, N=number)
Mercosul:   [L] [L] [L] [N] [L] [N] [N]
```

| Mode | Invalid Char Value | Use Case |
|------|-------------------|----------|
| Training | -100 | Soft masking for stable gradients |
| Hard Inference | -∞ | Guarantees valid format output |
| Soft Inference | -50 | Allows exceptions for damaged plates |

---

### GAN Training Architecture

**Discriminators (Stage 2 & 3):**

| Discriminator | Purpose | Loss Type |
|---------------|---------|-----------|
| **PatchDiscriminator** | Binary real/fake classification | LSGAN (MSE) |
| **OCR Discriminator** | Recognition-based quality | Confidence + CE |

**PatchDiscriminator Architecture:**
```python
PatchDiscriminator(
    in_channels=3,
    base_channels=64,
    num_layers=3,     # 64 → 128 → 256 → 512
    use_spectral_norm=True
)
# Output: (B, 1, H/8, W/8) patch scores
```

**OCR Discriminator:**
- Uses the model's own PARSeq recognizer
- Measures recognition confidence on generated images
- Skipped when `ocr_real_conf < 10%` (untrained recognizer)

---

### Training Stages

| Stage | Name | Modules Trained | Modules Frozen | Epochs |
|-------|------|-----------------|----------------|--------|
| 1 | STN | Encoder, STN, CornerPredictor | Generator, Recognizer | 50 |
| 2 | Restoration | Generator, LayoutClassifier, Fusion | Encoder, STN, Recognizer | 100 |
| 3 | Full | All modules | None | 50 |

**Loss Functions per Stage:**

| Stage | Losses | Formula |
|-------|--------|---------|
| **STN** | Self-supervised + Pixel | `L_identity + L_consistency + L_smoothness + L_pixel` |
| **Restoration** | Pixel + GAN + Layout + OCR Guidance | `L_pixel + 0.1×L_GAN + 0.1×L_layout + L_ocr_guidance` |
| **Full** | All | `L_pixel + 0.1×L_GAN + 0.5×L_OCR + 0.1×L_geo + 0.1×L_layout` |

---

### Data Flow Summary

```
Input:  (B, 5, 3, 16, 48)     # 5 LR frames, 16×48 RGB each
          ↓
Encoder:  (B, 5, 512, 4, 12)  # Feature maps
          ↓
STN:      (B, 5, 512, 4, 12)  # Rectified features
          ↓
Layout:   (B, 2)              # Brazilian/Mercosul logits
          ↓
Fusion:   (B, 512, 4, 12)     # Quality-weighted single feature
          ↓
SwinIR:   (B, 3, 64, 192)     # Super-resolved HR image (4× upscale)
          ↓
PARSeq:   (B, 7, 39)          # Raw logits (7 positions × 39 vocab)
          ↓
Mask:     (B, 7, 39)          # Valid chars only (position-constrained)
          ↓
Output:   ["ABC1234", ...]    # Decoded plate strings
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
| Restoration | Generator, Layout, Fusion | Pixel, GAN, Layout | 32 |
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
│   ├── lcofl_loss.py         # LCOFL loss with SSIM and confusion tracking
│   └── ocr_discriminator.py  # OCR-as-Discriminator for stable GAN training
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

## Reproducibility & Determinism

By default, this codebase enables **strict determinism** for exact reproducibility across training runs. This is controlled by `seed_everything()` in `train.py`.

### What's Enabled by Default

| Setting | Value | Purpose |
|---------|-------|---------|
| `torch.backends.cudnn.deterministic` | `True` | Force deterministic cuDNN algorithms |
| `torch.backends.cudnn.benchmark` | `False` | Disable cuDNN autotuning |
| `torch.use_deterministic_algorithms` | `True` | Error on non-deterministic ops |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Deterministic cuBLAS |
| TF32 | Disabled | Exact float32 precision |
| `PYTHONHASHSEED` | Set to seed | Deterministic Python hashing |

### Performance Impact

Strict determinism typically has a **5-15% performance overhead** compared to non-deterministic training. This is the tradeoff for exact reproducibility.

To disable strict determinism for faster training (at the cost of reproducibility):

```python
# In your training script
seed_everything(42, strict_determinism=False)
```

### Deterministic Inference

Inference is deterministic by default. The `--frame-noise-std` flag (default: `0.0`) controls whether extra frames have random noise added:

```bash
# Deterministic inference (default)
python inference.py --model checkpoint.pth --input image.jpg

# With frame augmentation (non-deterministic)
python inference.py --model checkpoint.pth --input image.jpg --frame-noise-std 0.01
```

## Checkpoint Security

**Warning:** This codebase uses `torch.load()` for checkpoint loading, which internally uses Python's `pickle` module. Pickle can execute arbitrary code during deserialization.

### Security Guidelines

1. **Only load checkpoints from trusted sources** - Never load `.pth` files from untrusted origins
2. **Verify checkpoint integrity** - Use checksums (SHA-256) when downloading checkpoints
3. **Local files only** - The loading functions reject non-file paths (e.g., URLs)

When loading a checkpoint, you'll see a security warning:

```
UserWarning: Loading checkpoint from 'path/to/model.pth'. 
torch.load() uses pickle which can execute arbitrary code. 
Only load checkpoints from trusted sources.
```

### Future Improvements

For enhanced security, consider migrating to [safetensors](https://github.com/huggingface/safetensors) format, which is a safe-by-design serialization format that doesn't support code execution.

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

### v1.5.0 (Latest)

**Stage 3 Anti-Collapse Training:**
- ✅ Frozen OCR for LCOFL classification - prevents mode collapse by using a frozen copy of PARSeq for loss computation
- ✅ SR Anchor Loss - anchors Stage 3 output to Stage 2 for visual quality preservation
- ✅ GAN disabled in Stage 3 - using OCR-only discriminator approach from original LCOFL paper
- ✅ LCOFL classification active from start (no curriculum) - matches original paper configuration

**Configuration Updates:**
- ✅ New config parameter: `use_frozen_ocr_for_lcofl` (default: True)
- ✅ LCOFL weight increased to 0.75 (matching original paper)
- ✅ Edge loss weight set to 0.5 for sharper character boundaries
- ✅ Stage 3 OCR parameters: warmup 6000 steps, ramp 6000 steps, max weight 0.5

**Documentation:**
- ✅ Updated AGENTS.md with Stage 3 training patterns
- ✅ Added frozen OCR integration guidance

### v1.4.0

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

### v1.4.0 (Current)

**GAN Training Stability Fixes:**
- 🐛 Fixed LSGAN mismatch: Generator used MSE but Discriminator used BCE
- 🐛 Fixed warm-up period: GAN loss now completely disabled during warm-up
- 🐛 Fixed GAN weight cap: Was capped at 0.01 instead of config value 0.05
- 🐛 Fixed R1 penalty: Reduced from 5.0 to 1.0 for LSGAN stability  
- 🐛 Fixed validation loss: Was returning 0.0 due to NaN check on all outputs
- ✅ Added `--reset-epoch` flag for resuming training with epoch reset
- ✅ Increased warm-up epochs from 5 to 10

**Documentation:**
- ✅ Updated AGENTS.md with known issues and unintegrated features
- ✅ Clarified features implemented but not yet integrated:
  - OCR-as-Discriminator (`losses/ocr_discriminator.py`) - NOT integrated
  - Deformable Conv (`models/deformable_conv.py`) - NOT integrated  
  - Shared Attention (`models/shared_attention.py`) - NOT integrated

### v1.3.0

**LCOFL Paper Implementation:**
- ✅ LCOFL Loss with 4 components (classification, layout penalty, SSIM, confusion tracking)
- ✅ Deformable Convolution v2 module (implemented, not integrated)
- ✅ PLTFAM-style Shared Attention Module (implemented, not integrated)
- ✅ OCR-as-Discriminator module (implemented, not integrated)
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

### v1.1.1 (2026-01-19)

**Bug Fixes:**
- Fixed PARSeq pretrained model staying in training mode → random outputs
- Fixed PARSeq charset adapter overwriting lowercase with uppercase logits
- Fixed VGG perceptual loss feature extraction feeding wrong shapes
- Fixed perceptual loss not being computed in restoration stage
- Enabled perceptual loss in Stage 2 training (weight=0.1)

**Improvements:**
- Added TensorBoard visualizations for STN, LR images, and OCR predictions
- Added OCR confidence logging per sample

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
