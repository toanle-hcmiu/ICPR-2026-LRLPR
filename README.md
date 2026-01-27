# Text-Prior Guided LPR System

A sophisticated Automatic License Plate Recognition (ALPR) system for Brazilian license plates, using text-prior guidance from frozen PARSeq during training. Designed for the ICPR 2026 conference.

## Overview

This system recognizes Brazilian license plates in two formats:
- **Brazilian (Old)**: `LLL-NNNN` (e.g., ABC-1234)
- **Mercosul (New)**: `LLLNLNN` (e.g., ABC1D23)

### Key Features

- **Multi-frame Processing**: Processes 5 LR frames for robust recognition
- **Spatial Transformer Network**: Corrects geometric distortions with self-supervised alignment
- **Layout Classification**: Automatically detects Brazilian vs Mercosul format
- **Text-Prior Guided Generator**: BiLSTM-based sequential blocks for character-aware super-resolution
- **Frozen PARSeq Guidance**: Text prior during training prevents mode collapse
- **Inference-Only HR Generation**: No PARSeq needed during inference (pure LR→HR)
- **Multi-Frame Fusion**: Inter-frame Cross-Attention Module (ICAM) for temporal aggregation
- **Syntax-Masked Recognition**: Enforces valid plate formats using symbolic constraints
- **LCOFL Loss**: Layout and Character Oriented Focal Loss with SSIM
- **Mixed Precision Training**: 2x faster training with automatic mixed precision (AMP)
- **EMA Model Averaging**: Exponential moving average for stable models
- **Stage-Aware Validation**: Prevents NaN losses during staged training

## Architecture

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
│  │  Layout     │     │ Inter-Frame Cross-Attn  │            │
│  │ Classifier  │     │ (ICAM Fusion)           │            │
│  └──────┬──────┘     └───────────┬─────────────┘            │
│         │                        │                          │
│    Layout Prob              Fused Feature                   │
│   (Brazilian/               (B,512,4,12)                    │
│    Mercosul)                     │                          │
└─────────────────────────────────┬┼──────────────────────────┘
                               │  │
                               ▼  ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: Text-Prior Guided Super-Resolution                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              TP Generator (BiLSTM-based)             │    │
│  │  ┌─────────────────────────────────────────────────┐│    │
│  │  │ 4 Sequential Residual Blocks (BiLSTM)           ││    │
│  │  │ • Horizontal sequential modeling (character)    ││    │
│  │  │ • Vertical sequential modeling (stroke)         ││    │
│  │  │ • Text-guided fusion (training only)           ││    │
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

### Text Prior Guidance (Training Only)

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Mode Only                        │
│  ┌─────────────┐     ┌─────────────────────────────────┐    │
│  │ LR Frames   │────▶│  Text Prior Extractor           │    │
│  │ (5×16×48)   │     │  • Frozen PARSeq                │    │
│  └─────────────┘     │  • Character probabilities       │    │
│                      │  • Spatial text features         │    │
│                      └───────────────┬─────────────────┘    │
│                                      │                       │
│                                      ▼                       │
│                              Text Prior Features             │
│                              (guides generator)             │
└─────────────────────────────────────────────────────────────┘
```

## TP Training Pipeline

### Three Training Stages

| Stage | Name | Modules Trained | Modules Frozen | Epochs |
|-------|------|-----------------|----------------|--------|
| 1 | tp_stn | STN, LayoutClassifier | Generator, Recognizer | 15 |
| 2 | tp_generator | Generator | STN, Recognizer | 50 |
| 3 | tp_full | All modules | None | 30 |

### Loss Functions per Stage

| Stage | Losses | Formula |
|-------|--------|---------|
| **tp_stn** | Self-supervised + Pixel + Layout | `L_stn + L_pixel + L_layout` |
| **tp_generator** | Pixel + Text Prior + Layout + SSIM | `L_pixel + 0.3×L_tp + 0.1×L_layout + 0.2×L_ssim` |
| **tp_full** | All (reduced OCR) | `L_pixel + 0.1×L_GAN + 0.05×L_OCR + 0.1×L_layout + 0.3×L_tp` |

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

The system uses a 3-stage training schedule:

```bash
# Train all stages
python train.py --stage all

# Train specific stage
python train.py --stage 1  # tp_stn (STN + Layout)
python train.py --stage 2  # tp_generator (Text-prior guided)
python train.py --stage 3  # tp_full (End-to-end)

# Also supports explicit TP stage names
python train.py --stage tp_stn
python train.py --stage tp_generator
python train.py --stage tp_full
```

#### Advanced Training Options

```bash
# Full training with all options
python train.py --stage all \
    --data-dir data/ \
    --output-dir outputs/ \
    --early-stopping 15

# Disable mixed precision
python train.py --no-amp --stage 1

# Resume from checkpoint
python train.py --stage 2 --resume checkpoints/tp_stn_best.pth
```

#### Training Features

| Feature | Default | Description |
|---------|---------|-------------|
| **Mixed Precision (AMP)** | Enabled | 2x faster training, lower memory |
| **EMA** | Enabled | Exponential moving average |
| **Early Stopping** | Disabled | Stop if no improvement |
| **Gradient Clipping** | 1.0 (0.5 for STN) | Prevents gradient explosion |

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
from models import TextPriorGuidedLPR
import torch

# Load model
device = torch.device('cuda')
model = TextPriorGuidedLPR(use_pretrained_parseq=True)
model.load_state_dict(torch.load('checkpoints/best.pth'))
model.to(device)
model.eval()

# Inference (no text prior needed)
lr_frames = torch.randn(1, 5, 3, 16, 48).to(device)  # 5 frames
with torch.no_grad():
    hr_image = model(lr_frames, use_text_prior=False)
    text = model.recognize(hr_image)
print(f"Plate: {text}")
```

## Project Structure

```
ICPR-2026-LRLPR/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script (3 TP stages)
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
│
├── models/
│   ├── __init__.py
│   ├── tp_lpr.py             # Text-Prior Guided LPR (main model)
│   ├── tp_guided_generator.py # TP Generator with BiLSTM blocks
│   ├── text_prior.py         # Text Prior Extractor (frozen PARSeq)
│   ├── sequential_blocks.py  # BiLSTM Sequential Residual Blocks
│   ├── multi_frame_fusion.py # Inter-frame Cross-Attention (ICAM)
│   ├── encoder.py            # Shared CNN encoder
│   ├── stn.py                # Spatial Transformer Network
│   ├── layout_classifier.py  # Layout classifier
│   ├── parseq.py             # PARSeq wrapper (pretrained)
│   └── syntax_mask.py        # Dynamic syntax mask
│
├── losses/
│   ├── __init__.py
│   ├── composite_loss.py     # Combined training loss
│   ├── lcofl_loss.py         # LCOFL loss
│   ├── gan_loss.py           # Adversarial losses
│   ├── corner_loss.py        # STN corner supervision
│   └── ssim_loss.py          # SSIM loss
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset classes
│   ├── augmentation.py       # Augmentation pipeline
│   ├── train/                # Training data
│   ├── val/                  # Validation data
│   └── test/                 # Test data
│
├── scripts/
│   └── split_dataset.py      # Dataset splitting
│
└── utils/
    ├── __init__.py
    └── visualization.py      # Visualization tools
```

## Key Components

### TextPriorGuidedGenerator

- **Sequential Residual Blocks (SRB)**: BiLSTM-based blocks for sequential character modeling
- **Text-Guided Fusion**: Incorporates text prior features during training
- **4× Upscaling**: 16×48 → 64×192

### TextPriorExtractor

- **Frozen PARSeq**: Extracts character probabilities without gradient flow
- **Spatial Features**: Projects probabilities to spatial feature maps
- **Training Only**: Not used during inference

### Multi-Frame Fusion

- **ICAM (Inter-frame Cross-Attention Module)**: Learns frame importance
- **Temporal Consistency**: Aggregates information across 5 frames

## Loss Functions

### Text Prior Loss

```python
L_tp = CrossEntropy(PARSeq(generator_output), ground_truth_text)
```

The text prior loss encourages the generator to produce images that PARSeq can recognize correctly.

### Composite Training Loss

```
L_total = L_pixel + 0.3×L_tp + 0.1×L_layout + 0.2×L_ssim
```

| Loss | Weight | Description |
|------|--------|-------------|
| L_pixel | 1.0 | L1 pixel reconstruction |
| L_tp | 0.3 | Text prior (PARSeq cross-entropy) |
| L_layout | 0.1 | Layout classification |
| L_ssim | 0.2 | Structural similarity |

## Plate Format Specifications

### Brazilian Format (Old)
- **Pattern**: `LLL-NNNN`
- **Position Constraints**: `[L, L, L, N, N, N, N]`
- **Example**: `ABC1234`

### Mercosul Format (New)
- **Pattern**: `LLLNLNN`
- **Position Constraints**: `[L, L, L, N, L, N, N]`
- **Example**: `ABC1D23`

### Hardcoded Constants

```python
PLATE_LENGTH = 7
VOCAB_SIZE = 39  # 36 chars + 3 special tokens
CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
```

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

## Reproducibility

The codebase enables **strict determinism** by default for exact reproducibility:

| Setting | Value |
|---------|-------|
| `torch.backends.cudnn.deterministic` | `True` |
| `torch.use_deterministic_algorithms` | `True` |
| TF32 | Disabled |

## Citation

```bibtex
@inproceedings{tpguidlpr2026,
  title={Text-Prior Guided License Plate Super-Resolution},
  author={Your Name},
  booktitle={ICPR 2026},
  year={2026}
}
```

## License

MIT License
