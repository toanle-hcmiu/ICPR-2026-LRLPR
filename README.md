# Neuro-Symbolic LPR System

A sophisticated Automatic License Plate Recognition (ALPR) system for Brazilian license plates, combining deep neural networks with symbolic reasoning.

## Overview

This system recognizes Brazilian license plates in two formats:
- **Brazilian (Old)**: `LLL-NNNN` (e.g., ABC-1234)
- **Mercosul (New)**: `LLLNLNN` (e.g., ABC1D23)

### Key Features

- **Multi-frame Processing**: Processes 5 LR frames for robust recognition
- **Spatial Transformer Network**: Corrects geometric distortions with corner-supervised alignment
- **Layout Classification**: Automatically detects Brazilian vs Mercosul format (with optional attention mechanism)
- **GAN Super-Resolution**: SwinIR-based image restoration with OCR-aware perceptual loss
- **Syntax-Masked Recognition**: Enforces valid plate formats using symbolic constraints
- **Mixed Precision Training**: 2x faster training with automatic mixed precision (AMP)
- **EMA Model Averaging**: Exponential moving average for more stable final models
- **Soft Inference Mode**: Robust handling of damaged/non-standard plates

## Architecture

```
Input (5 LR Frames)
       │
       ▼
┌─────────────────┐
│ Shared CNN      │  Phase 1: Feature Extraction
│ Encoder         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spatial         │  Phase 1: Geometric Alignment
│ Transformer Net │  (Corner Loss Supervision)
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
    │  │ SwinIR          │  Phase 3: Super-Resolution
    │  │ Generator (GAN) │
    │  └────────┬────────┘
    │           │
    │           ▼
    │  ┌─────────────────┐
    │  │ PARSeq          │  Phase 4: Recognition
    │  │ Recognizer      │
    │  └────────┬────────┘
    │           │
    ▼           ▼
┌─────────────────────────┐
│ Syntax Mask Layer       │  Neuro-Symbolic Integration
│ (Dynamic Constraints)   │
└────────────┬────────────┘
             │
             ▼
      Plate Text Output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ICPR-2026-LRLPR.git
cd ICPR-2026-LRLPR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

The system uses a staged training schedule with advanced training features:

```bash
# Train all stages (with defaults: AMP enabled, EMA enabled)
python train.py --data-dir data/ --output-dir outputs/

# Train specific stage
python train.py --stage 1 --data-dir data/  # STN only
python train.py --stage 3 --resume checkpoints/step2.pth  # Fine-tune

# Training stages:
# 0: Synthetic pre-training (PARSeq)
# 1: Geometry warm-up (STN)
# 2: Restoration + Layout (SwinIR, Classifier)
# 3: End-to-end fine-tuning (All)
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
| **Gradient Clipping** | 1.0 | Prevents gradient explosion |

#### Metrics Tracked

- **Plate Accuracy**: Exact match (all 7 characters correct)
- **Character Accuracy**: Per-character accuracy
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
)
```

#### Soft Inference Mode

By default, the syntax mask uses hard constraints (`-inf`) during inference, guaranteeing syntactically valid outputs. Enable `soft_inference=True` to use softer penalties (`-50.0`), allowing the model to predict "invalid" characters if visual evidence is overwhelming. This improves robustness to:

- Damaged or worn plates
- Non-standard custom plates
- Manufacturing defects

## Project Structure

```
ICPR-2026-LRLPR/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
│
├── models/
│   ├── __init__.py
│   ├── neuro_symbolic_lpr.py # Main model (4-phase pipeline)
│   ├── encoder.py            # Shared CNN encoder
│   ├── stn.py                # Spatial Transformer Network
│   ├── layout_classifier.py  # Layout Switch branch
│   ├── feature_fusion.py     # Quality scorer & fusion
│   ├── swinir.py             # SwinIR generator
│   ├── discriminator.py      # PatchGAN discriminator
│   ├── parseq.py             # PARSeq recognizer
│   └── syntax_mask.py        # Dynamic syntax mask
│
├── losses/
│   ├── __init__.py
│   ├── corner_loss.py        # STN corner supervision
│   ├── gan_loss.py           # Adversarial losses
│   ├── composite_loss.py     # Combined training loss
│   └── ocr_perceptual_loss.py # OCR-aware perceptual losses
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loaders
│   └── augmentation.py       # Augmentation pipeline
│
└── utils/
    ├── __init__.py
    ├── ema.py                # Exponential Moving Average
    ├── early_stopping.py     # Early stopping utility
    └── visualization.py      # Visualization tools
```

## Loss Functions

### Composite Training Loss

The total loss for end-to-end training:

```
L_total = L_pixel + 0.1 × L_GAN + 0.5 × L_OCR + 0.1 × L_geo
```

| Loss | Weight | Description |
|------|--------|-------------|
| L_pixel | 1.0 | L1 pixel reconstruction loss |
| L_GAN | 0.1 | Adversarial loss for realism |
| L_OCR | 0.5 | Character recognition loss |
| L_geo | 0.1 | Corner loss for STN |

### OCR-Aware Perceptual Losses

Additional losses for enhanced super-resolution (in `losses/ocr_perceptual_loss.py`):

| Loss Class | Description |
|------------|-------------|
| `OCRAwarePerceptualLoss` | Uses downstream OCR model to guide restoration. The generator learns to produce images optimized for OCR accuracy, not just visual similarity. |
| `CharacterFocusLoss` | Edge-aware loss using Sobel operators. Emphasizes character boundaries and high-frequency details critical for recognition. |
| `MultiScaleOCRLoss` | Evaluates OCR at multiple scales (1.0×, 0.5×, 0.25×) for robust recognition across different viewing distances. |

#### Using OCR-Aware Loss

```python
from losses import OCRAwarePerceptualLoss

# Wrap your OCR model for perceptual loss
ocr_loss = OCRAwarePerceptualLoss(
    ocr_model=recognizer,
    loss_type='cross_entropy',
    weight=1.0,
    freeze_ocr=True  # Freeze OCR weights
)

# In training loop
loss = ocr_loss(generated_hr, target_text)
```

## Plate Format Specifications (Hardcoded)

The system recognizes two Brazilian license plate formats with **hardcoded syntax rules**:

### Brazilian Format (Old)
- **Pattern**: `LLL-NNNN` (displayed with dash) or `LLLNNNN` (stored without dash)
- **Regex**: `^[A-Z]{3}[0-9]{4}$`
- **Position Constraints** (hardcoded in `config.py`):
  ```
  Position:  0  1  2  3  4  5  6
  Type:      L  L  L  N  N  N  N
  ```
  - Positions 0-2: **Letters** (A-Z)
  - Positions 3-6: **Digits** (0-9)
- **Example**: `ABC1234` → `ABC-1234`

### Mercosul Format (New)
- **Pattern**: `LLLNLNN` (no dash)
- **Regex**: `^[A-Z]{3}[0-9][A-Z][0-9]{2}$`
- **Position Constraints** (hardcoded in `config.py`):
  ```
  Position:  0  1  2  3  4  5  6
  Type:      L  L  L  N  L  N  N
  ```
  - Positions 0, 1, 2, 4: **Letters** (A-Z)
  - Positions 3, 5, 6: **Digits** (0-9)
- **Example**: `ABC1D23` → `ABC1D23`

### Hardcoded Constants

All format specifications are **hardcoded** in `config.py`:

```python
# Plate length (without dash)
PLATE_LENGTH = 7

# Regex patterns
BRAZILIAN_PATTERN = re.compile(r'^[A-Z]{3}[0-9]{4}$')
MERCOSUL_PATTERN = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')

# Position constraints function
def get_position_constraints(is_mercosul: bool) -> List[str]:
    if is_mercosul:
        return ['L', 'L', 'L', 'N', 'L', 'N', 'N']  # Mercosul
    else:
        return ['L', 'L', 'L', 'N', 'N', 'N', 'N']  # Brazilian
```

**Important**: These formats are **hardcoded** and cannot be changed without modifying:
- `config.py`: Pattern definitions and position constraints
- `models/syntax_mask.py`: Mask generation logic (lines 317, 320, 349, 352)
- `data/dataset.py`: Text validation and layout inference

## Syntax Mask

The key neuro-symbolic innovation. The mask enforces valid formats using the **hardcoded constraints** above:

```python
# Brazilian: LLL-NNNN (without dash: LLLNNNN)
# Position: 0  1  2  3  4  5  6
# Type:     L  L  L  N  N  N  N

# Mercosul: LLLNLNN  
# Position: 0  1  2  3  4  5  6
# Type:     L  L  L  N  L  N  N
```

### Masking Modes

| Mode | Mask Value | Use Case |
|------|------------|----------|
| **Training** | `-100` | Soft mask for gradient stability |
| **Hard Inference** | `-inf` | Guarantees valid output (default) |
| **Soft Inference** | `-50` | Allows invalid chars if evidence is strong |

### Enabling Soft Inference

```python
from models import NeuroSymbolicLPR

# For robustness to damaged/non-standard plates
model = NeuroSymbolicLPR(
    soft_inference=True,
    soft_inference_value=-50.0
)
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
└── val/
    ├── images/
    └── annotations.json
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

## Changelog

### v1.1.0 (Latest)

**Training Improvements:**
- ✅ Mixed precision training (AMP) for 2x faster training
- ✅ EMA (Exponential Moving Average) for stable model weights
- ✅ Early stopping with configurable patience
- ✅ Learning rate warmup (linear warmup before main scheduler)
- ✅ Character-level and plate-level accuracy metrics
- ✅ Comprehensive model parameter logging

**Model Enhancements:**
- ✅ Attention-enhanced layout classifier (`use_attention_layout=True`)
- ✅ Soft inference constraints for damaged plates (`soft_inference=True`)
- ✅ Input validation with clear error messages

**New Loss Functions:**
- ✅ `OCRAwarePerceptualLoss` - Optimizes SR for OCR performance
- ✅ `CharacterFocusLoss` - Edge-aware loss for character boundaries
- ✅ `MultiScaleOCRLoss` - Multi-scale OCR evaluation

**Bug Fixes:**
- 🐛 Fixed duplicate condition in `MultiFrameSTN`
- 🐛 Fixed invalid parameter in model initialization

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
