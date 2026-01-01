# Neuro-Symbolic LPR System

A sophisticated Automatic License Plate Recognition (ALPR) system for Brazilian license plates, combining deep neural networks with symbolic reasoning.

## Overview

This system recognizes Brazilian license plates in two formats:
- **Brazilian (Old)**: `LLL-NNNN` (e.g., ABC-1234)
- **Mercosul (New)**: `LLLNLNN` (e.g., ABC1D23)

### Key Features

- **Multi-frame Processing**: Processes 5 LR frames for robust recognition
- **Spatial Transformer Network**: Corrects geometric distortions with corner-supervised alignment
- **Layout Classification**: Automatically detects Brazilian vs Mercosul format
- **GAN Super-Resolution**: SwinIR-based image restoration
- **Syntax-Masked Recognition**: Enforces valid plate formats using symbolic constraints

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

The system uses a staged training schedule:

```bash
# Train all stages
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

# Load model
device = torch.device('cuda')
model = load_model('checkpoints/best.pth', device)

# Run inference
image = np.array(Image.open('plate.jpg').convert('RGB'))
text, confidence, sr_image, is_mercosul = predict_single(model, image, device)

print(f"Plate: {text}")
print(f"Format: {'Mercosul' if is_mercosul else 'Brazilian'}")
print(f"Confidence: {confidence:.2%}")
```

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
│   └── composite_loss.py     # Combined training loss
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loaders
│   └── augmentation.py       # Augmentation pipeline
│
└── utils/
    ├── __init__.py
    └── visualization.py      # Visualization tools
```

## Loss Function

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

## Syntax Mask

The key neuro-symbolic innovation. The mask enforces valid formats:

```python
# Brazilian: LLL-NNNN
# Position: 0  1  2  3  4  5  6
# Type:     L  L  L  N  N  N  N

# Mercosul: LLLNLNN  
# Position: 0  1  2  3  4  5  6
# Type:     L  L  L  N  L  N  N
```

Invalid characters at each position are masked with `-inf` (or `-100` during training for gradient stability).

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
