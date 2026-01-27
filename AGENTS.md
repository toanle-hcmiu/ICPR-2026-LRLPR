# AGENTS.md - Text-Prior Guided LPR System

This document provides guidelines for AI coding agents (Claude, GPT, Copilot, etc.) working on this codebase.

## Project Overview

**Text-Prior Guided License Plate Recognition (LPR) System** for Brazilian license plates, targeting ICPR 2026.

### Core Concept

Uses frozen PARSeq to provide **text prior guidance** during training:

- **Training Mode**: Frozen PARSeq extracts character probabilities → guides generator via text prior loss
- **Inference Mode**: Pure LR→HR super-resolution (no PARSeq needed)
- **Key Innovation**: Text prior provides categorical constraints preventing mode collapse

### Target Formats
- **Brazilian**: `LLLNNNN` (3 letters + 4 digits)
- **Mercosul**: `LLLNLNN` (3 letters + digit + letter + 2 digits)

## TP Architecture

```
LR Frames (5×16×48) → Encoder → STN → Layout+Fusion → TP Generator → HR (64×192)
                                                              ↓
                                                      Text Prior (training only)
                                                              ↓
                                                      PARSeq (frozen)
```

### Key Components

| Module | File | Purpose |
|--------|------|---------|
| TextPriorGuidedLPR | `models/tp_lpr.py` | Complete TP-guided system |
| TextPriorGuidedGenerator | `models/tp_guided_generator.py` | TP generator with BiLSTM blocks |
| TextPriorExtractor | `models/text_prior.py` | Text prior from frozen PARSeq |
| SequentialResidualBlock | `models/sequential_blocks.py` | BiLSTM sequential modeling |
| InterFrameCrossAttention | `models/multi_frame_fusion.py` | Multi-frame temporal fusion |
| SharedCNNEncoder | `models/encoder.py` | Shared CNN feature extraction |
| SpatialTransformerNetwork | `models/stn.py` | Geometric rectification |
| LayoutClassifier | `models/layout_classifier.py` | Brazilian vs Mercosul detection |
| PARSeq | `models/parseq.py` | Character recognition (pretrained, frozen) |
| CompositeLoss | `losses/composite_loss.py` | Combined training loss |
| LCOFL Loss | `losses/lcofl_loss.py` | Layout-aware character-oriented loss |

## Critical Constraints

### Hardcoded Values (DO NOT CHANGE without updating all references)

```python
# In config.py - Referenced throughout codebase
PLATE_LENGTH = 7
VOCAB_SIZE = 39  # 36 chars + 3 special tokens
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
CHAR_START_IDX = 3
```

### Image Sizes
```python
LR_SIZE = (16, 48)   # Low-resolution input
HR_SIZE = (64, 192)  # High-resolution output (4× upscale)
```

## TP Training Pipeline

### Three Training Stages

```
Stage 1 (tp_stn):     STN + Layout training (15 epochs)
Stage 2 (tp_generator): Generator with text prior (50 epochs)
Stage 3 (tp_full):     End-to-end fine-tuning (30 epochs)
```

### Stage-Specific Behavior

| Stage | Frozen Modules | Active Modules | Loss Functions |
|-------|----------------|----------------|----------------|
| tp_stn | Generator, Recognizer | STN, Layout | Self-supervised, Pixel, Corner, Layout |
| tp_generator | STN, Recognizer | Generator | Pixel, Text Prior, Layout, SSIM |
| tp_full | None | All | All losses (reduced OCR weight) |

### Usage

```bash
# Train STN + Layout
python train.py --stage 1

# Train Generator with text prior
python train.py --stage 2

# Full end-to-end fine-tuning
python train.py --stage 3

# Train all stages
python train.py --stage all

# Also supports explicit TP stage names
python train.py --stage tp_stn
python train.py --stage tp_generator
python train.py --stage tp_full
```

## Code Patterns

### TP Generator Forward

```python
# Training: use text prior guidance
hr_image = model(lr_frames, use_text_prior=True)

# Inference: no text prior needed
hr_image = model(lr_frames, use_text_prior=False)
```

### Model Forward Pass Pattern

```python
def forward(self, x, targets=None, return_intermediates=False):
    outputs = {}
    # Encode multi-frame LR
    encoded = self.encoder.forward_multi_frame(x)
    # Rectify with STN
    rectified, thetas, corners = self.stn(encoded, ...)
    # Classify layout
    layout_logits = self.layout_classifier(rectified)
    # Fuse multi-frame features
    fused = self.multi_frame_fusion(rectified)
    # Generate HR (with text prior if training)
    text_prior = None
    if self.training and use_text_prior:
        text_prior, text_features = self.text_prior_extractor(x)
    hr_image = self.generator(fused, text_features=text_features)
    # Recognize text
    logits = self.recognizer(hr_image)
    return {**outputs, 'hr_image': hr_image, 'masked_logits': logits, ...}
```

### Loss Computation Pattern

```python
def forward(self, outputs, targets, discriminator=None):
    loss_dict = {}
    total_loss = None

    # Check for NaN inputs before computing each loss
    if 'hr_image' in outputs and 'hr_image' in targets:
        if not torch.isnan(outputs['hr_image']).any():
            l_pixel = self.pixel_loss(...)
            l_pixel = self._clamp_loss(l_pixel)
            loss_dict['pixel'] = self._safe_loss_item(l_pixel)
            total_loss = weighted if total_loss is None else total_loss + weighted

    # Text prior loss (training only)
    if self.text_prior_loss is not None and 'hr_image' in outputs:
        l_tp = self.text_prior_loss(outputs['hr_image'], targets['text_indices'])
        loss_dict['tp_text_prior'] = self._safe_loss_item(l_tp)
        weighted = self.weights['tp_text_prior'] * l_tp
        total_loss = weighted if total_loss is None else total_loss + weighted

    return total_loss, loss_dict
```

### Numerical Stability Pattern

```python
def _clamp_loss(self, loss, max_val=100.0):
    if loss is None:
        return None
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return loss * 0.0  # Zero with gradient connection
    return torch.clamp(loss, min=0.0, max=max_val)
```

## Common Tasks

### Adding a New Loss Function

1. Create loss class in `losses/` directory
2. Import in `losses/__init__.py`
3. Integrate in `losses/composite_loss.py`
4. Add weight to `TrainingConfig` in `config.py`

### Modifying the Model

1. Edit component in `models/` directory
2. Update `models/__init__.py` if new module
3. Integrate in `models/tp_lpr.py`
4. Update `get_trainable_params()` for staged training

### Using LCOFL Loss

```python
# In config.py
config.training.use_lcofl = True      # Enable LCOFL
config.training.weight_lcofl = 0.5    # Weight for LCOFL
config.training.weight_ssim = 0.3     # SSIM component weight
config.training.lcofl_alpha = 1.0     # Confusion penalty increment
config.training.lcofl_beta = 2.0      # Layout violation penalty
```

## Known Issues & Gotchas

### 1. PARSeq Pretrained Model Lazy Loading

**Behavior**: `PretrainedPARSeq._load_model()` is called lazily on first forward
**Gotcha**: Model may not be on correct device until first forward pass
**Solution**: Always pass device to `_load_model(device=x.device)`

### 2. Multi-Frame STN Output Dimensions

**Shape**: `(B, T, 2, 3)` for thetas, `(B, T, 4, 2)` for corners
**Gotcha**: Some losses expect `(B, 2, 3)` or `(B, 4, 2)`
**Solution**: Average over frames when needed: `corners.mean(dim=1)`

### 3. Layout Label Invalid Values

**Value**: `-1` indicates invalid/unknown layout
**Gotcha**: Don't include in loss computation
**Solution**: Filter with `valid_mask = (layout >= 0)`

### 4. Text Indices Format

**Shape**: `(B, PLATE_LENGTH + 2)` with `[BOS, char1, ..., char7, EOS]`
**Gotcha**: Logits are `(B, PLATE_LENGTH, VOCAB_SIZE)` - no BOS/EOS
**Solution**: Slice targets: `targets[:, 1:PLATE_LENGTH + 1]`

### 5. Text Prior vs Inference

**Training**: Set `use_text_prior=True` to enable PARSeq guidance
**Inference**: Set `use_text_prior=False` - PARSeq not needed
**Gotcha**: Don't forget to toggle between modes

### 6. Frozen PARSeq in Text Prior

**Behavior**: `TextPriorExtractor` freezes PARSeq completely
**Gotcha**: No gradients flow through PARSeq (intentional)
**Solution**: Only projection layers are trainable

## Reproducibility & Determinism

### Strict Determinism (Default)

The codebase enables strict determinism by default via `seed_everything()` in `train.py`.

```python
# Default - strict determinism enabled
seed_everything(42)
```

### What Strict Determinism Enables

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG = ':4096:8'`
- TF32 disabled (exact float32)
- `PYTHONHASHSEED` set to seed

## Checkpoint Security

### Critical Security Rules

1. **NEVER load checkpoints from untrusted sources** - `torch.load()` uses pickle
2. **NEVER accept checkpoint paths from user input without validation**
3. **Always validate checkpoint paths are local files**

## Testing Changes

### Quick Validation
```bash
# Run training for 1 epoch to check for errors
python train.py --stage 1 --epochs 1
```

### Check for NaN
- Monitor validation logs for `nan` values in losses
- Check `train.py` NaN batch detection warnings

## Dependencies

### Required
- `torch>=2.0.0` - Core framework
- `timm>=0.9.0` - For PARSeq pretrained models
- `einops>=0.7.0` - Tensor operations

### For Pretrained PARSeq
- Requires internet on first run (torch.hub download)
- Falls back to custom implementation if download fails

## Verification

After training, verify:

1. **Character accuracy** >90%
2. **All characters represented** (no mode collapse)
3. **Inference works** without text input
4. **Visual quality** preserved (sharp characters)

## Contact

For questions about this codebase, refer to:
1. This AGENTS.md file
2. README.md for usage
3. Inline docstrings in each module
4. `config.py` for all hyperparameters and constants
