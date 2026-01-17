# AGENTS.md - AI Coding Agent Guidelines

This document provides guidelines for AI coding agents (Claude, GPT, Copilot, etc.) working on this codebase.

## Project Overview

**Neuro-Symbolic License Plate Recognition (LPR) System** for Brazilian license plates, targeting ICPR 2026.

### Core Concept
Combines deep neural networks with symbolic reasoning:
- **Neural**: CNN encoder, STN, SwinIR, PARSeq
- **Symbolic**: Syntax mask enforcing valid plate formats

### Target Formats
- **Brazilian**: `LLLNNNN` (3 letters + 4 digits)
- **Mercosul**: `LLLNLNN` (3 letters + digit + letter + 2 digits)

## Architecture Summary

```
LR Frames (5×16×48) → Encoder → STN → Layout+Fusion → SwinIR → PARSeq → Syntax Mask → Text
```

### Key Components

| Module | File | Purpose |
|--------|------|---------|
| Main Model | `models/neuro_symbolic_lpr.py` | Orchestrates 4-phase pipeline |
| Encoder | `models/encoder.py` | Shared CNN feature extraction |
| STN | `models/stn.py` | Geometric rectification |
| Layout | `models/layout_classifier.py` | Brazilian vs Mercosul detection |
| Fusion | `models/feature_fusion.py` | Quality-weighted frame fusion |
| SwinIR | `models/swinir.py` | Super-resolution (4× upscale) |
| PARSeq | `models/parseq.py` | Character recognition (pretrained) |
| Syntax Mask | `models/syntax_mask.py` | Enforces valid plate formats |
| Deformable Conv | `models/deformable_conv.py` | Adaptive spatial sampling |
| Shared Attention | `models/shared_attention.py` | PLTFAM-style attention module |
| Composite Loss | `losses/composite_loss.py` | Combined training loss |
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

### Files with Hardcoded Format Logic
- `config.py`: Pattern definitions, `get_position_constraints()`
- `models/syntax_mask.py`: Mask generation (lines ~317, 320, 349, 352)
- `data/dataset.py`: Text validation and layout inference

### Image Sizes
```python
LR_SIZE = (16, 48)   # Low-resolution input
HR_SIZE = (64, 192)  # High-resolution output (4× upscale)
```

## Training Pipeline

### Staged Training
```
Stage 0: Pretrain (SKIPPED with pretrained PARSeq)
Stage 1: STN - Geometry warm-up (encoder + STN)
Stage 2: Restoration - SwinIR + Layout + Fusion
Stage 3: Full - End-to-end fine-tuning
```

### Stage-Specific Behavior

| Stage | Frozen Modules | Loss Functions |
|-------|----------------|----------------|
| STN | Recognizer | Self-supervised, Pixel, Corner |
| Restoration | Encoder, STN, Recognizer | Pixel, GAN, Layout |
| Full | None | All losses |

### Important Training Details
- **Validation uses stage-aware loss** to prevent NaN (see `train.py:validate()`)
- **STN stage uses tighter gradient clipping** (0.5 vs 1.0)
- **Self-supervised STN loss** provides gradients without corner annotations

## Common Tasks

### Adding a New Loss Function

1. Create loss class in `losses/` directory
2. Import in `losses/__init__.py`
3. Integrate in `losses/composite_loss.py`
4. Add weight to `TrainingConfig` in `config.py`

### Modifying the Model

1. Edit component in `models/` directory
2. Update `models/__init__.py` if new module
3. Integrate in `models/neuro_symbolic_lpr.py`
4. Update `get_trainable_params()` for staged training

### Adding a Training Stage

1. Add stage config in `train.py:main()` stages list
2. Add stage-specific loss in `composite_loss.py:get_stage_loss()`
3. Update validation in `train.py:validate()` for stage-aware loss
4. Add freeze/unfreeze methods if needed

### Using LCOFL Loss (Layout and Character Oriented Focal Loss)

```python
# In config.py or training script
config.training.use_lcofl = True      # Enable LCOFL
config.training.weight_lcofl = 0.5    # Weight for LCOFL
config.training.weight_ssim = 0.3     # SSIM component weight
config.training.lcofl_alpha = 1.0     # Confusion penalty increment
config.training.lcofl_beta = 2.0      # Layout violation penalty
```

**Key files:**
- `losses/lcofl_loss.py` - LCOFL implementation
- `models/deformable_conv.py` - Deformable convolutions
- `models/shared_attention.py` - PLTFAM-style attention

## Code Patterns

### Model Forward Pass Pattern
```python
def forward(self, x, targets=None, return_intermediates=False):
    outputs = {}
    # Phase 1: Encode and rectify
    encoded = self.encoder.forward_multi_frame(x)
    rectified, thetas, corners = self.multi_frame_stn(encoded, ...)
    if return_intermediates:
        outputs['corners'] = corners
        outputs['thetas'] = thetas
    # ... more phases
    return outputs
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

## Known Issues & Gotchas

### 1. OCR Loss NaN During STN Stage
**Cause**: Pretrained PARSeq uses different charset; unmapped logits = -100  
**Solution**: Stage-aware validation skips OCR loss during STN stage

### 2. Pretrained PARSeq Lazy Loading
**Behavior**: `PretrainedPARSeq._load_model()` is called lazily on first forward  
**Gotcha**: Model may not be on correct device until first forward pass  
**Solution**: Always pass device to `_load_model(device=x.device)`

### 3. Multi-Frame STN Output Dimensions
**Shape**: `(B, T, 2, 3)` for thetas, `(B, T, 4, 2)` for corners  
**Gotcha**: Some losses expect `(B, 2, 3)` or `(B, 4, 2)`  
**Solution**: Average over frames when needed: `corners.mean(dim=1)`

### 4. Layout Label Invalid Values
**Value**: `-1` indicates invalid/unknown layout  
**Gotcha**: Don't include in loss computation  
**Solution**: Filter with `valid_mask = (layout >= 0)`

### 5. Text Indices Format
**Shape**: `(B, PLATE_LENGTH + 2)` with `[BOS, char1, ..., char7, EOS]`  
**Gotcha**: Logits are `(B, PLATE_LENGTH, VOCAB_SIZE)` - no BOS/EOS  
**Solution**: Slice targets: `targets[:, 1:PLATE_LENGTH + 1]`

### 6. GAN Training Stability
**Issues Fixed**: LSGAN mismatch, GAN weight cap, R1 penalty, warm-up  
**File**: `train.py`, `losses/composite_loss.py`  
**Details**: See commit history or walkthrough artifact for full list

## LCOFL Paper Features (Nascimento et al.)

> **INTEGRATED**: These features from "Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach" are now fully integrated.

### OCR-as-Discriminator
- **File**: `losses/ocr_discriminator.py`
- **Config**: `use_ocr_discriminator: True` (config.py line 204)
- **Status**: ✅ Integrated into `train.py:train_stage()` and `train_epoch()`
- **Purpose**: Uses OCR recognition confidence instead of binary real/fake for more stable GAN training
- **Usage**: Set `config.training.use_ocr_discriminator = True` to enable

### Shared Attention Module (PLTFAM-style)
- **File**: `models/shared_attention.py`
- **Config**: `use_shared_attention: True` (config.py line 199)
- **Status**: ✅ Integrated into `models/swinir.py` RSTB blocks
- **Purpose**: Three-fold attention (Channel, Positional, Geometrical) with shared weights across all RSTB blocks
- **Usage**: Set `config.training.use_shared_attention = True` to enable

### Deformable Convolutions
- **File**: `models/deformable_conv.py`
- **Config**: `use_deformable_conv: True` (config.py line 200)
- **Status**: ✅ Integrated via Shared Attention Module
- **Purpose**: Adaptive spatial sampling for character alignment
- **Usage**: Set `config.training.use_deformable_conv = True` (requires `use_shared_attention = True`)


## Testing Changes

### Quick Validation
```bash
# Run training for 1 epoch to check for errors
python train.py --data-dir data/ --output-dir test_outputs/ --stage 1
```

### Check for NaN
- Monitor validation logs for `nan` values in losses
- Check `train.py` NaN batch detection warnings

### Memory Check
```python
# Monitor GPU memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Dependencies

### Required
- `torch>=2.0.0` - Core framework
- `timm>=0.9.0` - For PARSeq pretrained models
- `einops>=0.7.0` - Tensor operations in SwinIR

### For Pretrained PARSeq
- Requires internet on first run (torch.hub download)
- Falls back to custom implementation if download fails

## File Modification Checklist

When modifying key files, ensure consistency:

- [ ] `config.py` - Update hyperparameters
- [ ] `train.py` - Update training logic
- [ ] `models/__init__.py` - Export new modules
- [ ] `losses/__init__.py` - Export new losses
- [ ] `README.md` - Update documentation
- [ ] Test with `python train.py --stage 1` (quick smoke test)

## Contact

For questions about this codebase, refer to:
1. This AGENTS.md file
2. README.md for usage
3. Inline docstrings in each module
4. `config.py` for all hyperparameters and constants
