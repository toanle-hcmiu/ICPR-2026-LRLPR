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
Stage 1.5: PARSeq Warm-up - Train recognizer on GT HR images
Stage 2: Restoration - SwinIR + Layout + Fusion
Stage 3: Full - End-to-end fine-tuning
```

### Stage-Specific Behavior

| Stage | Frozen Modules | Loss Functions |
|-------|----------------|----------------|
| STN | Recognizer | Self-supervised, Pixel, Corner |
| PARSeq Warm-up | All except Recognizer | OCR only |
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

### 9. Wavy/Checkerboard Artifact Prevention (FIXED)
**Issue 1**: UNetDiscriminator used `ConvTranspose2d` causing checkerboard artifacts  
**Solution**: Replaced with resize-convolution (`Upsample + Conv2d`) in `models/discriminator.py`  
**Issue 2**: Missing anti-aliased downsampling caused aliasing  
**Solution**: Created `models/blur_pool.py` with `BlurPool2d`, `MaxBlurPool2d`, `AntiAliasedConv2d`  
**Issue 3**: No explicit penalty for high-frequency wavy noise  
**Solution**: Added `TotalVariationLoss` to `losses/composite_loss.py` with `weight_tv` in config  
**Usage**: Set `config.training.weight_tv = 1e-5` to enable TV denoising

### 10. Stage 3 Mode Collapse (FIXED)
**Issue**: Generator learns to produce "adversarial" features that fool trainable OCR but aren't visually clear  
**Symptom**: Characters fade or become blurry while OCR loss decreases  
**Solution**: Use frozen copy of OCR for LCOFL classification loss computation  
**File**: `train.py:train_stage()` (lines 1459-1477)  
**Config**: `use_frozen_ocr_for_lcofl: True` (default), LCOFL weight = 0.75


### 7. PARSeq Pretrained Model Issues (FIXED)
**Issue 1**: Model loaded in training mode → random outputs due to dropout  
**Solution**: Added `.eval()` in `models/parseq.py:_load_model()` after loading  
**Issue 2**: Charset adapter overwrote lowercase logits with uppercase  
**Solution**: Use `logsumexp` to properly combine both in `_adapt_logits()`  
**File**: `models/parseq.py`

### 8. Perceptual Loss Not Working (FIXED)
**Issue 1**: VGG feature extraction fed wrong shapes (reused `x` across layers)  
**Solution**: Single forward pass with incremental layer extraction in `losses/gan_loss.py`  
**Issue 2**: Perceptual loss not computed in restoration stage  
**Solution**: Added perceptual loss to `composite_loss.py:get_stage_loss()` for restoration  
**Issue 3**: Perceptual loss not enabled in training  
**Solution**: Added `use_perceptual=True, weight_perceptual=0.1` in `train.py`

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

### Stage 3 Mode Collapse Prevention
- **File**: `train.py:train_stage()` (lines 1459-1477)
- **Config**: `use_frozen_ocr_for_lcofl: True` (config.py line 206)
- **Status**: ✅ Active by default for Stage 3 training
- **Purpose**: Prevents generator from learning adversarial features that fool trainable OCR
- **Mechanism**: 
  1. Creates deep copy of recognizer at Stage 3 start
  2. Freezes all parameters and sets `eval()` mode
  3. Uses frozen copy for LCOFL classification loss computation
- **Key Insight**: Generator cannot "cheat" by learning hidden features only trainable OCR detects

**Stage 3 Anti-Collapse Configuration:**
```python
# In config.py - TrainingConfig
use_frozen_ocr_for_lcofl: bool = True      # Enable frozen OCR copy
stage3_sr_anchor_weight: float = 1.0       # Anchor to Stage 2 output
stage3_ocr_warmup_steps: int = 6000        # Steps before OCR ramps up
stage3_ocr_ramp_steps: int = 6000          # Steps to ramp to max weight
stage3_ocr_max_weight: float = 0.5         # Max OCR loss weight
weight_lcofl: float = 0.75                 # LCOFL weight (original paper)
```

### 11. Stage 3 Character Collapse - CRITICAL FIX (2025-01-25)

**Problem:** Stage 3 training causes character destruction/averaging despite Stage 2 producing good results with readable 7 characters and PARSeq pretrained >90% accuracy on HR.

**Root Cause: Gradient Conflict Loss Imbalance (GCLI)**
- Stage 3 uses TWO different OCR sources with conflicting gradients:
  1. **Trainable OCR** (forward pass) - continues learning, representation drifts
  2. **Frozen OCR** (for LCOFL loss) - stuck at Stage 2 weights, provides static gradients
- Frozen OCR forward pass was NOT wrapped in `torch.no_grad()` or `.detach()`
- Computation graph was tracked, allowing gradients to flow back through frozen OCR to generator
- Trainable OCR's representation drifted from frozen OCR's static weights
- Generator received **conflicting gradient fields** from two diverging OCRs
- Impossible optimization → generator learned adversarial features → character collapse

**Fixes Applied:**

1. **Disable Frozen OCR Gradient Flow (CRITICAL)** - `train.py` lines 542-546:
   ```python
   # Wrapped frozen OCR forward in torch.no_grad() and added .detach()
   with torch.no_grad():
       frozen_logits = frozen_ocr(outputs['hr_image'])
   outputs['frozen_logits'] = frozen_logits.detach()
   ```
   - Eliminates gradient conflict between frozen and trainable OCR
   - LCOFL still provides character guidance via loss value
   - No adversarial gradient flow back through frozen OCR

2. **Rebalance Loss Weights** - `config.py` lines 191, 201-202, 218:
   ```python
   # Increased pixel anchor, reduced LCOFL domination
   weight_pixel = 0.50      # Increased from 0.25
   weight_lcofl = 0.25       # Decreased from 0.75
   weight_ssim = 0.05        # Decreased from 0.1
   weight_tv = 5e-6            # Decreased from 1e-5
   ```
   - Pixel loss anchors generator to maintain visual quality
   - LCOFL provides refinement (not domination)
   - Reduces SSIM/TV smoothing effects

3. **LCOFL Curriculum** - `train.py` lines 1600-1608:
   ```python
   # Gradual LCOFL classification introduction over 20 epochs
   if epoch < 5:
       lcofl_classification_weight = 0.0      # Pixel-only warmup
   elif epoch < 20:
       lcofl_classification_weight = (epoch - 5) / 15.0  # Ramp up
   else:
       lcofl_classification_weight = 1.0      # Full strength
   ```
   - Generator maintains Stage 2 quality in early epochs
   - Gradually adapts to LCOFL guidance
   - Prevents sudden gradient conflict shock

**Expected Results:**
- ✅ Stage 3 maintains Stage 2 character quality
- ✅ OCR accuracy improves to 85-92% without collapse
- ✅ Visual quality preserved (sharp characters)
- ✅ Smooth convergence without sudden degradation

**Total Changes:** 9 lines of code (5 in train.py, 4 in config.py)


## Reproducibility & Determinism

### Strict Determinism (Default)

The codebase enables strict determinism by default via `seed_everything()` in `train.py`. This ensures exact reproducibility across runs but has a 5-15% performance overhead.

**DO NOT disable strict determinism** unless you have a specific reason and document it.

```python
# Default - strict determinism enabled
seed_everything(42)

# Only if you need faster training and don't need reproducibility
seed_everything(42, strict_determinism=False)
```

### What Strict Determinism Enables

- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG = ':4096:8'`
- TF32 disabled (exact float32)
- `PYTHONHASHSEED` set to seed

### Non-Deterministic Operations

If you encounter a `RuntimeError` about non-deterministic operations:

1. First, try to find a deterministic alternative
2. If unavoidable, document the exception clearly
3. Consider using `torch.use_deterministic_algorithms(True, warn_only=True)` locally

## Checkpoint Security

### Critical Security Rules

1. **NEVER load checkpoints from untrusted sources** - `torch.load()` uses pickle which can execute arbitrary code
2. **NEVER accept checkpoint paths from user input without validation**
3. **Always validate checkpoint paths are local files** - The `load_checkpoint()` function enforces this

### Adding New Checkpoint Loading Code

If you need to add new checkpoint loading functionality:

```python
from pathlib import Path
import warnings

def load_something(checkpoint_path: str):
    # REQUIRED: Validate path is a local file
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found or not a local file: {checkpoint_path}. "
            "For security reasons, only local file paths are accepted."
        )
    
    # REQUIRED: Emit security warning
    warnings.warn(
        f"Loading checkpoint from '{checkpoint_path}'. "
        "torch.load() uses pickle which can execute arbitrary code. "
        "Only load checkpoints from trusted sources.",
        UserWarning
    )
    
    # Load with weights_only=False (we need optimizer state, etc.)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint
```

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
- [ ] Run `ruff check .` and `ruff format --check .` before committing
- [ ] Run `pytest` to verify no regressions

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Lint**: Ruff formatting and linting checks
- **Test**: pytest on Python 3.10 and 3.11
- **Smoke Test**: Quick training pipeline validation

See `.github/workflows/ci.yml` for details.

### Running CI Checks Locally

```bash
# Install dev dependencies
pip install ruff pytest

# Linting
ruff check .
ruff format --check .

# Testing
pytest -v

# Smoke tests only
pytest -v -m smoke
```

## Contact

For questions about this codebase, refer to:
1. This AGENTS.md file
2. README.md for usage
3. Inline docstrings in each module
4. `config.py` for all hyperparameters and constants
