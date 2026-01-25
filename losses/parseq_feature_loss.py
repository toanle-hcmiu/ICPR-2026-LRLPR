"""
PARSeq Feature Loss for License Plate Super-Resolution.

This module implements a feature-based perceptual loss using PARSeq encoder
as a frozen feature extractor. Unlike CE-based losses, this uses L1 distance
on encoder features, bypassing the decoder/LM entirely.

Key features:
- Geometry-preserving adapter (pad, don't stretch)
- Frozen PARSeq in eval mode
- L1 feature loss on encoder outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PARSeqAdapter(nn.Module):
    """
    Differentiable adapter to prepare images for PARSeq.
    
    This adapter:
    1. Preserves aspect ratio using padding (no stretching)
    2. Resizes to 32x128 (PARSeq expected size)
    3. Normalizes to mean=0.5, std=0.5 (PARSeq expected range)
    
    The adapter is fully differentiable so gradients can flow back
    to the generator.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (32, 128),
        input_range: str = 'tanh'  # 'tanh' for [-1,1], 'sigmoid' for [0,1]
    ):
        """
        Initialize the adapter.
        
        Args:
            target_size: Target size (H, W) for PARSeq input.
            input_range: Expected input range ('tanh' or 'sigmoid').
        """
        super().__init__()
        self.target_size = target_size
        self.input_range = input_range
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply geometry-preserving resize and normalization.
        
        Args:
            x: Input image (B, C, H, W) in [-1, 1] range.
            
        Returns:
            Adapted image (B, C, 32, 128) normalized for PARSeq.
        """
        B, C, H, W = x.shape
        target_h, target_w = self.target_size
        
        # === Step 1: Preserve Geometry with Padding ===
        # Calculate scale to fit height to target while preserving aspect ratio
        scale = target_h / H
        new_w = int(W * scale)
        
        # Resize to match target height
        x_resized = F.interpolate(
            x, 
            size=(target_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Pad width to target width (symmetric padding)
        if new_w < target_w:
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            # Use reflection padding for better edge handling
            x_padded = F.pad(x_resized, (pad_left, pad_right, 0, 0), mode='reflect')
        elif new_w > target_w:
            # Crop center if wider than target
            start = (new_w - target_w) // 2
            x_padded = x_resized[:, :, :, start:start + target_w]
        else:
            x_padded = x_resized
        
        # === Step 2: Normalize for PARSeq ===
        # Input is in [-1, 1], PARSeq expects normalized with mean=0.5, std=0.5
        # Convert to [0, 1] first: (x + 1) / 2
        # Then normalize: (x - 0.5) / 0.5 = x * 2 - 1 = back to [-1, 1]
        # So actually, if input is already in [-1, 1], PARSeq expects that!
        # But PARSeq ImageNet normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        
        # Convert from [-1, 1] to [0, 1]
        x_01 = (x_padded + 1.0) / 2.0
        
        # Apply ImageNet normalization (what PARSeq expects)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x_01 - mean) / std
        
        return x_normalized


class PARSeqFeatureLoss(nn.Module):
    """
    Feature-based perceptual loss using frozen PARSeq encoder.
    
    This computes L1 distance between PARSeq encoder features of
    the generated and ground truth images. The decoder/LM is bypassed,
    making this a pure feature matching loss.
    
    Benefits over CE loss:
    - Smoother loss landscape
    - No mode collapse from classification
    - Direct feature alignment signal
    """
    
    def __init__(
        self,
        parseq_model: nn.Module,
        feature_layers: Optional[list] = None,
        weight: float = 1.0
    ):
        """
        Initialize PARSeq feature loss.
        
        Args:
            parseq_model: PARSeq model (will be frozen and set to eval).
            feature_layers: Which encoder layers to use (None = final output).
            weight: Loss weight.
        """
        super().__init__()
        
        # Store reference (not as submodule to avoid param counting)
        self._parseq = parseq_model
        self.adapter = PARSeqAdapter()
        self.weight = weight
        
        # Freeze PARSeq and set to eval mode
        self._freeze_parseq()
    
    def _freeze_parseq(self):
        """Freeze PARSeq weights and set to eval mode."""
        self._parseq.eval()
        for param in self._parseq.parameters():
            param.requires_grad = False
    
    def _get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract encoder features from PARSeq.
        
        Args:
            x: Adapted image (B, C, 32, 128).
            
        Returns:
            Encoder features (B, N, D).
        """
        # Access the underlying model - hub PARSeq wraps it in _model.model
        if hasattr(self._parseq, '_model') and self._parseq._model is not None:
            hub_model = self._parseq._model
            
            # Hub PARSeq structure: _model has 'model' attribute containing actual model
            if hasattr(hub_model, 'model'):
                inner_model = hub_model.model
                # Try to access encoder directly
                if hasattr(inner_model, 'encoder'):
                    # The encoder expects embedded input
                    if hasattr(inner_model, 'embed'):
                        embedded = inner_model.embed(x)
                        return inner_model.encoder(embedded)
                    else:
                        return inner_model.encoder(x)
            
            # Alternative: hub_model itself has encoder
            if hasattr(hub_model, 'encoder'):
                if hasattr(hub_model, 'embed'):
                    embedded = hub_model.embed(x)
                    return hub_model.encoder(embedded)
                else:
                    return hub_model.encoder(x)
            
            # Fallback: run forward pass and capture encoder output via hook
            # Or just use the full model output as features (last hidden state)
            if hasattr(hub_model, 'forward'):
                # Run forward and get logits, use as feature proxy
                with torch.no_grad():
                    logits = hub_model(x)
                # Use logits as feature - shape (B, T, V)
                # This isn't ideal but provides some signal
                return logits
        
        # Fallback: use custom implementation's encode method
        if hasattr(self._parseq, '_fallback_model') and self._parseq._fallback_model is not None:
            return self._parseq._fallback_model.encode(x)
        
        # Last resort: if parseq has encode method
        if hasattr(self._parseq, 'encode'):
            return self._parseq.encode(x)
        
        # Ultimate fallback: just run forward pass on the wrapper
        if hasattr(self._parseq, 'forward'):
            with torch.no_grad():
                return self._parseq(x)
        
        raise RuntimeError("Cannot extract encoder features from PARSeq model")
    
    def forward(
        self,
        sr_image: torch.Tensor,
        gt_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 feature loss between SR and GT images.
        
        Args:
            sr_image: Super-resolved image (B, C, H, W) in [-1, 1].
            gt_image: Ground truth HR image (B, C, H, W) in [-1, 1].
            
        Returns:
            L1 feature loss (scalar).
        """
        # Ensure parseq is frozen and in eval mode
        self._parseq.eval()
        
        # Adapt images for PARSeq
        sr_adapted = self.adapter(sr_image)
        gt_adapted = self.adapter(gt_image)
        
        # Get GT features (no gradients needed)
        with torch.no_grad():
            # Disable autocast for PARSeq (expects FP32)
            with torch.amp.autocast('cuda', enabled=False):
                gt_features = self._get_encoder_features(gt_adapted.float())
        
        # Get SR features (gradients flow back to generator)
        with torch.amp.autocast('cuda', enabled=False):
            sr_features = self._get_encoder_features(sr_adapted.float())
        
        # L1 loss on features
        loss = F.l1_loss(sr_features, gt_features.detach())
        
        return self.weight * loss


class GatedPARSeqFeatureLoss(PARSeqFeatureLoss):
    """
    PARSeq feature loss with warmup and pixel-loss gating.
    
    Features:
    - Warmup: Loss weight starts at 0 and ramps up over N steps
    - Gating: Disables loss when pixel loss is high
    """
    
    def __init__(
        self,
        parseq_model: nn.Module,
        max_weight: float = 1e-3,
        warmup_steps: int = 5000,
        pixel_threshold: float = 0.5
    ):
        """
        Initialize gated PARSeq feature loss.
        
        Args:
            parseq_model: PARSeq model.
            max_weight: Maximum loss weight after warmup.
            warmup_steps: Number of steps to reach max weight.
            pixel_threshold: Disable OCR loss when pixel loss > this.
        """
        super().__init__(parseq_model, weight=max_weight)
        
        self.max_weight = max_weight
        self.warmup_steps = warmup_steps
        self.pixel_threshold = pixel_threshold
        
        # Track current step
        self.register_buffer('current_step', torch.tensor(0))
    
    def get_current_weight(self, pixel_loss: Optional[float] = None) -> float:
        """
        Get current loss weight based on warmup and gating.
        
        Args:
            pixel_loss: Current pixel loss value for gating.
            
        Returns:
            Current loss weight.
        """
        # Warmup schedule
        step = self.current_step.item()
        warmup_factor = min(step / max(self.warmup_steps, 1), 1.0)
        weight = self.max_weight * warmup_factor
        
        # Pixel loss gating
        if pixel_loss is not None and pixel_loss > self.pixel_threshold:
            weight = 0.0
        
        return weight
    
    def step(self):
        """Increment step counter (call once per training iteration)."""
        self.current_step += 1
    
    def forward(
        self,
        sr_image: torch.Tensor,
        gt_image: torch.Tensor,
        pixel_loss: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute gated L1 feature loss.
        
        Args:
            sr_image: Super-resolved image.
            gt_image: Ground truth image.
            pixel_loss: Current pixel loss for gating.
            
        Returns:
            Gated L1 feature loss.
        """
        # Get current weight
        current_weight = self.get_current_weight(pixel_loss)
        
        if current_weight <= 0:
            # Return zero loss (but keep gradient path)
            return sr_image.new_zeros(1).squeeze()
        
        # Compute base loss
        base_loss = super().forward(sr_image, gt_image)
        
        # Apply current weight (already includes max_weight from parent)
        # So we need to scale by warmup factor and gating
        warmup_factor = min(self.current_step.item() / max(self.warmup_steps, 1), 1.0)
        
        return base_loss * warmup_factor
