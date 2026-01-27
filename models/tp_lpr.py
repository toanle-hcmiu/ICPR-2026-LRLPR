"""
Complete Text-Prior Guided LPR System.

This module integrates the new text-prior guided generator with existing components:
- STN (Spatial Transformer Network) for geometric rectification
- Layout Classifier for Brazilian/Mercosul detection
- PARSeq recognizer for character recognition
- New Text-Prior Guided Generator for super-resolution

This is a drop-in replacement for NeuroSymbolicLPR that uses the new architecture.

Author: New Architecture Implementation (2026-01-27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

# Import existing components
from .encoder import SharedCNNEncoder
from .stn import SpatialTransformerNetwork
from .layout_classifier import LayoutClassifier
from .parseq import PARSeqRecognizer, PretrainedPARSeq, load_pretrained_parseq

# Import new components
from .tp_guided_generator import TextPriorGuidedGenerator, create_tp_guided_generator
from .text_prior import TextPriorExtractor, TextPriorLoss, create_text_prior_extractor
from .feature_fusion import QualityScorerFusion


class TextPriorGuidedLPR(nn.Module):
    """
    Complete Text-Prior Guided License Plate Recognition System.

    This system replaces SwinIR with the text-prior guided generator that:
    1. Uses Sequential Residual Blocks (BiLSTM) instead of SwinIR windows
    2. Uses multi-frame fusion with cross-attention (from LP-Diff)
    3. Uses text prior during training (from frozen PARSeq)
    4. Pure image-based during inference (no text input required)

    Architecture:
        LR Frames (5×16×48) → Encoder → STN → Layout Fusion → TP Generator → HR (64×192)
                                         ↓                    ↓
                                    Layout              Text Prior (training only)
                                     Class                   + PARSeq (frozen)

    Args:
        use_pretrained_parseq: Whether to use pretrained PARSeq
        num_frames: Number of input frames (default: 5)
        num_srb: Number of Sequential Residual Blocks (default: 4)
        generator_channels: Number of feature channels in generator (default: 64)
        use_layout: Whether to use layout classifier
        use_quality_fusion: Whether to use quality-based frame fusion
    """

    def __init__(
        self,
        use_pretrained_parseq: bool = True,
        num_frames: int = 5,
        num_srb: int = 4,
        generator_channels: int = 64,
        lstm_hidden: int = 32,
        use_layout: bool = True,
        use_quality_fusion: bool = True,
        lightweight: bool = False
    ):
        super().__init__()

        self.num_frames = num_frames
        self.use_layout = use_layout
        self.use_quality_fusion = use_quality_fusion

        # Shared encoder for feature extraction
        self.encoder = SharedCNNEncoder()

        # STN for geometric rectification
        # Encoder outputs (B, T, 512, 4, 12), so STN expects these dimensions
        self.stn = SpatialTransformerNetwork(
            in_channels=512,
            feature_height=4,
            feature_width=12
        )

        # Layout classifier (Brazilian vs Mercosul)
        # Encoder outputs (B, T, 512, 4, 12), so classifier expects in_channels=512
        if use_layout:
            self.layout_classifier = LayoutClassifier(in_channels=512)

        # Quality-based multi-frame fusion
        if use_quality_fusion:
            self.quality_fusion = QualityScorerFusion(in_channels=512)
        else:
            self.quality_fusion = None

        # Text-Prior Guided Generator (NEW - replaces SwinIR)
        self.generator = create_tp_guided_generator(
            num_frames=num_frames,
            num_srb=num_srb,
            channels=generator_channels,
            lstm_hidden=lstm_hidden,
            lightweight=lightweight
        )

        # PARSeq recognizer
        if use_pretrained_parseq:
            self.recognizer = PretrainedPARSeq(pretrained='parseq')
        else:
            vocab_size = 39  # 36 chars + 3 special tokens
            plate_length = 7
            self.recognizer = PARSeqRecognizer(
                vocab_size=vocab_size,
                plate_length=plate_length
            )

        # Text prior extractor (for training)
        # Uses the same PARSeq model
        self.text_prior_extractor = None  # Will be set during training setup

        # Syntax mask (reuse from existing system)
        self._register_syntax_mask()

    def _register_syntax_mask(self) -> None:
        """Register syntax mask from PARSeq if available."""
        if hasattr(self.recognizer, 'syntax_mask'):
            self.syntax_mask = self.recognizer.syntax_mask
        else:
            # Create a dummy syntax mask for compatibility
            self.syntax_mask = lambda *args, **kwargs: args[0] if args else None

    def setup_text_prior(self) -> None:
        """
        Setup text prior extractor using the frozen PARSeq.

        Call this before training to enable text prior guidance.
        """
        self.text_prior_extractor = TextPriorExtractor(self.recognizer)
        self.generator.set_text_extractor(self.recognizer)

        # Create text prior loss
        self.text_prior_loss = TextPriorLoss(self.recognizer)

    def forward(
        self,
        lr_frames: torch.Tensor,
        return_intermediates: bool = False,
        use_text_prior: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the LPR system.

        Args:
            lr_frames: Multi-frame LR input of shape (B, T, 3, 16, 48)
            return_intermediates: Whether to return intermediate outputs
            use_text_prior: Whether to use text prior (training=True, inference=False)

        Returns:
            Dictionary containing:
                - hr_image: Generated HR image (B, 3, 64, 192)
                - raw_logits: PARSeq logits (B, 7, vocab_size)
                - masked_logits: Syntax-masked logits (B, 7, vocab_size)
                - layout_logits: Layout classification logits (B, 1) [if use_layout]
        """
        B, T, C, H_in, W_in = lr_frames.shape

        outputs = {}
        intermediates = []

        # 1. Encode each frame
        frame_features = []
        for t in range(T):
            feat = self.encoder(lr_frames[:, t])  # (B, C, H, W)
            frame_features.append(feat)

        frame_features = torch.stack(frame_features, dim=1)  # (B, T, C, H, W)

        # 2. Apply STN to each frame
        stn_features = []
        thetas = []
        for t in range(T):
            feat, theta = self.stn(lr_frames[:, t], frame_features[:, t])
            stn_features.append(feat)
            thetas.append(theta)

        stn_features = torch.stack(stn_features, dim=1)  # (B, T, C, H, W)

        # 3. Layout classification (on first frame)
        if self.use_layout:
            layout_logits = self.layout_classifier(stn_features[:, 0])
            outputs['layout_logits'] = layout_logits
            outputs['layout'] = layout_logits  # For compatibility

        # 4. Quality-based fusion
        if self.quality_fusion is not None:
            fused_features, frame_weights = self.quality_fusion(
                stn_features,
                thetas,
                layout_logits if self.use_layout else None
            )
            outputs['frame_weights'] = frame_weights
        else:
            fused_features = stn_features.mean(dim=1)  # Simple average

        # 5. Generate HR image using TP-Guided Generator
        # The generator handles multi-frame fusion internally
        if use_text_prior:
            hr_image, text_prior = self.generator(
                stn_features,
                use_text_prior=True,
                return_text_prior=True
            )
            outputs['text_prior'] = text_prior
        else:
            hr_image = self.generator(
                stn_features,
                use_text_prior=False
            )

        outputs['hr_image'] = hr_image

        # 6. Recognize text from HR image
        raw_logits = self.recognizer.forward_parallel(hr_image)
        outputs['raw_logits'] = raw_logits

        # 7. Apply syntax mask
        if self.use_layout:
            layout_input = torch.sigmoid(layout_logits).squeeze(-1)
            layout_input = layout_input.round().long()
        else:
            layout_input = torch.zeros(B, dtype=torch.long, device=lr_frames.device)

        masked_logits = self.syntax_mask(raw_logits, layout_input, training=self.training)
        outputs['masked_logits'] = masked_logits

        # Store thetas if requested
        if return_intermediates:
            outputs['thetas'] = torch.stack(thetas, dim=1)
            outputs['stn_features'] = stn_features
            outputs['frame_features'] = frame_features

        return outputs

    def get_trainable_params(
        self,
        stage: str = 'full',
        freeze_stn: bool = False,
        freeze_recognizer: bool = False,
        freeze_generator: bool = False
    ) -> List[Dict]:
        """
        Get trainable parameters for different training stages.

        Args:
            stage: Training stage ('stn', 'tp_stn', 'generator', 'tp_generator', 'recognizer', 'full', 'tp_full')
            freeze_stn: Whether to freeze STN
            freeze_recognizer: Whether to freeze recognizer
            freeze_generator: Whether to freeze generator

        Returns:
            List of parameter groups
        """
        param_groups = []

        # STN parameters
        if stage in ['stn', 'tp_stn', 'full', 'tp_full'] and not freeze_stn:
            param_groups.append({
                'params': self.stn.parameters(),
                'name': 'stn',
                'lr_scale': 1.0
            })

        # Encoder parameters
        if stage in ['stn', 'tp_stn', 'full', 'tp_full']:
            param_groups.append({
                'params': self.encoder.parameters(),
                'name': 'encoder',
                'lr_scale': 1.0
            })

        # Layout classifier parameters
        if self.use_layout and stage in ['stn', 'tp_stn', 'full', 'tp_full']:
            param_groups.append({
                'params': self.layout_classifier.parameters(),
                'name': 'layout',
                'lr_scale': 1.0
            })

        # Quality fusion parameters
        if self.quality_fusion is not None:
            param_groups.append({
                'params': self.quality_fusion.parameters(),
                'name': 'quality_fusion',
                'lr_scale': 1.0
            })

        # Generator parameters (NEW - TP-Guided)
        if stage in ['generator', 'tp_generator', 'full', 'tp_full'] and not freeze_generator:
            gen_params = self.generator.get_trainable_params()
            param_groups.extend(gen_params)

        # Recognizer parameters
        if stage in ['recognizer', 'full', 'tp_full'] and not freeze_recognizer:
            param_groups.append({
                'params': self.recognizer.parameters(),
                'name': 'recognizer',
                'lr_scale': 1.0
            })

        return param_groups

    def freeze_stn(self) -> None:
        """Freeze STN parameters."""
        for param in self.stn.parameters():
            param.requires_grad = False

    def unfreeze_stn(self) -> None:
        """Unfreeze STN parameters."""
        for param in self.stn.parameters():
            param.requires_grad = True

    def freeze_recognizer(self) -> None:
        """Freeze recognizer parameters."""
        for param in self.recognizer.parameters():
            param.requires_grad = False

    def unfreeze_recognizer(self) -> None:
        """Unfreeze recognizer parameters."""
        for param in self.recognizer.parameters():
            param.requires_grad = True

    def freeze_generator(self) -> None:
        """Freeze generator parameters."""
        for param in self.generator.parameters():
            param.requires_grad = False

    def unfreeze_generator(self) -> None:
        """Unfreeze generator parameters."""
        for param in self.generator.parameters():
            param.requires_grad = True

    def init_weights(self) -> None:
        """Initialize network weights."""
        # Generator is already initialized in __init__
        self.generator.init_weights()

        # Other components use pretrained weights or default init

    @torch.no_grad()
    def sample(
        self,
        lr_frames: torch.Tensor,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples from LR frames (inference mode).

        Args:
            lr_frames: Multi-frame LR input
            num_samples: Number of samples to generate

        Returns:
            Dictionary with hr_image and predictions
        """
        self.eval()
        hr_image = self(lr_frames, use_text_prior=False)['hr_image']
        self.train()

        return {'hr_image': hr_image}

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        num_params = {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'stn': sum(p.numel() for p in self.stn.parameters()),
            'layout': sum(p.numel() for p in self.layout_classifier.parameters()) if self.use_layout else 0,
            'quality_fusion': sum(p.numel() for p in self.quality_fusion.parameters()) if self.quality_fusion else 0,
            'generator': sum(p.numel() for p in self.generator.parameters()),
            'recognizer': sum(p.numel() for p in self.recognizer.parameters()),
        }

        num_params['total'] = sum(num_params.values())
        num_params['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return num_params


def create_text_prior_lpr(
    use_pretrained_parseq: bool = True,
    num_frames: int = 5,
    num_srb: int = 4,
    generator_channels: int = 64,
    use_layout: bool = True,
    lightweight: bool = False
) -> TextPriorGuidedLPR:
    """
    Factory function to create Text-Prior Guided LPR system.

    Args:
        use_pretrained_parseq: Whether to use pretrained PARSeq
        num_frames: Number of input frames
        num_srb: Number of Sequential Residual Blocks
        generator_channels: Number of feature channels
        use_layout: Whether to use layout classifier
        lightweight: Whether to use lightweight variant

    Returns:
        TextPriorGuidedLPR model
    """
    model = TextPriorGuidedLPR(
        use_pretrained_parseq=use_pretrained_parseq,
        num_frames=num_frames,
        num_srb=num_srb,
        generator_channels=generator_channels,
        use_layout=use_layout,
        lightweight=lightweight
    )

    # Setup text prior extraction
    model.setup_text_prior()

    return model


# Test function
def _test_text_prior_lpr():
    """Quick test to verify the complete system works correctly."""
    import torch

    # Create model (lightweight for testing)
    model = create_text_prior_lpr(
        num_frames=3,  # Use fewer frames for testing
        num_srb=2,
        generator_channels=32,
        lightweight=True
    )

    # Test data
    B, T, C, H, W = 2, 3, 3, 16, 48
    lr_frames = torch.randn(B, T, C, H, W)

    # Forward pass
    outputs = model(lr_frames, use_text_prior=True)

    print(f"Input shape: {lr_frames.shape}")
    print(f"HR output shape: {outputs['hr_image'].shape}")
    print(f"Raw logits shape: {outputs['raw_logits'].shape}")
    print(f"Masked logits shape: {outputs['masked_logits'].shape}")
    print(f"Layout logits shape: {outputs['layout_logits'].shape}")

    # Get parameter counts
    num_params = model.get_num_params()
    print(f"\nParameter counts:")
    for name, count in num_params.items():
        print(f"  {name}: {count:,}")

    print("\n✓ Text-Prior Guided LPR system test passed!")


if __name__ == "__main__":
    _test_text_prior_lpr()
