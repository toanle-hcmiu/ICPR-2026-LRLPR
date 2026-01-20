"""
Main Neuro-Symbolic LPR Model.

This module assembles all components into the complete 4-phase pipeline:
1. Geometry-Constrained Alignment (STN)
2. Layout Classification and Feature Fusion
3. Image Restoration (GAN)
4. Syntax-Masked Recognition (PARSeq)

By default, uses the pretrained PARSeq from https://github.com/baudm/parseq
for better OCR accuracy. Set use_pretrained_parseq=False to use custom impl.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .encoder import SharedCNNEncoder
from .stn import SpatialTransformerNetwork, MultiFrameSTN
from .layout_classifier import LayoutClassifier, LayoutClassifierWithAttention
from .feature_fusion import QualityScorerFusion
from .swinir import SwinIRGenerator
from .discriminator import PatchDiscriminator
from .parseq import PARSeqRecognizer, PretrainedPARSeq
from .syntax_mask import SyntaxMaskLayer


class NeuroSymbolicLPR(nn.Module):
    """
    Complete Neuro-Symbolic License Plate Recognition System.
    
    This system combines deep neural networks with symbolic reasoning
    to recognize Brazilian license plates (both old Brazilian and Mercosul
    formats) from low-resolution video frames.
    
    Pipeline:
        1. Phase 1: Multi-frame encoding and geometric rectification
        2. Phase 2: Layout classification and quality-weighted fusion
        3. Phase 3: GAN-based super-resolution
        4. Phase 4: Syntax-constrained character recognition
    """
    
    def __init__(
        self,
        # Input configuration
        num_frames: int = 5,
        in_channels: int = 3,
        lr_size: Tuple[int, int] = (16, 48),
        hr_size: Tuple[int, int] = (64, 192),
        
        # Encoder configuration
        encoder_base_channels: int = 64,
        
        # STN configuration
        use_corner_predictor: bool = True,
        
        # Layout classifier configuration
        layout_hidden_dim: int = 256,
        use_attention_layout: bool = False,  # Use attention-enhanced layout classifier
        
        # Feature fusion configuration
        fusion_type: str = 'weighted_avg',
        
        # SwinIR configuration (full implementation, no lightweight option)
        swinir_embed_dim: int = 180,  # Full SwinIR uses 180
        swinir_depths: list = None,   # Default: [6, 6, 6, 6, 6, 6]
        swinir_num_heads: list = None,  # Default: [6, 6, 6, 6, 6, 6]
        swinir_window_size: int = 8,
        
        # Shared Attention configuration
        use_shared_attention: bool = False,
        use_deformable_conv: bool = False,
        
        # PARSeq configuration
        use_pretrained_parseq: bool = True,  # Use pretrained (recommended)
        parseq_model_name: str = 'parseq',   # 'parseq' or 'parseq_tiny'
        parseq_freeze_backbone: bool = False,
        parseq_embed_dim: int = 384,  # Only used if use_pretrained_parseq=False
        parseq_num_heads: int = 6,
        parseq_depth: int = 12,
        
        # Syntax mask configuration
        soft_mask_value: float = -100.0,
        soft_inference: bool = False,  # Use soft constraints during inference
        soft_inference_value: float = -50.0
    ):
        """
        Initialize the Neuro-Symbolic LPR system.
        
        Args:
            num_frames: Number of input LR frames.
            in_channels: Number of input channels (3 for RGB).
            lr_size: Low-resolution input size (H, W).
            hr_size: High-resolution output size (H, W).
            encoder_base_channels: Base channels for CNN encoder.
            use_corner_predictor: Whether to use corner prediction in STN.
            layout_hidden_dim: Hidden dimension for layout classifier.
            use_attention_layout: Whether to use attention-enhanced layout classifier.
            fusion_type: Type of feature fusion.
            swinir_embed_dim: Embedding dimension for SwinIR (default: 180 for full model).
            swinir_depths: Depths for SwinIR transformer blocks (default: [6]*6).
            swinir_num_heads: Number of heads for SwinIR (default: [6]*6).
            swinir_window_size: Window size for SwinIR attention.
            use_shared_attention: Enable shared attention in SwinIR.
            use_deformable_conv: Enable deformable convolutions in shared attention.
            use_pretrained_parseq: Whether to use pretrained PARSeq (recommended).
            parseq_model_name: Pretrained model name ('parseq' or 'parseq_tiny').
            parseq_freeze_backbone: Whether to freeze PARSeq encoder.
            parseq_embed_dim: Embedding dimension for custom PARSeq.
            parseq_num_heads: Number of heads for custom PARSeq.
            parseq_depth: Depth of custom PARSeq encoder.
            soft_mask_value: Soft mask value for training.
            soft_inference: Whether to use soft constraints during inference.
            soft_inference_value: Penalty value for soft inference mode.
        """
        super().__init__()
        
        self.num_frames = num_frames
        self.lr_size = lr_size
        self.hr_size = hr_size
        
        if swinir_depths is None:
            swinir_depths = [6, 6, 6, 6, 6, 6]  # Full SwinIR: 6 RSTB blocks
        if swinir_num_heads is None:
            swinir_num_heads = [6, 6, 6, 6, 6, 6]
        
        # ==========================================
        # Phase 1: Shared Encoder and STN
        # ==========================================
        
        self.encoder = SharedCNNEncoder(
            in_channels=in_channels,
            base_channels=encoder_base_channels,
            num_blocks=4
        )
        
        # Get encoder output dimensions
        encoder_out_channels, feat_h, feat_w = self.encoder.get_output_size(
            lr_size[0], lr_size[1]
        )
        
        self.stn = SpatialTransformerNetwork(
            in_channels=encoder_out_channels,
            feature_height=feat_h,
            feature_width=feat_w,
            output_size=(feat_h, feat_w),
            use_corner_predictor=use_corner_predictor
        )
        
        self.multi_frame_stn = MultiFrameSTN(self.stn)
        
        # ==========================================
        # Phase 2: Layout Classifier and Fusion
        # ==========================================
        
        # Use attention-enhanced classifier if requested
        if use_attention_layout:
            self.layout_classifier = LayoutClassifierWithAttention(
                in_channels=encoder_out_channels,
                hidden_dim=layout_hidden_dim
            )
        else:
            self.layout_classifier = LayoutClassifier(
                in_channels=encoder_out_channels,
                fc_dim=layout_hidden_dim
            )
        
        self.quality_fusion = QualityScorerFusion(
            in_channels=encoder_out_channels,
            fusion_type=fusion_type
        )
        
        # ==========================================
        # Phase 3: Super-Resolution (Full SwinIR)
        # ==========================================
        
        # Decoder from feature space to image space
        self.feature_to_image = nn.Sequential(
            nn.Conv2d(encoder_out_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Upsampling from LR to HR using full SwinIR architecture
        # With optional shared attention
        upscale = hr_size[0] // lr_size[0]
        
        self.generator = SwinIRGenerator(
            in_channels=in_channels,
            out_channels=in_channels,
            embed_dim=swinir_embed_dim,
            depths=swinir_depths,
            num_heads=swinir_num_heads,
            window_size=swinir_window_size,
            upscale=upscale,
            img_size=lr_size,
            use_shared_attention=use_shared_attention,
            use_deformable=use_deformable_conv
        )
        
        # ==========================================
        # Phase 4: PARSeq Recognizer and Syntax Mask
        # ==========================================
        
        # Use pretrained PARSeq for better accuracy (recommended)
        # Falls back to custom implementation if torch.hub fails
        self.use_pretrained_parseq = use_pretrained_parseq
        
        if use_pretrained_parseq:
            self.recognizer = PretrainedPARSeq(
                pretrained=True,
                model_name=parseq_model_name,
                freeze_backbone=parseq_freeze_backbone,
                img_size=hr_size  # For fallback custom implementation
            )
        else:
            self.recognizer = PARSeqRecognizer(
                img_size=hr_size,
                embed_dim=parseq_embed_dim,
                num_heads=parseq_num_heads,
                encoder_depth=parseq_depth
            )
        
        self.syntax_mask = SyntaxMaskLayer(
            soft_mask_value=soft_mask_value,
            soft_inference=soft_inference,
            soft_inference_value=soft_inference_value
        )
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the pipeline.
        
        Args:
            x: Input LR frames of shape (B, num_frames, C, H, W).
            targets: Target token indices for training (B, seq_len).
            return_intermediates: Whether to return intermediate outputs.
            
        Returns:
            Dictionary containing:
                - 'logits': Raw recognition logits
                - 'masked_logits': Syntax-masked logits
                - 'layout_logits': Layout classification logits
                - 'hr_image': Generated HR image
                - (optional) 'corners': Predicted corner coordinates
                - (optional) 'quality_scores': Quality scores per frame
                - (optional) 'rectified_features': Rectified feature maps
        
        Raises:
            ValueError: If input shape is incorrect.
        """
        # Input validation
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D input (B, T, C, H, W), got {x.dim()}D tensor with shape {x.shape}"
            )
        
        B, T, C, H, W = x.shape
        
        if T != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {T}"
            )
        
        if (H, W) != self.lr_size:
            raise ValueError(
                f"Expected LR size {self.lr_size}, got ({H}, {W})"
            )
        
        outputs = {}
        
        # ==========================================
        # Phase 1: Encode and Rectify
        # ==========================================
        
        # Encode all frames with shared weights
        encoded = self.encoder.forward_multi_frame(x)  # (B, T, F, H', W')
        
        # Apply STN to each frame
        if return_intermediates:
            rectified, thetas, corners = self.multi_frame_stn(
                encoded, return_theta=True, return_corners=True
            )
            outputs['corners'] = corners
            outputs['thetas'] = thetas
            outputs['rectified_features'] = rectified
        else:
            rectified = self.multi_frame_stn(encoded)
        
        # ==========================================
        # Phase 2: Layout Classification and Fusion
        # ==========================================
        
        # Use first (or best) frame for layout classification
        # Could also use fused features
        layout_input = rectified[:, 0, :, :, :]  # (B, F, H', W')
        layout_logits = self.layout_classifier(layout_input)  # (B, 1)
        outputs['layout_logits'] = layout_logits
        
        # Convert to probability for mask selection
        is_mercosul = torch.sigmoid(layout_logits).squeeze(-1)  # (B,)
        
        # Quality-weighted fusion
        fused, quality_scores = self.quality_fusion(rectified, return_scores=True)
        if return_intermediates and quality_scores is not None:
            outputs['quality_scores'] = quality_scores
        
        # ==========================================
        # Phase 3: Super-Resolution
        # ==========================================
        
        # Convert features to image space
        lr_image = self.feature_to_image(fused)  # (B, C, H', W')
        
        # Upsample to fused feature size if needed
        if lr_image.shape[2:] != (self.lr_size[0], self.lr_size[1]):
            lr_image = F.interpolate(
                lr_image, 
                size=self.lr_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Super-resolve
        hr_image = self.generator(lr_image)  # (B, C, HR_H, HR_W)
        outputs['hr_image'] = hr_image
        outputs['lr_image'] = lr_image
        
        # ==========================================
        # Phase 4: Recognition with Syntax Mask
        # ==========================================
        
        # Get raw logits from recognizer
        if targets is not None:
            raw_logits = self.recognizer(hr_image, targets)
        else:
            raw_logits = self.recognizer.forward_parallel(hr_image)
        
        outputs['raw_logits'] = raw_logits
        
        # Apply syntax mask
        masked_logits = self.syntax_mask(
            raw_logits, 
            is_mercosul,
            training=self.training
        )
        outputs['masked_logits'] = masked_logits
        
        return outputs
    
    def inference(
        self,
        x: torch.Tensor
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """
        Inference mode for getting text predictions.
        
        Args:
            x: Input LR frames of shape (B, num_frames, C, H, W).
            
        Returns:
            Tuple containing:
                - List of predicted plate strings
                - HR images
                - Layout predictions (0=Brazilian, 1=Mercosul)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_intermediates=False)
            
            # Get layout decisions
            layout_probs = torch.sigmoid(outputs['layout_logits'])
            is_mercosul = (layout_probs > 0.5).squeeze(-1)
            
            # Decode with hard constraints
            texts = self.syntax_mask.decode_to_text(
                outputs['raw_logits'],
                is_mercosul
            )
            
            return texts, outputs['hr_image'], is_mercosul
    
    def freeze_recognizer(self):
        """Freeze the PARSeq recognizer weights."""
        for param in self.recognizer.parameters():
            param.requires_grad = False
    
    def unfreeze_recognizer(self):
        """Unfreeze the PARSeq recognizer weights."""
        for param in self.recognizer.parameters():
            param.requires_grad = True
    
    def freeze_stn(self):
        """Freeze the STN weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.stn.parameters():
            param.requires_grad = False
    
    def unfreeze_stn(self):
        """Unfreeze the STN weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.stn.parameters():
            param.requires_grad = True
    
    def freeze_all_except_recognizer(self):
        """Freeze all modules except the recognizer (for pretrain stage)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.stn.parameters():
            param.requires_grad = False
        for param in self.layout_classifier.parameters():
            param.requires_grad = False
        for param in self.quality_fusion.parameters():
            param.requires_grad = False
        for param in self.feature_to_image.parameters():
            param.requires_grad = False
        for param in self.generator.parameters():
            param.requires_grad = False
        # Leave recognizer trainable
        for param in self.recognizer.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self, stage: str) -> list:
        """
        Get trainable parameters for a specific training stage.
        
        Args:
            stage: Training stage name.
                'pretrain': Only recognizer parameters (OCR pre-training)
                'stn': Only STN parameters
                'restoration': Generator and layout classifier
                'full': All parameters
                
        Returns:
            List of parameter groups.
        """
        if stage == 'pretrain' or stage == 'parseq_warmup':
            # Pretrain/parseq_warmup: only train the recognizer (PARSeq) on OCR task
            # parseq_warmup specifically trains on GT HR to learn Brazilian plate fonts
            return [
                {'params': self.recognizer.parameters()}
            ]
        elif stage == 'stn':
            return [
                {'params': self.encoder.parameters()},
                {'params': self.stn.parameters()}
            ]
        elif stage == 'restoration':
            # Train generator + OCR together so OCR learns as image quality improves
            # This prevents 0% plate accuracy during restoration stage
            # Note: syntax_mask has no parameters (only buffers), so it's not included
            return [
                {'params': self.generator.parameters()},
                {'params': self.layout_classifier.parameters()},
                {'params': self.quality_fusion.parameters()},
                {'params': self.feature_to_image.parameters()},
                {'params': self.recognizer.parameters(), 'lr_scale': 0.5},  # Joint OCR training
            ]
        elif stage == 'full':
            return [
                {'params': self.encoder.parameters(), 'lr_scale': 0.1},
                {'params': self.stn.parameters(), 'lr_scale': 0.1},
                {'params': self.layout_classifier.parameters()},
                {'params': self.quality_fusion.parameters()},
                {'params': self.feature_to_image.parameters()},
                {'params': self.generator.parameters()},
                {'params': self.recognizer.parameters(), 'lr_scale': 0.1}
            ]
        else:
            raise ValueError(f"Unknown training stage: {stage}")


class NeuroSymbolicLPRWithDiscriminator(nn.Module):
    """
    Neuro-Symbolic LPR with integrated discriminator for GAN training.
    """
    
    def __init__(
        self,
        generator: NeuroSymbolicLPR,
        discriminator: PatchDiscriminator = None
    ):
        super().__init__()
        
        self.generator = generator
        
        if discriminator is None:
            discriminator = PatchDiscriminator(
                in_channels=3,
                base_channels=64,
                num_layers=3,
                use_spectral_norm=True
            )
        
        self.discriminator = discriminator
    
    def forward_generator(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through generator."""
        return self.generator(x, targets, return_intermediates)
    
    def forward_discriminator(
        self,
        real_hr: torch.Tensor,
        fake_hr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through discriminator.
        
        Args:
            real_hr: Real HR images.
            fake_hr: Generated HR images.
            
        Returns:
            Tuple of (real_scores, fake_scores).
        """
        real_scores = self.discriminator(real_hr)
        fake_scores = self.discriminator(fake_hr.detach())
        return real_scores, fake_scores
    
    def inference(self, x: torch.Tensor) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """Inference through generator."""
        return self.generator.inference(x)
