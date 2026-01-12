"""
OCR-as-Discriminator for GAN-based License Plate Super-Resolution.

Implements the approach from "Enhancing License Plate Super-Resolution:
A Layout-Aware and Character-Driven Approach" (Nascimento et al.).

Key insight: Instead of using a binary classifier as discriminator,
use the OCR model to provide recognition-based feedback. This:
1. Trains the generator to produce recognizable plates, not just visually pleasing ones
2. Eliminates GAN instability (NaN issues) since recognition is well-defined
3. Naturally handles the multi-modal nature of license plate recognition

The generator is trained to maximize OCR recognition confidence on generated images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from config import VOCAB_SIZE, PLATE_LENGTH, CHAR_START_IDX, CHARSET


class OCRDiscriminator(nn.Module):
    """
    OCR-based Discriminator for LP Super-Resolution GAN.
    
    Uses a pretrained/frozen OCR model to evaluate image quality
    based on recognition confidence rather than binary real/fake classification.
    
    Key benefits over binary discriminator:
    1. More stable gradients (no mode collapse)
    2. Directly optimizes for recognition accuracy
    3. Provides character-level feedback
    
    The discriminator outputs:
    - Recognition confidence (higher = more recognizable = more "real")
    - Character-level probabilities for detailed feedback
    """
    
    def __init__(
        self,
        ocr_model: nn.Module,
        freeze_ocr: bool = True,
        confidence_mode: str = 'mean',  # 'mean', 'min', 'product'
        temperature: float = 1.0
    ):
        """
        Initialize OCR discriminator.
        
        Args:
            ocr_model: Pretrained OCR model (e.g., PARSeq).
            freeze_ocr: Whether to freeze OCR model weights.
            confidence_mode: How to aggregate character confidences.
            temperature: Temperature for softmax (lower = sharper).
        """
        super().__init__()
        
        self.ocr_model = ocr_model
        self.confidence_mode = confidence_mode
        self.temperature = temperature
        
        if freeze_ocr:
            for param in ocr_model.parameters():
                param.requires_grad = False
            ocr_model.eval()
        
        self._frozen = freeze_ocr
    
    def train(self, mode: bool = True):
        """Override train to keep OCR frozen if specified."""
        super().train(mode)
        if self._frozen:
            self.ocr_model.eval()
        return self
    
    def get_recognition_confidence(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute recognition confidence from OCR logits.
        
        Args:
            logits: OCR logits (B, L, V).
            targets: Optional ground truth indices for targeted confidence.
            
        Returns:
            Tuple of (confidence, per_char_confidence).
        """
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Get probabilities
        probs = F.softmax(scaled_logits, dim=-1)  # (B, L, V)
        
        if targets is not None:
            # Compute confidence for target characters
            B, L = targets.shape
            per_char_conf = torch.gather(
                probs, 
                dim=2, 
                index=targets.unsqueeze(-1)
            ).squeeze(-1)  # (B, L)
        else:
            # Compute confidence for predicted characters (max prob)
            per_char_conf = probs.max(dim=-1)[0]  # (B, L)
        
        # Aggregate confidence across characters
        if self.confidence_mode == 'mean':
            confidence = per_char_conf.mean(dim=1)
        elif self.confidence_mode == 'min':
            confidence = per_char_conf.min(dim=1)[0]
        elif self.confidence_mode == 'product':
            # Product of probabilities (log-space for stability)
            log_conf = torch.log(per_char_conf + 1e-10)
            confidence = torch.exp(log_conf.mean(dim=1))
        else:
            raise ValueError(f"Unknown confidence mode: {self.confidence_mode}")
        
        return confidence, per_char_conf
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate images using OCR-based discrimination.
        
        Args:
            images: Input images (B, C, H, W).
            targets: Optional ground truth character indices (B, L).
            return_logits: Whether to include raw logits in output.
            
        Returns:
            Dictionary with 'confidence', 'per_char_confidence', optionally 'logits'.
        """
        # Get OCR predictions
        with torch.set_grad_enabled(not self._frozen):
            logits = self.ocr_model(images)
        
        # Compute confidence
        confidence, per_char_conf = self.get_recognition_confidence(logits, targets)
        
        result = {
            'confidence': confidence,
            'per_char_confidence': per_char_conf
        }
        
        if return_logits:
            result['logits'] = logits
        
        return result
    
    def discriminate(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compare real and fake images for discrimination.
        
        Args:
            real_images: Ground truth HR images.
            fake_images: Generated SR images.
            targets: Ground truth character indices.
            
        Returns:
            Dictionary with confidence scores for both.
        """
        real_output = self.forward(real_images, targets)
        fake_output = self.forward(fake_images, targets)
        
        return {
            'real_confidence': real_output['confidence'],
            'fake_confidence': fake_output['confidence'],
            'real_per_char': real_output['per_char_confidence'],
            'fake_per_char': fake_output['per_char_confidence'],
        }


class OCRDiscriminatorLoss(nn.Module):
    """
    Loss functions for OCR-based GAN training.
    
    Provides both generator and discriminator losses based on
    OCR recognition confidence.
    """
    
    def __init__(
        self,
        ocr_discriminator: OCRDiscriminator,
        generator_loss_type: str = 'confidence',  # 'confidence', 'cross_entropy', 'combined'
        margin: float = 0.1,  # Margin for hinge-style losses
        lambda_ce: float = 1.0,  # Weight for cross-entropy component
        lambda_conf: float = 1.0  # Weight for confidence component
    ):
        """
        Initialize OCR discriminator loss.
        
        Args:
            ocr_discriminator: The OCR discriminator instance.
            generator_loss_type: Type of generator loss.
            margin: Margin for discrimination.
            lambda_ce: Weight for cross-entropy loss.
            lambda_conf: Weight for confidence loss.
        """
        super().__init__()
        
        self.discriminator = ocr_discriminator
        self.generator_loss_type = generator_loss_type
        self.margin = margin
        self.lambda_ce = lambda_ce
        self.lambda_conf = lambda_conf
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def generator_loss(
        self,
        fake_images: torch.Tensor,
        targets: torch.Tensor,
        real_images: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator loss to maximize recognition on fake images.
        
        The generator should produce images that the OCR can recognize.
        
        Args:
            fake_images: Generated SR images (B, C, H, W).
            targets: Ground truth character indices (B, L).
            real_images: Optional real HR images for comparison.
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Get OCR output on generated images
        output = self.discriminator(fake_images, targets, return_logits=True)
        
        metrics = {
            'fake_confidence': output['confidence'].mean().item(),
        }
        
        if self.generator_loss_type == 'confidence':
            # Maximize recognition confidence (minimize 1 - confidence)
            loss = 1.0 - output['confidence'].mean()
            
        elif self.generator_loss_type == 'cross_entropy':
            # Direct cross-entropy on OCR predictions
            logits = output['logits']  # (B, L, V)
            B, L, V = logits.shape
            loss = self.ce_loss(logits.view(-1, V), targets.view(-1))
            
        elif self.generator_loss_type == 'combined':
            # Combine confidence and cross-entropy
            conf_loss = 1.0 - output['confidence'].mean()
            
            logits = output['logits']
            B, L, V = logits.shape
            ce_loss = self.ce_loss(logits.view(-1, V), targets.view(-1))
            
            loss = self.lambda_conf * conf_loss + self.lambda_ce * ce_loss
            metrics['ce_loss'] = ce_loss.item()
            metrics['conf_loss'] = conf_loss.item()
            
        else:
            raise ValueError(f"Unknown generator loss type: {self.generator_loss_type}")
        
        # Optional: add real comparison for relative improvement
        if real_images is not None:
            real_output = self.discriminator(real_images, targets)
            metrics['real_confidence'] = real_output['confidence'].mean().item()
            # Could add margin loss: fake should approach real confidence
            margin_loss = F.relu(real_output['confidence'] - output['confidence'] - self.margin).mean()
            metrics['margin_loss'] = margin_loss.item()
        
        metrics['total'] = loss.item()
        
        return loss, metrics
    
    def discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss (mainly for logging, OCR is frozen).
        
        Since OCR is frozen, this is mainly for metrics tracking.
        If OCR is unfrozen, it learns to distinguish real from generated.
        
        Args:
            real_images: Ground truth HR images.
            fake_images: Generated SR images.
            targets: Ground truth character indices.
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        comparison = self.discriminator.discriminate(real_images, fake_images, targets)
        
        # Hinge-style loss: real should have higher confidence than fake
        loss = F.relu(
            self.margin - (comparison['real_confidence'] - comparison['fake_confidence'])
        ).mean()
        
        metrics = {
            'real_confidence': comparison['real_confidence'].mean().item(),
            'fake_confidence': comparison['fake_confidence'].mean().item(),
            'confidence_gap': (comparison['real_confidence'] - comparison['fake_confidence']).mean().item(),
            'total': loss.item()
        }
        
        return loss, metrics
    
    def forward(
        self,
        fake_images: torch.Tensor,
        targets: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        mode: str = 'generator'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss based on mode.
        
        Args:
            fake_images: Generated images.
            targets: Ground truth indices.
            real_images: Real images (required for discriminator mode).
            mode: 'generator' or 'discriminator'.
            
        Returns:
            Tuple of (loss, metrics).
        """
        if mode == 'generator':
            return self.generator_loss(fake_images, targets, real_images)
        elif mode == 'discriminator':
            if real_images is None:
                raise ValueError("real_images required for discriminator mode")
            return self.discriminator_loss(real_images, fake_images, targets)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class RecognitionGuidedLoss(nn.Module):
    """
    Combined loss that uses OCR feedback to guide super-resolution.
    
    This replaces the binary GAN discriminator with OCR-based guidance,
    providing more stable and recognition-focused training.
    
    Total loss = λ_pixel * L_pixel + λ_ssim * L_ssim + λ_ocr * L_ocr_guidance
    """
    
    def __init__(
        self,
        ocr_model: nn.Module,
        weight_pixel: float = 1.0,
        weight_ssim: float = 0.3,
        weight_ocr_guidance: float = 1.0,
        freeze_ocr: bool = True,
        confidence_mode: str = 'mean'
    ):
        """
        Initialize recognition-guided loss.
        
        Args:
            ocr_model: Pretrained OCR model.
            weight_pixel: Weight for pixel loss (L1).
            weight_ssim: Weight for SSIM loss.
            weight_ocr_guidance: Weight for OCR-based guidance.
            freeze_ocr: Whether to freeze OCR model.
            confidence_mode: How to aggregate confidence.
        """
        super().__init__()
        
        self.weight_pixel = weight_pixel
        self.weight_ssim = weight_ssim
        self.weight_ocr_guidance = weight_ocr_guidance
        
        # OCR discriminator
        self.ocr_discriminator = OCRDiscriminator(
            ocr_model, 
            freeze_ocr=freeze_ocr,
            confidence_mode=confidence_mode
        )
        
        # OCR-based generator loss
        self.ocr_loss = OCRDiscriminatorLoss(
            self.ocr_discriminator,
            generator_loss_type='combined'
        )
        
        # Pixel loss
        self.l1_loss = nn.L1Loss()
        
        # Import SSIM from lcofl if available
        try:
            from .lcofl_loss import SSIMLoss
            self.ssim_loss = SSIMLoss()
        except ImportError:
            self.ssim_loss = None
    
    def forward(
        self,
        generated_hr: torch.Tensor,
        gt_hr: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute recognition-guided loss.
        
        Args:
            generated_hr: Generated HR images (B, C, H, W).
            gt_hr: Ground truth HR images (B, C, H, W).
            targets: Ground truth character indices (B, L).
            
        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=generated_hr.device)
        
        # Pixel loss (L1)
        if self.weight_pixel > 0:
            l_pixel = self.l1_loss(generated_hr, gt_hr)
            metrics['pixel'] = l_pixel.item()
            total_loss = total_loss + self.weight_pixel * l_pixel
        
        # SSIM loss
        if self.weight_ssim > 0 and self.ssim_loss is not None:
            l_ssim = self.ssim_loss(generated_hr, gt_hr)
            metrics['ssim'] = l_ssim.item()
            total_loss = total_loss + self.weight_ssim * l_ssim
        
        # OCR guidance loss
        if self.weight_ocr_guidance > 0:
            l_ocr, ocr_metrics = self.ocr_loss.generator_loss(
                generated_hr, targets, real_images=gt_hr
            )
            metrics['ocr_guidance'] = l_ocr.item()
            metrics['recognition_confidence'] = ocr_metrics.get('fake_confidence', 0.0)
            total_loss = total_loss + self.weight_ocr_guidance * l_ocr
        
        metrics['total'] = total_loss.item()
        
        return total_loss, metrics


def create_ocr_discriminator(
    ocr_model: nn.Module,
    freeze_ocr: bool = True,
    confidence_mode: str = 'mean'
) -> OCRDiscriminator:
    """
    Factory function to create OCR discriminator.
    
    Args:
        ocr_model: Pretrained OCR model.
        freeze_ocr: Whether to freeze weights.
        confidence_mode: Confidence aggregation mode.
        
    Returns:
        Configured OCRDiscriminator.
    """
    return OCRDiscriminator(
        ocr_model=ocr_model,
        freeze_ocr=freeze_ocr,
        confidence_mode=confidence_mode
    )
