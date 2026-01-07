"""
OCR-Aware Perceptual Loss for License Plate Super-Resolution.

This module implements an OCR-aware loss function that optimizes the 
image restoration network based on recognition performance, as proposed
in LPSRGAN. The generator learns to produce images that are not just 
visually similar to ground truth, but also optimized for OCR accuracy.

Reference: "License Plate Super-Resolution GAN" (LPSRGAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OCRAwarePerceptualLoss(nn.Module):
    """
    OCR-aware perceptual loss that optimizes the generator based on
    recognition performance of a downstream OCR model.
    
    The core idea is to use the CTC loss (or cross-entropy) from an OCR
    network as a training signal for the generator. This encourages the
    generator to produce images with clear, recognizable characters rather
    than just pixel-accurate reconstructions.
    
    Benefits:
        - Improved character readability in generated images
        - Focus on high-frequency details important for OCR
        - Task-specific optimization beyond perceptual similarity
    """
    
    def __init__(
        self,
        ocr_model: nn.Module,
        loss_type: str = 'cross_entropy',
        weight: float = 1.0,
        freeze_ocr: bool = True,
        blank_idx: int = 0
    ):
        """
        Initialize OCR-aware perceptual loss.
        
        Args:
            ocr_model: Pre-trained OCR model that outputs logits.
            loss_type: Type of OCR loss ('cross_entropy' or 'ctc').
            weight: Weight for this loss component.
            freeze_ocr: Whether to freeze the OCR model weights.
            blank_idx: Blank index for CTC loss.
        """
        super().__init__()
        
        self.ocr_model = ocr_model
        self.loss_type = loss_type
        self.weight = weight
        
        if freeze_ocr:
            for param in self.ocr_model.parameters():
                param.requires_grad = False
            self.ocr_model.eval()
        
        if loss_type == 'ctc':
            self.criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        generated_hr: torch.Tensor,
        target_text: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute OCR-aware loss on generated HR images.
        
        Args:
            generated_hr: Generated HR images (B, C, H, W).
            target_text: Target text indices (B, T) for CE or (sum(target_lengths),) for CTC.
            input_lengths: Input sequence lengths for CTC loss.
            target_lengths: Target sequence lengths for CTC loss.
            
        Returns:
            OCR perceptual loss value.
        """
        # Get OCR predictions on generated image
        with torch.set_grad_enabled(True):  # Need gradients to flow through
            logits = self.ocr_model(generated_hr)
        
        if self.loss_type == 'ctc':
            # CTC expects (T, B, C) log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            if logits.dim() == 3 and logits.size(0) != logits.size(1):
                # Assume (B, T, C) format, need to transpose
                log_probs = log_probs.transpose(0, 1)
            
            loss = self.criterion(log_probs, target_text, input_lengths, target_lengths)
        else:
            # Cross-entropy loss
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = target_text.view(-1)
            loss = self.criterion(logits_flat, targets_flat)
        
        return self.weight * loss


class CharacterFocusLoss(nn.Module):
    """
    Character-focused perceptual loss that emphasizes edges and 
    high-frequency details important for character recognition.
    
    Uses gradient-based feature extraction to focus on character boundaries.
    """
    
    def __init__(self, edge_weight: float = 0.5, l1_weight: float = 0.5):
        """
        Initialize character focus loss.
        
        Args:
            edge_weight: Weight for edge/gradient loss.
            l1_weight: Weight for L1 reconstruction loss.
        """
        super().__init__()
        
        self.edge_weight = edge_weight
        self.l1_weight = l1_weight
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge magnitude using Sobel operator.
        
        Args:
            x: Input image (B, C, H, W).
            
        Returns:
            Edge magnitude (B, C, H, W).
        """
        B, C, H, W = x.shape
        
        # Convert to grayscale if RGB
        if C == 3:
            x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            x_gray = x
        
        # Apply Sobel filters
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return edges
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute character-focused loss.
        
        Args:
            pred: Predicted/generated image (B, C, H, W).
            target: Target image (B, C, H, W).
            
        Returns:
            Combined loss value.
        """
        # L1 reconstruction loss
        l1_loss = F.l1_loss(pred, target)
        
        # Edge loss
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.edge_weight * edge_loss
        
        return total_loss


class MultiScaleOCRLoss(nn.Module):
    """
    Multi-scale OCR loss that evaluates recognition at multiple resolutions.
    
    This helps the generator learn to produce images that are recognizable
    even when viewed at different scales, improving robustness.
    """
    
    def __init__(
        self,
        ocr_model: nn.Module,
        scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
        freeze_ocr: bool = True
    ):
        """
        Initialize multi-scale OCR loss.
        
        Args:
            ocr_model: Pre-trained OCR model.
            scales: Tuple of scales to evaluate at.
            freeze_ocr: Whether to freeze OCR model.
        """
        super().__init__()
        
        self.ocr_model = ocr_model
        self.scales = scales
        
        if freeze_ocr:
            for param in self.ocr_model.parameters():
                param.requires_grad = False
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        generated_hr: torch.Tensor,
        target_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-scale OCR loss.
        
        Args:
            generated_hr: Generated HR image (B, C, H, W).
            target_text: Target text indices (B, T).
            
        Returns:
            Multi-scale OCR loss.
        """
        total_loss = 0.0
        
        for scale in self.scales:
            if scale != 1.0:
                # Resize image
                scaled_size = (
                    int(generated_hr.size(2) * scale),
                    int(generated_hr.size(3) * scale)
                )
                scaled_img = F.interpolate(
                    generated_hr, size=scaled_size,
                    mode='bilinear', align_corners=False
                )
                # Resize back to original size for OCR model
                scaled_img = F.interpolate(
                    scaled_img, size=(generated_hr.size(2), generated_hr.size(3)),
                    mode='bilinear', align_corners=False
                )
            else:
                scaled_img = generated_hr
            
            # Get OCR predictions
            logits = self.ocr_model(scaled_img)
            
            # Compute loss
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = target_text.view(-1)
            loss = self.criterion(logits_flat, targets_flat)
            
            total_loss = total_loss + loss
        
        return total_loss / len(self.scales)

