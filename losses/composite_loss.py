"""
Composite Loss for End-to-End Training.

This module implements the complete loss function that combines all
individual losses for training the Neuro-Symbolic LPR system:
    L_total = L_pixel + 0.1 * L_GAN + 0.5 * L_OCR + 0.1 * L_geo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .corner_loss import CornerLoss
from .gan_loss import GANLoss, PixelLoss, PerceptualLoss


class OCRLoss(nn.Module):
    """
    OCR loss for character recognition.
    
    Supports both CTC loss and cross-entropy loss depending on
    the recognition architecture.
    """
    
    def __init__(
        self,
        loss_type: str = 'cross_entropy',
        blank_idx: int = 0,
        label_smoothing: float = 0.0
    ):
        """
        Initialize OCR loss.
        
        Args:
            loss_type: Type of loss ('cross_entropy' or 'ctc').
            blank_idx: Blank index for CTC loss.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.blank_idx = blank_idx
        
        if loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing
            )
        elif loss_type == 'ctc':
            self.loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute OCR loss.
        
        Args:
            logits: Predicted logits of shape (B, T, C) for CE or (T, B, C) for CTC.
            targets: Target indices of shape (B, T) for CE or (sum(target_lengths),) for CTC.
            input_lengths: Input sequence lengths for CTC.
            target_lengths: Target sequence lengths for CTC.
            
        Returns:
            OCR loss value.
        """
        if self.loss_type == 'cross_entropy':
            # Reshape logits: (B, T, C) -> (B*T, C)
            # Reshape targets: (B, T) -> (B*T,)
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            targets_flat = targets.view(-1)
            
            return self.loss(logits_flat, targets_flat)
        
        elif self.loss_type == 'ctc':
            # CTC expects (T, B, C) log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            if logits.dim() == 3 and logits.size(0) != logits.size(1):
                # Assume (B, T, C) format, need to transpose
                log_probs = log_probs.transpose(0, 1)
            
            return self.loss(log_probs, targets, input_lengths, target_lengths)


class LayoutLoss(nn.Module):
    """
    Binary cross-entropy loss for layout classification.
    """
    
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute layout classification loss.
        
        Args:
            logits: Layout logits of shape (B, 1).
            targets: Layout labels of shape (B,) or (B, 1).
            
        Returns:
            Layout loss value.
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        targets = targets.float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy_with_logits(logits, targets)


class CompositeLoss(nn.Module):
    """
    Complete composite loss for end-to-end training.
    
    Combines:
        - Pixel loss (L1) for image reconstruction
        - GAN loss for perceptual quality
        - OCR loss for character recognition
        - Geometry loss (corner loss) for STN supervision
        - Layout loss for classifier supervision
    
    Total loss:
        L_total = w_pixel * L_pixel + w_gan * L_GAN + 
                  w_ocr * L_OCR + w_geo * L_geo + w_layout * L_layout
    """
    
    def __init__(
        self,
        weight_pixel: float = 1.0,
        weight_gan: float = 0.1,
        weight_ocr: float = 0.5,
        weight_geometry: float = 0.1,
        weight_layout: float = 0.1,
        weight_perceptual: float = 0.0,
        gan_mode: str = 'vanilla',
        ocr_loss_type: str = 'cross_entropy',
        use_perceptual: bool = False
    ):
        """
        Initialize composite loss.
        
        Args:
            weight_pixel: Weight for pixel loss.
            weight_gan: Weight for GAN loss.
            weight_ocr: Weight for OCR loss.
            weight_geometry: Weight for geometry/corner loss.
            weight_layout: Weight for layout loss.
            weight_perceptual: Weight for perceptual loss.
            gan_mode: Type of GAN loss.
            ocr_loss_type: Type of OCR loss.
            use_perceptual: Whether to use perceptual loss.
        """
        super().__init__()
        
        self.weights = {
            'pixel': weight_pixel,
            'gan': weight_gan,
            'ocr': weight_ocr,
            'geometry': weight_geometry,
            'layout': weight_layout,
            'perceptual': weight_perceptual
        }
        
        # Individual losses
        self.pixel_loss = PixelLoss(criterion='l1')
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.ocr_loss = OCRLoss(loss_type=ocr_loss_type)
        self.corner_loss = CornerLoss()
        self.layout_loss = LayoutLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator: Optional[nn.Module] = None,
        compute_components: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss.
        
        Args:
            outputs: Model outputs containing:
                - 'hr_image': Generated HR image
                - 'masked_logits': Syntax-masked logits
                - 'layout_logits': Layout classification logits
                - 'corners': Predicted corners (optional)
            targets: Ground-truth targets containing:
                - 'hr_image': Real HR image
                - 'text_indices': Character indices
                - 'layout': Layout labels (0=Brazilian, 1=Mercosul)
                - 'corners': Corner coordinates (optional)
            discriminator: Discriminator network for GAN loss.
            compute_components: Whether to compute component losses.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Pixel loss
        if 'hr_image' in outputs and 'hr_image' in targets:
            l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
            loss_dict['pixel'] = l_pixel.item()
            total_loss = total_loss + self.weights['pixel'] * l_pixel
        
        # GAN loss (generator side)
        if discriminator is not None and 'hr_image' in outputs:
            pred_fake = discriminator(outputs['hr_image'])
            l_gan = self.gan_loss(pred_fake, target_is_real=True)
            loss_dict['gan'] = l_gan.item()
            total_loss = total_loss + self.weights['gan'] * l_gan
        
        # OCR loss
        if 'masked_logits' in outputs and 'text_indices' in targets:
            l_ocr = self.ocr_loss(outputs['masked_logits'], targets['text_indices'])
            loss_dict['ocr'] = l_ocr.item()
            total_loss = total_loss + self.weights['ocr'] * l_ocr
        
        # Geometry loss
        if 'corners' in outputs and 'corners' in targets:
            # Handle multi-frame corners
            pred_corners = outputs['corners']
            gt_corners = targets['corners']
            
            if pred_corners.dim() == 4:  # (B, T, 4, 2)
                # Average over frames
                pred_corners = pred_corners.mean(dim=1)
            
            l_geo = self.corner_loss(pred_corners, gt_corners)
            loss_dict['geometry'] = l_geo.item()
            total_loss = total_loss + self.weights['geometry'] * l_geo
        
        # Layout loss
        if 'layout_logits' in outputs and 'layout' in targets:
            l_layout = self.layout_loss(outputs['layout_logits'], targets['layout'])
            loss_dict['layout'] = l_layout.item()
            total_loss = total_loss + self.weights['layout'] * l_layout
        
        # Perceptual loss
        if self.perceptual_loss is not None and 'hr_image' in outputs and 'hr_image' in targets:
            l_perceptual = self.perceptual_loss(outputs['hr_image'], targets['hr_image'])
            loss_dict['perceptual'] = l_perceptual.item()
            total_loss = total_loss + self.weights['perceptual'] * l_perceptual
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict
    
    def get_stage_loss(
        self,
        stage: str,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get loss for a specific training stage.
        
        Args:
            stage: Training stage ('stn', 'restoration', 'full').
            outputs: Model outputs.
            targets: Ground-truth targets.
            discriminator: Discriminator for GAN loss.
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        if stage == 'stn':
            # Only corner loss for STN training
            if 'corners' in outputs and 'corners' in targets:
                pred_corners = outputs['corners']
                gt_corners = targets['corners']
                
                if pred_corners.dim() == 4:
                    pred_corners = pred_corners.mean(dim=1)
                
                loss = self.corner_loss(pred_corners, gt_corners)
                return loss, {'geometry': loss.item()}
            else:
                # Create a zero loss that maintains gradient flow through model outputs
                # This allows backward() to work even when corner annotations are missing
                if 'corners' in outputs:
                    loss = outputs['corners'].sum() * 0.0
                elif 'hr_image' in outputs:
                    loss = outputs['hr_image'].sum() * 0.0
                else:
                    # Fallback: use any available output tensor
                    any_output = next(iter(outputs.values()))
                    loss = any_output.sum() * 0.0
                return loss, {'geometry': 0.0}
        
        elif stage == 'restoration':
            # Pixel + GAN + Layout losses
            loss_dict = {}
            total_loss = 0.0
            
            if 'hr_image' in outputs and 'hr_image' in targets:
                l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
                loss_dict['pixel'] = l_pixel.item()
                total_loss = total_loss + l_pixel
            
            if discriminator is not None and 'hr_image' in outputs:
                pred_fake = discriminator(outputs['hr_image'])
                l_gan = self.gan_loss(pred_fake, target_is_real=True)
                loss_dict['gan'] = l_gan.item()
                total_loss = total_loss + 0.1 * l_gan
            
            if 'layout_logits' in outputs and 'layout' in targets:
                l_layout = self.layout_loss(outputs['layout_logits'], targets['layout'])
                loss_dict['layout'] = l_layout.item()
                total_loss = total_loss + 0.1 * l_layout
            
            loss_dict['total'] = total_loss.item()
            return total_loss, loss_dict
        
        else:  # 'full'
            return self.forward(outputs, targets, discriminator)


class StagedTrainingManager:
    """
    Manager for staged training schedule.
    
    Handles the 4-step training process:
        Step 0: Synthetic pre-training (PARSeq)
        Step 1: Geometry warm-up (STN)
        Step 2: Restoration & Layout (SwinIR, Classifier)
        Step 3: End-to-end fine-tuning
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict
    ):
        """
        Initialize training manager.
        
        Args:
            model: NeuroSymbolicLPR model.
            config: Training configuration.
        """
        self.model = model
        self.config = config
        self.current_stage = 0
        
        # Create loss function
        self.criterion = CompositeLoss(
            weight_pixel=config.get('weight_pixel', 1.0),
            weight_gan=config.get('weight_gan', 0.1),
            weight_ocr=config.get('weight_ocr', 0.5),
            weight_geometry=config.get('weight_geometry', 0.1)
        )
    
    def get_stage_config(self, stage: int) -> dict:
        """Get configuration for a specific training stage."""
        stage_configs = {
            0: {
                'name': 'pretrain_recognizer',
                'modules': ['recognizer'],
                'lr': self.config.get('lr_pretrain', 1e-4),
                'epochs': self.config.get('epochs_pretrain', 50),
                'loss_type': 'ocr_only'
            },
            1: {
                'name': 'geometry_warmup',
                'modules': ['encoder', 'stn'],
                'lr': self.config.get('lr_stn', 1e-4),
                'epochs': self.config.get('epochs_stn', 20),
                'loss_type': 'geometry_only'
            },
            2: {
                'name': 'restoration_layout',
                'modules': ['generator', 'layout_classifier', 'quality_fusion'],
                'lr': self.config.get('lr_restoration', 2e-4),
                'epochs': self.config.get('epochs_restoration', 30),
                'loss_type': 'restoration'
            },
            3: {
                'name': 'end_to_end',
                'modules': ['all'],
                'lr': self.config.get('lr_finetune', 1e-5),
                'epochs': self.config.get('epochs_finetune', 100),
                'loss_type': 'full'
            }
        }
        return stage_configs[stage]
    
    def prepare_stage(self, stage: int):
        """
        Prepare model for a specific training stage.
        
        Args:
            stage: Stage number (0-3).
        """
        self.current_stage = stage
        config = self.get_stage_config(stage)
        
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze stage-specific modules
        if config['modules'] == ['all']:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for module_name in config['modules']:
                if hasattr(self.model, module_name):
                    for param in getattr(self.model, module_name).parameters():
                        param.requires_grad = True
        
        return config
    
    def get_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get loss for current training stage.
        
        Args:
            outputs: Model outputs.
            targets: Ground-truth targets.
            discriminator: Discriminator for GAN training.
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        config = self.get_stage_config(self.current_stage)
        loss_type = config['loss_type']
        
        if loss_type == 'ocr_only':
            loss = self.criterion.ocr_loss(
                outputs['masked_logits'], 
                targets['text_indices']
            )
            return loss, {'ocr': loss.item()}
        
        elif loss_type == 'geometry_only':
            return self.criterion.get_stage_loss('stn', outputs, targets)
        
        elif loss_type == 'restoration':
            return self.criterion.get_stage_loss('restoration', outputs, targets, discriminator)
        
        else:  # 'full'
            return self.criterion(outputs, targets, discriminator)
