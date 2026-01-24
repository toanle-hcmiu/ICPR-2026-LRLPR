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
from .lcofl_loss import LCOFLLoss, SSIMLoss, ConfusionMatrixTracker


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness.
    
    Penalizes high-frequency noise and "wavy" artifacts while preserving sharp edges.
    Based on Rudin-Osher-Fatemi denoising model.
    
    This is the recommended fix for wavy distortions caused by:
    - Gibbs phenomenon ringing near edges
    - Aliasing from downsampling
    - Checkerboard artifacts from upsampling
    
    Reference: Rudin, Osher, Fatemi "Nonlinear total variation based noise removal algorithms" (1992)
    
    Args:
        weight: Loss weight (recommended: 1e-5 to 1e-4 for subtle smoothing).
        reduction: Loss reduction mode ('mean' or 'sum').
    """
    
    def __init__(self, weight: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Total Variation loss.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            TV loss value (scalar).
        """
        # Compute horizontal differences (left-right gradients)
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        
        # Compute vertical differences (top-bottom gradients)
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        
        # Sum of absolute differences (L1 Total Variation)
        # L1 preserves sharp edges better than L2 (allows jump discontinuities)
        if self.reduction == 'mean':
            tv_loss = diff_h.mean() + diff_w.mean()
        else:
            tv_loss = diff_h.sum() + diff_w.sum()
        
        return self.weight * tv_loss

class SelfSupervisedSTNLoss(nn.Module):
    """
    Self-supervised loss for STN training when corner annotations are not available.
    
    This loss encourages the STN to learn meaningful geometric transformations by:
    1. Multi-frame consistency: Different frames of the same plate should have similar transformations
    2. Temporal smoothness: Adjacent frames should have smooth transformation changes
    3. Geometric plausibility: Transformations should be within reasonable bounds
    
    The key insight is that we want the STN to LEARN to rectify while maintaining
    consistency across frames. The loss provides regularization without collapsing
    to identity (which would be trivially consistent).
    
    Includes numerical stability safeguards to prevent NaN/Inf losses.
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.5,  # Reduced to prevent dominating
        temporal_weight: float = 0.3,     # Reduced for smoother learning
        plausibility_weight: float = 0.1, # Light regularization
        max_loss_value: float = 20.0,     # Tighter clamp for STN stability
        eps: float = 1e-8
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temporal_weight = temporal_weight
        self.plausibility_weight = plausibility_weight
        self.max_loss_value = max_loss_value
        self.eps = eps
        
        # Identity transformation matrix (2x3): [[1, 0, 0], [0, 1, 0]]
        self.register_buffer(
            'identity_theta',
            torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
        )
    
    def _safe_loss(self, loss: torch.Tensor, name: str = "") -> torch.Tensor:
        """Ensure loss is valid and clamped."""
        if loss is None:
            return loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return loss * 0.0
        return torch.clamp(loss, min=0.0, max=self.max_loss_value)
    
    def forward(
        self,
        thetas: torch.Tensor,
        corners: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute self-supervised STN loss.
        
        Args:
            thetas: Affine transformation parameters of shape (B, T, 2, 3) or (B, 2, 3).
            corners: Predicted corners of shape (B, T, 4, 2) or (B, 4, 2) (optional).
            
        Returns:
            Loss value (clamped to prevent explosion).
        """
        # Check for NaN/Inf inputs and return zero loss
        if torch.isnan(thetas).any() or torch.isinf(thetas).any():
            return thetas.new_zeros(1, requires_grad=True).squeeze() + thetas.sum() * 0.0
        
        # Handle single-frame case
        if thetas.dim() == 3:
            thetas = thetas.unsqueeze(1)  # (B, 1, 2, 3)
        
        B, T = thetas.shape[:2]
        
        # Clamp theta values to reasonable range (tighter bounds for stability)
        thetas_clamped = torch.clamp(thetas, min=-2.0, max=2.0)
        
        total_loss = thetas.new_zeros(1).squeeze()
        
        # 1. Multi-frame consistency loss
        # All frames of the same license plate should have similar transformations
        if T > 1:
            mean_theta = thetas_clamped.mean(dim=1, keepdim=True)  # (B, 1, 2, 3)
            # Use smooth L1 for robustness to outliers
            consistency_loss = F.smooth_l1_loss(thetas_clamped, mean_theta.expand_as(thetas_clamped))
            consistency_loss = self._safe_loss(consistency_loss, "consistency")
            total_loss = total_loss + self.consistency_weight * consistency_loss
        
        # 2. Temporal smoothness loss
        # Adjacent frames should have smooth transformation changes
        if T > 1:
            # Difference between adjacent frames
            theta_diff = thetas_clamped[:, 1:] - thetas_clamped[:, :-1]  # (B, T-1, 2, 3)
            # Use smooth L1 for robustness
            temporal_loss = F.smooth_l1_loss(theta_diff, torch.zeros_like(theta_diff))
            temporal_loss = self._safe_loss(temporal_loss, "temporal")
            total_loss = total_loss + self.temporal_weight * temporal_loss
        
        # 3. Geometric plausibility loss (light regularization)
        # Only penalize extreme values, not moderate transformations
        scale_x = thetas_clamped[:, :, 0, 0]
        scale_y = thetas_clamped[:, :, 1, 1]
        
        # Penalize scales very far from 1 (threshold at 0.7 to allow rectification)
        scale_deviation = (scale_x - 1).abs() + (scale_y - 1).abs()
        scale_penalty = F.relu(scale_deviation - 0.7) ** 2
        
        # Shear should be small (off-diagonal elements)
        shear_x = thetas_clamped[:, :, 0, 1]
        shear_y = thetas_clamped[:, :, 1, 0]
        # Only penalize large shear (threshold at 0.4)
        shear_penalty = F.relu(shear_x.abs() - 0.4) ** 2 + F.relu(shear_y.abs() - 0.4) ** 2
        
        # Translation penalty only for extreme values
        trans_x = thetas_clamped[:, :, 0, 2]
        trans_y = thetas_clamped[:, :, 1, 2]
        trans_penalty = F.relu(trans_x.abs() - 0.7) ** 2 + F.relu(trans_y.abs() - 0.7) ** 2
        
        plausibility_loss = (scale_penalty + shear_penalty + trans_penalty).mean()
        plausibility_loss = self._safe_loss(plausibility_loss, "plausibility")
        total_loss = total_loss + self.plausibility_weight * plausibility_loss
        
        # 4. Corner-based consistency if corners are available
        if corners is not None and not (torch.isnan(corners).any() or torch.isinf(corners).any()):
            if corners.dim() == 3:
                corners = corners.unsqueeze(1)  # (B, 1, 4, 2)
            
            # Clamp corners for stability
            corners = torch.clamp(corners, min=-2.0, max=2.0)
            
            if corners.size(1) == T and T > 1:
                # Corner consistency: all frames should predict similar corners
                mean_corners = corners.mean(dim=1, keepdim=True)
                corner_consistency = F.smooth_l1_loss(corners, mean_corners.expand_as(corners))
                corner_consistency = self._safe_loss(corner_consistency, "corner_consistency")
                total_loss = total_loss + 0.3 * corner_consistency
                
                # Corner rectangularity: encourage plate-like aspect ratio
                # Top edge should be horizontal-ish, left edge should be vertical-ish
                top_edge = corners[:, :, 1] - corners[:, :, 0]
                left_edge = corners[:, :, 3] - corners[:, :, 0]
                
                # Edges should be roughly perpendicular (dot product near zero)
                dot_product = (top_edge * left_edge).sum(dim=-1)
                orthogonality_loss = (dot_product ** 2).mean()
                orthogonality_loss = self._safe_loss(orthogonality_loss, "orthogonality")
                total_loss = total_loss + 0.1 * orthogonality_loss
        
        # Final clamp
        total_loss = torch.clamp(total_loss, min=0.0, max=self.max_loss_value)
        
        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return thetas.new_zeros(1, requires_grad=True).squeeze() + thetas.sum() * 0.0
        
        return total_loss


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
            logits: Predicted logits of shape (B, T_logits, C) for CE or (T, B, C) for CTC.
            targets: Target indices of shape (B, T_targets) for CE or (sum(target_lengths),) for CTC.
                     For CE, T_targets may be T_logits + 2 if targets include BOS and EOS tokens.
            input_lengths: Input sequence lengths for CTC.
            target_lengths: Target sequence lengths for CTC.
            
        Returns:
            OCR loss value.
        """
        if self.loss_type == 'cross_entropy':
            # Reshape logits: (B, T_logits, C) -> (B*T_logits, C)
            # Reshape targets: (B, T_targets) -> (B*T_targets,) 
            # Note: T_targets may include BOS/EOS tokens that logits don't have
            B, T_logits, C = logits.shape
            T_targets = targets.shape[1]
            
            # Handle sequence length mismatch:
            # If targets is longer (has BOS/EOS), slice to match logits length
            # Targets format: [BOS, char1, ..., charN, EOS, PAD, ...]
            # Logits format: [char1, ..., charN] (no BOS/EOS)
            if T_targets > T_logits:
                # Slice targets to extract character positions (skip BOS at start)
                # Take T_logits positions starting from index 1 (after BOS)
                targets = targets[:, 1:1 + T_logits]
            elif T_targets < T_logits:
                # Unlikely case: pad targets to match logits length
                # Use ignore_index (-100) for padding
                padding = torch.full(
                    (B, T_logits - T_targets), 
                    -100, 
                    dtype=targets.dtype, 
                    device=targets.device
                )
                targets = torch.cat([targets, padding], dim=1)
            
            logits_flat = logits.view(-1, C)
            targets_flat = targets.reshape(-1)
            
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
    
    Handles invalid layout labels (layout=-1) by excluding them from loss computation.
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
                     Values should be 0 (Brazilian), 1 (Mercosul), or -1 (invalid).
            
        Returns:
            Layout loss value. Returns 0 if no valid samples.
        """
        if targets.dim() == 1:
            targets_expanded = targets.unsqueeze(-1)
        else:
            targets_expanded = targets
        
        # Create mask for valid layout labels (0 or 1, not -1)
        valid_mask = (targets >= 0).view(-1)
        
        if not valid_mask.any():
            # No valid samples - return zero loss with gradient connection
            return logits.sum() * 0.0
        
        # Filter to valid samples only
        valid_logits = logits.view(-1, 1)[valid_mask]
        valid_targets = targets_expanded.view(-1, 1)[valid_mask].float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            valid_targets = valid_targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets)


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
        weight_lcofl: float = 0.0,
        weight_ssim: float = 0.0,
        weight_tv: float = 0.0,  # Total Variation loss weight for wavy artifact suppression
        gan_mode: str = 'vanilla',
        ocr_loss_type: str = 'cross_entropy',
        use_perceptual: bool = False,
        use_lcofl: bool = False,
        lcofl_alpha: float = 1.0,
        lcofl_beta: float = 2.0
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
            weight_lcofl: Weight for LCOFL loss (0 to disable).
            weight_ssim: Weight for SSIM loss (0 to disable).
            weight_tv: Weight for Total Variation loss (0 to disable, 1e-5 recommended for wavy suppression).
            gan_mode: Type of GAN loss.
            ocr_loss_type: Type of OCR loss.
            use_perceptual: Whether to use perceptual loss.
            use_lcofl: Whether to use LCOFL loss.
            lcofl_alpha: LCOFL penalty increment for confused characters.
            lcofl_beta: LCOFL layout violation penalty.
        """
        super().__init__()
        
        self.weights = {
            'pixel': weight_pixel,
            'gan': weight_gan,
            'ocr': weight_ocr,
            'geometry': weight_geometry,
            'layout': weight_layout,
            'perceptual': weight_perceptual,
            'lcofl': weight_lcofl,
            'ssim': weight_ssim,
            'tv': weight_tv  # Total Variation for wavy artifact suppression
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
        
        # LCOFL loss (from paper: "Enhancing LP Super-Resolution")
        if use_lcofl or weight_lcofl > 0:
            self.lcofl_loss = LCOFLLoss(
                weight_classification=1.0,
                weight_layout=1.0,
                weight_ssim=0.3,
                alpha=lcofl_alpha,
                beta=lcofl_beta
            )
        else:
            self.lcofl_loss = None
        
        # Standalone SSIM loss (can be used without full LCOFL)
        if weight_ssim > 0 and self.lcofl_loss is None:
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None
        
        # Total Variation loss for suppressing wavy/checkerboard artifacts
        # Recommended weight: 1e-5 to 1e-4 for subtle smoothing without blur
        if weight_tv > 0:
            self.tv_loss = TotalVariationLoss(weight=1.0)  # Weight applied via self.weights
        else:
            self.tv_loss = None
    
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
        total_loss = None
        
        # Pixel loss
        if 'hr_image' in outputs and 'hr_image' in targets:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
                l_pixel = self._clamp_loss(l_pixel)
                loss_dict['pixel'] = self._safe_loss_item(l_pixel)
                weighted = self.weights['pixel'] * l_pixel
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # GAN loss (generator side)
        if discriminator is not None and 'hr_image' in outputs:
            if not torch.isnan(outputs['hr_image']).any():
                pred_fake = discriminator(outputs['hr_image'])
                if not torch.isnan(pred_fake).any():
                    l_gan = self.gan_loss(pred_fake, target_is_real=True)
                    l_gan = self._clamp_loss(l_gan)
                    loss_dict['gan'] = self._safe_loss_item(l_gan)
                    weighted = self.weights['gan'] * l_gan
                    total_loss = weighted if total_loss is None else total_loss + weighted
        
        # OCR loss
        if 'masked_logits' in outputs and 'text_indices' in targets:
            if not (torch.isnan(outputs['masked_logits']).any() or torch.isnan(targets['text_indices'].float()).any()):
                l_ocr = self.ocr_loss(outputs['masked_logits'], targets['text_indices'])
                l_ocr = self._clamp_loss(l_ocr)
                loss_dict['ocr'] = self._safe_loss_item(l_ocr)
                weighted = self.weights['ocr'] * l_ocr
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # Geometry loss
        if 'corners' in outputs and 'corners' in targets:
            pred_corners = outputs['corners']
            gt_corners = targets['corners']
            
            if not (torch.isnan(pred_corners).any() or torch.isnan(gt_corners).any()):
                # Handle multi-frame corners
                if pred_corners.dim() == 4:  # (B, T, 4, 2)
                    # Average over frames
                    pred_corners = pred_corners.mean(dim=1)
                
                l_geo = self.corner_loss(pred_corners, gt_corners)
                l_geo = self._clamp_loss(l_geo)
                loss_dict['geometry'] = self._safe_loss_item(l_geo)
                weighted = self.weights['geometry'] * l_geo
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # Layout loss
        if 'layout_logits' in outputs and 'layout' in targets:
            if not torch.isnan(outputs['layout_logits']).any():
                l_layout = self.layout_loss(outputs['layout_logits'], targets['layout'])
                l_layout = self._clamp_loss(l_layout)
                loss_dict['layout'] = self._safe_loss_item(l_layout)
                weighted = self.weights['layout'] * l_layout
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # Perceptual loss
        if self.perceptual_loss is not None and 'hr_image' in outputs and 'hr_image' in targets:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                l_perceptual = self.perceptual_loss(outputs['hr_image'], targets['hr_image'])
                l_perceptual = self._clamp_loss(l_perceptual)
                loss_dict['perceptual'] = self._safe_loss_item(l_perceptual)
                weighted = self.weights['perceptual'] * l_perceptual
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # LCOFL loss (from Nascimento et al. paper)
        if self.lcofl_loss is not None and 'masked_logits' in outputs and 'text_indices' in targets:
            if not torch.isnan(outputs['masked_logits']).any():
                # Get layout info for LCOFL
                is_mercosul = targets.get('layout', torch.zeros(outputs['masked_logits'].size(0), device=outputs['masked_logits'].device))
                
                # Strip BOS/EOS from targets to match masked_logits shape
                # text_indices: (B, PLATE_LENGTH+2) with [BOS, char1, ..., char7, EOS]
                # masked_logits: (B, PLATE_LENGTH, V)
                plate_length = outputs['masked_logits'].size(1)
                lcofl_targets = targets['text_indices'][:, 1:plate_length+1]  # (B, PLATE_LENGTH)
                
                l_lcofl, lcofl_dict = self.lcofl_loss(
                    logits=outputs['masked_logits'],
                    targets=lcofl_targets,
                    is_mercosul=is_mercosul,
                    generated_hr=outputs.get('hr_image'),
                    gt_hr=targets.get('hr_image'),
                    compute_ssim=self.weights.get('ssim', 0) > 0 or 'hr_image' in targets
                )
                l_lcofl = self._clamp_loss(l_lcofl)
                loss_dict['lcofl'] = self._safe_loss_item(l_lcofl)
                # Add LCOFL sub-components to loss dict for logging
                for k, v in lcofl_dict.items():
                    if k != 'total':
                        loss_dict[f'lcofl_{k}'] = v
                weighted = self.weights['lcofl'] * l_lcofl
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # Standalone SSIM loss (when not using full LCOFL)
        if self.ssim_loss is not None and 'hr_image' in outputs and 'hr_image' in targets:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                l_ssim = self.ssim_loss(outputs['hr_image'], targets['hr_image'])
                l_ssim = self._clamp_loss(l_ssim)
                loss_dict['ssim'] = self._safe_loss_item(l_ssim)
                weighted = self.weights['ssim'] * l_ssim
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # Total Variation loss for suppressing wavy/checkerboard artifacts
        if self.tv_loss is not None and 'hr_image' in outputs:
            if not torch.isnan(outputs['hr_image']).any():
                l_tv = self.tv_loss(outputs['hr_image'])
                l_tv = self._clamp_loss(l_tv)
                loss_dict['tv'] = self._safe_loss_item(l_tv)
                weighted = self.weights['tv'] * l_tv
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # If no loss computed, create zero loss with gradient
        if total_loss is None:
            if 'hr_image' in outputs:
                total_loss = outputs['hr_image'].sum() * 0.0
            else:
                any_output = next(iter(outputs.values()))
                total_loss = any_output.sum() * 0.0
        
        # Final clamp
        total_loss = self._clamp_loss(total_loss, max_val=100.0)
        
        loss_dict['total'] = self._safe_loss_item(total_loss)
        
        return total_loss, loss_dict
    
    def _safe_loss_item(self, loss: torch.Tensor) -> float:
        """Safely get loss item, returning NaN-safe float."""
        if loss is None:
            return 0.0
        val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        # Check for NaN/Inf and return 0 instead
        if not (val == val) or val == float('inf') or val == float('-inf'):
            return float('nan')
        return val
    
    def _clamp_loss(self, loss: torch.Tensor, max_val: float = 100.0) -> torch.Tensor:
        """Clamp loss to prevent explosion while maintaining gradients."""
        if loss is None:
            return None
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            # Return zero tensor with gradient connection
            return loss * 0.0
        return torch.clamp(loss, min=0.0, max=max_val)
    
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
            stage: Training stage ('pretrain', 'stn', 'restoration', 'full').
            outputs: Model outputs.
            targets: Ground-truth targets.
            discriminator: Discriminator for GAN loss.
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        if stage == 'pretrain':
            # Pretrain stage: OCR loss only (train recognizer on synthetic data)
            loss_dict = {}
            total_loss = None
            
            # OCR loss - the only loss used during pretrain
            if 'masked_logits' in outputs and 'text_indices' in targets:
                logits = outputs['masked_logits']
                text_targets = targets['text_indices']
                
                # Check for NaN inputs
                if not (torch.isnan(logits).any() or torch.isnan(text_targets.float()).any()):
                    l_ocr = self.ocr_loss(logits, text_targets)
                    l_ocr = self._clamp_loss(l_ocr)
                    loss_dict['ocr'] = self._safe_loss_item(l_ocr)
                    total_loss = l_ocr
            
            # Fallback if no loss was computed
            if total_loss is None:
                # Create zero loss with gradient connection
                # Use nan_to_num to handle NaN values and ensure finite fallback loss
                if 'masked_logits' in outputs:
                    sanitized = torch.nan_to_num(outputs['masked_logits'], nan=0.0, posinf=0.0, neginf=0.0)
                    total_loss = sanitized.sum() * 0.0
                elif 'raw_logits' in outputs:
                    sanitized = torch.nan_to_num(outputs['raw_logits'], nan=0.0, posinf=0.0, neginf=0.0)
                    total_loss = sanitized.sum() * 0.0
                else:
                    any_output = next(iter(outputs.values()))
                    sanitized = torch.nan_to_num(any_output, nan=0.0, posinf=0.0, neginf=0.0)
                    total_loss = sanitized.sum() * 0.0
            
            loss_dict['total'] = self._safe_loss_item(total_loss)
            return total_loss, loss_dict
        
        elif stage == 'stn':
            loss_dict = {}
            total_loss = None
            has_corner_supervision = False
            
            # Supervised corner loss if ground-truth corners are available
            if 'corners' in outputs and 'corners' in targets:
                pred_corners = outputs['corners']
                gt_corners = targets['corners']
                
                # Check for NaN inputs
                if not (torch.isnan(pred_corners).any() or torch.isnan(gt_corners).any()):
                    if pred_corners.dim() == 4:
                        pred_corners = pred_corners.mean(dim=1)
                    
                    corner_loss = self.corner_loss(pred_corners, gt_corners)
                    corner_loss = self._clamp_loss(corner_loss)
                    loss_dict['corner'] = self._safe_loss_item(corner_loss)
                    total_loss = corner_loss if total_loss is None else total_loss + corner_loss
                    has_corner_supervision = True
            
            # Self-supervised STN loss using transformation parameters
            # This provides meaningful gradients even without corner annotations
            if 'thetas' in outputs:
                if not hasattr(self, 'stn_self_supervised_loss'):
                    self.stn_self_supervised_loss = SelfSupervisedSTNLoss(
                        consistency_weight=0.5,   # Moderate multi-frame consistency
                        temporal_weight=0.3,      # Smooth temporal changes
                        plausibility_weight=0.1,  # Light regularization on geometry
                        max_loss_value=20.0       # Tighter clamp for STN stability
                    )
                    # Move to same device as outputs
                    device = outputs['thetas'].device
                    self.stn_self_supervised_loss = self.stn_self_supervised_loss.to(device)
                
                self_sup_loss = self.stn_self_supervised_loss(
                    outputs['thetas'],
                    outputs.get('corners', None)
                )
                self_sup_loss = self._clamp_loss(self_sup_loss, max_val=20.0)
                loss_dict['self_supervised'] = self._safe_loss_item(self_sup_loss)
                # Use moderate weight for self-supervised loss (regularizer)
                # Not too small (needs to provide signal) or too large (shouldn't dominate)
                weighted_self_sup = 0.2 * self_sup_loss
                total_loss = weighted_self_sup if total_loss is None else total_loss + weighted_self_sup
            
            # ALWAYS use pixel reconstruction loss as the PRIMARY learning signal
            # This is the main gradient source for STN learning
            # The STN learns to rectify by minimizing reconstruction error
            if 'hr_image' in outputs and 'hr_image' in targets:
                # Check for NaN inputs
                if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                    l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
                    l_pixel = self._clamp_loss(l_pixel)
                    loss_dict['pixel'] = self._safe_loss_item(l_pixel)
                    # Pixel loss is the main learning signal for STN
                    # Use full weight (1.0) to drive actual learning
                    weighted_pixel = 1.0 * l_pixel
                    total_loss = weighted_pixel if total_loss is None else total_loss + weighted_pixel
            
            # If no useful loss computed, create a minimal loss with gradient flow
            if total_loss is None:
                if 'corners' in outputs:
                    total_loss = outputs['corners'].sum() * 0.0
                elif 'thetas' in outputs:
                    total_loss = outputs['thetas'].sum() * 0.0
                elif 'hr_image' in outputs:
                    total_loss = outputs['hr_image'].sum() * 0.0
                else:
                    any_output = next(iter(outputs.values()))
                    total_loss = any_output.sum() * 0.0
            
            # Final clamp and NaN check
            total_loss = self._clamp_loss(total_loss, max_val=100.0)
            
            loss_dict['geometry'] = self._safe_loss_item(total_loss)
            return total_loss, loss_dict
        
        elif stage == 'restoration':
            # Pixel + GAN + Layout losses
            loss_dict = {}
            total_loss = None
            
            # Get a reference tensor that definitely has gradients for fallback
            ref_tensor = None
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    ref_tensor = v
                    break
            
            # For restoration, only check hr_image for NaN (other outputs like masked_logits may have NaN during early training)
            hr_image_has_nan = False
            if 'hr_image' in outputs:
                hr_image_has_nan = torch.isnan(outputs['hr_image']).any() or torch.isinf(outputs['hr_image']).any()
            
            if hr_image_has_nan:
                # HR image is corrupted, return zero loss
                if ref_tensor is not None:
                    zero_loss = ref_tensor.sum() * 0.0
                else:
                    any_output = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
                    zero_loss = (any_output.detach() * 0.0).sum().requires_grad_(True)
                loss_dict['total'] = 0.0
                loss_dict['pixel'] = 0.0
                return zero_loss, loss_dict
            
            if 'hr_image' in outputs and 'hr_image' in targets:
                # Check for NaN in targets as well
                if not torch.isnan(targets['hr_image']).any():
                    l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
                    l_pixel = self._clamp_loss(l_pixel)
                    loss_dict['pixel'] = self._safe_loss_item(l_pixel)
                    total_loss = l_pixel
            
            if discriminator is not None and 'hr_image' in outputs:
                if not torch.isnan(outputs['hr_image']).any():
                    pred_fake = discriminator(outputs['hr_image'])
                    if not torch.isnan(pred_fake).any():
                        l_gan = self.gan_loss(pred_fake, target_is_real=True)
                        l_gan = self._clamp_loss(l_gan)
                        loss_dict['gan'] = self._safe_loss_item(l_gan)
                        # Use config GAN weight (removed artificial cap of 0.01)
                        weighted_gan = self.weights['gan'] * l_gan
                        total_loss = weighted_gan if total_loss is None else total_loss + weighted_gan
            
            if 'layout_logits' in outputs and 'layout' in targets:
                if not (torch.isnan(outputs['layout_logits']).any()):
                    l_layout = self.layout_loss(outputs['layout_logits'], targets['layout'])
                    l_layout = self._clamp_loss(l_layout)
                    loss_dict['layout'] = self._safe_loss_item(l_layout)
                    weighted_layout = 0.1 * l_layout
                    total_loss = weighted_layout if total_loss is None else total_loss + weighted_layout
            
            # Perceptual loss for sharper images
            if self.perceptual_loss is not None and 'hr_image' in outputs and 'hr_image' in targets:
                if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                    # Denormalize from [-1, 1] to [0, 1] for VGG
                    pred_for_vgg = (outputs['hr_image'] + 1) / 2
                    target_for_vgg = (targets['hr_image'] + 1) / 2
                    l_perceptual = self.perceptual_loss(pred_for_vgg, target_for_vgg)
                    l_perceptual = self._clamp_loss(l_perceptual)
                    loss_dict['perceptual'] = self._safe_loss_item(l_perceptual)
                    weighted_perceptual = self.weights['perceptual'] * l_perceptual
                    total_loss = weighted_perceptual if total_loss is None else total_loss + weighted_perceptual
            
            # If no loss was computed, create zero loss with gradient
            if total_loss is None:
                if ref_tensor is not None:
                    total_loss = ref_tensor.sum() * 0.0
                elif 'hr_image' in outputs:
                    total_loss = outputs['hr_image'].sum() * 0.0
                else:
                    any_output = next(iter(outputs.values()))
                    if isinstance(any_output, torch.Tensor):
                        total_loss = any_output.sum() * 0.0
                    else:
                        # Create a minimal grad-enabled tensor
                        total_loss = torch.tensor(0.0, requires_grad=True, device=next(iter(outputs.values())).device if outputs else 'cuda')
            
            # Ensure total_loss requires grad
            if not total_loss.requires_grad:
                # This shouldn't happen, but add safeguard
                if ref_tensor is not None:
                    total_loss = total_loss + ref_tensor.sum() * 0.0
                else:
                    total_loss = total_loss.requires_grad_(True)
            
            # Final clamp
            total_loss = self._clamp_loss(total_loss, max_val=100.0)
            
            loss_dict['total'] = self._safe_loss_item(total_loss)
            return total_loss, loss_dict
        
        else:  # 'full' - Stage 3 with anti-collapse mechanisms
            return self.get_stage3_loss(
                outputs, targets, discriminator,
                global_step=0,  # Default, should be passed from train.py
                sr_stage2=None,  # Default, should be passed from train.py
                ocr_baseline=None  # Default OCR loss from Stage 2
            )
    
    def get_stage3_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        discriminator: Optional[nn.Module] = None,
        global_step: int = 0,
        sr_stage2: Optional[torch.Tensor] = None,
        ocr_baseline: Optional[float] = None,
        config: Optional['TrainingConfig'] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage 3 loss with anti-collapse mechanisms.
        
        Prevents OCR gradients from dominating and causing visual quality degradation.
        
        Anti-collapse mechanisms:
        1. SR Anchor Loss: Anchors Stage 3 SR output to Stage 2 SR output
        2. OCR Warmup: Gates OCR loss for first N steps, then ramps up gradually
        3. Hinge-Style OCR: Only penalizes if OCR gets worse than Stage 2 baseline
        
        Args:
            outputs: Model outputs (hr_image, masked_logits, etc.)
            targets: Ground truth targets
            discriminator: Optional discriminator for GAN loss
            global_step: Current training step (for OCR warmup scheduling)
            sr_stage2: Detached SR output from Stage 2 model (for anchoring)
            ocr_baseline: OCR loss value from Stage 2 (for hinge constraint)
            config: Training config with Stage 3 parameters
        """
        loss_dict = {}
        total_loss = None
        
        # Default Stage 3 parameters (can be overridden by config)
        sr_anchor_weight = 1.0
        ocr_warmup_steps = 3000
        ocr_ramp_steps = 3000
        ocr_max_weight = 0.1
        use_ocr_hinge = True
        
        if config is not None:
            sr_anchor_weight = getattr(config, 'stage3_sr_anchor_weight', 1.0)
            ocr_warmup_steps = getattr(config, 'stage3_ocr_warmup_steps', 3000)
            ocr_ramp_steps = getattr(config, 'stage3_ocr_ramp_steps', 3000)
            ocr_max_weight = getattr(config, 'stage3_ocr_max_weight', 0.1)
            use_ocr_hinge = getattr(config, 'stage3_use_ocr_hinge', True)
        
        # =================================================================
        # 1. PIXEL LOSS (Primary anchor - prevents mode collapse)
        # =================================================================
        if 'hr_image' in outputs and 'hr_image' in targets:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                l_pixel = self.pixel_loss(outputs['hr_image'], targets['hr_image'])
                l_pixel = self._clamp_loss(l_pixel)
                loss_dict['pixel'] = self._safe_loss_item(l_pixel)
                weighted = self.weights['pixel'] * l_pixel
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 2. SR ANCHOR LOSS (Anchors to Stage 2 output - prevents drift)
        # =================================================================
        if sr_stage2 is not None and 'hr_image' in outputs:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(sr_stage2).any()):
                # L1 distance from Stage 2 output (detached)
                l_sr_anchor = F.l1_loss(outputs['hr_image'], sr_stage2.detach())
                l_sr_anchor = self._clamp_loss(l_sr_anchor)
                loss_dict['sr_anchor'] = self._safe_loss_item(l_sr_anchor)
                weighted = sr_anchor_weight * l_sr_anchor
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 3. PERCEPTUAL LOSS (Increased weight for Stage 3)
        # =================================================================
        if self.perceptual_loss is not None and 'hr_image' in outputs and 'hr_image' in targets:
            if not (torch.isnan(outputs['hr_image']).any() or torch.isnan(targets['hr_image']).any()):
                l_perceptual = self.perceptual_loss(outputs['hr_image'], targets['hr_image'])
                l_perceptual = self._clamp_loss(l_perceptual)
                loss_dict['perceptual'] = self._safe_loss_item(l_perceptual)
                # Use config weight directly (Real-ESRGAN recommends 1.0)
                perceptual_weight = self.weights.get('perceptual', 1.0)
                weighted = perceptual_weight * l_perceptual
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 4. GAN LOSS (Generator side, with warm-up)
        # =================================================================
        if discriminator is not None and 'hr_image' in outputs:
            if not torch.isnan(outputs['hr_image']).any():
                pred_fake = discriminator(outputs['hr_image'])
                if not torch.isnan(pred_fake).any():
                    l_gan = self.gan_loss(pred_fake, target_is_real=True)
                    l_gan = self._clamp_loss(l_gan)
                    loss_dict['gan'] = self._safe_loss_item(l_gan)
                    weighted = self.weights['gan'] * l_gan
                    total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 5. OCR LOSS with WARMUP and HINGE CONSTRAINT
        # =================================================================
        if 'masked_logits' in outputs and 'text_indices' in targets:
            if not (torch.isnan(outputs['masked_logits']).any() or torch.isnan(targets['text_indices'].float()).any()):
                # Compute raw OCR loss
                l_ocr_raw = self.ocr_loss(outputs['masked_logits'], targets['text_indices'])
                l_ocr_raw = self._clamp_loss(l_ocr_raw)
                loss_dict['ocr_raw'] = self._safe_loss_item(l_ocr_raw)
                
                # Use OCR weight from curriculum (set in train.py)
                # Note: OCR is now disabled (replaced by LCOFL), so weight is expected to be 0
                ocr_weight = self.weights.get('ocr', 0.0)
                
                loss_dict['ocr_weight'] = ocr_weight
                
                # Apply OCR loss (optionally with hinge constraint)
                if ocr_weight > 0:
                    if use_ocr_hinge and ocr_baseline is not None:
                        # Hinge constraint: only penalize if worse than Stage 2
                        # relu(ocr_current - ocr_baseline) -> 0 if better, positive if worse
                        l_ocr = F.relu(l_ocr_raw - ocr_baseline)
                        loss_dict['ocr_hinge'] = self._safe_loss_item(l_ocr)
                        loss_dict['ocr_baseline'] = ocr_baseline  # Log Stage-2 baseline for diagnostics
                    else:
                        # Standard OCR loss (scaled by warmup weight)
                        l_ocr = l_ocr_raw
                    
                    loss_dict['ocr'] = self._safe_loss_item(l_ocr)
                    weighted = ocr_weight * l_ocr
                    total_loss = weighted if total_loss is None else total_loss + weighted
                else:
                    loss_dict['ocr'] = 0.0
        
        # =================================================================
        # 6. LAYOUT LOSS
        # =================================================================
        if 'layout_logits' in outputs and 'layout' in targets:
            if not torch.isnan(outputs['layout_logits']).any():
                l_layout = self.layout_loss(outputs['layout_logits'], targets['layout'])
                l_layout = self._clamp_loss(l_layout)
                loss_dict['layout'] = self._safe_loss_item(l_layout)
                weighted = self.weights['layout'] * l_layout
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 7. TOTAL VARIATION LOSS (for artifact suppression)
        # =================================================================
        if self.tv_loss is not None and 'hr_image' in outputs:
            if not torch.isnan(outputs['hr_image']).any():
                l_tv = self.tv_loss(outputs['hr_image'])
                l_tv = self._clamp_loss(l_tv)
                loss_dict['tv'] = self._safe_loss_item(l_tv)
                weighted = self.weights['tv'] * l_tv
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # 8. LCOFL LOSS (Character-aware loss with curriculum)
        # =================================================================
        # LCOFL replaces separate OCR loss with a unified approach:
        # - Classification: character recognition (curriculum-controlled in train.py)
        # - Layout: digit/letter position enforcement (always active)
        # - SSIM: structural similarity for visual quality (always active)
        if self.lcofl_loss is not None and 'masked_logits' in outputs and 'text_indices' in targets:
            if not torch.isnan(outputs['masked_logits']).any():
                # Get layout info
                is_mercosul = targets.get('layout', torch.zeros(
                    outputs['masked_logits'].size(0), 
                    device=outputs['masked_logits'].device
                ))
                
                # Strip BOS/EOS from targets
                plate_length = outputs['masked_logits'].size(1)
                lcofl_targets = targets['text_indices'][:, 1:plate_length+1]
                
                # Compute LCOFL loss (classification weight is set by curriculum in train.py)
                l_lcofl, lcofl_dict = self.lcofl_loss(
                    logits=outputs['masked_logits'],
                    targets=lcofl_targets,
                    is_mercosul=is_mercosul,
                    generated_hr=outputs.get('hr_image'),
                    gt_hr=targets.get('hr_image'),
                    compute_ssim=True  # Always compute SSIM in Stage 3
                )
                l_lcofl = self._clamp_loss(l_lcofl)
                loss_dict['lcofl'] = self._safe_loss_item(l_lcofl)
                
                # Log LCOFL sub-components for monitoring
                for k, v in lcofl_dict.items():
                    if k != 'total':
                        loss_dict[f'lcofl_{k}'] = v
                
                # Apply LCOFL weight from config
                weighted = self.weights.get('lcofl', 0.5) * l_lcofl
                total_loss = weighted if total_loss is None else total_loss + weighted
        
        # =================================================================
        # Fallback if no loss was computed
        # =================================================================
        if total_loss is None:
            if 'hr_image' in outputs:
                total_loss = outputs['hr_image'].sum() * 0.0
            else:
                any_output = next(iter(outputs.values()))
                total_loss = any_output.sum() * 0.0
        
        # Final clamp
        total_loss = self._clamp_loss(total_loss, max_val=100.0)
        
        loss_dict['total'] = self._safe_loss_item(total_loss)
        
        return total_loss, loss_dict


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
