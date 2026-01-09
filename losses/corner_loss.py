"""
Corner Loss for STN Supervision.

This module implements the corner loss that provides explicit supervision
for the Spatial Transformer Network using ground-truth corner annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CornerLoss(nn.Module):
    """
    Corner Loss for supervising the STN's geometric alignment.
    
    Calculates the Mean Squared Error between predicted corner coordinates
    and ground-truth corner annotations. This provides direct supervision
    for accurate license plate rectification.
    
    The corners are expected in normalized coordinates [-1, 1] in the order:
    [top-left, top-right, bottom-right, bottom-left]
    
    Includes numerical stability safeguards to prevent NaN/Inf losses.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        corner_weight: Optional[torch.Tensor] = None,
        max_loss_value: float = 50.0,
        eps: float = 1e-8
    ):
        """
        Initialize the corner loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
            corner_weight: Optional weights for each corner (4,).
            max_loss_value: Maximum loss value to clamp to (prevents explosion).
            eps: Small epsilon for numerical stability.
        """
        super().__init__()
        
        self.reduction = reduction
        self.max_loss_value = max_loss_value
        self.eps = eps
        
        if corner_weight is not None:
            self.register_buffer('corner_weight', corner_weight)
        else:
            self.corner_weight = None
    
    def forward(
        self,
        pred_corners: torch.Tensor,
        gt_corners: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute corner loss with numerical stability safeguards.
        
        Args:
            pred_corners: Predicted corners of shape (B, 4, 2).
            gt_corners: Ground-truth corners of shape (B, 4, 2).
            mask: Optional mask for valid samples of shape (B,).
            
        Returns:
            Corner loss value (clamped to prevent explosion).
        """
        # Check for NaN/Inf inputs and return zero loss with gradient connection
        if torch.isnan(pred_corners).any() or torch.isinf(pred_corners).any():
            return pred_corners.sum() * 0.0
        if torch.isnan(gt_corners).any() or torch.isinf(gt_corners).any():
            return pred_corners.sum() * 0.0
        
        # Clamp inputs to reasonable range for stability
        pred_corners = torch.clamp(pred_corners, min=-2.0, max=2.0)
        gt_corners = torch.clamp(gt_corners, min=-2.0, max=2.0)
        
        # Compute squared error
        squared_error = (pred_corners - gt_corners) ** 2  # (B, 4, 2)
        
        # Sum over x,y coordinates
        corner_errors = squared_error.sum(dim=-1)  # (B, 4)
        
        # Apply corner weights if provided
        if self.corner_weight is not None:
            corner_errors = corner_errors * self.corner_weight.unsqueeze(0)
        
        # Sum over corners
        sample_errors = corner_errors.sum(dim=-1)  # (B,)
        
        # Apply sample mask if provided
        if mask is not None:
            sample_errors = sample_errors * mask
            num_valid = mask.sum().clamp(min=1)
        else:
            num_valid = max(sample_errors.size(0), 1)
        
        # Reduce
        if self.reduction == 'mean':
            loss = sample_errors.sum() / (num_valid + self.eps)
        elif self.reduction == 'sum':
            loss = sample_errors.sum()
        else:  # 'none'
            loss = sample_errors
        
        # Final clamp and NaN check
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return pred_corners.sum() * 0.0
        
        return torch.clamp(loss, min=0.0, max=self.max_loss_value)
    
    def forward_from_theta(
        self,
        theta: torch.Tensor,
        gt_corners: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute corner loss from affine transformation parameters.
        
        Transforms canonical corners using theta and compares with GT.
        
        Args:
            theta: Affine parameters of shape (B, 2, 3).
            gt_corners: Ground-truth corners of shape (B, 4, 2).
            mask: Optional mask for valid samples.
            
        Returns:
            Corner loss value.
        """
        batch_size = theta.size(0)
        device = theta.device
        
        # Canonical corners in [-1, 1]
        canonical = torch.tensor([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ], dtype=torch.float, device=device)
        
        # Add homogeneous coordinate
        ones = torch.ones(4, 1, device=device)
        canonical_h = torch.cat([canonical, ones], dim=-1)  # (4, 3)
        
        # Transform for each sample in batch
        # theta: (B, 2, 3), canonical_h: (4, 3)
        # Result: (B, 4, 2)
        pred_corners = torch.bmm(
            canonical_h.unsqueeze(0).expand(batch_size, -1, -1),
            theta.transpose(1, 2)
        )
        
        return self.forward(pred_corners, gt_corners, mask)


class IoUCornerLoss(nn.Module):
    """
    IoU-based loss for corner prediction.
    
    Uses the Intersection over Union of the quadrilaterals defined by
    predicted and ground-truth corners as a loss signal.
    """
    
    def __init__(self):
        super().__init__()
    
    def _shoelace_area(self, corners: torch.Tensor) -> torch.Tensor:
        """
        Compute area of polygon using shoelace formula.
        
        Args:
            corners: Corner coordinates of shape (B, 4, 2).
            
        Returns:
            Areas of shape (B,).
        """
        # Shift corners for cross product
        corners_shifted = torch.roll(corners, 1, dims=1)
        
        # Shoelace formula
        cross = corners[:, :, 0] * corners_shifted[:, :, 1] - \
                corners_shifted[:, :, 0] * corners[:, :, 1]
        
        area = torch.abs(cross.sum(dim=1)) / 2
        
        return area
    
    def forward(
        self,
        pred_corners: torch.Tensor,
        gt_corners: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU-based loss.
        
        Note: This is a simplified version that uses area ratio
        rather than true IoU (which requires polygon intersection).
        
        Args:
            pred_corners: Predicted corners of shape (B, 4, 2).
            gt_corners: Ground-truth corners of shape (B, 4, 2).
            
        Returns:
            Loss value.
        """
        pred_area = self._shoelace_area(pred_corners)
        gt_area = self._shoelace_area(gt_corners)
        
        # Simplified IoU approximation using area ratio
        area_ratio = torch.min(pred_area, gt_area) / torch.max(pred_area, gt_area).clamp(min=1e-6)
        
        # Combine with L1 corner distance
        l1_dist = F.l1_loss(pred_corners, gt_corners, reduction='none').sum(dim=[1, 2])
        
        # Combined loss
        loss = l1_dist + (1 - area_ratio)
        
        return loss.mean()


class SmoothL1CornerLoss(nn.Module):
    """
    Smooth L1 (Huber) loss for corner prediction.
    
    More robust to outliers than MSE loss.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        pred_corners: torch.Tensor,
        gt_corners: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Smooth L1 corner loss.
        
        Args:
            pred_corners: Predicted corners of shape (B, 4, 2).
            gt_corners: Ground-truth corners of shape (B, 4, 2).
            
        Returns:
            Loss value.
        """
        return F.smooth_l1_loss(pred_corners, gt_corners, beta=self.beta)


class MultiFrameCornerLoss(nn.Module):
    """
    Corner loss for multi-frame STN outputs.
    
    Computes corner loss for each frame and aggregates.
    """
    
    def __init__(self, aggregation: str = 'mean'):
        super().__init__()
        self.corner_loss = CornerLoss()
        self.aggregation = aggregation
    
    def forward(
        self,
        pred_corners: torch.Tensor,
        gt_corners: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-frame corner loss.
        
        Args:
            pred_corners: Predicted corners of shape (B, T, 4, 2).
            gt_corners: Ground-truth corners of shape (B, T, 4, 2) or (B, 4, 2).
            
        Returns:
            Loss value.
        """
        B, T = pred_corners.shape[:2]
        
        # Expand gt_corners if shared across frames
        if gt_corners.dim() == 3:
            gt_corners = gt_corners.unsqueeze(1).expand(-1, T, -1, -1)
        
        # Compute loss for each frame
        losses = []
        for t in range(T):
            loss = self.corner_loss(pred_corners[:, t], gt_corners[:, t])
            losses.append(loss)
        
        losses = torch.stack(losses)
        
        if self.aggregation == 'mean':
            return losses.mean()
        elif self.aggregation == 'sum':
            return losses.sum()
        else:
            return losses
