"""Losses package for Neuro-Symbolic LPR System."""

from .corner_loss import CornerLoss
from .gan_loss import GANLoss, DiscriminatorLoss, PerceptualLoss
from .composite_loss import CompositeLoss, OCRLoss

__all__ = [
    'CornerLoss',
    'GANLoss',
    'DiscriminatorLoss',
    'PerceptualLoss',
    'CompositeLoss',
    'OCRLoss',
]
