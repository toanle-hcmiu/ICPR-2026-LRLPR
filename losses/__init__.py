"""Losses package for Neuro-Symbolic LPR System."""

from .corner_loss import CornerLoss
from .gan_loss import GANLoss, DiscriminatorLoss, PerceptualLoss
from .composite_loss import CompositeLoss, OCRLoss
from .ocr_perceptual_loss import (
    OCRAwarePerceptualLoss,
    CharacterFocusLoss,
    MultiScaleOCRLoss,
)
from .lcofl_loss import (
    LCOFLLoss,
    LCOFLWithOCR,
    SSIMLoss,
    LayoutPenalty,
    ConfusionMatrixTracker,
)
from .ocr_discriminator import (
    OCRDiscriminator,
    OCRDiscriminatorLoss,
    RecognitionGuidedLoss,
    create_ocr_discriminator,
)

__all__ = [
    'CornerLoss',
    'GANLoss',
    'DiscriminatorLoss',
    'PerceptualLoss',
    'CompositeLoss',
    'OCRLoss',
    'OCRAwarePerceptualLoss',
    'CharacterFocusLoss',
    'MultiScaleOCRLoss',
    # LCOFL losses
    'LCOFLLoss',
    'LCOFLWithOCR',
    'SSIMLoss',
    'LayoutPenalty',
    'ConfusionMatrixTracker',
    # OCR-as-Discriminator
    'OCRDiscriminator',
    'OCRDiscriminatorLoss',
    'RecognitionGuidedLoss',
    'create_ocr_discriminator',
]


