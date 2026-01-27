"""Models package for Text-Prior Guided LPR System."""

from .encoder import SharedCNNEncoder
from .stn import SpatialTransformerNetwork, MultiFrameSTN
from .layout_classifier import LayoutClassifier
from .feature_fusion import QualityScorer, FeatureFusion, QualityScorerFusion
from .parseq import PARSeqRecognizer, PretrainedPARSeq, load_pretrained_parseq
from .syntax_mask import SyntaxMaskLayer
from .deformable_conv import DeformableConv2d, DeformableConv2dSimple, create_deformable_conv
from .shared_attention import (
    SharedAttentionModule,
    PixelLevelAttentionBlock,
    ChannelAttentionUnit,
    PositionalAttentionUnit,
    GeometricalPerceptionUnit,
    SubPixelConvolutionBlock,
    create_shared_attention_network,
)
from .blur_pool import BlurPool2d, MaxBlurPool2d, AntiAliasedConv2d

# Text-Prior Guided Architecture (2026-01-27)
from .multi_frame_fusion import (
    InterFrameCrossAttentionModule,
    TemporalFusionModule,
    AdaptiveFrameFusion,
    create_multi_frame_fusion,
)
from .text_prior import (
    TextPriorExtractor,
    TextPriorLoss,
    MultiFrameTextPriorExtractor,
    create_text_prior_extractor,
)
from .sequential_blocks import (
    SequentialResidualBlock,
    TextGuidedSRB,
    SequentialResidualBlocks,
    create_srb,
)
from .tp_guided_generator import (
    TextPriorGuidedGenerator,
    LightweightTPGenerator,
    CrossAttentionBlock,
    create_tp_guided_generator,
)
from .tp_lpr import (
    TextPriorGuidedLPR,
    create_text_prior_lpr,
)

__all__ = [
    # Core components
    'SharedCNNEncoder',
    'SpatialTransformerNetwork',
    'MultiFrameSTN',
    'LayoutClassifier',
    'QualityScorer',
    'FeatureFusion',
    'QualityScorerFusion',
    # OCR
    'PARSeqRecognizer',
    'PretrainedPARSeq',
    'load_pretrained_parseq',
    'SyntaxMaskLayer',
    # Deformable convolutions
    'DeformableConv2d',
    'DeformableConv2dSimple',
    'create_deformable_conv',
    # Shared attention
    'SharedAttentionModule',
    'PixelLevelAttentionBlock',
    'ChannelAttentionUnit',
    'PositionalAttentionUnit',
    'GeometricalPerceptionUnit',
    'SubPixelConvolutionBlock',
    'create_shared_attention_network',
    # Anti-aliased downsampling
    'BlurPool2d',
    'MaxBlurPool2d',
    'AntiAliasedConv2d',
    # Text-Prior Guided Architecture (2026-01-27)
    'InterFrameCrossAttentionModule',
    'TemporalFusionModule',
    'AdaptiveFrameFusion',
    'create_multi_frame_fusion',
    'TextPriorExtractor',
    'TextPriorLoss',
    'MultiFrameTextPriorExtractor',
    'create_text_prior_extractor',
    'SequentialResidualBlock',
    'TextGuidedSRB',
    'SequentialResidualBlocks',
    'create_srb',
    'TextPriorGuidedGenerator',
    'LightweightTPGenerator',
    'CrossAttentionBlock',
    'create_tp_guided_generator',
    'TextPriorGuidedLPR',
    'create_text_prior_lpr',
]
