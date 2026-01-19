"""Models package for Neuro-Symbolic LPR System."""

from .neuro_symbolic_lpr import NeuroSymbolicLPR, NeuroSymbolicLPRWithDiscriminator
from .encoder import SharedCNNEncoder
from .stn import SpatialTransformerNetwork, MultiFrameSTN
from .layout_classifier import LayoutClassifier
from .feature_fusion import QualityScorer, FeatureFusion, QualityScorerFusion
from .swinir import SwinIRGenerator, LightweightSRGenerator
from .discriminator import PatchDiscriminator, MultiScaleDiscriminator
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

__all__ = [
    'NeuroSymbolicLPR',
    'NeuroSymbolicLPRWithDiscriminator',
    'SharedCNNEncoder',
    'SpatialTransformerNetwork',
    'MultiFrameSTN',
    'LayoutClassifier',
    'QualityScorer',
    'FeatureFusion',
    'QualityScorerFusion',
    'SwinIRGenerator',
    'LightweightSRGenerator',
    'PatchDiscriminator',
    'MultiScaleDiscriminator',
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
    # Anti-aliased downsampling for preventing wavy artifacts
    'BlurPool2d',
    'MaxBlurPool2d',
    'AntiAliasedConv2d',
]

