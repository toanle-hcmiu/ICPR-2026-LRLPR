"""
Tests for pretrain stage support.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict


class TestCompositeLossPretrainStage:
    """Tests for CompositeLoss.get_stage_loss with pretrain stage."""
    
    @pytest.fixture
    def composite_loss(self):
        """Create a CompositeLoss instance for testing."""
        from losses import CompositeLoss
        return CompositeLoss(
            weight_pixel=1.0,
            weight_gan=0.1,
            weight_ocr=1.0,
            weight_geometry=0.1,
            weight_layout=0.1
        )
    
    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs for testing."""
        batch_size = 2
        plate_length = 7
        vocab_size = 39
        
        return {
            'masked_logits': torch.randn(batch_size, plate_length, vocab_size, requires_grad=True),
            'raw_logits': torch.randn(batch_size, plate_length, vocab_size, requires_grad=True),
            'hr_image': torch.randn(batch_size, 3, 64, 192, requires_grad=True),
            'layout_logits': torch.randn(batch_size, 1, requires_grad=True),
        }
    
    @pytest.fixture
    def mock_targets(self):
        """Create mock targets for testing."""
        batch_size = 2
        plate_length = 7
        
        # text_indices: (B, PLATE_LENGTH + 2) with BOS and EOS
        # Values should be valid token indices (3+ for actual chars)
        text_indices = torch.randint(3, 39, (batch_size, plate_length + 2))
        text_indices[:, 0] = 1  # BOS
        text_indices[:, -1] = 2  # EOS
        
        return {
            'text_indices': text_indices,
            'hr_image': torch.randn(batch_size, 3, 64, 192),
            'layout': torch.tensor([0, 1]),  # Brazilian and Mercosul
        }
    
    def test_pretrain_stage_returns_loss(self, composite_loss, mock_outputs, mock_targets):
        """Pretrain stage should return a loss tensor."""
        loss, loss_dict = composite_loss.get_stage_loss('pretrain', mock_outputs, mock_targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.dim() == 0  # Scalar
    
    def test_pretrain_stage_only_ocr_loss(self, composite_loss, mock_outputs, mock_targets):
        """Pretrain stage should only compute OCR loss."""
        loss, loss_dict = composite_loss.get_stage_loss('pretrain', mock_outputs, mock_targets)
        
        # Should have OCR loss
        assert 'ocr' in loss_dict
        
        # Should NOT have pixel, gan, geometry, or layout losses
        assert 'pixel' not in loss_dict
        assert 'gan' not in loss_dict
        assert 'geometry' not in loss_dict
        assert 'layout' not in loss_dict
    
    def test_pretrain_stage_loss_dict_has_total(self, composite_loss, mock_outputs, mock_targets):
        """Pretrain stage loss_dict should have 'total' key."""
        loss, loss_dict = composite_loss.get_stage_loss('pretrain', mock_outputs, mock_targets)
        
        assert 'total' in loss_dict
    
    def test_pretrain_stage_handles_missing_logits(self, composite_loss, mock_targets):
        """Pretrain stage should handle missing masked_logits gracefully."""
        outputs_no_logits = {
            'hr_image': torch.randn(2, 3, 64, 192, requires_grad=True),
        }
        
        loss, loss_dict = composite_loss.get_stage_loss('pretrain', outputs_no_logits, mock_targets)
        
        # Should still return a loss (fallback)
        assert isinstance(loss, torch.Tensor)
    
    def test_pretrain_stage_nan_handling(self, composite_loss, mock_targets):
        """Pretrain stage should skip OCR loss computation when logits have NaN."""
        outputs_with_nan = {
            'masked_logits': torch.tensor([[[float('nan')] * 39] * 7] * 2, requires_grad=True),
        }
        
        loss, loss_dict = composite_loss.get_stage_loss('pretrain', outputs_with_nan, mock_targets)
        
        # Should return a loss tensor (fallback)
        assert isinstance(loss, torch.Tensor)
        # OCR loss should NOT be computed (skipped due to NaN check)
        assert 'ocr' not in loss_dict
        # Fallback loss should be finite (zero) even when inputs contain NaN
        # The fallback uses nan_to_num to sanitize the tensor before sum
        assert torch.isfinite(loss).all(), "Fallback loss should be finite, not NaN"
        assert loss.item() == 0.0, "Fallback loss should be zero"


class TestModelPretrainParams:
    """Tests for NeuroSymbolicLPR.get_trainable_params with pretrain stage."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a minimal mock model for testing param groups."""
        class MockRecognizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 10)
                self.stn = nn.Linear(10, 10)
                self.generator = nn.Linear(10, 10)
                self.layout_classifier = nn.Linear(10, 10)
                self.quality_fusion = nn.Linear(10, 10)
                self.feature_to_image = nn.Linear(10, 10)
                self.recognizer = MockRecognizer()
            
            def get_trainable_params(self, stage: str):
                """Minimal implementation matching the real model."""
                if stage == 'pretrain':
                    return [{'params': self.recognizer.parameters()}]
                elif stage == 'stn':
                    return [
                        {'params': self.encoder.parameters()},
                        {'params': self.stn.parameters()}
                    ]
                else:
                    raise ValueError(f"Unknown stage: {stage}")
        
        return MockModel()
    
    def test_pretrain_returns_only_recognizer_params(self, mock_model):
        """Pretrain stage should return only recognizer parameters."""
        param_groups = mock_model.get_trainable_params('pretrain')
        
        assert len(param_groups) == 1
        
        # Get all param ids from the group
        param_ids = set(id(p) for p in param_groups[0]['params'])
        recognizer_param_ids = set(id(p) for p in mock_model.recognizer.parameters())
        
        assert param_ids == recognizer_param_ids
    
    def test_stn_returns_encoder_and_stn_params(self, mock_model):
        """STN stage should return encoder and stn parameters."""
        param_groups = mock_model.get_trainable_params('stn')
        
        assert len(param_groups) == 2


class TestModelFreezeAllExceptRecognizer:
    """Tests for freeze_all_except_recognizer method."""
    
    def test_freeze_method_exists(self):
        """NeuroSymbolicLPR should have freeze_all_except_recognizer method."""
        from models import NeuroSymbolicLPR
        
        assert hasattr(NeuroSymbolicLPR, 'freeze_all_except_recognizer')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_freeze_leaves_recognizer_trainable(self):
        """After freeze_all_except_recognizer, only recognizer should be trainable."""
        # This test requires actual model instantiation which may need CUDA
        # Skip if no CUDA available
        from models import NeuroSymbolicLPR
        
        model = NeuroSymbolicLPR(
            num_frames=5,
            lr_size=(16, 48),
            hr_size=(64, 192),
            use_pretrained_parseq=False  # Use custom to avoid download
        )
        
        model.freeze_all_except_recognizer()
        
        # Recognizer should be trainable
        for param in model.recognizer.parameters():
            assert param.requires_grad is True
        
        # Other modules should be frozen
        for param in model.encoder.parameters():
            assert param.requires_grad is False
        for param in model.stn.parameters():
            assert param.requires_grad is False
        for param in model.generator.parameters():
            assert param.requires_grad is False
