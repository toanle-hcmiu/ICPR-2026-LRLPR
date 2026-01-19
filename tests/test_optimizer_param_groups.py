"""
Tests for optimizer param group safety.

Verifies that create_optimizer() doesn't crash when encountering:
- Empty param groups
- Param groups where all params have requires_grad=False
- None param groups
"""

import pytest
import torch
import torch.nn as nn


class TestCreateOptimizer:
    """Tests for train.create_optimizer with edge cases."""
    
    @pytest.fixture
    def mock_model_with_empty_group(self):
        """Create a mock model that returns an empty param group."""
        class EmptyModule(nn.Module):
            """Module with no parameters (like SyntaxMaskLayer)."""
            def __init__(self):
                super().__init__()
                # Only registers a buffer, no parameters
                self.register_buffer('mask', torch.ones(10))
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(10, 10)
                self.empty_module = EmptyModule()
                self.decoder = nn.Linear(10, 10)
            
            def get_trainable_params(self, stage: str):
                """Return param groups including empty module."""
                if stage == 'test_empty':
                    return [
                        {'params': self.encoder.parameters()},
                        {'params': self.empty_module.parameters()},  # Empty!
                        {'params': self.decoder.parameters()},
                    ]
                elif stage == 'test_none':
                    return [
                        {'params': self.encoder.parameters()},
                        None,  # None group
                        {'params': self.decoder.parameters()},
                    ]
                elif stage == 'test_frozen':
                    # Freeze encoder
                    for p in self.encoder.parameters():
                        p.requires_grad = False
                    return [
                        {'params': self.encoder.parameters()},  # All frozen
                        {'params': self.decoder.parameters()},
                    ]
                return [{'params': self.parameters()}]
        
        return MockModel()
    
    @pytest.fixture
    def mock_config(self):
        """Create a minimal config for optimizer creation."""
        class TrainingConfig:
            optimizer = 'adamw'
            weight_decay = 0.01
            betas = (0.9, 0.999)
            lr_pretrain = 1e-4
            lr_stn = 1e-5
            lr_restoration = 5e-5
            lr_finetune = 2e-5
        
        class Config:
            training = TrainingConfig()
        
        return Config()
    
    def test_create_optimizer_filters_empty_groups(self, mock_model_with_empty_group, mock_config):
        """create_optimizer should filter out empty param groups."""
        from train import create_optimizer
        
        # This should NOT raise ValueError("optimizer got an empty parameter list")
        optimizer = create_optimizer(mock_model_with_empty_group, 'test_empty', mock_config)
        
        # Should have 2 param groups (encoder + decoder, not empty_module)
        assert len(optimizer.param_groups) == 2
    
    def test_create_optimizer_filters_none_groups(self, mock_model_with_empty_group, mock_config):
        """create_optimizer should filter out None param groups."""
        from train import create_optimizer
        
        optimizer = create_optimizer(mock_model_with_empty_group, 'test_none', mock_config)
        
        # Should have 2 param groups (encoder + decoder, not None)
        assert len(optimizer.param_groups) == 2
    
    def test_create_optimizer_filters_frozen_groups(self, mock_model_with_empty_group, mock_config):
        """create_optimizer should filter out groups where all params are frozen."""
        from train import create_optimizer
        
        optimizer = create_optimizer(mock_model_with_empty_group, 'test_frozen', mock_config)
        
        # Should have 1 param group (only decoder, encoder is frozen)
        assert len(optimizer.param_groups) == 1
        
        # Verify the remaining group has trainable params
        for group in optimizer.param_groups:
            for param in group['params']:
                assert param.requires_grad


class TestSyntaxMaskHasNoParams:
    """Verify SyntaxMaskLayer has no trainable parameters."""
    
    def test_syntax_mask_has_no_parameters(self):
        """SyntaxMaskLayer should have no trainable parameters."""
        from models.syntax_mask import SyntaxMaskLayer
        
        mask = SyntaxMaskLayer()
        params = list(mask.parameters())
        
        assert len(params) == 0, "SyntaxMaskLayer should have no trainable parameters"
    
    def test_syntax_mask_has_buffers(self):
        """SyntaxMaskLayer should have registered buffers for masks."""
        from models.syntax_mask import SyntaxMaskLayer
        
        mask = SyntaxMaskLayer()
        buffers = list(mask.buffers())
        
        # Should have at least the Brazilian and Mercosul masks
        assert len(buffers) >= 2, "SyntaxMaskLayer should have registered buffers"
