"""
Tests for apply_config_overrides function in train.py.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_default_config


# Import apply_config_overrides from train.py
# We need to do this carefully to avoid running the whole training script
def get_apply_config_overrides():
    """Import apply_config_overrides without running train.py's main."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    
    # We need to execute the module to get the function definitions
    # but we don't want to run main()
    original_name = None
    try:
        spec.loader.exec_module(train_module)
    except SystemExit:
        pass  # In case argparse exits
    
    return train_module.apply_config_overrides


class TestApplyConfigOverrides:
    """Tests for apply_config_overrides function."""
    
    @pytest.fixture
    def apply_overrides(self):
        """Get the apply_config_overrides function."""
        return get_apply_config_overrides()
    
    def test_empty_overrides(self, apply_overrides):
        """Empty overrides should not change config."""
        config = get_default_config()
        original_seed = config.training.seed
        
        result = apply_overrides(config, {})
        
        assert result.training.seed == original_seed
    
    def test_none_overrides(self, apply_overrides):
        """None overrides should not change config."""
        config = get_default_config()
        original_seed = config.training.seed
        
        result = apply_overrides(config, None)
        
        assert result.training.seed == original_seed
    
    def test_training_seed_override(self, apply_overrides):
        """Should override training.seed."""
        config = get_default_config()
        overrides = {
            'training': {
                'seed': 12345
            }
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.training.seed == 12345
    
    def test_training_lr_override(self, apply_overrides):
        """Should override training learning rates."""
        config = get_default_config()
        overrides = {
            'training': {
                'lr_stn': 1e-6,
                'lr_restoration': 2e-5
            }
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.training.lr_stn == 1e-6
        assert result.training.lr_restoration == 2e-5
    
    def test_model_config_override(self, apply_overrides):
        """Should override model config values."""
        config = get_default_config()
        overrides = {
            'model': {
                'num_frames': 3,
                'swinir_embed_dim': 128
            }
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.model.num_frames == 3
        assert result.model.swinir_embed_dim == 128
    
    def test_data_config_override(self, apply_overrides):
        """Should override data config values."""
        config = get_default_config()
        overrides = {
            'data': {
                'num_frames': 10,
                'use_augmentation': False
            }
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.data.num_frames == 10
        assert result.data.use_augmentation is False
    
    def test_multiple_sections_override(self, apply_overrides):
        """Should override multiple sections at once."""
        config = get_default_config()
        overrides = {
            'training': {'seed': 999},
            'model': {'num_frames': 7},
            'data': {'use_augmentation': True}
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.training.seed == 999
        assert result.model.num_frames == 7
        assert result.data.use_augmentation is True
    
    def test_unknown_section_ignored(self, apply_overrides, capsys):
        """Unknown section should be ignored with warning."""
        config = get_default_config()
        original_seed = config.training.seed
        overrides = {
            'unknown_section': {
                'some_key': 'some_value'
            }
        }
        
        result = apply_overrides(config, overrides)
        
        # Config should be unchanged
        assert result.training.seed == original_seed
        
        # Should print a warning (captured by capsys)
        captured = capsys.readouterr()
        assert "Unknown config section 'unknown_section'" in captured.out
    
    def test_unknown_key_ignored(self, apply_overrides, capsys):
        """Unknown key within valid section should be ignored with warning."""
        config = get_default_config()
        overrides = {
            'training': {
                'nonexistent_key': 42
            }
        }
        
        result = apply_overrides(config, overrides)
        
        # Should print a warning
        captured = capsys.readouterr()
        assert "Unknown config key 'training.nonexistent_key'" in captured.out
    
    def test_top_level_value_override(self, apply_overrides):
        """Should handle top-level non-dict values like experiment_name."""
        config = get_default_config()
        overrides = {
            'experiment_name': 'my_experiment'
        }
        
        result = apply_overrides(config, overrides)
        
        assert result.experiment_name == 'my_experiment'
    
    def test_returns_same_config_instance(self, apply_overrides):
        """Should return the same config instance (mutated in place)."""
        config = get_default_config()
        overrides = {'training': {'seed': 111}}
        
        result = apply_overrides(config, overrides)
        
        assert result is config
