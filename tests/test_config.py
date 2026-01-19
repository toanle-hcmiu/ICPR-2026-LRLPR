"""
Tests for config.py helper functions.
"""

import pytest
from config import (
    get_position_constraints,
    validate_plate_text,
    infer_layout_from_text,
    get_default_config,
    PLATE_LENGTH,
    VOCAB_SIZE,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    CHAR_START_IDX,
)


class TestGetPositionConstraints:
    """Tests for get_position_constraints function."""
    
    def test_brazilian_format(self):
        """Brazilian format should be LLLNNNN (3 letters + 4 digits)."""
        constraints = get_position_constraints(is_mercosul=False)
        assert constraints == ['L', 'L', 'L', 'N', 'N', 'N', 'N']
        assert len(constraints) == PLATE_LENGTH
    
    def test_mercosul_format(self):
        """Mercosul format should be LLLNLNN (3 letters, digit, letter, 2 digits)."""
        constraints = get_position_constraints(is_mercosul=True)
        assert constraints == ['L', 'L', 'L', 'N', 'L', 'N', 'N']
        assert len(constraints) == PLATE_LENGTH


class TestValidatePlateText:
    """Tests for validate_plate_text function."""
    
    def test_valid_brazilian_plate(self):
        """Valid Brazilian plate should return (True, 'brazilian')."""
        valid, plate_type = validate_plate_text("ABC1234")
        assert valid is True
        assert plate_type == 'brazilian'
    
    def test_valid_brazilian_plate_lowercase(self):
        """Lowercase Brazilian plate should also be valid (auto-uppercased)."""
        valid, plate_type = validate_plate_text("abc1234")
        assert valid is True
        assert plate_type == 'brazilian'
    
    def test_valid_mercosul_plate(self):
        """Valid Mercosul plate should return (True, 'mercosul')."""
        valid, plate_type = validate_plate_text("ABC1D23")
        assert valid is True
        assert plate_type == 'mercosul'
    
    def test_valid_mercosul_plate_lowercase(self):
        """Lowercase Mercosul plate should also be valid."""
        valid, plate_type = validate_plate_text("abc1d23")
        assert valid is True
        assert plate_type == 'mercosul'
    
    def test_invalid_plate_wrong_pattern(self):
        """Invalid plate pattern should return (False, None)."""
        valid, plate_type = validate_plate_text("AB123CD")
        assert valid is False
        assert plate_type is None
    
    def test_invalid_plate_too_short(self):
        """Too short plate should return (False, None)."""
        valid, plate_type = validate_plate_text("ABC123")
        assert valid is False
        assert plate_type is None
    
    def test_invalid_plate_too_long(self):
        """Too long plate should return (False, None)."""
        valid, plate_type = validate_plate_text("ABC12345")
        assert valid is False
        assert plate_type is None
    
    def test_plate_with_dash_stripped(self):
        """Plate with dash should be stripped and validated."""
        valid, plate_type = validate_plate_text("ABC-1234")
        assert valid is True
        assert plate_type == 'brazilian'
    
    def test_plate_with_spaces_stripped(self):
        """Plate with spaces should be stripped and validated."""
        valid, plate_type = validate_plate_text("ABC 1234")
        assert valid is True
        assert plate_type == 'brazilian'


class TestInferLayoutFromText:
    """Tests for infer_layout_from_text function."""
    
    def test_brazilian_layout(self):
        """Brazilian plate should return layout 0."""
        layout = infer_layout_from_text("ABC1234")
        assert layout == 0
    
    def test_mercosul_layout(self):
        """Mercosul plate should return layout 1."""
        layout = infer_layout_from_text("ABC1D23")
        assert layout == 1
    
    def test_invalid_layout(self):
        """Invalid plate should return layout -1."""
        layout = infer_layout_from_text("INVALID")
        assert layout == -1
    
    def test_empty_string(self):
        """Empty string should return layout -1."""
        layout = infer_layout_from_text("")
        assert layout == -1


class TestConfigConstants:
    """Tests for config constants."""
    
    def test_plate_length(self):
        """PLATE_LENGTH should be 7."""
        assert PLATE_LENGTH == 7
    
    def test_vocab_size(self):
        """VOCAB_SIZE should be 39 (36 chars + 3 special tokens)."""
        assert VOCAB_SIZE == 39
    
    def test_special_token_indices(self):
        """Special token indices should be sequential starting from 0."""
        assert PAD_IDX == 0
        assert BOS_IDX == 1
        assert EOS_IDX == 2
        assert CHAR_START_IDX == 3


class TestGetDefaultConfig:
    """Tests for get_default_config function."""
    
    def test_returns_config_object(self):
        """get_default_config should return a Config object."""
        from config import Config
        config = get_default_config()
        assert isinstance(config, Config)
    
    def test_has_model_config(self):
        """Config should have model sub-config."""
        config = get_default_config()
        assert hasattr(config, 'model')
        assert config.model.num_frames == 5
    
    def test_has_training_config(self):
        """Config should have training sub-config."""
        config = get_default_config()
        assert hasattr(config, 'training')
        assert config.training.seed == 42
    
    def test_has_data_config(self):
        """Config should have data sub-config."""
        config = get_default_config()
        assert hasattr(config, 'data')
        assert config.data.lr_height == 16
        assert config.data.lr_width == 48
