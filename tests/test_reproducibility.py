"""
Tests for reproducibility and determinism.

These tests verify that:
1. seed_everything() sets deterministic flags correctly
2. Two runs with same seed produce identical results
3. Inference preprocessing is deterministic by default
"""

import os
import sys
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSeedEverything:
    """Tests for seed_everything function."""

    @pytest.fixture
    def seed_everything(self):
        """Import seed_everything from train.py."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("train", "train.py")
        train_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(train_module)
        except SystemExit:
            pass
        return train_module.seed_everything

    def test_sets_python_random_seed(self, seed_everything):
        """seed_everything should set Python's random module seed."""
        seed_everything(12345, strict_determinism=False)

        # Generate some random numbers
        vals1 = [random.random() for _ in range(5)]

        # Re-seed and generate again
        seed_everything(12345, strict_determinism=False)
        vals2 = [random.random() for _ in range(5)]

        assert vals1 == vals2

    def test_sets_numpy_seed(self, seed_everything):
        """seed_everything should set NumPy's random seed."""
        seed_everything(12345, strict_determinism=False)
        vals1 = np.random.rand(5).tolist()

        seed_everything(12345, strict_determinism=False)
        vals2 = np.random.rand(5).tolist()

        assert vals1 == vals2

    def test_sets_torch_seed(self, seed_everything):
        """seed_everything should set PyTorch's seed."""
        seed_everything(12345, strict_determinism=False)
        vals1 = torch.rand(5).tolist()

        seed_everything(12345, strict_determinism=False)
        vals2 = torch.rand(5).tolist()

        assert vals1 == vals2

    def test_strict_determinism_sets_cudnn_flags(self, seed_everything):
        """With strict_determinism=True, cuDNN flags should be set."""
        seed_everything(42, strict_determinism=True)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_strict_determinism_sets_env_vars(self, seed_everything):
        """With strict_determinism=True, environment variables should be set."""
        seed_everything(42, strict_determinism=True)

        assert os.environ.get("PYTHONHASHSEED") == "42"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_strict_determinism_disables_tf32(self, seed_everything):
        """With strict_determinism=True, TF32 should be disabled on CUDA."""
        seed_everything(42, strict_determinism=True)

        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False


class TestDeterministicTraining:
    """Tests for deterministic training runs."""

    @pytest.fixture
    def seed_everything(self):
        """Import seed_everything from train.py."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("train", "train.py")
        train_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(train_module)
        except SystemExit:
            pass
        return train_module.seed_everything

    def test_identical_weight_init(self, seed_everything):
        """Two models initialized with same seed should have identical weights."""
        seed_everything(999, strict_determinism=False)
        model1 = nn.Linear(10, 5)

        seed_everything(999, strict_determinism=False)
        model2 = nn.Linear(10, 5)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Weights should be identical with same seed"

    def test_identical_training_step_cpu(self, seed_everything):
        """Two training steps with same seed should produce identical results on CPU."""
        # This test uses strict_determinism=False to avoid potential issues
        # with torch.use_deterministic_algorithms on CPU

        def train_step(seed):
            seed_everything(seed, strict_determinism=False)

            model = nn.Linear(10, 5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # Fixed input data
            torch.manual_seed(seed)
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)

            # Training step
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()

            return {name: p.clone() for name, p in model.named_parameters()}

        weights1 = train_step(42)
        weights2 = train_step(42)

        for name in weights1:
            assert torch.equal(
                weights1[name], weights2[name]
            ), f"Weights for {name} should be identical after training step"


class TestInferencePreprocessDeterminism:
    """Tests for deterministic inference preprocessing."""

    @pytest.fixture
    def preprocess_image(self):
        """Import preprocess_image from inference.py."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("inference", "inference.py")
        inference_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(inference_module)
        except SystemExit:
            pass
        return inference_module.preprocess_image

    def test_default_preprocessing_is_deterministic(self, preprocess_image):
        """preprocess_image with default args should be deterministic."""
        # Create a test image
        image = np.random.randint(0, 255, (32, 96, 3), dtype=np.uint8)

        # Preprocess twice with default settings (frame_noise_std=0.0)
        result1 = preprocess_image(image, (16, 48), num_frames=5)
        result2 = preprocess_image(image, (16, 48), num_frames=5)

        assert torch.equal(result1, result2), "Default preprocessing should be deterministic"

    def test_all_frames_identical_without_noise(self, preprocess_image):
        """Without noise, all frames should be identical."""
        image = np.random.randint(0, 255, (32, 96, 3), dtype=np.uint8)

        result = preprocess_image(image, (16, 48), num_frames=5, frame_noise_std=0.0)

        # Check all frames are identical
        for i in range(1, 5):
            assert torch.equal(
                result[0, 0], result[0, i]
            ), f"Frame {i} should be identical to frame 0 without noise"

    def test_frames_differ_with_noise(self, preprocess_image):
        """With noise, extra frames should differ from first frame."""
        image = np.random.randint(0, 255, (32, 96, 3), dtype=np.uint8)

        result = preprocess_image(image, (16, 48), num_frames=5, frame_noise_std=0.01)

        # Check extra frames differ from first (with noise)
        for i in range(1, 5):
            assert not torch.equal(
                result[0, 0], result[0, i]
            ), f"Frame {i} should differ from frame 0 with noise"

    def test_output_shape_correct(self, preprocess_image):
        """Output shape should be (1, num_frames, C, H, W)."""
        image = np.random.randint(0, 255, (64, 192, 3), dtype=np.uint8)

        result = preprocess_image(image, (16, 48), num_frames=5)

        assert result.shape == (1, 5, 3, 16, 48), f"Unexpected shape: {result.shape}"


class TestCheckpointSecurity:
    """Tests for checkpoint loading security checks."""

    @pytest.fixture
    def load_checkpoint(self):
        """Import load_checkpoint from train.py."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("train", "train.py")
        train_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(train_module)
        except SystemExit:
            pass
        return train_module.load_checkpoint

    def test_rejects_nonexistent_file(self, load_checkpoint):
        """load_checkpoint should reject nonexistent files."""
        model = nn.Linear(10, 5)

        with pytest.raises(FileNotFoundError) as exc_info:
            load_checkpoint(model, "/nonexistent/path/model.pth")

        assert "not found or not a local file" in str(exc_info.value)

    def test_emits_security_warning(self, load_checkpoint, tmp_path):
        """load_checkpoint should emit security warning."""
        import warnings

        # Create a minimal valid checkpoint
        model = nn.Linear(10, 5)
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        # Load and check for warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_checkpoint(model, str(checkpoint_path))

            # Check that a security warning was issued
            security_warnings = [
                warning for warning in w if "pickle" in str(warning.message).lower()
            ]
            assert len(security_warnings) > 0, "Should emit security warning about pickle"


@pytest.mark.smoke
class TestSmokeReproducibility:
    """Quick smoke tests for reproducibility."""

    def test_torch_manual_seed_works(self):
        """Basic check that torch seeding works."""
        torch.manual_seed(42)
        a = torch.rand(3)

        torch.manual_seed(42)
        b = torch.rand(3)

        assert torch.equal(a, b)

    def test_numpy_seed_works(self):
        """Basic check that numpy seeding works."""
        np.random.seed(42)
        a = np.random.rand(3)

        np.random.seed(42)
        b = np.random.rand(3)

        assert np.array_equal(a, b)
