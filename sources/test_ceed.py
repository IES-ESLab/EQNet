# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
# ]
# ///
"""
Test suite for CEED (California Earthquake Event Dataset).

Tests cover:
1. Transform correctness and composability
2. Label generation accuracy
3. Sample data structure integrity
4. Dataset loading and iteration
5. Stacking augmentation behavior

Run with: uv run pytest test_ceed.py -v
Or standalone: uv run test_ceed.py
"""

from __future__ import annotations

import numpy as np
import torch

# Optional pytest import for running with pytest
try:
    import pytest
except ImportError:
    pytest = None

# Provide fixture decorator for standalone running
if pytest is None:
    class _fixture:
        def __call__(self, func):
            return func

    class _PytestMock:
        fixture = _fixture()

    pytest = _PytestMock()

from ceed import (
    # Core data structures
    Sample,
    LabelConfig,
    # Transforms
    Transform,
    Compose,
    Identity,
    Normalize,
    RandomCrop,
    CenterCrop,
    RandomShift,
    TimeStretch,
    FlipPolarity,
    RandomAmplitudeScale,
    DropChannel,
    AddGaussianNoise,
    StackEvents,
    StackNoise,
    # Label generation
    generate_gaussian_label,
    generate_phase_labels,
    # Presets
    default_train_transforms,
    default_eval_transforms,
    minimal_transforms,
    # Data loading
    record_to_sample,
    SampleBuffer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_basic() -> Sample:
    """Create a basic sample for testing."""
    np.random.seed(42)
    return Sample(
        waveform=np.random.randn(3, 8192).astype(np.float32),
        p_indices=[2000, 5000],
        s_indices=[2500, 5800],
        polarity_up=[2000],
        polarity_down=[5000],
        event_center=[2250.0, 5400.0],
        event_time=[1800.0, 4800.0],
        snr=10.0,
        amp_signal=1.0,
        amp_noise=0.1,
        trace_id="test/XX.STA01",
        sensor="HH",
    )


@pytest.fixture
def sample_short() -> Sample:
    """Create a short sample for crop testing."""
    np.random.seed(42)
    return Sample(
        waveform=np.random.randn(3, 2048).astype(np.float32),
        p_indices=[500],
        s_indices=[800],
        snr=5.0,
        amp_signal=1.0,
        amp_noise=0.2,
    )


@pytest.fixture
def sample_buffer(sample_basic: Sample) -> SampleBuffer:
    """Create a sample buffer with test samples."""
    buffer = SampleBuffer(max_size=100)
    for i in range(50):
        s = sample_basic.copy()
        s.trace_id = f"test/XX.STA{i:02d}"
        buffer.add(s)
    return buffer


# =============================================================================
# Sample Tests
# =============================================================================

class TestSample:
    """Test Sample dataclass."""

    def test_properties(self, sample_basic: Sample):
        """Test sample properties."""
        assert sample_basic.nt == 8192
        assert sample_basic.nch == 3

    def test_copy(self, sample_basic: Sample):
        """Test deep copy."""
        copy = sample_basic.copy()

        # Modify original
        sample_basic.waveform[0, 0] = 999.0
        sample_basic.p_indices.append(9999)

        # Copy should be unchanged
        assert copy.waveform[0, 0] != 999.0
        assert 9999 not in copy.p_indices

    def test_empty_lists(self):
        """Test sample with empty lists."""
        sample = Sample(waveform=np.zeros((3, 1000), dtype=np.float32))
        assert sample.p_indices == []
        assert sample.s_indices == []
        assert sample.nt == 1000


# =============================================================================
# Transform Tests
# =============================================================================

class TestBasicTransforms:
    """Test basic waveform transforms."""

    def test_identity(self, sample_basic: Sample):
        """Test identity transform."""
        t = Identity()
        result = t(sample_basic)
        np.testing.assert_array_equal(result.waveform, sample_basic.waveform)

    def test_normalize(self, sample_basic: Sample):
        """Test normalization."""
        t = Normalize()
        result = t(sample_basic.copy())

        # Check mean ~0 and std ~1
        assert np.abs(result.waveform.mean()) < 1e-6
        assert np.abs(result.waveform.std() - 1.0) < 0.1

    def test_normalize_handles_zeros(self):
        """Test normalization handles zero arrays."""
        sample = Sample(waveform=np.zeros((3, 1000), dtype=np.float32))
        t = Normalize()
        result = t(sample)
        assert not np.any(np.isnan(result.waveform))

    def test_normalize_handles_nan(self):
        """Test normalization handles NaN values."""
        waveform = np.random.randn(3, 1000).astype(np.float32)
        waveform[0, 100:200] = np.nan
        sample = Sample(waveform=waveform)
        t = Normalize()
        result = t(sample)
        assert not np.any(np.isnan(result.waveform))


class TestTemporalTransforms:
    """Test temporal transforms that modify indices."""

    def test_random_crop(self, sample_basic: Sample):
        """Test random crop."""
        t = RandomCrop(length=4096)
        result = t(sample_basic.copy())

        assert result.nt == 4096
        # All indices should be in valid range
        for idx in result.p_indices + result.s_indices:
            assert 0 <= idx < 4096

    def test_random_crop_preserves_phases(self, sample_basic: Sample):
        """Test random crop keeps at least min_phases."""
        t = RandomCrop(length=4096, min_phases=1, max_tries=100)
        result = t(sample_basic.copy())

        total_phases = len(result.p_indices) + len(result.s_indices)
        assert total_phases >= 1

    def test_random_crop_short_waveform(self, sample_short: Sample):
        """Test crop when waveform is shorter than target."""
        t = RandomCrop(length=4096)
        result = t(sample_short.copy())

        # Should not change
        assert result.nt == 2048

    def test_center_crop(self, sample_basic: Sample):
        """Test center crop."""
        t = CenterCrop(length=4096)
        result = t(sample_basic.copy())

        assert result.nt == 4096

    def test_random_shift_circular(self, sample_basic: Sample):
        """Test circular shift."""
        t = RandomShift(max_shift=100, mode="circular")
        result = t(sample_basic.copy())

        # Length should be preserved
        assert result.nt == sample_basic.nt
        # All indices should be in valid range
        for idx in result.p_indices + result.s_indices:
            assert 0 <= idx < result.nt

    def test_random_shift_zero(self, sample_basic: Sample):
        """Test zero-pad shift removes out-of-bounds phases."""
        # Create sample with phase near edge
        sample = Sample(
            waveform=np.random.randn(3, 1000).astype(np.float32),
            p_indices=[50],  # Near start
            s_indices=[950],  # Near end
        )

        # Large negative shift should remove p_indices[0]
        t = RandomShift(max_shift=100, mode="zero")
        # Run multiple times to test edge cases
        for _ in range(10):
            result = t(sample.copy())
            for idx in result.p_indices + result.s_indices:
                assert 0 <= idx < result.nt


class TestAmplitudeTransforms:
    """Test amplitude/polarity transforms."""

    def test_flip_polarity_always(self):
        """Test flip polarity with p=1."""
        sample = Sample(
            waveform=np.ones((3, 100), dtype=np.float32),
            polarity_up=[10],
            polarity_down=[20],
        )
        t = FlipPolarity(p=1.0)
        result = t(sample.copy())

        # Waveform should be negated
        np.testing.assert_array_equal(result.waveform, -sample.waveform)
        # Polarity labels should be swapped
        assert result.polarity_up == [20]
        assert result.polarity_down == [10]

    def test_flip_polarity_never(self, sample_basic: Sample):
        """Test flip polarity with p=0."""
        t = FlipPolarity(p=0.0)
        result = t(sample_basic.copy())
        np.testing.assert_array_equal(result.waveform, sample_basic.waveform)

    def test_random_amplitude_scale(self, sample_basic: Sample):
        """Test amplitude scaling."""
        t = RandomAmplitudeScale(min_scale=2.0, max_scale=2.0, log_scale=False)
        result = t(sample_basic.copy())

        # Should be scaled by 2
        np.testing.assert_array_almost_equal(result.waveform, sample_basic.waveform * 2)
        assert result.amp_signal == sample_basic.amp_signal * 2

    def test_drop_channel(self, sample_basic: Sample):
        """Test channel dropping."""
        t = DropChannel(p=1.0)  # Always apply

        # Run multiple times to ensure at least one drops channels
        dropped = False
        for _ in range(10):
            result = t(sample_basic.copy())
            zero_channels = np.sum(np.all(result.waveform == 0, axis=-1))
            if zero_channels >= 1:
                dropped = True
                break

        assert dropped, "DropChannel should drop at least one channel in 10 tries"


class TestNoiseTransforms:
    """Test noise augmentation transforms."""

    def test_add_gaussian_noise(self, sample_basic: Sample):
        """Test Gaussian noise addition."""
        t = AddGaussianNoise(snr_db_range=(10, 10))  # Fixed SNR
        original = sample_basic.waveform.copy()
        result = t(sample_basic.copy())

        # Waveform should be modified
        assert not np.allclose(result.waveform, original)


# =============================================================================
# Compose Tests
# =============================================================================

class TestCompose:
    """Test transform composition."""

    def test_compose_basic(self, sample_basic: Sample):
        """Test basic composition."""
        t = Compose([
            Normalize(),
            RandomCrop(4096),
        ])
        result = t(sample_basic.copy())

        assert result.nt == 4096
        assert np.abs(result.waveform.std() - 1.0) < 0.2  # Approximately normalized

    def test_compose_empty(self, sample_basic: Sample):
        """Test empty composition."""
        t = Compose([])
        result = t(sample_basic.copy())
        np.testing.assert_array_equal(result.waveform, sample_basic.waveform)

    def test_compose_repr(self):
        """Test compose string representation."""
        t = Compose([Normalize(), RandomCrop(4096)])
        repr_str = repr(t)
        assert "Compose" in repr_str
        assert "Normalize" in repr_str
        assert "RandomCrop" in repr_str


# =============================================================================
# Stacking Tests
# =============================================================================

class TestStackEvents:
    """Test event stacking augmentation."""

    def test_stack_events_basic(self, sample_basic: Sample, sample_buffer: SampleBuffer):
        """Test basic event stacking."""
        t = StackEvents(max_events=1, max_shift=1000)
        t.set_sample_fn(sample_buffer.get_random)

        result = t(sample_basic.copy())

        # Should have more phases after stacking (or same if stacking failed)
        assert len(result.p_indices) >= len(sample_basic.p_indices)

    def test_stack_events_no_buffer(self, sample_basic: Sample):
        """Test stacking without buffer returns unchanged."""
        t = StackEvents(max_events=2)
        # Don't set sample_fn
        result = t(sample_basic.copy())

        # Should be unchanged
        assert len(result.p_indices) == len(sample_basic.p_indices)

    def test_stack_events_different_shapes(self, sample_basic: Sample):
        """Test stacking skips mismatched shapes."""
        # Create buffer with different-sized samples
        buffer = SampleBuffer(10)
        short_sample = Sample(
            waveform=np.random.randn(3, 1000).astype(np.float32),  # Different size
            p_indices=[200],
            s_indices=[400],
            amp_signal=1.0,
            amp_noise=0.1,
        )
        buffer.add(short_sample)

        t = StackEvents(max_events=1)
        t.set_sample_fn(buffer.get_random)

        result = t(sample_basic.copy())

        # Should be unchanged (size mismatch)
        assert result.nt == sample_basic.nt


class TestStackNoise:
    """Test noise stacking augmentation."""

    def test_stack_noise_basic(self, sample_basic: Sample, sample_buffer: SampleBuffer):
        """Test basic noise stacking."""
        t = StackNoise(max_ratio=1.0)
        t.set_sample_fn(sample_buffer.get_random)

        original = sample_basic.waveform.copy()
        result = t(sample_basic.copy())

        # Waveform may be modified (depends on random sampling)
        # At minimum, shape should be preserved
        assert result.waveform.shape == original.shape


# =============================================================================
# Label Generation Tests
# =============================================================================

class TestLabelGeneration:
    """Test label generation functions."""

    def test_gaussian_label_basic(self):
        """Test basic Gaussian label generation."""
        indices = [500]
        label = generate_gaussian_label(indices, length=1000, width=60)

        assert label.shape == (1000,)
        assert label[500] > 0.9  # Peak should be near 1
        assert label[0] < 0.01  # Far from peak should be near 0
        assert label[999] < 0.01

    def test_gaussian_label_multiple(self):
        """Test multiple peaks."""
        indices = [200, 800]
        label = generate_gaussian_label(indices, length=1000, width=60)

        assert label[200] > 0.9
        assert label[800] > 0.9
        assert label[500] < 0.1  # Between peaks

    def test_gaussian_label_empty(self):
        """Test empty indices."""
        label = generate_gaussian_label([], length=1000, width=60)
        assert np.all(label == 0)

    def test_generate_phase_labels(self, sample_basic: Sample):
        """Test full label generation."""
        # Crop first to get consistent length
        t = RandomCrop(4096)
        sample = t(sample_basic.copy())

        labels = generate_phase_labels(sample)

        # Check shapes
        assert labels["phase_pick"].shape == (3, sample.nt)
        assert labels["phase_mask"].shape == (sample.nt,)
        assert labels["polarity"].shape == (sample.nt,)
        assert labels["polarity_mask"].shape == (sample.nt,)
        assert labels["event_center"].shape == (sample.nt,)
        assert labels["event_time"].shape == (sample.nt,)
        assert labels["event_mask"].shape == (sample.nt,)

        # Check phase_pick sums to ~1
        phase_sum = labels["phase_pick"].sum(axis=0)
        # Near picks, sum should be close to 1
        assert np.all(phase_sum >= 0.9)
        assert np.all(phase_sum <= 1.1)

        # Check polarity range
        assert np.all(labels["polarity"] >= 0)
        assert np.all(labels["polarity"] <= 1)

    def test_labels_with_config(self, sample_basic: Sample):
        """Test label generation with custom config."""
        config = LabelConfig(
            phase_width=30,  # Narrower
            polarity_width=10,
            event_width=100,
        )

        t = RandomCrop(4096)
        sample = t(sample_basic.copy())

        labels = generate_phase_labels(sample, config)

        # Labels should still be valid
        assert labels["phase_pick"].shape == (3, sample.nt)


# =============================================================================
# Preset Tests
# =============================================================================

class TestPresets:
    """Test transform presets."""

    def test_default_train_transforms(self, sample_basic: Sample):
        """Test default training transforms."""
        t = default_train_transforms(crop_length=4096, enable_stacking=False)
        result = t(sample_basic.copy())

        assert result.nt == 4096

    def test_default_eval_transforms(self, sample_basic: Sample):
        """Test default eval transforms."""
        t = default_eval_transforms(crop_length=4096)
        result = t(sample_basic.copy())

        assert result.nt == 4096

    def test_minimal_transforms(self, sample_basic: Sample):
        """Test minimal transforms."""
        t = minimal_transforms()
        result = t(sample_basic.copy())

        # Only normalization, length unchanged
        assert result.nt == sample_basic.nt
        assert np.abs(result.waveform.std() - 1.0) < 0.1


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestRecordConversion:
    """Test record to sample conversion."""

    def test_record_to_sample_basic(self):
        """Test basic record conversion."""
        record = {
            "waveform": np.random.randn(3, 4096).astype(np.float32).tolist(),
            "p_phase_index": 1000,
            "s_phase_index": 1500,
            "p_phase_polarity": "U",
            "event_id": "ev001",
            "network": "XX",
            "station": "STA01",
            "snr": 10.0,
        }

        sample = record_to_sample(record)

        assert sample.nt == 4096
        assert sample.p_indices == [1000]
        assert sample.s_indices == [1500]
        assert sample.polarity_up == [1000]
        assert sample.snr == 10.0

    def test_record_to_sample_missing_fields(self):
        """Test conversion with missing optional fields."""
        record = {
            "waveform": np.random.randn(3, 4096).astype(np.float32).tolist(),
            "p_phase_index": None,
            "s_phase_index": None,
        }

        sample = record_to_sample(record)

        assert sample.p_indices == []
        assert sample.s_indices == []


class TestSampleBuffer:
    """Test sample buffer for stacking."""

    def test_buffer_add_and_get(self):
        """Test adding and retrieving samples."""
        buffer = SampleBuffer(max_size=10)

        for i in range(5):
            sample = Sample(
                waveform=np.full((3, 100), i, dtype=np.float32),
                trace_id=f"test_{i}",
            )
            buffer.add(sample)

        assert len(buffer) == 5

        retrieved = buffer.get_random()
        assert retrieved is not None
        assert retrieved.trace_id.startswith("test_")

    def test_buffer_reservoir_sampling(self):
        """Test reservoir sampling when buffer is full."""
        buffer = SampleBuffer(max_size=10)

        for i in range(100):
            sample = Sample(
                waveform=np.full((3, 100), i, dtype=np.float32),
                trace_id=f"test_{i}",
            )
            buffer.add(sample)

        assert len(buffer) == 10
        assert buffer.count == 100

    def test_buffer_empty(self):
        """Test empty buffer returns None."""
        buffer = SampleBuffer(max_size=10)
        assert buffer.get_random() is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_training_pipeline(self, sample_basic: Sample, sample_buffer: SampleBuffer):
        """Test complete training pipeline."""
        # Create transforms with stacking
        transforms = default_train_transforms(
            crop_length=4096,
            enable_stacking=True,
            enable_noise_stacking=True,
        )

        # Connect to buffer
        for t in transforms.transforms:
            if isinstance(t, (StackEvents, StackNoise)):
                t.set_sample_fn(sample_buffer.get_random)

        # Process sample
        result = transforms(sample_basic.copy())

        # Generate labels
        labels = generate_phase_labels(result)

        # Should produce valid output
        assert result.nt == 4096
        assert labels["phase_pick"].shape == (3, 4096)

    def test_full_eval_pipeline(self, sample_basic: Sample):
        """Test complete evaluation pipeline."""
        transforms = default_eval_transforms(crop_length=4096)
        result = transforms(sample_basic.copy())
        labels = generate_phase_labels(result)

        assert result.nt == 4096
        assert labels["phase_pick"].shape == (3, 4096)

    def test_output_tensor_shapes(self, sample_basic: Sample):
        """Test that output matches expected tensor shapes for model."""
        transforms = default_eval_transforms(crop_length=4096)
        result = transforms(sample_basic.copy())
        labels = generate_phase_labels(result)

        # Convert to tensors (as dataset would)
        data = torch.from_numpy(result.waveform[:, np.newaxis, :]).float()
        phase_pick = torch.from_numpy(labels["phase_pick"][:, np.newaxis, :]).float()

        # Check shapes match expected
        assert data.shape == (3, 1, 4096)  # (C, 1, T)
        assert phase_pick.shape == (3, 1, 4096)  # (C, 1, T)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_amplitude_sample(self):
        """Test handling of zero amplitude samples."""
        sample = Sample(
            waveform=np.zeros((3, 4096), dtype=np.float32),
            p_indices=[1000],
            s_indices=[1500],
        )

        transforms = minimal_transforms()
        result = transforms(sample)

        # Should not crash
        assert not np.any(np.isnan(result.waveform))

    def test_single_sample_point(self):
        """Test sample with minimum size."""
        sample = Sample(
            waveform=np.random.randn(3, 100).astype(np.float32),
            p_indices=[50],
            s_indices=[75],
        )

        transforms = Compose([Normalize()])
        result = transforms(sample)

        assert result.nt == 100

    def test_many_phases(self):
        """Test sample with many phases."""
        np.random.seed(42)
        sample = Sample(
            waveform=np.random.randn(3, 10000).astype(np.float32),
            p_indices=list(range(100, 9000, 200)),  # Many P picks
            s_indices=list(range(200, 9100, 200)),  # Many S picks
        )

        transforms = default_train_transforms(crop_length=4096, enable_stacking=False)
        result = transforms(sample)

        # Should handle many phases
        labels = generate_phase_labels(result)
        assert labels["phase_pick"].shape == (3, 4096)


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    import sys
    import traceback

    test_classes = [
        TestSample,
        TestBasicTransforms,
        TestTemporalTransforms,
        TestAmplitudeTransforms,
        TestNoiseTransforms,
        TestCompose,
        TestStackEvents,
        TestStackNoise,
        TestLabelGeneration,
        TestPresets,
        TestRecordConversion,
        TestSampleBuffer,
        TestIntegration,
        TestEdgeCases,
    ]

    # Create fixtures
    np.random.seed(42)
    sample_basic = Sample(
        waveform=np.random.randn(3, 8192).astype(np.float32),
        p_indices=[2000, 5000],
        s_indices=[2500, 5800],
        polarity_up=[2000],
        polarity_down=[5000],
        event_center=[2250.0, 5400.0],
        event_time=[1800.0, 4800.0],
        snr=10.0,
        amp_signal=1.0,
        amp_noise=0.1,
        trace_id="test/XX.STA01",
        sensor="HH",
    )

    sample_short = Sample(
        waveform=np.random.randn(3, 2048).astype(np.float32),
        p_indices=[500],
        s_indices=[800],
        snr=5.0,
        amp_signal=1.0,
        amp_noise=0.2,
    )

    sample_buffer = SampleBuffer(max_size=100)
    for i in range(50):
        s = sample_basic.copy()
        s.trace_id = f"test/XX.STA{i:02d}"
        sample_buffer.add(s)

    fixtures = {
        "sample_basic": sample_basic,
        "sample_short": sample_short,
        "sample_buffer": sample_buffer,
    }

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("CEED Test Suite")
    print("=" * 60)

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)

        instance = test_class()
        for name in dir(instance):
            if name.startswith("test_"):
                method = getattr(instance, name)
                try:
                    # Get fixture arguments
                    import inspect
                    sig = inspect.signature(method)
                    kwargs = {}
                    for param in sig.parameters:
                        if param in fixtures:
                            # Create fresh copy of fixture
                            if hasattr(fixtures[param], "copy"):
                                kwargs[param] = fixtures[param].copy() if hasattr(fixtures[param], "copy") else fixtures[param]
                            else:
                                kwargs[param] = fixtures[param]

                    method(**kwargs)
                    print(f"  {name}: PASSED")
                    passed += 1
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")
                    errors.append((test_class.__name__, name, traceback.format_exc()))
                    failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for cls, name, tb in errors:
            print(f"\n{cls}.{name}:")
            print(tb)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
