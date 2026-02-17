"""Comprehensive tests for eqnet.data.das transforms and label generation."""

import random

import numpy as np
import pytest

from eqnet.data.das import (
    ColoredNoise,
    Compose,
    FlipLR,
    Identity,
    LabelConfig,
    Masking,
    MedianFilter,
    Normalize,
    RandomApply,
    RandomBandpass,
    RandomCrop,
    RandomGain,
    ResampleSpace,
    ResampleTime,
    Sample,
    StackEvents,
    StackNoise,
    Target,
    generate_labels,
)

# =============================================================================
# Factories
# =============================================================================

NCH, NX, NT = 1, 64, 512


def make_target(
    p_times=None, s_times=None, channels=None,
    snr=10.0, amp_signal=1.0, amp_noise=0.1,
):
    """Create a DAS Target with picks on specified channels."""
    p_times = p_times or [150]
    s_times = s_times or [250]
    channels = channels or [10, 20, 30, 40, 50]

    p_picks = [(ch, t) for ch in channels for t in p_times]
    s_picks = [(ch, t) for ch in channels for t in s_times]

    event_centers = []
    ps_intervals = []
    for ch in channels:
        for pt, st in zip(p_times, s_times):
            center = (pt + st) / 2
            event_centers.append((ch, center))
            ps_intervals.append((ch, st - pt))

    return Target(
        p_picks=p_picks, s_picks=s_picks,
        event_centers=event_centers, ps_intervals=ps_intervals,
        snr=snr, amp_signal=amp_signal, amp_noise=amp_noise,
    )


def make_sample(nt=NT, nx=NX, target=None, seed=42):
    """Create a DAS Sample with synthetic waveform and one target."""
    rng = np.random.RandomState(seed)
    waveform = rng.randn(NCH, nx, nt).astype(np.float32)
    if target is None:
        target = make_target()
    return Sample(waveform=waveform, targets=[target], dt_s=0.01, dx_m=10.0)


# =============================================================================
# Target method tests
# =============================================================================


class TestTargetCrop:
    def test_picks_shifted(self):
        t = Target(
            p_picks=[(5, 100), (20, 200), (70, 300)],
            s_picks=[(5, 150), (20, 250), (70, 350)],
            event_centers=[(5, 125), (20, 225), (70, 325)],
            ps_intervals=[(5, 50), (20, 50), (70, 50)],
        )
        # Crop: t0=100, nt=200, x0=10, nx=30 → keep [10,40) x [100,300)
        t.crop(t0=100, nt=200, x0=10, nx=30)
        # Only (20, 200) and (20, 250) are in range
        assert t.p_picks == [(10, 100)]  # ch=20-10=10, t=200-100=100
        assert t.s_picks == [(10, 150)]  # ch=20-10=10, t=250-100=150

    def test_all_outside_is_empty(self):
        t = Target(p_picks=[(5, 50)], s_picks=[(5, 80)])
        t.crop(t0=200, nt=100, x0=0, nx=64)
        assert t.is_empty


class TestTargetShiftTime:
    def test_wrap(self):
        t = Target(p_picks=[(10, 400)], s_picks=[(10, 450)])
        t.shift_time(200, 500, wrap=True)
        assert t.p_picks == [(10, 100)]  # (400+200) % 500
        assert t.s_picks == [(10, 150)]

    def test_nowrap_drops(self):
        t = Target(
            p_picks=[(10, 400), (20, 100)],
            s_picks=[(10, 450), (20, 200)],
            event_centers=[(10, 425), (20, 150)],
            ps_intervals=[(10, 50), (20, 100)],
        )
        t.shift_time(200, 500, wrap=False)
        assert t.p_picks == [(20, 300)]
        assert t.s_picks == [(20, 400)]
        assert t.event_centers == [(20, 350)]


class TestTargetScaleTime:
    def test_basic(self):
        t = Target(
            p_picks=[(10, 100)], s_picks=[(10, 200)],
            event_centers=[(10, 150)], ps_intervals=[(10, 100)],
        )
        t.scale_time(2.0)
        assert t.p_picks == [(10, 200.0)]
        assert t.s_picks == [(10, 400.0)]
        assert t.event_centers == [(10, 300.0)]
        assert t.ps_intervals == [(10, 200.0)]


class TestTargetFlipSpace:
    def test_basic(self):
        t = Target(
            p_picks=[(0, 100), (63, 200)],
            s_picks=[(0, 150), (63, 250)],
            event_centers=[(0, 125)],
            ps_intervals=[(0, 50)],
        )
        t.flip_space(64)
        assert t.p_picks == [(63, 100), (0, 200)]
        assert t.s_picks == [(63, 150), (0, 250)]
        assert t.event_centers == [(63, 125)]


class TestTargetScaleSpace:
    def test_basic(self):
        t = Target(p_picks=[(10, 100)], s_picks=[(10, 200)])
        t.scale_space(2.0)
        assert t.p_picks == [(20, 100)]
        assert t.s_picks == [(20, 200)]


class TestTargetMaskTime:
    def test_picks_removed(self):
        t = Target(
            p_picks=[(10, 100), (10, 300)],
            s_picks=[(10, 200), (10, 400)],
            event_centers=[(10, 150), (10, 350)],
            ps_intervals=[(10, 100), (10, 100)],
        )
        t.mask_time(90, 210)
        # picks at 100 and 200 removed (in [90, 210))
        assert t.p_picks == [(10, 300)]
        assert t.s_picks == [(10, 400)]
        assert t.event_centers == [(10, 350)]

    def test_no_overlap(self):
        t = Target(p_picks=[(10, 300)], s_picks=[(10, 400)])
        t.mask_time(0, 100)
        assert t.p_picks == [(10, 300)]
        assert t.s_picks == [(10, 400)]


# =============================================================================
# Transform tests
# =============================================================================


class TestNormalize:
    def test_global(self):
        sample = make_sample()
        result = Normalize(mode="global")(sample)
        assert result.waveform.shape == (NCH, NX, NT)
        assert abs(result.waveform.std() - 1.0) < 0.05

    def test_channel(self):
        sample = make_sample()
        result = Normalize(mode="channel")(sample)
        assert result.waveform.shape == (NCH, NX, NT)
        # Each channel (spatial position) should have std ~1
        for ch in range(min(5, NX)):
            std = result.waveform[0, ch].std()
            assert abs(std - 1.0) < 0.1 or std < 0.01  # allow near-zero channels

    def test_nan(self):
        sample = make_sample()
        sample.waveform[0, 0, :5] = np.nan
        result = Normalize()(sample)
        assert not np.any(np.isnan(result.waveform))

    def test_zero_std(self):
        sample = make_sample()
        sample.waveform[:] = 3.0
        result = Normalize()(sample)
        assert np.all(np.isfinite(result.waveform))


class TestMedianFilter:
    def test_removes_common_mode(self):
        sample = make_sample()
        # Add a constant signal across all channels
        common = np.sin(np.linspace(0, 2 * np.pi, NT)).astype(np.float32)
        sample.waveform[0, :, :] += common[np.newaxis, :]
        result = MedianFilter()(sample)
        # The median should have been subtracted
        median_after = np.median(result.waveform[0], axis=0)
        assert np.allclose(median_after, 0.0, atol=1e-5)

    def test_preserves_shape(self):
        sample = make_sample()
        result = MedianFilter()(sample)
        assert result.waveform.shape == (NCH, NX, NT)


class TestRandomCrop:
    def test_output_shape(self):
        random.seed(0)
        sample = make_sample(nt=1024, nx=128)
        result = RandomCrop(nt=256, nx=64)(sample)
        assert result.waveform.shape == (NCH, 64, 256)

    def test_pads_if_smaller(self):
        """Reflect padding requires pad < input dim, so use moderate sizes."""
        random.seed(0)
        sample = make_sample(nt=400, nx=50)
        result = RandomCrop(nt=512, nx=64)(sample)
        assert result.waveform.shape == (NCH, 64, 512)

    def test_picks_adjusted(self):
        random.seed(0)
        sample = make_sample(nt=1024, nx=128, target=make_target(
            p_times=[300], s_times=[500], channels=[30, 50, 70],
        ))
        result = RandomCrop(nt=512, nx=64)(sample)
        for target in result.targets:
            for ch, t in target.p_picks + target.s_picks:
                assert 0 <= ch < 64
                assert 0 <= t < 512

    def test_no_op_when_exact(self):
        sample = make_sample(nt=NT, nx=NX)
        result = RandomCrop(nt=NT, nx=NX)(sample)
        assert result.waveform.shape == (NCH, NX, NT)


class TestFlipLR:
    def test_waveform_flipped(self):
        sample = make_sample()
        # Make spatial axis distinguishable
        sample.waveform[0, :, 0] = np.arange(NX, dtype=np.float32)
        result = FlipLR(p=1.0)(sample)
        expected = np.arange(NX - 1, -1, -1, dtype=np.float32)
        np.testing.assert_array_equal(result.waveform[0, :, 0], expected)

    def test_picks_flipped(self):
        target = make_target(channels=[0, 10, 63])
        sample = make_sample(target=target)
        result = FlipLR(p=1.0)(sample)
        flipped_chs = {ch for ch, _ in result.targets[0].p_picks}
        assert 63 in flipped_chs  # 0 → 63
        assert 53 in flipped_chs  # 10 → 53
        assert 0 in flipped_chs   # 63 → 0

    def test_p_zero_never_flips(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = FlipLR(p=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_p_one_always_flips(self):
        for _ in range(5):
            sample = make_sample()
            original = sample.waveform.copy()
            result = FlipLR(p=1.0)(sample)
            # Should be flipped along axis -2
            np.testing.assert_array_equal(
                result.waveform, np.flip(original, axis=-2)
            )


class TestResampleTime:
    def test_shape(self):
        random.seed(42)
        sample = make_sample()
        result = ResampleTime(min_factor=2.0, max_factor=2.0)(sample)
        assert result.nt == int(NT * 2)
        assert result.nx == NX

    def test_picks_scaled(self):
        random.seed(42)
        target = make_target(p_times=[100], s_times=[200], channels=[10])
        sample = make_sample(target=target)
        result = ResampleTime(min_factor=2.0, max_factor=2.0)(sample)
        for t in result.targets:
            for _, pick_t in t.p_picks:
                assert abs(pick_t - 200.0) < 1.0
            for _, pick_t in t.s_picks:
                assert abs(pick_t - 400.0) < 1.0

    def test_near_one_no_op(self):
        random.seed(42)
        sample = make_sample()
        original_nt = sample.nt
        result = ResampleTime(min_factor=1.0, max_factor=1.0)(sample)
        assert result.nt == original_nt


class TestResampleSpace:
    def test_shape(self):
        random.seed(42)
        sample = make_sample()
        result = ResampleSpace(min_factor=2.0, max_factor=2.0)(sample)
        assert result.nx == NX * 2
        assert result.nt == NT

    def test_picks_scaled(self):
        random.seed(42)
        target = make_target(p_times=[100], s_times=[200], channels=[10])
        sample = make_sample(target=target)
        result = ResampleSpace(min_factor=2.0, max_factor=2.0)(sample)
        for t in result.targets:
            for ch, _ in t.p_picks:
                assert ch == 20  # 10 * 2


class TestMasking:
    def test_zeros_window(self):
        random.seed(0)
        sample = make_sample()
        sample.waveform[:] = 1.0
        result = Masking(max_mask_nt=128, p=1.0)(sample)
        # Some region should be zeroed
        assert np.any(result.waveform == 0.0)
        # But not all
        assert np.any(result.waveform != 0.0)

    def test_picks_removed(self):
        random.seed(42)
        target = make_target(p_times=[100], s_times=[200], channels=[10])
        sample = make_sample(target=target)
        # Force mask to cover pick at t=100
        # We set a specific seed and large mask to increase chance of covering picks
        masked = False
        for seed in range(100):
            random.seed(seed)
            s = make_sample(target=make_target(p_times=[100], s_times=[200], channels=[10]))
            result = Masking(max_mask_nt=400, p=1.0)(s)
            p_times_after = [t for _, t in result.targets[0].p_picks] if result.targets else []
            if 100 not in p_times_after:
                masked = True
                break
        assert masked, "Expected masking to remove pick in at least one trial"

    def test_p_zero_never_masks(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = Masking(p=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestRandomGain:
    def test_per_channel_variation(self):
        np.random.seed(42)
        sample = make_sample()
        sample.waveform[:] = 1.0
        result = RandomGain(min_gain=0.5, max_gain=2.0)(sample)
        # Each channel should have a different gain
        ch_vals = [result.waveform[0, ch, 0] for ch in range(NX)]
        assert len(set(ch_vals)) == NX

    def test_range(self):
        np.random.seed(42)
        sample = make_sample()
        sample.waveform[:] = 1.0
        result = RandomGain(min_gain=0.5, max_gain=2.0)(sample)
        gains = result.waveform[0, :, 0]
        assert np.all(gains >= 0.49)
        assert np.all(gains <= 2.01)


class TestColoredNoise:
    def test_shape_preserved(self):
        random.seed(42)
        np.random.seed(42)
        sample = make_sample()
        result = ColoredNoise(p=1.0)(sample)
        assert result.waveform.shape == (NCH, NX, NT)

    def test_modifies_waveform(self):
        random.seed(42)
        np.random.seed(42)
        sample = make_sample()
        original = sample.waveform.copy()
        result = ColoredNoise(p=1.0)(sample)
        assert not np.allclose(result.waveform, original)

    def test_p_zero_no_change(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = ColoredNoise(p=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_zero_signal(self):
        random.seed(42)
        sample = make_sample()
        sample.waveform[:] = 0.0
        result = ColoredNoise(p=1.0)(sample)
        np.testing.assert_array_equal(result.waveform, 0.0)


class TestRandomBandpass:
    def test_shape_preserved(self):
        random.seed(42)
        sample = make_sample()
        result = RandomBandpass(p=1.0)(sample)
        assert result.waveform.shape == (NCH, NX, NT)

    def test_modifies_waveform(self):
        random.seed(42)
        sample = make_sample()
        original = sample.waveform.copy()
        result = RandomBandpass(p=1.0)(sample)
        assert not np.allclose(result.waveform, original)

    def test_p_zero_no_change(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = RandomBandpass(p=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestStackEvents:
    def test_no_fn_unchanged(self):
        sample = make_sample()
        n_targets = len(sample.targets)
        result = StackEvents(p=1.0)(sample)
        assert len(result.targets) == n_targets

    def test_adds_target(self):
        random.seed(0)
        sample = make_sample(target=make_target(snr=20.0))
        donor = make_sample(seed=99, target=make_target(snr=20.0))

        stacker = StackEvents(p=1.0, min_snr=0.0)
        stacker.set_sample_fn(lambda: donor.copy())
        result = stacker(sample)
        assert len(result.targets) >= 2

    def test_shape_mismatch_skip(self):
        random.seed(0)
        sample = make_sample(nt=512)
        donor = make_sample(nt=256)

        stacker = StackEvents(p=1.0, min_snr=0.0)
        stacker.set_sample_fn(lambda: donor.copy())
        result = stacker(sample)
        assert len(result.targets) == 1


class TestStackNoise:
    def test_no_fn_unchanged(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = StackNoise(p=1.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_modifies_waveform(self):
        random.seed(0)
        np.random.seed(0)
        sample = make_sample(target=make_target(amp_signal=1.0))
        original = sample.waveform.copy()
        noise = np.random.randn(NCH, NX, NT).astype(np.float32)
        sn = StackNoise(p=1.0, max_ratio=1.0)
        sn.set_noise_fn(lambda: noise)
        result = sn(sample)
        assert not np.allclose(result.waveform, original)

    def test_shape_mismatch_skip(self):
        sample = make_sample()
        original = sample.waveform.copy()
        wrong = np.random.randn(NCH, NX, NT + 50).astype(np.float32)
        sn = StackNoise(p=1.0)
        sn.set_noise_fn(lambda: wrong)
        result = sn(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestRandomApply:
    def test_p_zero(self):
        sample = make_sample()
        original = sample.waveform.copy()
        t = RandomApply(FlipLR(p=1.0), p=0.0)
        result = t(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_p_one(self):
        sample = make_sample()
        original = sample.waveform.copy()
        t = RandomApply(FlipLR(p=1.0), p=1.0)
        result = t(sample)
        np.testing.assert_array_equal(
            result.waveform, np.flip(original, axis=-2)
        )


class TestCompose:
    def test_chains(self):
        sample = make_sample()
        original = sample.waveform.copy()
        c = Compose([Identity(), FlipLR(p=1.0)])
        result = c(sample)
        np.testing.assert_array_equal(
            result.waveform, np.flip(original, axis=-2)
        )


# =============================================================================
# Label generation tests
# =============================================================================


class TestGenerateLabels:
    def test_shapes(self):
        sample = make_sample()
        labels = generate_labels(sample)
        assert labels["phase_pick"].shape == (3, NX, NT)
        assert labels["phase_mask"].shape == (1, NX, NT)
        assert labels["event_center"].shape == (1, NX, NT)
        assert labels["event_time"].shape == (1, NX, NT)
        assert labels["event_center_mask"].shape == (1, NX, NT)
        assert labels["event_time_mask"].shape == (1, NX, NT)

    def test_gaussian_peak_at_pick(self):
        p_time, s_time = 150, 250
        target = make_target(p_times=[p_time], s_times=[s_time], channels=[10])
        sample = make_sample(target=target)
        labels = generate_labels(sample)
        p_label = labels["phase_pick"][1, 10]
        assert np.argmax(p_label) == p_time
        assert p_label[p_time] > 0.9
        s_label = labels["phase_pick"][2, 10]
        assert np.argmax(s_label) == s_time
        assert s_label[s_time] > 0.9

    def test_noise_complement(self):
        sample = make_sample()
        labels = generate_labels(sample)
        noise = labels["phase_pick"][0]
        p_label = labels["phase_pick"][1]
        s_label = labels["phase_pick"][2]
        expected = np.maximum(0, 1.0 - p_label - s_label)
        np.testing.assert_array_almost_equal(noise, expected)

    def test_empty_targets(self):
        sample = make_sample()
        sample.targets = []
        labels = generate_labels(sample)
        assert np.allclose(labels["phase_pick"][1], 0.0)
        assert np.allclose(labels["phase_pick"][2], 0.0)
        assert np.allclose(labels["phase_pick"][0], 1.0)
        assert np.allclose(labels["phase_mask"], 0.0)

    def test_phase_mask_both_ps(self):
        """When both P and S on a channel, mask is full trace."""
        target = make_target(p_times=[150], s_times=[250], channels=[10])
        sample = make_sample(target=target)
        labels = generate_labels(sample)
        assert np.all(labels["phase_mask"][0, 10] == 1.0)

    def test_phase_mask_single_pick(self):
        """When only one phase on a channel, mask is narrow window."""
        target = Target(p_picks=[(10, 150)], s_picks=[])
        sample = make_sample(target=target)
        labels = generate_labels(sample)
        mask = labels["phase_mask"][0, 10]
        assert not np.all(mask == 1.0)
        assert mask[150] == 1.0
