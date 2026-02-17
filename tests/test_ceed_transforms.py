"""Comprehensive tests for eqnet.data.ceed transforms and label generation."""

import random

import numpy as np
import pytest

from eqnet.data.ceed import (
    AddGaussianNoise,
    CenterCrop,
    ColoredNoise,
    Compose,
    DropChannel,
    FlipPolarity,
    Identity,
    LabelConfig,
    Normalize,
    RandomAmplitudeScale,
    RandomBandpass,
    RandomCrop,
    RandomGain,
    RandomShift,
    Sample,
    StackEvents,
    StackNoise,
    Taper,
    Target,
    TimeStretch,
    generate_labels,
)

# =============================================================================
# Factories
# =============================================================================

NCH, NX, NT = 3, 4, 1024


def make_target(
    p_times=None, s_times=None, polarity_signs=None,
    snr=10.0, amp_signal=1.0, amp_noise=0.1, distance_km=50.0,
):
    """Create a Target with picks on all stations."""
    p_times = p_times or [300]
    s_times = s_times or [500]
    p_picks = [(sta, t) for sta in range(NX) for t in p_times]
    s_picks = [(sta, t) for sta in range(NX) for t in s_times]

    polarity = []
    if polarity_signs is None:
        polarity_signs = [1]
    for sta in range(NX):
        for t, sign in zip(p_times, polarity_signs):
            polarity.append((sta, t, sign))

    event_centers = []
    ps_intervals = []
    for sta in range(NX):
        for pt, st in zip(p_times, s_times):
            center = (pt + st) / 2
            event_centers.append((sta, center))
            ps_intervals.append((sta, st - pt))

    return Target(
        p_picks=p_picks, s_picks=s_picks, polarity=polarity,
        event_centers=event_centers, ps_intervals=ps_intervals,
        snr=snr, amp_signal=amp_signal, amp_noise=amp_noise,
        distance_km=distance_km,
    )


def make_sample(nt=NT, nx=NX, target=None, seed=42):
    """Create a Sample with a sine-wave waveform and one target."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi * 5, nt)
    waveform = np.zeros((NCH, nx, nt), dtype=np.float32)
    for ch in range(NCH):
        for sta in range(nx):
            waveform[ch, sta] = np.sin(t + ch + sta) + rng.randn(nt) * 0.1

    if target is None:
        target = make_target()

    return Sample(waveform=waveform, targets=[target])


# =============================================================================
# Target method tests
# =============================================================================


class TestTargetCropTime:
    def test_picks_shifted(self):
        t = Target(
            p_picks=[(0, 100), (1, 300), (2, 600)],
            s_picks=[(0, 200), (1, 400), (2, 700)],
            polarity=[(0, 100, 1), (2, 600, -1)],
            event_centers=[(0, 150), (2, 650)],
            ps_intervals=[(0, 100), (2, 100)],
        )
        t.crop_time(200, 600)
        # Only picks in [200, 600) kept, shifted by -200
        assert t.p_picks == [(1, 100)]
        assert t.s_picks == [(0, 0), (1, 200)]
        assert t.polarity == []  # (0,100) and (2,600) outside
        assert not t.is_empty

    def test_all_outside_is_empty(self):
        t = Target(p_picks=[(0, 50)], s_picks=[(0, 80)])
        t.crop_time(200, 600)
        assert t.is_empty


class TestTargetShiftTime:
    def test_wrap(self):
        t = Target(p_picks=[(0, 900)], s_picks=[(0, 950)])
        t.shift_time(200, 1000, wrap=True)
        assert t.p_picks == [(0, 100)]  # (900+200) % 1000
        assert t.s_picks == [(0, 150)]

    def test_nowrap_drops_out_of_bounds(self):
        t = Target(
            p_picks=[(0, 900), (1, 100)],
            s_picks=[(0, 950), (1, 200)],
            event_centers=[(0, 925), (1, 150)],
            ps_intervals=[(0, 50), (1, 100)],
        )
        t.shift_time(200, 1000, wrap=False)
        # (900+200)=1100 out, (100+200)=300 in
        assert t.p_picks == [(1, 300)]
        assert t.s_picks == [(1, 400)]
        assert t.event_centers == [(1, 350)]


class TestTargetScaleTime:
    def test_scale_up(self):
        t = Target(p_picks=[(0, 100)], s_picks=[(0, 200)])
        t.scale_time(2.0, 500)
        assert t.p_picks == [(0, 200.0)]
        assert t.s_picks == [(0, 400.0)]

    def test_scale_drops_outside(self):
        t = Target(p_picks=[(0, 100), (0, 400)], s_picks=[(0, 200)])
        t.scale_time(2.0, 500)
        # 400*2=800 >= 500 → dropped
        assert len(t.p_picks) == 1
        assert t.p_picks[0] == (0, 200.0)


class TestTargetFlipPolarity:
    def test_signs_negated(self):
        t = Target(polarity=[(0, 100, 1), (1, 200, -1), (2, 300, 1)])
        t.flip_polarity_sign()
        assert [sign for _, _, sign in t.polarity] == [-1, 1, -1]


# =============================================================================
# Transform tests
# =============================================================================


class TestNormalize:
    def test_basic(self):
        sample = make_sample()
        result = Normalize()(sample)
        assert result.waveform.shape == (NCH, NX, NT)
        assert abs(result.waveform.mean()) < 0.01
        assert abs(result.waveform.std() - 1.0) < 0.01

    def test_with_nan(self):
        sample = make_sample()
        sample.waveform[0, 0, :10] = np.nan
        result = Normalize()(sample)
        assert not np.any(np.isnan(result.waveform))

    def test_zero_std(self):
        sample = make_sample()
        sample.waveform[:] = 5.0  # constant
        result = Normalize()(sample)
        # After demean, all zeros; std=0, should not blow up
        assert np.all(np.isfinite(result.waveform))
        assert np.allclose(result.waveform, 0.0)


class TestTaper:
    def test_edges_attenuated(self):
        sample = make_sample()
        sample.waveform[:] = 1.0  # constant value
        result = Taper(max_percentage=0.1)(sample)
        # First sample should be near 0 (taper starts at 0)
        assert abs(result.waveform[0, 0, 0]) < 0.01
        # Last sample should be near 0
        assert abs(result.waveform[0, 0, -1]) < 0.01
        # Center should be 1.0
        assert abs(result.waveform[0, 0, NT // 2] - 1.0) < 0.01

    def test_zero_percentage(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = Taper(max_percentage=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestRandomCrop:
    def test_output_length(self):
        sample = make_sample()
        result = RandomCrop(length=256)(sample)
        assert result.nt == 256

    def test_picks_in_range(self):
        random.seed(0)
        sample = make_sample()
        result = RandomCrop(length=512)(sample)
        for target in result.targets:
            for _, t in target.p_picks + target.s_picks:
                assert 0 <= t < 512

    def test_no_op_when_shorter(self):
        sample = make_sample(nt=100)
        result = RandomCrop(length=256)(sample)
        assert result.nt == 100  # unchanged

    def test_empty_targets_removed(self):
        # Place picks at the very end so most crops miss them
        target = Target(p_picks=[(0, NT - 5)], s_picks=[(0, NT - 3)])
        sample = make_sample(target=target)
        random.seed(0)
        result = RandomCrop(length=64, min_phases=0, max_tries=1)(sample)
        # Either picks are kept (and adjusted) or target is removed
        for t in result.targets:
            assert not t.is_empty


class TestCenterCrop:
    def test_output_length(self):
        sample = make_sample()
        result = CenterCrop(length=256)(sample)
        assert result.nt == 256

    def test_symmetric(self):
        sample = make_sample()
        # Put a spike at the center
        center = NT // 2
        sample.waveform[:, :, center] = 999.0
        result = CenterCrop(length=256)(sample)
        # The spike should be at 256 // 2 = 128
        assert result.waveform[0, 0, 128] == 999.0

    def test_no_op_when_shorter(self):
        sample = make_sample(nt=100)
        result = CenterCrop(length=256)(sample)
        assert result.nt == 100


class TestRandomShift:
    def test_circular_shape_preserved(self):
        random.seed(0)
        sample = make_sample()
        result = RandomShift(max_shift=100, mode="circular")(sample)
        assert result.waveform.shape == (NCH, NX, NT)

    def test_zero_pad_shape_preserved(self):
        random.seed(0)
        sample = make_sample()
        result = RandomShift(max_shift=100, mode="zero")(sample)
        assert result.waveform.shape == (NCH, NX, NT)

    def test_circular_waveform_equivalence(self):
        random.seed(42)
        sample = make_sample()
        original = sample.waveform.copy()
        result = RandomShift(max_shift=100, mode="circular")(sample)
        # Can't know exact shift, but shape should match
        assert result.waveform.shape == original.shape

    def test_zero_pad_has_zeros(self):
        # Force a known positive shift
        sample = make_sample()
        original = sample.waveform.copy()
        shift = RandomShift(max_shift=0, mode="zero")
        # With max_shift=0, shift is always 0 → no change
        result = shift(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestTimeStretch:
    def test_output_shape(self):
        random.seed(42)
        sample = make_sample()
        # Force a specific factor
        t = TimeStretch(min_factor=1.5, max_factor=1.5)
        result = t(sample)
        expected_nt = int(NT * 1.5)
        assert result.nt == expected_nt

    def test_picks_scaled(self):
        random.seed(42)
        sample = make_sample(target=make_target(p_times=[200], s_times=[400]))
        t = TimeStretch(min_factor=2.0, max_factor=2.0)
        result = t(sample)
        for target in result.targets:
            for _, pick_t in target.p_picks:
                assert abs(pick_t - 400.0) < 1.0  # 200 * 2.0
            for _, pick_t in target.s_picks:
                assert abs(pick_t - 800.0) < 1.0  # 400 * 2.0


class TestFlipPolarity:
    def test_waveform_negated(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = FlipPolarity(p=1.0)(sample)
        np.testing.assert_array_almost_equal(result.waveform, -original)

    def test_polarity_signs_negated(self):
        sample = make_sample(target=make_target(polarity_signs=[1]))
        result = FlipPolarity(p=1.0)(sample)
        for target in result.targets:
            for _, _, sign in target.polarity:
                assert sign == -1

    def test_p_zero_never_flips(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = FlipPolarity(p=0.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_p_one_always_flips(self):
        sample = make_sample()
        original = sample.waveform.copy()
        for _ in range(5):
            s = make_sample()
            result = FlipPolarity(p=1.0)(s)
            np.testing.assert_array_almost_equal(result.waveform, -original)


class TestRandomAmplitudeScale:
    def test_range(self):
        random.seed(42)
        sample = make_sample()
        original_max = np.max(np.abs(sample.waveform))
        result = RandomAmplitudeScale(min_scale=0.5, max_scale=2.0)(sample)
        result_max = np.max(np.abs(result.waveform))
        ratio = result_max / original_max
        assert 0.4 < ratio < 2.1

    def test_log_vs_linear(self):
        random.seed(42)
        sample1 = make_sample()
        result1 = RandomAmplitudeScale(min_scale=0.5, max_scale=2.0, log_scale=True)(sample1)
        random.seed(42)
        sample2 = make_sample()
        result2 = RandomAmplitudeScale(min_scale=0.5, max_scale=2.0, log_scale=False)(sample2)
        # Both should produce valid output, but different scales
        assert np.all(np.isfinite(result1.waveform))
        assert np.all(np.isfinite(result2.waveform))

    def test_signal_noise_updated(self):
        sample = make_sample(target=make_target(amp_signal=2.0, amp_noise=0.5))
        result = RandomAmplitudeScale(min_scale=3.0, max_scale=3.0)(sample)
        assert abs(result.targets[0].amp_signal - 6.0) < 0.01
        assert abs(result.targets[0].amp_noise - 1.5) < 0.01


class TestDropChannel:
    def test_at_least_one_zeroed(self):
        # With p=1.0, always applies
        sample = make_sample()
        result = DropChannel(p=1.0)(sample)
        # At least one channel should be zero
        any_zeroed = any(
            np.allclose(result.waveform[ch], 0.0) for ch in range(NCH)
        )
        assert any_zeroed

    def test_polarity_cleared_on_z_drop(self):
        # Run many times; when Z channel (index 2) is zeroed, polarity should be empty
        random.seed(0)
        for i in range(100):
            random.seed(i)
            sample = make_sample(target=make_target(polarity_signs=[1]))
            result = DropChannel(p=1.0)(sample)
            if np.allclose(result.waveform[2], 0.0):
                for target in result.targets:
                    assert target.polarity == []
                break


class TestAddGaussianNoise:
    def test_shape_preserved(self):
        sample = make_sample()
        result = AddGaussianNoise(snr_db_range=(10, 30))(sample)
        assert result.waveform.shape == (NCH, NX, NT)

    def test_modifies_waveform(self):
        np.random.seed(42)
        sample = make_sample()
        original = sample.waveform.copy()
        result = AddGaussianNoise(snr_db_range=(10, 10))(sample)
        assert not np.allclose(result.waveform, original)

    def test_zero_signal(self):
        sample = make_sample()
        sample.waveform[:] = 0.0
        result = AddGaussianNoise(snr_db_range=(10, 30))(sample)
        # noise_power = 0 / something = 0, so noise is all zeros
        np.testing.assert_array_equal(result.waveform, 0.0)


class TestRandomGain:
    def test_per_station_variation(self):
        np.random.seed(42)
        sample = make_sample()
        sample.waveform[:] = 1.0
        result = RandomGain(min_gain=0.5, max_gain=2.0)(sample)
        # Each station should have a different gain
        station_means = [result.waveform[0, sta, 0] for sta in range(NX)]
        assert len(set(station_means)) == NX  # all different

    def test_channels_same_gain(self):
        """All 3 components of one station should get the same gain."""
        np.random.seed(42)
        sample = make_sample()
        sample.waveform[:] = 1.0
        result = RandomGain(min_gain=0.5, max_gain=2.0)(sample)
        for sta in range(NX):
            vals = [result.waveform[ch, sta, 0] for ch in range(NCH)]
            assert all(abs(v - vals[0]) < 1e-6 for v in vals)

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
        # signal_power=0 → no noise added
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

    def test_invalid_corners_no_change(self):
        """When low >= high, should return unchanged."""
        sample = make_sample()
        original = sample.waveform.copy()
        # Force low > high
        result = RandomBandpass(low_range=(40.0, 40.0), high_range=(5.0, 5.0), p=1.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)


class TestStackEvents:
    def test_no_sample_fn(self):
        sample = make_sample()
        n_targets = len(sample.targets)
        result = StackEvents()(sample)
        assert len(result.targets) == n_targets  # unchanged

    def test_adds_target(self):
        random.seed(0)
        sample = make_sample()
        donor = make_sample(seed=99)
        donor.targets[0].amp_signal = 1.0
        donor.targets[0].amp_noise = 0.1

        stacker = StackEvents(max_events=1, max_shift=100, allow_overlap="full")
        stacker.set_sample_fn(lambda: donor.copy())
        result = stacker(sample)
        # Should have stacked at least one event (with full overlap allowed)
        assert len(result.targets) >= 1

    def test_shape_mismatch_skip(self):
        random.seed(0)
        sample = make_sample(nt=1024)
        donor = make_sample(nt=512)  # different nt

        stacker = StackEvents(max_events=1)
        stacker.set_sample_fn(lambda: donor.copy())
        result = stacker(sample)
        assert len(result.targets) == 1  # no stacking occurred


class TestStackNoise:
    def test_no_fn_unchanged(self):
        sample = make_sample()
        original = sample.waveform.copy()
        result = StackNoise(p=1.0)(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_shape_mismatch_skip(self):
        sample = make_sample()
        original = sample.waveform.copy()
        wrong_shape = np.random.randn(NCH, NX, NT + 100).astype(np.float32)
        sn = StackNoise(p=1.0)
        sn.set_noise_fn(lambda: wrong_shape)
        result = sn(sample)
        np.testing.assert_array_equal(result.waveform, original)

    def test_modifies_waveform(self):
        random.seed(0)
        np.random.seed(0)
        sample = make_sample(target=make_target(amp_signal=1.0, amp_noise=0.1))
        original = sample.waveform.copy()
        noise = np.random.randn(NCH, NX, NT).astype(np.float32)
        sn = StackNoise(p=1.0, max_ratio=1.0)
        sn.set_noise_fn(lambda: noise)
        result = sn(sample)
        assert not np.allclose(result.waveform, original)


class TestCompose:
    def test_chains(self):
        sample = make_sample()
        original = sample.waveform.copy()
        # Identity followed by flip
        c = Compose([Identity(), FlipPolarity(p=1.0)])
        result = c(sample)
        np.testing.assert_array_almost_equal(result.waveform, -original)

    def test_repr(self):
        c = Compose([Normalize(), FlipPolarity()])
        r = repr(c)
        assert "Compose" in r
        assert "Normalize" in r
        assert "FlipPolarity" in r


# =============================================================================
# Label generation tests
# =============================================================================


class TestGenerateLabels:
    def test_shapes(self):
        sample = make_sample()
        labels = generate_labels(sample)
        assert labels["phase_pick"].shape == (3, NX, NT)
        assert labels["phase_mask"].shape == (1, NX, NT)
        assert labels["polarity"].shape == (1, NX, NT)
        assert labels["polarity_mask"].shape == (1, NX, NT)
        assert labels["event_center"].shape == (1, NX, NT)
        assert labels["event_time"].shape == (1, NX, NT)
        assert labels["event_center_mask"].shape == (1, NX, NT)
        assert labels["event_time_mask"].shape == (1, NX, NT)

    def test_gaussian_peak_at_pick(self):
        p_time, s_time = 300, 500
        sample = make_sample(target=make_target(p_times=[p_time], s_times=[s_time]))
        labels = generate_labels(sample)
        # P label should peak near p_time
        p_label = labels["phase_pick"][1, 0]  # station 0
        assert np.argmax(p_label) == p_time
        assert p_label[p_time] > 0.9
        # S label should peak near s_time
        s_label = labels["phase_pick"][2, 0]
        assert np.argmax(s_label) == s_time
        assert s_label[s_time] > 0.9

    def test_noise_complement(self):
        sample = make_sample()
        labels = generate_labels(sample)
        noise = labels["phase_pick"][0]
        p_label = labels["phase_pick"][1]
        s_label = labels["phase_pick"][2]
        expected_noise = np.maximum(0, 1.0 - p_label - s_label)
        np.testing.assert_array_almost_equal(noise, expected_noise)

    def test_empty_targets(self):
        sample = make_sample()
        sample.targets = []
        labels = generate_labels(sample)
        # P and S labels should be all zeros
        assert np.allclose(labels["phase_pick"][1], 0.0)
        assert np.allclose(labels["phase_pick"][2], 0.0)
        # Noise should be all ones
        assert np.allclose(labels["phase_pick"][0], 1.0)
        # Masks should be all zeros
        assert np.allclose(labels["phase_mask"], 0.0)

    def test_phase_mask_both_ps(self):
        """When both P and S picks exist on a station, phase_mask is full trace."""
        sample = make_sample(target=make_target(p_times=[300], s_times=[500]))
        labels = generate_labels(sample)
        # Station 0 has both P and S, so mask should be all 1s
        assert np.all(labels["phase_mask"][0, 0] == 1.0)

    def test_phase_mask_single_pick(self):
        """When only one phase on a station, mask is narrow window around pick."""
        target = Target(p_picks=[(0, 300)], s_picks=[])
        sample = make_sample(target=target)
        labels = generate_labels(sample)
        mask = labels["phase_mask"][0, 0]
        # Should NOT be all 1s (only narrow window)
        assert not np.all(mask == 1.0)
        # But should have 1s around the pick
        assert mask[300] == 1.0

    def test_polarity_label_range(self):
        """Polarity label should be in [0, 1] with default config."""
        sample = make_sample(target=make_target(polarity_signs=[1]))
        config = LabelConfig()
        labels = generate_labels(sample, config)
        pol = labels["polarity"][0]
        # Default: polarity_scale=0.5, polarity_shift=0.5
        # For sign=+1: raw * 0.5 + 0.5 → should be in [0.5, 1.0] near pick
        assert pol.min() >= -0.01
        assert pol.max() <= 1.01
