# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "numpy",
#     "torch",
#     "gcsfs",
# ]
# ///
"""
CEED: California Earthquake Event Dataset

A modern, efficient dataset implementation for seismic phase picking following
best practices from computer vision (torchvision, timm, albumentations).

Design Principles:
1. COCO-style Target annotations: each event is a separate Target object,
   making multi-event stacking trivial and per-event metadata preserved
2. Unified (nch, nx, nt) internal format: (3, nx, nt) for multi-station seismic
3. Transform-based augmentation operating on raw picks before label generation
4. Compose pattern for chaining transforms
5. Clean separation: loading -> transforms -> label generation

Example:
    >>> from ceed import CEEDDataset, default_train_transforms
    >>> dataset = CEEDDataset(
    ...     region="SC", years=[2025], days=[1, 2, 3],
    ...     transforms=default_train_transforms(),
    ...     streaming=False,
    ... )
    >>> sample = dataset[0]
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# =============================================================================
# Configuration
# =============================================================================

BUCKET = "gs://quakeflow_dataset"
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
DEFAULT_SAMPLING_RATE = 100  # Hz
DEFAULT_WINDOW_LENGTH = 12288  # samples at 100Hz (~122.88s)


@dataclass
class LabelConfig:
    """Configuration for label generation."""
    phase_width: int = 60  # Gaussian width for phase picks (samples)
    polarity_width: int = 20  # Gaussian width for polarity
    polarity_shift: float = 0.5  # output = raw * scale + shift; default maps [-1,1] → [0,1]
    polarity_scale: float = 0.5
    event_width: int = 150  # Gaussian width for event center
    mask_width_factor: float = 1.5  # mask_width = phase_width * factor
    gaussian_threshold: float = 0.1  # Values below this are zeroed
    vp: float = 6.0  # P-wave velocity (km/s)
    vp_vs_ratio: float = 1.73  # Vp/Vs ratio


# =============================================================================
# Data Structures — COCO-style Target + Sample
# =============================================================================

@dataclass
class Target:
    """Per-event annotation, following COCO/torchvision detection conventions.

    Each Target represents ONE seismic event's picks on the waveform.
    A Sample can have multiple Targets (e.g. from event stacking).

    Picks are (station_idx, time_sample) tuples where station_idx is the
    index in the sorted station array (0 to nx-1).
    """
    p_picks: list[tuple[int, float]] = field(default_factory=list)
    s_picks: list[tuple[int, float]] = field(default_factory=list)
    polarity: list[tuple[int, float, int]] = field(default_factory=list)  # (sta, time, sign)
    event_centers: list[tuple[int, float]] = field(default_factory=list)  # (sta, center_time)
    ps_intervals: list[tuple[int, float]] = field(default_factory=list)  # (sta, S-P interval)

    # Per-event metadata
    snr: float = 0.0
    amp_signal: float = 0.0
    amp_noise: float = 0.0
    distance_km: float = 0.0
    event_id: str = ""

    def copy(self) -> "Target":
        return Target(
            p_picks=self.p_picks.copy(),
            s_picks=self.s_picks.copy(),
            polarity=self.polarity.copy(),
            event_centers=self.event_centers.copy(),
            ps_intervals=self.ps_intervals.copy(),
            snr=self.snr,
            amp_signal=self.amp_signal,
            amp_noise=self.amp_noise,
            distance_km=self.distance_km,
            event_id=self.event_id,
        )

    @property
    def is_empty(self) -> bool:
        return not self.p_picks and not self.s_picks

    def all_times(self) -> list[float]:
        """All pick times (P + S)."""
        return [t for _, t in self.p_picks] + [t for _, t in self.s_picks]

    # -- Pick adjustment methods (in-place, return self for chaining) --

    def crop_time(self, start: int, end: int) -> "Target":
        """Keep picks in [start, end), shift to new origin."""
        self.p_picks = [(s, t - start) for s, t in self.p_picks if start <= t < end]
        self.s_picks = [(s, t - start) for s, t in self.s_picks if start <= t < end]
        self.polarity = [(s, t - start, sign) for s, t, sign in self.polarity if start <= t < end]
        ec, ps = [], []
        for (s, t), (_, d) in zip(self.event_centers, self.ps_intervals):
            if start <= t < end:
                ec.append((s, t - start))
                ps.append((s, d))
        self.event_centers, self.ps_intervals = ec, ps
        return self

    def shift_time(self, shift: int, nt: int, wrap: bool = True) -> "Target":
        """Shift all time indices. Wraps modulo nt or clips to [0, nt)."""
        if wrap:
            self.p_picks = [(s, (t + shift) % nt) for s, t in self.p_picks]
            self.s_picks = [(s, (t + shift) % nt) for s, t in self.s_picks]
            self.polarity = [(s, (t + shift) % nt, sign) for s, t, sign in self.polarity]
            self.event_centers = [(s, (t + shift) % nt) for s, t in self.event_centers]
            # ps_intervals unchanged by circular shift
        else:
            self.p_picks = [(s, t + shift) for s, t in self.p_picks if 0 <= t + shift < nt]
            self.s_picks = [(s, t + shift) for s, t in self.s_picks if 0 <= t + shift < nt]
            self.polarity = [(s, t + shift, sign) for s, t, sign in self.polarity if 0 <= t + shift < nt]
            ec, ps = [], []
            for (s, t), (_, d) in zip(self.event_centers, self.ps_intervals):
                if 0 <= t + shift < nt:
                    ec.append((s, t + shift))
                    ps.append((s, d))
            self.event_centers, self.ps_intervals = ec, ps
        return self

    def scale_time(self, factor: float, nt_new: int) -> "Target":
        """Scale time indices by factor, drop picks outside [0, nt_new)."""
        self.p_picks = [(s, t * factor) for s, t in self.p_picks if t * factor < nt_new]
        self.s_picks = [(s, t * factor) for s, t in self.s_picks if t * factor < nt_new]
        self.polarity = [(s, t * factor, sign) for s, t, sign in self.polarity if t * factor < nt_new]
        ec, ps = [], []
        for (s, t), (_, d) in zip(self.event_centers, self.ps_intervals):
            if t * factor < nt_new:
                ec.append((s, t * factor))
                ps.append((s, d * factor))
        self.event_centers, self.ps_intervals = ec, ps
        return self

    def flip_polarity_sign(self) -> "Target":
        """Negate polarity signs (for waveform polarity flip)."""
        self.polarity = [(s, t, -sign) for s, t, sign in self.polarity]
        return self


@dataclass
class Sample:
    """A seismic sample with waveform and per-event annotations.

    Waveform shape: (nch, nx, nt) = (3, num_stations, num_time_samples).
    Stations are sorted by distance for spatial coherence.

    targets: list of Target, one per seismic event in the window.
    """
    waveform: np.ndarray  # (nch, nx, nt) — always 3D
    targets: list[Target] = field(default_factory=list)

    # Metadata
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    trace_ids: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)

    @property
    def nch(self) -> int:
        return self.waveform.shape[0]

    @property
    def nx(self) -> int:
        return self.waveform.shape[1]

    @property
    def nt(self) -> int:
        return self.waveform.shape[2]

    @property
    def amp_signal(self) -> float:
        """Weakest event signal (conservative for stacking ratio)."""
        vals = [t.amp_signal for t in self.targets if t.amp_signal > 0]
        return min(vals) if vals else 0.0

    @property
    def amp_noise(self) -> float:
        """Worst-case noise level across events."""
        vals = [t.amp_noise for t in self.targets if t.amp_noise > 0]
        return max(vals) if vals else 0.0

    def copy(self) -> "Sample":
        return Sample(
            waveform=self.waveform.copy(),
            targets=[t.copy() for t in self.targets],
            sampling_rate=self.sampling_rate,
            trace_ids=self.trace_ids.copy(),
            sensors=self.sensors.copy(),
        )


# =============================================================================
# Transforms — Following CV Best Practices (torchvision/albumentations pattern)
# =============================================================================

class Transform(ABC):
    """Base class for all transforms. Operates on Sample objects."""

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """Compose multiple transforms together."""

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


class Identity(Transform):
    def __call__(self, sample: Sample) -> Sample:
        return sample


# -----------------------------------------------------------------------------
# Basic Waveform Transforms
# -----------------------------------------------------------------------------

class Normalize(Transform):
    """Normalize waveform to zero mean and unit std."""

    def __init__(self, eps: float = 1e-10):
        self.eps = eps

    def __call__(self, sample: Sample) -> Sample:
        data = np.nan_to_num(sample.waveform)
        data = data - data.mean(axis=-1, keepdims=True)
        std = data.std()
        if std > self.eps:
            data = data / std
        sample.waveform = data
        return sample

    def __repr__(self) -> str:
        return f"Normalize(eps={self.eps})"


class HighpassFilter(Transform):
    """Apply highpass filter to waveform."""

    def __init__(self, freq: float = 1.0, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        from scipy import signal
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.sos = signal.butter(4, freq, btype='high', fs=sampling_rate, output='sos')

    def __call__(self, sample: Sample) -> Sample:
        from scipy import signal
        sample.waveform = signal.sosfilt(self.sos, sample.waveform, axis=-1).astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"HighpassFilter(freq={self.freq})"


class Taper(Transform):
    """Apply cosine taper to waveform edges."""

    def __init__(self, max_percentage: float = 0.05):
        self.max_percentage = max_percentage

    def __call__(self, sample: Sample) -> Sample:
        nt = sample.nt
        taper_len = int(nt * self.max_percentage)
        if taper_len > 0:
            taper = np.ones(nt, dtype=np.float32)
            taper[:taper_len] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len) / taper_len))
            taper[-taper_len:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_len, 0, -1) / taper_len))
            sample.waveform = sample.waveform * taper
        return sample

    def __repr__(self) -> str:
        return f"Taper(max_percentage={self.max_percentage})"


# -----------------------------------------------------------------------------
# Temporal Transforms (modify both waveform and picks via Target methods)
# -----------------------------------------------------------------------------

class RandomCrop(Transform):
    """Randomly crop waveform to fixed length, adjusting picks.

    Args:
        length: Target length in samples
        min_phases: Minimum number of phase picks required in crop
        max_tries: Maximum attempts to find valid crop
    """

    def __init__(self, length: int = 4096, min_phases: int = 1, max_tries: int = 100):
        self.length = length
        self.min_phases = min_phases
        self.max_tries = max_tries

    def __call__(self, sample: Sample) -> Sample:
        nt = sample.nt
        if nt <= self.length:
            return sample

        all_times = [t for tgt in sample.targets for t in tgt.all_times()]

        start = 0
        for _ in range(self.max_tries):
            start = random.randint(0, nt - self.length)
            end = start + self.length
            if sum(1 for t in all_times if start <= t < end) >= self.min_phases:
                break

        end = start + self.length
        sample.waveform = sample.waveform[:, :, start:end]
        for target in sample.targets:
            target.crop_time(start, end)
        sample.targets = [t for t in sample.targets if not t.is_empty]
        return sample

    def __repr__(self) -> str:
        return f"RandomCrop(length={self.length})"


class CenterCrop(Transform):
    """Crop waveform from center to fixed length."""

    def __init__(self, length: int = 4096):
        self.length = length

    def __call__(self, sample: Sample) -> Sample:
        nt = sample.nt
        if nt <= self.length:
            return sample

        start = (nt - self.length) // 2
        end = start + self.length
        sample.waveform = sample.waveform[:, :, start:end]
        for target in sample.targets:
            target.crop_time(start, end)
        sample.targets = [t for t in sample.targets if not t.is_empty]
        return sample

    def __repr__(self) -> str:
        return f"CenterCrop(length={self.length})"


class RandomShift(Transform):
    """Randomly shift waveform and picks (circular or zero-pad).

    Args:
        max_shift: Maximum shift in samples
        mode: "circular" or "zero" padding
    """

    def __init__(self, max_shift: int = 1024, mode: str = "circular"):
        self.max_shift = max_shift
        self.mode = mode

    def __call__(self, sample: Sample) -> Sample:
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return sample

        nt = sample.nt
        wrap = self.mode == "circular"

        if wrap:
            sample.waveform = np.roll(sample.waveform, shift, axis=-1)
        elif shift > 0:
            sample.waveform = np.concatenate([
                np.zeros_like(sample.waveform[:, :, :shift]),
                sample.waveform[:, :, :-shift]
            ], axis=-1)
        else:
            sample.waveform = np.concatenate([
                sample.waveform[:, :, -shift:],
                np.zeros_like(sample.waveform[:, :, :(-shift)])
            ], axis=-1)

        for target in sample.targets:
            target.shift_time(shift, nt, wrap=wrap)
        if not wrap:
            sample.targets = [t for t in sample.targets if not t.is_empty]
        return sample

    def __repr__(self) -> str:
        return f"RandomShift(max_shift={self.max_shift}, mode='{self.mode}')"


class TimeStretch(Transform):
    """Randomly stretch/compress time axis. Phase indices are scaled accordingly.

    Args:
        min_factor: Minimum stretch factor
        max_factor: Maximum stretch factor
    """

    def __init__(self, min_factor: float = 0.9, max_factor: float = 1.1):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: Sample) -> Sample:
        from scipy import ndimage

        factor = random.uniform(self.min_factor, self.max_factor)
        if abs(factor - 1.0) < 1e-6:
            return sample

        nt_new = int(sample.nt * factor)
        zoom_factors = [1.0, 1.0, factor]  # (nch, nx, nt)
        sample.waveform = ndimage.zoom(sample.waveform, zoom_factors, order=1).astype(np.float32)

        for target in sample.targets:
            target.scale_time(factor, nt_new)
        sample.targets = [t for t in sample.targets if not t.is_empty]
        return sample

    def __repr__(self) -> str:
        return f"TimeStretch(min_factor={self.min_factor}, max_factor={self.max_factor})"


# -----------------------------------------------------------------------------
# Amplitude/Polarity Transforms
# -----------------------------------------------------------------------------

class FlipPolarity(Transform):
    """Randomly flip waveform polarity (multiply by -1). Also negates polarity signs."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.waveform = -sample.waveform
            for target in sample.targets:
                target.flip_polarity_sign()
        return sample

    def __repr__(self) -> str:
        return f"FlipPolarity(p={self.p})"


class RandomAmplitudeScale(Transform):
    """Randomly scale waveform amplitude.

    Args:
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        log_scale: If True, sample from log-uniform distribution
    """

    def __init__(self, min_scale: float = 0.5, max_scale: float = 2.0, log_scale: bool = True):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.log_scale = log_scale

    def __call__(self, sample: Sample) -> Sample:
        if self.log_scale:
            scale = 10 ** random.uniform(np.log10(self.min_scale), np.log10(self.max_scale))
        else:
            scale = random.uniform(self.min_scale, self.max_scale)
        sample.waveform = sample.waveform * scale
        for target in sample.targets:
            target.amp_signal *= scale
            target.amp_noise *= scale
        return sample

    def __repr__(self) -> str:
        return f"RandomAmplitudeScale(min_scale={self.min_scale}, max_scale={self.max_scale})"


class DropChannel(Transform):
    """Randomly drop (zero out) channels.

    Args:
        p: Probability of applying the transform
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        drop_z = False
        r = random.random()
        if r < 0.15:
            sample.waveform[0] = 0
        elif r < 0.30:
            sample.waveform[1] = 0
        elif r < 0.45:
            sample.waveform[:2] = 0
        elif r < 0.60:
            sample.waveform[2] = 0
            drop_z = True
        elif r < 0.75:
            sample.waveform[0] = 0
            sample.waveform[2] = 0
            drop_z = True
        elif r < 0.90:
            sample.waveform[1] = 0
            sample.waveform[2] = 0
            drop_z = True
        else:
            ch = random.randint(0, 2)
            sample.waveform[ch] = 0
            if ch == 2:
                drop_z = True

        if drop_z:
            for target in sample.targets:
                target.polarity = []

        return sample

    def __repr__(self) -> str:
        return f"DropChannel(p={self.p})"


# -----------------------------------------------------------------------------
# Noise Augmentation
# -----------------------------------------------------------------------------

class AddGaussianNoise(Transform):
    """Add Gaussian noise to waveform.

    Args:
        snr_db_range: Range of SNR in dB (min, max)
    """

    def __init__(self, snr_db_range: tuple[float, float] = (10, 30)):
        self.snr_db_range = snr_db_range

    def __call__(self, sample: Sample) -> Sample:
        snr_db = random.uniform(*self.snr_db_range)
        signal_power = np.mean(sample.waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*sample.waveform.shape).astype(np.float32) * np.sqrt(noise_power)
        sample.waveform = sample.waveform + noise
        return sample

    def __repr__(self) -> str:
        return f"AddGaussianNoise(snr_db_range={self.snr_db_range})"


# -----------------------------------------------------------------------------
# Stacking Transforms — Core Seismic Augmentation
# -----------------------------------------------------------------------------

DEFAULT_VP = 6.0
DEFAULT_VS = 3.5


def predict_ps_times(
    distance_km: float,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    vp: float = DEFAULT_VP,
    vs: float = DEFAULT_VS,
) -> tuple[float, float]:
    """Predict P and S arrival times (in samples) from distance."""
    return distance_km / vp * sampling_rate, distance_km / vs * sampling_rate


class StackEvents(Transform):
    """Stack multiple events onto a single waveform (Mixup-inspired).

    Each stacked event becomes a separate Target in sample.targets.

    Args:
        max_events: Maximum number of additional events to stack
        max_shift: Maximum shift in samples for alignment
        min_ratio: Minimum amplitude ratio (log10 scale)
        max_ratio: Maximum amplitude ratio (log10 scale)
        max_tries: Maximum attempts to find valid position
        allow_overlap: Overlap policy ("none", "partial", "full")
        vp: P-wave velocity for window prediction (km/s)
        vs: S-wave velocity for window prediction (km/s)
    """

    def __init__(
        self,
        max_events: int = 2,
        max_shift: int = 4096,
        min_ratio: float = -2.0,
        max_ratio: float = 2.0,
        max_tries: int = 10,
        allow_overlap: str = "none",
        vp: float = DEFAULT_VP,
        vs: float = DEFAULT_VS,
    ):
        self.max_events = max_events
        self.max_shift = max_shift
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_tries = max_tries
        self.allow_overlap = allow_overlap
        self.vp = vp
        self.vs = vs
        self._sample_fn: Callable[[], Sample | None] | None = None

    def set_sample_fn(self, fn: Callable[[], Sample | None]):
        self._sample_fn = fn

    def _get_target_windows(self, target: Target, sampling_rate: float) -> list[tuple[int, int]]:
        """Get event windows (P to S) for a target."""
        if target.distance_km > 0:
            p_travel, s_travel = predict_ps_times(target.distance_km, sampling_rate, self.vp, self.vs)
            ps_diff = s_travel - p_travel
        else:
            ps_diff = 300

        windows = []
        for _, p_time in target.p_picks:
            windows.append((int(p_time), int(p_time + ps_diff)))
        if len(target.s_picks) > len(target.p_picks):
            for _, s_time in target.s_picks[len(target.p_picks):]:
                windows.append((int(s_time - ps_diff), int(s_time)))
        return windows

    def _check_overlap(
        self,
        windows_existing: list[tuple[int, int]],
        windows_new: list[tuple[int, int]],
        picks_existing: list[float],
        picks_new: list[float],
    ) -> bool:
        """Check if stacking is valid based on overlap policy."""
        if self.allow_overlap == "full":
            return True

        # New picks must not fall in existing windows
        if any(w0 <= t <= w1 for t in picks_new for w0, w1 in windows_existing):
            return False

        if self.allow_overlap == "none":
            # Existing picks must not fall in new windows
            if any(w0 <= t <= w1 for t in picks_existing for w0, w1 in windows_new):
                return False
        return True

    def __call__(self, sample: Sample) -> Sample:
        if self._sample_fn is None:
            return sample

        n_events = random.randint(1, self.max_events)

        for _ in range(n_events):
            sample2 = self._sample_fn()
            if sample2 is None or not sample2.targets:
                continue
            if sample2.nt != sample.nt:
                continue

            target2 = sample2.targets[0]
            if target2.amp_signal == 0 or target2.amp_noise == 0:
                continue
            if sample.amp_signal == 0 or sample.amp_noise == 0:
                continue

            # Existing picks/windows
            all_times_existing = [t for tgt in sample.targets for t in tgt.all_times()]
            windows_existing = [
                w for tgt in sample.targets
                for w in self._get_target_windows(tgt, sample.sampling_rate)
            ]

            # Donor first arrival for alignment
            times2 = target2.all_times()
            first1 = min(all_times_existing) if all_times_existing else sample.nt // 2
            first2 = min(times2) if times2 else sample2.nt // 2

            for _ in range(self.max_tries):
                shift = random.randint(-self.max_shift, self.max_shift) + int(first1 - first2)
                nt = sample.nt

                # Shifted donor picks and windows
                p2_shifted = [(t + shift) % nt for _, t in target2.p_picks]
                s2_shifted = [(t + shift) % nt for _, t in target2.s_picks]
                windows2 = self._get_target_windows(target2, sample.sampling_rate)
                windows2_shifted = [((w0 + shift) % nt, (w1 + shift) % nt) for w0, w1 in windows2]

                if not self._check_overlap(
                    windows_existing, windows2_shifted,
                    all_times_existing, p2_shifted + s2_shifted,
                ):
                    continue

                # Calculate amplitude ratio
                min_r = max(self.min_ratio, np.log10(sample.amp_noise * 2 / target2.amp_signal + 1e-10))
                max_r = min(self.max_ratio, np.log10(sample.amp_signal / 2 / target2.amp_noise + 1e-10))
                if min_r > max_r:
                    continue

                ratio = 10 ** random.uniform(min_r, max_r)
                flip = random.choice([-1.0, 1.0])

                # Stack waveforms — pad smaller nx to match larger
                donor = np.roll(sample2.waveform, shift, axis=-1) * ratio * flip
                nx1, nx2 = sample.nx, sample2.nx
                if nx1 < nx2:
                    pad = np.zeros((sample.nch, nx2 - nx1, sample.nt), dtype=sample.waveform.dtype)
                    sample.waveform = np.concatenate([sample.waveform, pad], axis=1)
                elif nx2 < nx1:
                    pad = np.zeros((sample2.nch, nx1 - nx2, sample2.nt), dtype=donor.dtype)
                    donor = np.concatenate([donor, pad], axis=1)
                sample.waveform = sample.waveform + donor

                # Create shifted target and append
                sign_flip = 1 if flip > 0 else -1
                new_target = Target(
                    p_picks=[(s, (t + shift) % nt) for s, t in target2.p_picks],
                    s_picks=[(s, (t + shift) % nt) for s, t in target2.s_picks],
                    polarity=[(s, (t + shift) % nt, sign * sign_flip) for s, t, sign in target2.polarity],
                    event_centers=[(s, (t + shift) % nt) for s, t in target2.event_centers],
                    ps_intervals=target2.ps_intervals.copy(),
                    snr=target2.snr,
                    amp_signal=target2.amp_signal * ratio,
                    amp_noise=target2.amp_noise * ratio,
                    distance_km=target2.distance_km,
                    event_id=target2.event_id,
                )
                sample.targets.append(new_target)
                break

        return sample

    def __repr__(self) -> str:
        return f"StackEvents(max_events={self.max_events}, max_shift={self.max_shift})"


class StackNoise(Transform):
    """Stack noise onto the waveform.

    Supports two noise sources (tried in order):
    1. Dedicated noise files via set_noise_fn
    2. Pre-arrival extraction from other samples via set_sample_fn

    Args:
        max_ratio: Maximum noise amplitude ratio relative to signal
        p: Probability of applying
    """

    def __init__(self, max_ratio: float = 2.0, p: float = 0.5):
        self.max_ratio = max_ratio
        self.p = p
        self._noise_fn: Callable[[], np.ndarray | None] | None = None
        self._sample_fn: Callable[[], Sample | None] | None = None

    def set_noise_fn(self, fn: Callable[[], np.ndarray | None]):
        """Set function to get random noise arrays (e.g., from dedicated noise files)."""
        self._noise_fn = fn

    def set_sample_fn(self, fn: Callable[[], Sample | None]):
        """Set function to get random samples (for pre-arrival noise extraction)."""
        self._sample_fn = fn

    def _extract_head_noise(self, donor: Sample, length: int) -> np.ndarray | None:
        """Extract noise from the beginning of the donor waveform, before first pick."""
        nt = donor.waveform.shape[-1]
        if nt < length:
            return None
        all_times = [t for tgt in donor.targets for t in tgt.all_times()]
        first_pick = int(min(all_times)) if all_times else nt
        if first_pick < length + 100:  # margin for label Gaussian tail
            return None
        return donor.waveform[..., :length]

    def _get_noise(self, sample: Sample) -> np.ndarray | None:
        """Get noise: dedicated files first, then head extraction."""
        if self._noise_fn is not None:
            noise = self._noise_fn()
            if noise is not None:
                return noise
        if self._sample_fn is not None:
            donor = self._sample_fn()
            if donor is not None:
                return self._extract_head_noise(donor, sample.nt)
        return None

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p or sample.amp_signal == 0:
            return sample
        noise = self._get_noise(sample)
        if noise is None or noise.shape != sample.waveform.shape:
            return sample
        noise_std = np.std(noise)
        if noise_std <= 0:
            return sample
        ratio = random.uniform(0, self.max_ratio) * sample.amp_signal
        sample.waveform = sample.waveform + noise / noise_std * ratio
        return sample

    def __repr__(self) -> str:
        return f"StackNoise(max_ratio={self.max_ratio}, p={self.p})"


# =============================================================================
# Label Generation
# =============================================================================

def generate_gaussian_1d(
    time_idx: float,
    length: int,
    width: int,
    threshold: float = 0.1,
) -> np.ndarray:
    """Generate a single Gaussian peak centered at time_idx."""
    sigma = width / 6
    t = np.arange(length)
    gaussian = np.exp(-((t - time_idx) ** 2) / (2 * sigma ** 2))
    gaussian[gaussian < threshold] = 0.0
    return gaussian


def generate_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
) -> dict[str, np.ndarray]:
    """Generate all labels from Sample.targets.

    Iterates over all targets (events) in the sample, accumulating
    Gaussian labels. Naturally handles multi-event windows from stacking.

    Returns:
        All arrays have shape (label_ch, nx, nt) matching the unified format.
        - phase_pick: (3, nx, nt) — [noise, P, S]
        - phase_mask: (1, nx, nt)
        - polarity: (1, nx, nt)
        - polarity_mask: (1, nx, nt)
        - event_center: (1, nx, nt)
        - event_time: (1, nx, nt)
        - event_center_mask: (1, nx, nt)
        - event_time_mask: (1, nx, nt)
    """
    nx, nt = sample.nx, sample.nt
    sigma_phase = config.phase_width / 6
    sigma_pol = config.polarity_width / 6
    sigma_event = config.event_width / 6
    mask_w = int(config.phase_width * config.mask_width_factor)
    event_mask_w = int(config.event_width * config.mask_width_factor)
    t = np.arange(nt)

    p_label = np.zeros((nx, nt), dtype=np.float32)
    s_label = np.zeros((nx, nt), dtype=np.float32)
    polarity_raw = np.zeros((nx, nt), dtype=np.float32)
    polarity_mask = np.zeros((nx, nt), dtype=np.float32)
    event_center_label = np.zeros((nx, nt), dtype=np.float32)
    event_time_label = np.zeros((nx, nt), dtype=np.float32)
    event_center_mask = np.zeros((nx, nt), dtype=np.float32)
    event_time_mask = np.zeros((nx, nt), dtype=np.float32)
    phase_mask = np.zeros((nx, nt), dtype=np.float32)

    has_p = np.zeros(nx, dtype=bool)
    has_s = np.zeros(nx, dtype=bool)

    vp = config.vp
    vs = vp / config.vp_vs_ratio
    dt_s = 1.0 / sample.sampling_rate

    picks_per_station: dict[int, list[float]] = {}

    def add_phase_picks(picks, label, has_flag):
        for sta, ti in picks:
            sta = int(sta)
            if 0 <= sta < nx:
                g = np.exp(-((t - ti) ** 2) / (2 * sigma_phase ** 2))
                g[g < config.gaussian_threshold] = 0.0
                label[sta] += g
                has_flag[sta] = True
                picks_per_station.setdefault(sta, []).append(ti)

    for target in sample.targets:
        add_phase_picks(target.p_picks, p_label, has_p)
        add_phase_picks(target.s_picks, s_label, has_s)

        # Polarity labels
        for sta, ti, sign in target.polarity:
            sta = int(sta)
            if 0 <= sta < nx:
                g = np.exp(-((t - ti) ** 2) / (2 * sigma_pol ** 2))
                g[g < config.gaussian_threshold] = 0.0
                polarity_raw[sta] += sign * g
                t0 = max(0, int(ti) - mask_w)
                t1 = min(nt, int(ti) + mask_w)
                polarity_mask[sta, t0:t1] = 1.0

        # Event labels
        for (sta, center), (_, ps_int) in zip(target.event_centers, target.ps_intervals):
            sta = int(sta)
            if 0 <= sta < nx:
                g = np.exp(-((t - center) ** 2) / (2 * sigma_event ** 2))
                g[g < 0.05] = 0.0
                event_center_label[sta] += g

                ps_seconds = ps_int * dt_s
                distance = ps_seconds * vp * vs / (vp - vs)
                center_travel = distance * (1 / vp + 1 / vs) / 2
                shift = center_travel / dt_s

                event_center_mask[sta, :] = 1.0
                t0 = max(0, int(center) - event_mask_w)
                t1 = min(nt, int(center) + event_mask_w)
                event_time_mask[sta, t0:t1] = 1.0
                event_time_label[sta, t0:t1] = (t[t0:t1] - center) + shift

    # Phase mask: full trace if both P and S present, narrow window otherwise
    for sta in picks_per_station:
        if has_p[sta] and has_s[sta]:
            phase_mask[sta, :] = 1.0
        else:
            for ti in picks_per_station[sta]:
                t0 = max(0, int(ti) - mask_w)
                t1 = min(nt, int(ti) + mask_w)
                phase_mask[sta, t0:t1] = 1.0

    noise_label = np.maximum(0, 1.0 - p_label - s_label)
    polarity_label = polarity_raw * config.polarity_scale + config.polarity_shift

    # All outputs: (label_ch, nx, nt)
    return {
        "phase_pick": np.stack([noise_label, p_label, s_label], axis=0),  # (3, nx, nt)
        "phase_mask": phase_mask[np.newaxis],  # (1, nx, nt)
        "polarity": polarity_label[np.newaxis],  # (1, nx, nt)
        "polarity_mask": polarity_mask[np.newaxis],  # (1, nx, nt)
        "event_center": event_center_label[np.newaxis],  # (1, nx, nt)
        "event_time": event_time_label[np.newaxis],  # (1, nx, nt)
        "event_center_mask": event_center_mask[np.newaxis],  # (1, nx, nt)
        "event_time_mask": event_time_mask[np.newaxis],  # (1, nx, nt)
    }


# =============================================================================
# Default Transform Presets
# =============================================================================

def default_train_transforms(
    crop_length: int = 4096,
    enable_stacking: bool = True,
    enable_noise_stacking: bool = True,
) -> Compose:
    """Default transforms for training."""
    transforms = [Normalize()]

    if enable_stacking:
        transforms.append(StackEvents(max_events=2, max_shift=4096))

    transforms.append(RandomCrop(crop_length))

    if enable_noise_stacking:
        transforms.append(StackNoise(max_ratio=2.0))

    transforms.extend([
        FlipPolarity(p=0.5),
        RandomAmplitudeScale(min_scale=0.5, max_scale=2.0),
        DropChannel(p=0.1),
        Normalize(),
    ])
    return Compose(transforms)


def default_eval_transforms(crop_length: int = 4096) -> Compose:
    """Default transforms for evaluation."""
    return Compose([
        Normalize(),
        CenterCrop(crop_length),
        Normalize(),
    ])


def minimal_transforms() -> Compose:
    """Minimal transforms — just normalize."""
    return Compose([Normalize()])


# =============================================================================
# Data Loading
# =============================================================================

def get_gcs_storage_options() -> dict:
    """Load GCS credentials for authenticated access."""
    if os.path.exists(GCS_CREDENTIALS_PATH):
        with open(GCS_CREDENTIALS_PATH, "r") as f:
            token = json.load(f)
        return {"token": token}
    return {}


def load_quakeflow_dataset(
    region: str = "SC",
    years: list[int] | None = None,
    days: list[int] | None = None,
    streaming: bool = False,
):
    """Load QuakeFlow dataset from GCS.

    Args:
        region: "SC" or "NC". Comma-separated for multiple regions.
        years: List of years to load. None = all available.
        days: List of days to load. None = all days.
        streaming: If True, stream without downloading.

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_dataset

    storage_options = get_gcs_storage_options()
    regions = [r.strip() for r in region.split(",")]

    patterns = []
    for reg in regions:
        if years is None:
            patterns.append(f"{BUCKET}/{reg}EDC/dataset/**/*.parquet")
        elif days is None:
            patterns.extend(f"{BUCKET}/{reg}EDC/dataset/{year}/*.parquet" for year in years)
        else:
            patterns.extend(
                f"{BUCKET}/{reg}EDC/dataset/{year}/{day:03d}.parquet"
                for year in years for day in days
            )

    pattern = patterns if len(patterns) > 1 else patterns[0]
    dataset = load_dataset(
        "parquet", data_files=pattern, streaming=streaming, storage_options=storage_options,
    )
    return dataset["train"]


def calc_snr(
    waveform: np.ndarray,
    p_picks: list[tuple[int, float]],
    noise_window: int = 300,
    signal_window: int = 300,
    gap_window: int = 50,
) -> tuple[float, float, float]:
    """Calculate SNR from waveform and P picks.

    Args:
        waveform: (nch, nx, nt)
        p_picks: [(station_idx, time_idx), ...]
    """
    nch, nx, nt = waveform.shape
    snrs, signals, noises = [], [], []
    for sta, t in p_picks:
        sta, t = int(sta), int(t)
        if 0 <= sta < nx and gap_window < t < nt - gap_window:
            for c in range(nch):
                noise = np.std(waveform[c, sta, max(0, t - noise_window):t - gap_window])
                signal = np.std(waveform[c, sta, t + gap_window:t + signal_window])
                if noise > 0 and signal > 0:
                    snrs.append(signal / noise)
                    signals.append(signal)
                    noises.append(noise)

    if not snrs:
        return 0.0, 0.0, 0.0
    idx = np.argmax(snrs)
    return snrs[idx], signals[idx], noises[idx]


def records_to_sample(records: list[dict]) -> Sample:
    """Convert HuggingFace records for one event into a multi-station Sample.

    All records must belong to the same event. Stations are sorted by distance
    to give spatial coherence along the nx axis.

    Args:
        records: List of HF records (one per station, same event)

    Returns:
        Sample with waveform shape (3, nx, nt) where nx = len(records)
    """
    # Sort by distance for spatial coherence
    records = sorted(records, key=lambda r: r.get("distance_km", 0.0) or 0.0)

    waveforms = []
    p_picks, s_picks, polarity_picks = [], [], []
    event_centers, ps_intervals = [], []
    trace_ids, sensors = [], []
    event_id = records[0].get("event_id", "")

    for sta_idx, record in enumerate(records):
        w = np.array(record["waveform"], dtype=np.float32)
        if w.ndim == 1:
            w = w.reshape(3, -1)
        waveforms.append(w)  # (3, nt)

        p_idx = record.get("p_phase_index")
        s_idx = record.get("s_phase_index")

        if p_idx is not None:
            p_picks.append((sta_idx, int(p_idx)))
        if s_idx is not None:
            s_picks.append((sta_idx, int(s_idx)))

        pol = record.get("p_phase_polarity")
        if pol == "U" and p_idx is not None:
            polarity_picks.append((sta_idx, int(p_idx), 1))
        elif pol == "D" and p_idx is not None:
            polarity_picks.append((sta_idx, int(p_idx), -1))

        if p_idx is not None and s_idx is not None:
            center = (int(p_idx) + int(s_idx)) / 2
            ps_int = int(s_idx) - int(p_idx)
            event_centers.append((sta_idx, center))
            ps_intervals.append((sta_idx, ps_int))

        net = record.get("network", "")
        sta = record.get("station", "")
        trace_ids.append(f"{event_id}/{net}.{sta}")
        sensors.append(record.get("instrument", ""))

    waveform = np.stack(waveforms, axis=1)  # (3, nx, nt)

    distance_km = records[0].get("distance_km", 0.0) or 0.0
    target = Target(
        p_picks=p_picks,
        s_picks=s_picks,
        polarity=polarity_picks,
        event_centers=event_centers,
        ps_intervals=ps_intervals,
        distance_km=distance_km,
        event_id=event_id,
    )

    return Sample(
        waveform=waveform,
        targets=[target] if not target.is_empty else [],
        sampling_rate=DEFAULT_SAMPLING_RATE,
        trace_ids=trace_ids,
        sensors=sensors,
    )


def record_to_sample(record: dict) -> Sample:
    """Convert a single HF record to a Sample. Thin wrapper around records_to_sample."""
    return records_to_sample([record])


# =============================================================================
# Dataset Classes
# =============================================================================

class SampleBuffer:
    """Efficient buffer for random sample access during stacking.
    Uses reservoir sampling for streaming datasets.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: list[Sample] = []
        self.count = 0

    def add(self, sample: Sample):
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample.copy())
        else:
            idx = random.randint(0, self.count)
            if idx < self.max_size:
                self.buffer[idx] = sample.copy()
        self.count += 1

    def get_random(self) -> Sample | None:
        if not self.buffer:
            return None
        return random.choice(self.buffer).copy()

    def __len__(self) -> int:
        return len(self.buffer)


def _pad_crop_nx(tensors: dict[str, torch.Tensor], target_nx: int, training: bool = True) -> dict[str, torch.Tensor]:
    """Pad or crop all tensors along the nx (axis=1) dimension to target_nx."""
    result = {}
    for key, val in tensors.items():
        if not isinstance(val, torch.Tensor):
            result[key] = val
            continue
        if val.ndim < 2:
            result[key] = val
            continue
        nx = val.shape[-2]
        if nx == target_nx:
            result[key] = val
        elif nx < target_nx:
            pad_size = target_nx - nx
            # Pad along nx dimension (second-to-last): F.pad takes (last_dim_right, last_dim_left, ...)
            result[key] = torch.nn.functional.pad(val, (0, 0, 0, pad_size))
        else:
            # Crop
            if training:
                start = random.randint(0, nx - target_nx)
            else:
                start = (nx - target_nx) // 2
            result[key] = val[..., start:start + target_nx, :]
    return result


def _to_output(
    sample: Sample,
    label_config: LabelConfig,
    event_feature_scale: int,
    target_nx: int | None = None,
    training: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert transformed Sample to output dict with labels."""
    labels = generate_labels(sample, label_config)
    s = event_feature_scale
    out = {
        "data": torch.nan_to_num(torch.from_numpy(sample.waveform).float()),
        "phase_pick": torch.from_numpy(labels["phase_pick"]).float(),
        "phase_mask": torch.from_numpy(labels["phase_mask"]).float(),
        "polarity": torch.from_numpy(labels["polarity"]).float(),
        "polarity_mask": torch.from_numpy(labels["polarity_mask"]).float(),
        "event_center": torch.from_numpy(labels["event_center"]).float()[:, :, ::s],
        "event_time": torch.from_numpy(labels["event_time"]).float()[:, :, ::s],
        "event_center_mask": torch.from_numpy(labels["event_center_mask"]).float()[:, :, ::s],
        "event_time_mask": torch.from_numpy(labels["event_time_mask"]).float()[:, :, ::s],
    }
    if target_nx is not None:
        out = _pad_crop_nx(out, target_nx, training=training)
    return out


class CEEDDataset(Dataset):
    """California Earthquake Event Dataset — Map-style Dataset.

    Args:
        region: "SC" or "NC"
        years: List of years to load
        days: List of days to load
        transforms: Transform pipeline
        label_config: Configuration for label generation
        min_snr: Minimum SNR to include sample
        buffer_size: Size of sample buffer for stacking
        target_nx: If set, pad/crop nx dimension to this size for batching
        event_feature_scale: Downsampling factor for event labels along time
    """

    def __init__(
        self,
        region: str = "SC",
        years: list[int] | None = None,
        days: list[int] | None = None,
        transforms: Transform | None = None,
        label_config: LabelConfig = LabelConfig(),
        min_snr: float = 0.0,
        buffer_size: int = 1000,
        target_nx: int | None = None,
        event_feature_scale: int = 16,
        preload: bool = False,
    ):
        self.transforms = transforms or minimal_transforms()
        self.label_config = label_config
        self.min_snr = min_snr
        self.target_nx = target_nx
        self.event_feature_scale = event_feature_scale

        hf_dataset = load_quakeflow_dataset(region, years, days, streaming=False)

        # Group records by event_id → multi-station Samples
        records_by_event: dict[str, list] = defaultdict(list)
        for record in hf_dataset:
            eid = record.get("event_id", "")
            if eid:
                records_by_event[eid].append(record)

        self.samples: list[Sample] = []
        for eid, event_records in records_by_event.items():
            sample = records_to_sample(event_records)
            target = sample.targets[0] if sample.targets else None
            if target is None or not target.p_picks or not target.s_picks:
                continue

            target.snr, target.amp_signal, target.amp_noise = calc_snr(
                sample.waveform, target.p_picks,
            )
            if target.snr >= min_snr and target.snr > 0:
                self.samples.append(sample)

        self.sample_buffer = SampleBuffer(buffer_size)
        for s in random.sample(self.samples, min(buffer_size, len(self.samples))):
            self.sample_buffer.add(s)

        self._setup_stacking_transforms()
        print(f"CEEDDataset: loaded {len(self.samples)} samples")

    def _setup_stacking_transforms(self):
        if isinstance(self.transforms, Compose):
            for t in self.transforms.transforms:
                if isinstance(t, (StackEvents, StackNoise)):
                    t.set_sample_fn(self.sample_buffer.get_random)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx].copy()
        sample = self.transforms(sample)
        return _to_output(sample, self.label_config, self.event_feature_scale, self.target_nx, training=True)


class CEEDIterableDataset(IterableDataset):
    """California Earthquake Event Dataset — Iterable/Streaming Dataset.

    Args:
        region: "SC" or "NC"
        years: List of years to load
        days: List of days to load
        transforms: Transform pipeline
        label_config: Configuration for label generation
        min_snr: Minimum SNR to include sample
        buffer_size: Size of sample buffer for stacking
        shuffle_buffer_size: Size of shuffle buffer
        target_nx: If set, pad/crop nx dimension to this size for batching
        event_feature_scale: Downsampling factor for event labels along time
    """

    def __init__(
        self,
        region: str = "SC",
        years: list[int] | None = None,
        days: list[int] | None = None,
        transforms: Transform | None = None,
        label_config: LabelConfig = LabelConfig(),
        min_snr: float = 0.0,
        buffer_size: int = 1000,
        shuffle_buffer_size: int = 1000,
        target_nx: int | None = None,
        event_feature_scale: int = 16,
    ):
        self.region = region
        self.years = years
        self.days = days
        self.transforms = transforms or minimal_transforms()
        self.label_config = label_config
        self.min_snr = min_snr
        self.buffer_size = buffer_size
        self.event_feature_scale = event_feature_scale
        self.shuffle_buffer_size = shuffle_buffer_size
        self.target_nx = target_nx

    def _emit_event(self, event_records: list[dict], sample_buffer: SampleBuffer) -> Sample | None:
        """Convert buffered records for one event into a validated Sample."""
        sample = records_to_sample(event_records)
        target = sample.targets[0] if sample.targets else None
        if target is None or not target.p_picks or not target.s_picks:
            return None
        target.snr, target.amp_signal, target.amp_noise = calc_snr(
            sample.waveform, target.p_picks,
        )
        if target.snr < self.min_snr or target.snr == 0:
            return None
        sample_buffer.add(sample)
        return sample

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        hf_dataset = load_quakeflow_dataset(
            self.region, self.years, self.days, streaming=True
        )

        sample_buffer = SampleBuffer(self.buffer_size)

        if isinstance(self.transforms, Compose):
            for t in self.transforms.transforms:
                if isinstance(t, (StackEvents, StackNoise)):
                    t.set_sample_fn(sample_buffer.get_random)

        shuffle_buffer: list[Sample] = []

        # Buffer records by event_id, emit when event changes
        current_event_id = None
        current_event_records: list[dict] = []
        event_count = 0

        for i, record in enumerate(hf_dataset):
            eid = record.get("event_id", "")
            if not eid:
                continue

            if eid != current_event_id:
                # Emit previous event
                if current_event_records:
                    # Distribute events across workers
                    if worker_info is None or event_count % worker_info.num_workers == worker_info.id:
                        sample = self._emit_event(current_event_records, sample_buffer)
                        if sample is not None:
                            shuffle_buffer.append(sample)
                    event_count += 1

                    if len(shuffle_buffer) >= self.shuffle_buffer_size:
                        random.shuffle(shuffle_buffer)
                        for s in shuffle_buffer:
                            yield self._process_sample(s.copy())
                        shuffle_buffer.clear()

                current_event_id = eid
                current_event_records = []

            current_event_records.append(record)

        # Emit last event
        if current_event_records:
            if worker_info is None or event_count % worker_info.num_workers == worker_info.id:
                sample = self._emit_event(current_event_records, sample_buffer)
                if sample is not None:
                    shuffle_buffer.append(sample)

        if shuffle_buffer:
            random.shuffle(shuffle_buffer)
            for s in shuffle_buffer:
                yield self._process_sample(s.copy())

    def _process_sample(self, sample: Sample) -> dict[str, torch.Tensor]:
        sample = self.transforms(sample)
        return _to_output(sample, self.label_config, self.event_feature_scale, self.target_nx, training=True)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_train_dataset(
    region: str = "SC",
    years: list[int] | None = None,
    days: list[int] | None = None,
    crop_length: int = 4096,
    streaming: bool = False,
    **kwargs,
) -> Dataset | IterableDataset:
    """Create a training dataset with default augmentation."""
    transforms = default_train_transforms(crop_length=crop_length)

    if streaming:
        return CEEDIterableDataset(
            region=region, years=years, days=days, transforms=transforms, **kwargs,
        )
    else:
        return CEEDDataset(
            region=region, years=years, days=days, transforms=transforms, **kwargs,
        )


def create_eval_dataset(
    region: str = "SC",
    years: list[int] | None = None,
    days: list[int] | None = None,
    crop_length: int = 4096,
    streaming: bool = False,
    **kwargs,
) -> Dataset | IterableDataset:
    """Create an evaluation dataset without augmentation."""
    transforms = default_eval_transforms(crop_length=crop_length)

    if streaming:
        return CEEDIterableDataset(
            region=region, years=years, days=days, transforms=transforms, **kwargs,
        )
    else:
        return CEEDDataset(
            region=region, years=years, days=days, transforms=transforms, **kwargs,
        )


# =============================================================================
# Plotting
# =============================================================================

def plot_trace(
    sample: Sample,
    labels: dict[str, np.ndarray],
    sta: int = 0,
    title: str = "",
    save_path: str = "ceed_trace.png",
):
    """Plot a single station from a multi-station Sample with labels (5-panel vertical stack).

    Layout:
        [1] Waveform (E/N/Z components) with P/S pick lines
        [2] P/S zoom-in (1s waveform at P and S picks, side by side)
        [3] Phase labels + phase_mask
        [4] Polarity label + polarity_mask
        [5] Event center + event_time + masks
    """
    import matplotlib.pyplot as plt

    nt = sample.nt
    t = np.arange(nt)
    waveform_z = sample.waveform[2, sta]  # Z component at station sta
    one_sec = int(sample.sampling_rate)

    p_times = [ti for tgt in sample.targets for s, ti in tgt.p_picks if int(s) == sta]
    s_times = [ti for tgt in sample.targets for s, ti in tgt.s_picks if int(s) == sta]

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})

    # [1] Waveform
    ax = axes[0]
    for i, name in enumerate(["E", "N", "Z"]):
        ax.plot(t, sample.waveform[i, sta] / 3 + i, label=name, alpha=0.7, linewidth=0.5)
    for pt in p_times:
        ax.axvline(pt, color="red", alpha=0.5, linewidth=0.8)
    for st in s_times:
        ax.axvline(st, color="blue", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Waveform")
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)

    # [2] P/S zoom-in
    ax = axes[1]
    ax.set_axis_off()
    for pick_t, color, label, x_pos in [
        (p_times[0] if p_times else None, "red", "P", 0.0),
        (s_times[0] if s_times else None, "blue", "S", 0.52),
    ]:
        if pick_t is None:
            continue
        inset = ax.inset_axes([x_pos, 0.0, 0.46, 1.0])
        t0 = max(0, int(pick_t) - one_sec // 4)
        t1 = min(nt, t0 + one_sec)
        inset.plot(t[t0:t1], waveform_z[t0:t1], "k", linewidth=0.8)
        inset.axvline(pick_t, color=color, linewidth=1.0, alpha=0.8)
        inset.set_title(f"{label} zoom (t={pick_t:.0f})", fontsize=9)
        inset.tick_params(labelsize=7)

    # [3] Phase labels + mask — labels are (label_ch, nx, nt)
    ax = axes[2]
    ax.plot(t, labels["phase_pick"][1, sta], label="P", color="red")
    ax.plot(t, labels["phase_pick"][2, sta], label="S", color="blue")
    ax.fill_between(t, labels["phase_mask"][0, sta], alpha=0.15, color="green", label="mask")
    ax.set_ylabel("Phase Labels")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)

    # [4] Polarity + mask
    ax = axes[3]
    ax.plot(t, labels["polarity"][0, sta], label="Polarity", color="green")
    ax.fill_between(t, labels["polarity_mask"][0, sta], alpha=0.15, color="green")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Polarity")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)

    # [5] Event center + event_time + masks
    ax = axes[4]
    ax.plot(t, labels["event_center"][0, sta], label="Event Center", color="purple")
    ax.fill_between(t, labels["event_center_mask"][0, sta], alpha=0.1, color="green", label="Center Mask")
    ax.fill_between(t, labels["event_time_mask"][0, sta], alpha=0.2, color="green", label="Time Mask")
    ax.set_ylabel("Event")
    ax.set_xlabel("Time Sample")
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    mask = labels["event_time_mask"][0, sta] > 0
    if mask.any():
        ax2.plot(t[mask], labels["event_time"][0, sta][mask], color="orange", linewidth=1.0, alpha=0.7)
        ax2.set_ylabel("Event Time", color="orange")
        ax2.set_ylim(min(0, labels["event_time"][0, sta][mask].min()), None)
    ax2.tick_params(axis="y", labelcolor="orange")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
    title: str = "",
    save_path: str = "ceed_overview.png",
):
    """Plot multi-station view of one event in a 2x2 grid.

    Sample has waveform (3, nx, nt). Stations already sorted by distance.

    Layout:
        [1,1] Waveform (Z component)
        [1,2] Waveform + P/S/Event picks
        [2,1] Phase labels + event center + phase mask
        [2,2] Event time + center + event mask
    """
    import matplotlib.pyplot as plt

    labels = generate_labels(sample, config)
    ns = sample.nx
    nt = sample.nt

    # Data is (nch, nx, nt). For display: transpose to (nt, nx)
    waveform_z = sample.waveform[2]  # (nx, nt)

    # Labels: (label_ch, nx, nt) → transpose to (nt, nx) for display
    p_disp = labels["phase_pick"][1].T  # (nt, nx)
    s_disp = labels["phase_pick"][2].T  # (nt, nx)
    phase_mask = labels["phase_mask"][0].T  # (nt, nx)
    event_center = labels["event_center"][0].T  # (nt, nx)
    event_center_mask = labels["event_center_mask"][0].T  # (nt, nx)
    event_time_mask = labels["event_time_mask"][0].T  # (nt, nx)

    vp, vs = config.vp, config.vp / config.vp_vs_ratio
    t = np.arange(nt)
    dt_s = 1.0 / sample.sampling_rate
    event_time_arr = np.zeros((ns, nt), dtype=np.float32)
    for tgt in sample.targets:
        for (sta, center), (_, ps_int) in zip(tgt.event_centers, tgt.ps_intervals):
            sta = int(sta)
            if 0 <= sta < ns:
                ps_seconds = ps_int * dt_s
                distance = ps_seconds * vp * vs / (vp - vs)
                shift = distance * (1 / vp + 1 / vs) / (2 * dt_s)
                event_time_arr[sta, :] = (t - center) + shift
    event_time_disp = event_time_arr.T  # (nt, nx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    imshow_kwargs = dict(aspect="auto", interpolation="nearest")

    # Normalize Z for wiggle display
    norm_z = waveform_z / (np.max(np.abs(waveform_z), axis=1, keepdims=True) + 1e-10)  # (nx, nt)
    scale = 0.5

    # [1,1] Waveform (Z) — wiggle traces
    ax = axes[0, 0]
    for i in range(ns):
        ax.plot(norm_z[i] * scale + i, np.arange(nt), "k", linewidth=0.3)
    ax.set_xlim(-1, ns)
    ax.set_ylim(nt, 0)
    ax.set_title("Waveform (Z)")
    ax.set_ylabel("Time Sample")

    # [1,2] Waveform + picks
    ax = axes[0, 1]
    for i in range(ns):
        ax.plot(norm_z[i] * scale + i, np.arange(nt), "k", linewidth=0.3)
    hw = 0.4
    for tgt in sample.targets:
        for sta, idx in tgt.p_picks:
            ax.plot([int(sta) - hw, int(sta) + hw], [idx, idx], c="red", linewidth=0.8, alpha=0.7, zorder=5)
        for sta, idx in tgt.s_picks:
            ax.plot([int(sta) - hw, int(sta) + hw], [idx, idx], c="blue", linewidth=0.8, alpha=0.7, zorder=5)
    ax.plot([], [], c="red", linewidth=1.5, label="P")
    ax.plot([], [], c="blue", linewidth=1.5, label="S")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    ax.set_xlim(-1, ns)
    ax.set_ylim(nt, 0)
    ax.set_title("Waveform + Picks")

    # [2,1] Phase labels + event center + phase mask
    ax = axes[1, 0]
    rgb = np.ones((nt, ns, 3))
    rgb[:, :, 1] = np.clip(1.0 - p_disp * 0.9, 0, 1)
    rgb[:, :, 2] = np.clip(1.0 - p_disp * 0.9, 0, 1)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] - s_disp * 0.9, 0, 1)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] - s_disp * 0.9, 0, 1)
    rgb[:, :, 0] = np.where(phase_mask > 0, rgb[:, :, 0] * 0.85, rgb[:, :, 0])
    rgb[:, :, 2] = np.where(phase_mask > 0, rgb[:, :, 2] * 0.85, rgb[:, :, 2])
    ax.imshow(rgb, **imshow_kwargs)
    event_rgba = np.zeros((nt, ns, 4))
    event_rgba[:, :, 0] = 1.0
    event_rgba[:, :, 1] = 1.0
    event_rgba[:, :, 3] = event_center * 0.8
    ax.imshow(event_rgba, **imshow_kwargs)
    ax.set_title("Phase Labels + Event Center")
    ax.set_ylabel("Time Sample")
    ax.set_xlabel("Station (sorted by distance)")

    # [2,2] Event time + center + event mask
    ax = axes[1, 1]
    bg = np.ones((nt, ns, 3))
    bg[:, :, 0] = np.where(event_center_mask > 0, 0.85, 1.0)
    bg[:, :, 2] = np.where(event_center_mask > 0, 0.85, 1.0)
    ax.imshow(bg, **imshow_kwargs)
    event_time_display = np.where(event_center_mask > 0, event_time_disp, np.nan)
    with np.errstate(all="ignore"):
        vabs = np.nanmax(np.abs(event_time_display)) or 1.0
    vabs = vabs if np.isfinite(vabs) else 1.0
    im = ax.imshow(event_time_display, cmap="seismic", vmin=-vabs, vmax=vabs, **imshow_kwargs)
    mask_rgba = np.zeros((nt, ns, 4))
    mask_rgba[:, :, 1] = 1.0
    mask_rgba[:, :, 3] = event_time_mask * 0.3
    ax.imshow(mask_rgba, **imshow_kwargs)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Event Time + Center")
    ax.set_xlabel("Station (sorted by distance)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_demo(
    sample: Sample,
    transforms: Compose,
    output_dir: str = "figures",
    event_id: str = "ceed_event",
    n_augmented: int = 5,
    n_traces: int = 5,
    config: LabelConfig = LabelConfig(),
):
    """Generate demo plots: overview, individual traces, and augmented overviews."""
    event_dir = os.path.join(output_dir, event_id)
    trace_dir = os.path.join(event_dir, "traces")
    aug_dir = os.path.join(event_dir, "augmented")
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)

    print(f"\nRaw data: {sample.waveform.shape}")
    plot_overview(sample, config, title=event_id, save_path=os.path.join(event_dir, "overview.png"))

    # Find stations with both P and S picks for trace plots
    labeled_stations = set()
    for tgt in sample.targets:
        p_stas = {int(s) for s, _ in tgt.p_picks}
        s_stas = {int(s) for s, _ in tgt.s_picks}
        labeled_stations |= (p_stas & s_stas)
    labeled_stations = sorted(labeled_stations)

    if labeled_stations:
        raw_labels = generate_labels(sample, config)
        n = min(n_traces, len(labeled_stations))
        stations = [labeled_stations[i] for i in np.linspace(0, len(labeled_stations) - 1, n, dtype=int)]
        for j, sta in enumerate(stations):
            trace_id = sample.trace_ids[sta] if sta < len(sample.trace_ids) else f"station_{sta}"
            plot_trace(
                sample, raw_labels, sta=sta,
                title=f"{event_id} | {trace_id}",
                save_path=os.path.join(trace_dir, f"{j:03d}.png"),
            )
        print(f"  Saved {n} trace plots to {trace_dir}/")

    print(f"\nGenerating {n_augmented} augmented views...")
    seed_offset = 0
    for i in range(n_augmented):
        for seed in range(seed_offset, seed_offset + 100):
            random.seed(seed)
            np.random.seed(seed)
            aug = transforms(sample.copy())
            if any(not t.is_empty for t in aug.targets):
                seed_offset = seed + 1
                break
        plot_overview(
            aug, config,
            title=f"{event_id} | augmented #{i}",
            save_path=os.path.join(aug_dir, f"{i:03d}.png"),
        )
    print(f"  Saved {n_augmented} augmented overviews to {aug_dir}/")


# =============================================================================
# Main — Demo and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CEED Dataset Demo")
    print("=" * 60)

    print("\nLoading data (streaming from GCS)...")
    try:
        ds = load_quakeflow_dataset(region="SC", years=[2025], days=[9], streaming=True)

        records_by_event = defaultdict(list)
        for i, record in enumerate(ds):
            records_by_event[record["event_id"]].append(record)
            if i >= 500:
                break

        best_event = max(records_by_event, key=lambda k: len(records_by_event[k]))
        event_records = records_by_event[best_event]
        print(f"Event {best_event}: {len(event_records)} stations")

        sample = records_to_sample(event_records)
        print(f"Waveform: {sample.waveform.shape}, targets: {len(sample.targets)}")

        transforms = default_train_transforms(crop_length=4096, enable_stacking=False)
        plot_demo(sample, transforms, event_id=best_event, n_augmented=5)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Could not load real data: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
