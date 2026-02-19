# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fsspec",
#     "gcsfs",
#     "h5py",
#     "numpy",
#     "pandas",
#     "scipy",
#     "torch",
# ]
# ///
"""
DAS (Distributed Acoustic Sensing) Data Loading Module

A modern, efficient dataset implementation for DAS phase picking following
best practices from computer vision (torchvision, timm, albumentations).

Design Principles:
1. COCO-style Target annotations: each event is a separate Target object,
   making multi-event stacking trivial and per-event metadata preserved
2. Unified (nch, nx, nt) internal format: (1, num_channels, num_time_samples)
3. Transform-based augmentation operating on raw picks before label generation
4. Compose pattern for chaining transforms
5. Clean separation: loading -> transforms -> label generation
6. Support for both local filesystem and Google Cloud Storage (GCS)

Example:
    >>> from eqnet.data.das import Sample, default_train_transforms, DASIterableDataset
    >>> transforms = default_train_transforms(nt=3072, nx=5120)
    >>> dataset = DASIterableDataset(
    ...     data_path="gs://quakeflow_das/ridgecrest_north",
    ...     label_path="gs://quakeflow_das/ridgecrest_north",
    ...     training=True,
    ...     transforms=transforms,
    ... )
"""
from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from glob import glob
from typing import Any, Callable, Iterator, Sequence

import fsspec
import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

# =============================================================================
# Configuration
# =============================================================================

BUCKET_DAS = "gs://quakeflow_das"
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
DEFAULT_SAMPLING_RATE = 100  # Hz
DEFAULT_SPATIAL_INTERVAL = 10.0  # meters


@dataclass
class LabelConfig:
    """Configuration for DAS label generation."""
    phase_width: int = 150  # Gaussian width for phase picks (samples)
    event_width: int = 150  # Gaussian width for event center
    mask_width_factor: float = 1.5  # mask_width = width * factor
    gaussian_threshold: float = 0.1  # Values below this are zeroed
    vp: float = 6.0  # P-wave velocity (km/s)
    vp_vs_ratio: float = 1.73  # Vp/Vs ratio


# =============================================================================
# Data Structures — COCO-style Target + Sample
# =============================================================================

@dataclass
class Target:
    """Per-event annotation for DAS data, following COCO/torchvision conventions.

    Each Target represents ONE seismic event's picks on the waveform.
    A Sample can have multiple Targets (e.g. from event stacking).

    Picks are (channel_index, time_sample) tuples.
    """
    p_picks: list[tuple[int, float]] = field(default_factory=list)
    s_picks: list[tuple[int, float]] = field(default_factory=list)
    event_centers: list[tuple[int, float]] = field(default_factory=list)  # (ch, center_time)
    ps_intervals: list[tuple[int, float]] = field(default_factory=list)  # (ch, S-P interval)

    # Per-event metadata
    snr: float = 0.0
    amp_signal: float = 0.0
    amp_noise: float = 0.0

    def copy(self) -> "Target":
        return Target(
            p_picks=self.p_picks.copy(),
            s_picks=self.s_picks.copy(),
            event_centers=self.event_centers.copy(),
            ps_intervals=self.ps_intervals.copy(),
            snr=self.snr,
            amp_signal=self.amp_signal,
            amp_noise=self.amp_noise,
        )

    @property
    def is_empty(self) -> bool:
        return not self.p_picks and not self.s_picks

    def all_times(self) -> list[float]:
        """All pick times (P + S)."""
        return [t for _, t in self.p_picks] + [t for _, t in self.s_picks]

    # -- Pick adjustment methods (in-place, return self for chaining) --

    def crop(self, t0: int, nt: int, x0: int, nx: int) -> "Target":
        """Keep picks in [x0, x0+nx) x [t0, t0+nt), shift to new origin."""
        self.p_picks = [
            (ch - x0, t - t0) for ch, t in self.p_picks
            if x0 <= ch < x0 + nx and t0 <= t < t0 + nt
        ]
        self.s_picks = [
            (ch - x0, t - t0) for ch, t in self.s_picks
            if x0 <= ch < x0 + nx and t0 <= t < t0 + nt
        ]
        ec, ps = [], []
        for (ch, t), (_, d) in zip(self.event_centers, self.ps_intervals):
            if x0 <= ch < x0 + nx and t0 <= t < t0 + nt:
                ec.append((ch - x0, t - t0))
                ps.append((ch - x0, d))
        self.event_centers, self.ps_intervals = ec, ps
        return self

    def shift_time(self, shift: int, nt: int, wrap: bool = True) -> "Target":
        """Shift all time indices. Wraps modulo nt or clips to [0, nt)."""
        if wrap:
            self.p_picks = [(ch, (t + shift) % nt) for ch, t in self.p_picks]
            self.s_picks = [(ch, (t + shift) % nt) for ch, t in self.s_picks]
            self.event_centers = [(ch, (t + shift) % nt) for ch, t in self.event_centers]
        else:
            self.p_picks = [(ch, t + shift) for ch, t in self.p_picks if 0 <= t + shift < nt]
            self.s_picks = [(ch, t + shift) for ch, t in self.s_picks if 0 <= t + shift < nt]
            ec, ps = [], []
            for (ch, t), (_, d) in zip(self.event_centers, self.ps_intervals):
                if 0 <= t + shift < nt:
                    ec.append((ch, t + shift))
                    ps.append((ch, d))
            self.event_centers, self.ps_intervals = ec, ps
        return self

    def scale_time(self, factor: float) -> "Target":
        """Scale time indices by factor."""
        self.p_picks = [(ch, t * factor) for ch, t in self.p_picks]
        self.s_picks = [(ch, t * factor) for ch, t in self.s_picks]
        ec, ps = [], []
        for (ch, t), (_, d) in zip(self.event_centers, self.ps_intervals):
            ec.append((ch, t * factor))
            ps.append((ch, d * factor))
        self.event_centers, self.ps_intervals = ec, ps
        return self

    def flip_space(self, nx: int) -> "Target":
        """Flip channel indices for spatial flip."""
        self.p_picks = [(nx - 1 - ch, t) for ch, t in self.p_picks]
        self.s_picks = [(nx - 1 - ch, t) for ch, t in self.s_picks]
        self.event_centers = [(nx - 1 - ch, t) for ch, t in self.event_centers]
        self.ps_intervals = [(nx - 1 - ch, d) for ch, d in self.ps_intervals]
        return self

    def scale_space(self, factor: float) -> "Target":
        """Scale channel indices by factor."""
        self.p_picks = [(int(ch * factor), t) for ch, t in self.p_picks]
        self.s_picks = [(int(ch * factor), t) for ch, t in self.s_picks]
        self.event_centers = [(int(ch * factor), t) for ch, t in self.event_centers]
        self.ps_intervals = [(int(ch * factor), d) for ch, d in self.ps_intervals]
        return self

    def mask_time(self, t0: int, t1: int) -> "Target":
        """Remove picks in time window [t0, t1)."""
        self.p_picks = [(ch, t) for ch, t in self.p_picks if not (t0 <= t < t1)]
        self.s_picks = [(ch, t) for ch, t in self.s_picks if not (t0 <= t < t1)]
        ec, ps = [], []
        for (ch, t), (_, d) in zip(self.event_centers, self.ps_intervals):
            if not (t0 <= t < t1):
                ec.append((ch, t))
                ps.append((ch, d))
        self.event_centers, self.ps_intervals = ec, ps
        return self


@dataclass
class Sample:
    """A DAS sample with waveform and per-event annotations.

    Waveform shape: (nch, nx, nt) = (1, num_channels, num_time_samples).
    targets: list of Target, one per seismic event in the window.
    """
    waveform: np.ndarray  # (nch, nx, nt) — always 3D
    targets: list[Target] = field(default_factory=list)

    # Metadata
    dt_s: float = 0.01
    dx_m: float = 10.0
    file_name: str = ""
    begin_time: datetime | None = None

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
    def snr(self) -> float:
        """Max SNR across all targets."""
        vals = [t.snr for t in self.targets if t.snr > 0]
        return max(vals) if vals else 0.0

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
            dt_s=self.dt_s,
            dx_m=self.dx_m,
            file_name=self.file_name,
            begin_time=self.begin_time,
        )


# =============================================================================
# Transforms — Following CV Best Practices (torchvision/albumentations pattern)
# =============================================================================

class Transform(ABC):
    """Base class for all DAS transforms. Operates on Sample objects."""

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """Compose multiple transforms together."""

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = list(transforms)

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
    """Normalize DAS waveform.

    Args:
        mode: "global" for global normalization, "channel" for per-channel
        eps: Small value for numerical stability
    """

    def __init__(self, mode: str = "global", eps: float = 1e-10):
        self.mode = mode
        self.eps = eps

    def __call__(self, sample: Sample) -> Sample:
        data = np.nan_to_num(sample.waveform)
        data = data - data.mean(axis=-1, keepdims=True)  # demean along time

        if self.mode == "global":
            std = data.std()
        else:
            std = data.std(axis=-1, keepdims=True)

        if np.any(std > self.eps):
            data = data / np.maximum(std, self.eps)

        sample.waveform = data.astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"Normalize(mode='{self.mode}')"


class MedianFilter(Transform):
    """Remove median along spatial axis (common-mode rejection).

    Args:
        p: Probability of applying
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample
        sample.waveform = sample.waveform - np.median(sample.waveform, axis=-2, keepdims=True)
        return sample

    def __repr__(self) -> str:
        return f"MedianFilter(p={self.p})"


class HighpassFilter(Transform):
    """Apply highpass filter to waveform."""

    def __init__(self, freq: float = 1.0, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.sos = scipy.signal.butter(4, freq, btype='high', fs=sampling_rate, output='sos')

    def __call__(self, sample: Sample) -> Sample:
        sample.waveform = scipy.signal.sosfilt(self.sos, sample.waveform, axis=-1).astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"HighpassFilter(freq={self.freq})"


# -----------------------------------------------------------------------------
# Spatial-Temporal Transforms
# -----------------------------------------------------------------------------

class RandomCrop(Transform):
    """Randomly crop waveform to fixed size, adjusting picks via Target.crop().

    If the data is smaller than the target size, it is padded with reflection.
    When picks exist, a random pick is chosen and the crop window is constrained
    to include it, guaranteeing every crop contains at least one pick.

    Args:
        nt: Target time samples
        nx: Target spatial samples
    """

    def __init__(self, nt: int = 3072, nx: int = 5120):
        self.nt = nt
        self.nx = nx

    def _pad_if_needed(self, sample: Sample) -> Sample:
        """Pad waveform with reflection if smaller than target."""
        _, nx_orig, nt_orig = sample.waveform.shape

        if nt_orig >= self.nt and nx_orig >= self.nx:
            return sample

        data_tensor = torch.from_numpy(sample.waveform).unsqueeze(0)  # (1, 1, nx, nt)
        pad_nx = max(0, self.nx - nx_orig)
        pad_nt = max(0, self.nt - nt_orig)
        # F.pad for 4D: (last_right, last_left, second_last_right, second_last_left)
        if pad_nt > 0 or pad_nx > 0:
            data_tensor = F.pad(data_tensor, (0, pad_nt, 0, pad_nx), mode="reflect")
            sample.waveform = data_tensor.squeeze(0).numpy().astype(np.float32)

        return sample

    def __call__(self, sample: Sample) -> Sample:
        sample = self._pad_if_needed(sample)
        _, nx_orig, nt_orig = sample.waveform.shape

        if nt_orig <= self.nt and nx_orig <= self.nx:
            return sample

        all_picks = [
            (ch, t) for target in sample.targets
            for ch, t in target.p_picks + target.s_picks
        ]

        if all_picks:
            ch_ref, t_ref = random.choice(all_picks)
            ch_ref, t_ref = int(ch_ref), int(t_ref)
            x0_lo = max(0, ch_ref - self.nx + 1)
            x0_hi = min(nx_orig - self.nx, ch_ref)
            t0_lo = max(0, t_ref - self.nt + 1)
            t0_hi = min(nt_orig - self.nt, t_ref)
            x0 = random.randint(x0_lo, max(x0_lo, x0_hi))
            t0 = random.randint(t0_lo, max(t0_lo, t0_hi))
        else:
            x0 = random.randint(0, max(0, nx_orig - self.nx))
            t0 = random.randint(0, max(0, nt_orig - self.nt))

        sample.waveform = sample.waveform[:, x0:x0 + self.nx, t0:t0 + self.nt].copy()

        for target in sample.targets:
            target.crop(t0, self.nt, x0, self.nx)
        sample.targets = [t for t in sample.targets if not t.is_empty]

        return sample

    def __repr__(self) -> str:
        return f"RandomCrop(nt={self.nt}, nx={self.nx})"


class FlipLR(Transform):
    """Randomly flip waveform horizontally (spatial axis).

    Args:
        p: Probability of flipping
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            nx = sample.nx
            sample.waveform = np.flip(sample.waveform, axis=-2).copy()  # flip spatial axis
            for target in sample.targets:
                target.flip_space(nx)
        return sample

    def __repr__(self) -> str:
        return f"FlipLR(p={self.p})"


class ResampleTime(Transform):
    """Resample time axis by a random factor.

    Args:
        min_factor: Minimum scale factor
        max_factor: Maximum scale factor
    """

    def __init__(self, min_factor: float = 0.5, max_factor: float = 3.0):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: Sample) -> Sample:
        factor = random.uniform(self.min_factor, self.max_factor)
        if abs(factor - 1.0) < 0.01:
            return sample

        # Resample waveform: (1, nx, nt) -> unsqueeze -> (1, 1, nx, nt)
        data_tensor = torch.from_numpy(sample.waveform).unsqueeze(0)
        data_resampled = F.interpolate(
            data_tensor, scale_factor=(1, factor), mode="bilinear", align_corners=False
        ).squeeze(0).numpy()
        sample.waveform = data_resampled.astype(np.float32)

        for target in sample.targets:
            target.scale_time(factor)

        return sample

    def __repr__(self) -> str:
        return f"ResampleTime(min_factor={self.min_factor}, max_factor={self.max_factor})"


class ResampleSpace(Transform):
    """Resample spatial axis by a random factor.

    Args:
        min_factor: Minimum scale factor
        max_factor: Maximum scale factor
    """

    def __init__(self, min_factor: float = 0.5, max_factor: float = 5.0):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: Sample) -> Sample:
        factor = random.uniform(self.min_factor, self.max_factor)
        if abs(factor - 1.0) < 0.01:
            return sample

        # Resample waveform: (1, nx, nt) -> unsqueeze -> (1, 1, nx, nt)
        data_tensor = torch.from_numpy(sample.waveform).unsqueeze(0)
        data_resampled = F.interpolate(
            data_tensor, scale_factor=(factor, 1), mode="nearest"
        ).squeeze(0).numpy()
        sample.waveform = data_resampled.astype(np.float32)

        for target in sample.targets:
            target.scale_space(factor)

        return sample

    def __repr__(self) -> str:
        return f"ResampleSpace(min_factor={self.min_factor}, max_factor={self.max_factor})"


# -----------------------------------------------------------------------------
# Masking Transforms
# -----------------------------------------------------------------------------

class Masking(Transform):
    """Randomly mask a time window in the waveform.

    Args:
        max_mask_nt: Maximum mask width in time samples
        p: Probability of applying mask
    """

    def __init__(self, max_mask_nt: int = 256, p: float = 0.2):
        self.max_mask_nt = max_mask_nt
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        nt = sample.nt
        mask_nt = random.randint(32, min(self.max_mask_nt, nt // 2))
        t0 = random.randint(0, nt - mask_nt)

        sample.waveform[:, :, t0:t0 + mask_nt] = 0.0

        for target in sample.targets:
            target.mask_time(t0, t0 + mask_nt)
        sample.targets = [t for t in sample.targets if not t.is_empty]

        return sample

    def __repr__(self) -> str:
        return f"Masking(max_mask_nt={self.max_mask_nt}, p={self.p})"


# -----------------------------------------------------------------------------
# Gain, Noise, Filter Transforms
# -----------------------------------------------------------------------------

class RandomGain(Transform):
    """Apply independent random gain to each DAS channel.

    Forces the model to learn waveform shape rather than absolute amplitude,
    simulating variation in fiber coupling and local site conditions.

    Args:
        min_gain: Minimum per-channel gain factor
        max_gain: Maximum per-channel gain factor
    """

    def __init__(self, min_gain: float = 0.5, max_gain: float = 2.0):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, sample: Sample) -> Sample:
        log_min = np.log10(self.min_gain)
        log_max = np.log10(self.max_gain)
        gains = 10 ** np.random.uniform(log_min, log_max, size=sample.nx)
        # gains shape: (nx,) → broadcast over (nch, nx, nt)
        sample.waveform = sample.waveform * gains[np.newaxis, :, np.newaxis].astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"RandomGain(min_gain={self.min_gain}, max_gain={self.max_gain})"


class ColoredNoise(Transform):
    """Add colored (1/f^alpha) noise to waveform.

    More realistic than white Gaussian noise. DAS noise is spectrally
    colored with contributions from ocean waves, traffic, and instrument noise.

    Args:
        snr_db_range: Range of SNR in dB (min, max)
        alpha_range: Range of spectral exponent (0=white, 1=pink, 2=brown)
        p: Probability of applying
    """

    def __init__(
        self,
        snr_db_range: tuple[float, float] = (5, 30),
        alpha_range: tuple[float, float] = (0.5, 2.0),
        p: float = 0.5,
    ):
        self.snr_db_range = snr_db_range
        self.alpha_range = alpha_range
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        alpha = random.uniform(*self.alpha_range)
        nt = sample.nt
        white = np.random.randn(*sample.waveform.shape).astype(np.float32)

        # Shape spectrum: multiply FFT by f^(-alpha/2)
        freqs = np.fft.rfftfreq(nt)
        freqs[0] = 1.0  # avoid division by zero at DC
        spectrum = np.fft.rfft(white, axis=-1)
        spectrum *= (freqs ** (-alpha / 2)).astype(np.float32)
        noise = np.fft.irfft(spectrum, n=nt, axis=-1).astype(np.float32)

        # Scale to target SNR
        snr_db = random.uniform(*self.snr_db_range)
        signal_power = np.mean(sample.waveform ** 2)
        if signal_power > 0:
            noise_power = signal_power / (10 ** (snr_db / 10))
            current_power = np.mean(noise ** 2)
            if current_power > 0:
                noise *= np.sqrt(noise_power / current_power)
                sample.waveform = sample.waveform + noise

        return sample

    def __repr__(self) -> str:
        return f"ColoredNoise(snr_db_range={self.snr_db_range}, alpha_range={self.alpha_range})"


class RandomBandpass(Transform):
    """Apply bandpass filter with random corner frequencies.

    Simulates different preprocessing choices and DAS system responses.

    Args:
        low_range: Range for low corner frequency (Hz)
        high_range: Range for high corner frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        p: Probability of applying
    """

    def __init__(
        self,
        low_range: tuple[float, float] = (0.1, 2.0),
        high_range: tuple[float, float] = (10.0, 45.0),
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        p: float = 0.3,
    ):
        self.low_range = low_range
        self.high_range = high_range
        self.sampling_rate = sampling_rate
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p:
            return sample

        low = random.uniform(*self.low_range)
        high = random.uniform(*self.high_range)
        if low >= high or high >= self.sampling_rate / 2:
            return sample

        sos = scipy.signal.butter(2, [low, high], btype='band', fs=self.sampling_rate, output='sos')
        sample.waveform = scipy.signal.sosfiltfilt(sos, sample.waveform, axis=-1).astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"RandomBandpass(low_range={self.low_range}, high_range={self.high_range})"


# -----------------------------------------------------------------------------
# Stacking Transforms
# -----------------------------------------------------------------------------

class StackEvents(Transform):
    """Stack multiple events onto a single waveform.

    Each stacked event becomes a separate Target in sample.targets.
    Similar to Mixup/CutMix in computer vision.

    Args:
        p: Probability of stacking
        min_snr: Minimum SNR required to apply stacking
        max_shift: Maximum temporal shift for alignment
    """

    def __init__(self, p: float = 0.3, min_snr: float = 10.0, max_shift: int = 2048):
        self.p = p
        self.min_snr = min_snr
        self.max_shift = max_shift
        self._sample_fn: Callable[[], Sample | None] | None = None

    def set_sample_fn(self, fn: Callable[[], Sample | None]):
        """Set function to get random samples for stacking."""
        self._sample_fn = fn

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p or sample.snr < self.min_snr:
            return sample
        if self._sample_fn is None:
            return sample

        sample2 = self._sample_fn()
        if sample2 is None:
            return sample

        if sample.waveform.shape != sample2.waveform.shape:
            return sample

        # Random shift and amplitude scaling
        shift = random.randint(-self.max_shift, self.max_shift)
        scale = 1 + random.random() * 2
        nt = sample.nt

        # Stack waveforms: roll along time axis
        waveform2 = np.roll(sample2.waveform, shift, axis=-1)
        sample.waveform = sample.waveform * scale + waveform2 * scale

        # Create shifted targets for stacked event
        for target2 in sample2.targets:
            new_target = target2.copy()
            new_target.shift_time(shift, nt, wrap=True)
            sample.targets.append(new_target)

        return sample

    def __repr__(self) -> str:
        return f"StackEvents(p={self.p}, min_snr={self.min_snr})"


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

    def _extract_tail_noise(self, donor: Sample, length: int) -> np.ndarray | None:
        """Extract noise from the end of the donor waveform, past all picks."""
        nt = donor.waveform.shape[-1]
        if nt < length:
            return None
        all_times = [t for tgt in donor.targets for t in tgt.all_times()]
        last_pick = int(max(all_times)) if all_times else 0
        start = nt - length
        if start < last_pick + 100:  # margin for label Gaussian tail
            return None
        return donor.waveform[..., start:]

    def _get_noise(self, sample: Sample) -> np.ndarray | None:
        """Get noise: dedicated files first, then tail extraction."""
        if self._noise_fn is not None:
            noise = self._noise_fn()
            if noise is not None:
                return noise
        if self._sample_fn is not None:
            donor = self._sample_fn()
            if donor is not None:
                return self._extract_tail_noise(donor, sample.nt)
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

def generate_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
    phases: list[str] = ["P", "S"],
) -> dict[str, np.ndarray]:
    """Generate all labels from Sample.targets.

    Iterates over all targets (events) in the sample, accumulating
    Gaussian labels. Naturally handles multi-event windows from stacking.

    Returns:
        All arrays have shape (label_ch, nx, nt) matching the unified format.
        - phase_pick: (3, nx, nt) — [noise, P, S]
        - phase_mask: (1, nx, nt)
        - event_center: (1, nx, nt)
        - event_time: (1, nx, nt)
        - event_center_mask: (1, nx, nt)
        - event_time_mask: (1, nx, nt)
    """
    nx, nt = sample.nx, sample.nt
    sigma = config.phase_width / 6
    sigma_event = config.event_width / 6
    mask_w = int(config.phase_width * config.mask_width_factor)
    event_mask_w = int(config.event_width * config.mask_width_factor)
    t = np.arange(nt)

    p_label = np.zeros((nx, nt), dtype=np.float32)
    s_label = np.zeros((nx, nt), dtype=np.float32)
    event_center_label = np.zeros((nx, nt), dtype=np.float32)
    event_time_label = np.zeros((nx, nt), dtype=np.float32)
    event_center_mask = np.zeros((nx, nt), dtype=np.float32)
    event_time_mask = np.zeros((nx, nt), dtype=np.float32)
    phase_mask = np.zeros((nx, nt), dtype=np.float32)

    has_p = np.zeros(nx, dtype=bool)
    has_s = np.zeros(nx, dtype=bool)

    vp = config.vp
    vs = vp / config.vp_vs_ratio

    picks_per_channel: dict[int, list[float]] = {}

    def add_phase_picks(picks, label, has_flag):
        for ch, ti in picks:
            ch = int(ch)
            if 0 <= ch < nx:
                g = np.exp(-((t - ti) ** 2) / (2 * sigma ** 2))
                g[g < config.gaussian_threshold] = 0.0
                label[ch] += g
                has_flag[ch] = True
                picks_per_channel.setdefault(ch, []).append(ti)

    for target in sample.targets:
        add_phase_picks(target.p_picks, p_label, has_p)
        add_phase_picks(target.s_picks, s_label, has_s)

        # Event labels
        for (ch, center), (_, ps_int) in zip(target.event_centers, target.ps_intervals):
            ch = int(ch)
            if 0 <= ch < nx:
                g = np.exp(-((t - center) ** 2) / (2 * sigma_event ** 2))
                g[g < 0.05] = 0.0
                event_center_label[ch] += g

                ps_seconds = ps_int * sample.dt_s
                distance = ps_seconds * vp * vs / (vp - vs)
                center_travel = distance * (1 / vp + 1 / vs) / 2
                shift = center_travel / sample.dt_s

                event_center_mask[ch, :] = 1.0
                t0 = max(0, int(center) - event_mask_w)
                t1 = min(nt, int(center) + event_mask_w)
                event_time_mask[ch, t0:t1] = 1.0
                event_time_label[ch, t0:t1] = (t[t0:t1] - center) + shift

    # Phase mask: full channel if both P and S, narrow window otherwise
    for ch in picks_per_channel:
        if has_p[ch] and has_s[ch]:
            phase_mask[ch, :] = 1.0
        else:
            for ti in picks_per_channel[ch]:
                t0 = max(0, int(ti) - mask_w)
                t1 = min(nt, int(ti) + mask_w)
                phase_mask[ch, t0:t1] = 1.0

    noise_label = np.maximum(0, 1.0 - p_label - s_label)

    # All outputs: (label_ch, nx, nt)
    return {
        "phase_pick": np.stack([noise_label, p_label, s_label], axis=0),  # (3, nx, nt)
        "phase_mask": phase_mask[np.newaxis],  # (1, nx, nt)
        "event_center": event_center_label[np.newaxis],  # (1, nx, nt)
        "event_time": event_time_label[np.newaxis],  # (1, nx, nt)
        "event_center_mask": event_center_mask[np.newaxis],  # (1, nx, nt)
        "event_time_mask": event_time_mask[np.newaxis],  # (1, nx, nt)
    }


# =============================================================================
# Default Transform Presets
# =============================================================================

def default_train_transforms(
    nt: int = 3072,
    nx: int = 5120,
    enable_stacking: bool = True,
    enable_noise_stacking: bool = True,
    enable_resample_time: bool = False,
    enable_resample_space: bool = False,
    enable_masking: bool = True,
    augment: bool = True,
) -> Compose:
    """Default transforms for DAS training.

    When augment=False, only Normalize + RandomCrop are applied (for overfit testing).
    """
    if not augment:
        return Compose([Normalize(), RandomCrop(nt=nt, nx=nx), Normalize()])

    transforms = [Normalize()]

    if enable_resample_time:
        transforms.append(ResampleTime(0.5, 3.0))

    if enable_resample_space:
        transforms.append(ResampleSpace(0.5, 5.0))

    if enable_stacking:
        transforms.append(StackEvents(p=0.3, min_snr=10.0))

    transforms.append(RandomCrop(nt=nt, nx=nx))

    if enable_noise_stacking:
        transforms.append(StackNoise(max_ratio=2.0, p=0.5))

    transforms.extend([
        ColoredNoise(snr_db_range=(5, 30), p=0.5),
        FlipLR(p=0.5),
    ])

    if enable_masking:
        transforms.append(Masking(max_mask_nt=256, p=0.2))

    transforms.extend([
        RandomBandpass(p=0.3),
        MedianFilter(p=0.5),
        Normalize(),  # Final normalization
    ])

    return Compose(transforms)


def default_eval_transforms() -> Compose:
    """Default transforms for DAS evaluation."""
    return Compose([Normalize()])


def minimal_transforms() -> Compose:
    """Minimal transforms - just normalize."""
    return Compose([Normalize()])


def _to_output(
    sample: Sample,
    label_config: LabelConfig,
    phases: list[str],
    event_feature_scale: int,
) -> dict[str, torch.Tensor]:
    """Convert transformed Sample to output dict with labels."""
    labels = generate_labels(sample, label_config, phases)
    s = event_feature_scale
    data = torch.from_numpy(sample.waveform).float()
    return {
        "data": torch.nan_to_num(data),
        "phase_pick": torch.from_numpy(labels["phase_pick"]).float(),
        "phase_mask": torch.from_numpy(labels["phase_mask"]).float(),
        "event_center": torch.from_numpy(labels["event_center"]).float()[:, :, ::s],
        "event_time": torch.from_numpy(labels["event_time"]).float()[:, :, ::s],
        "event_center_mask": torch.from_numpy(labels["event_center_mask"]).float()[:, :, ::s],
        "event_time_mask": torch.from_numpy(labels["event_time_mask"]).float()[:, :, ::s],
        "file_name": sample.file_name,
        "height": data.shape[-2],
        "width": data.shape[-1],
        "dt_s": sample.dt_s,
        "dx_m": sample.dx_m,
    }


# =============================================================================
# GCS Utilities
# =============================================================================

def get_gcs_storage_options() -> dict:
    """Load GCS credentials for authenticated access."""
    if os.path.exists(GCS_CREDENTIALS_PATH):
        with open(GCS_CREDENTIALS_PATH, "r") as f:
            token = json.load(f)
        return {"token": token}
    return {}


def get_filesystem(path: str):
    """Get appropriate filesystem for the given path."""
    if path.startswith("gs://"):
        storage_options = get_gcs_storage_options()
        fs = fsspec.filesystem("gcs", **storage_options)
        path_clean = path[5:]
        return fs, path_clean
    else:
        fs = fsspec.filesystem("file")
        return fs, path


def open_file(path: str, mode: str = "rb"):
    """Open a file from local or GCS path."""
    if path.startswith("gs://"):
        storage_options = get_gcs_storage_options()
        return fsspec.open(path, mode, **storage_options)
    else:
        return fsspec.open(path, mode)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def calc_snr(
    data: np.ndarray,
    picks: list[tuple[int, float]],
    noise_window: int = 200,
    signal_window: int = 200,
) -> tuple[float, float, float]:
    """Calculate SNR from waveform and picks.

    Args:
        data: Waveform array (nch, nx, nt)
        picks: List of (channel_index, time_index) tuples

    Returns:
        (snr, signal_std, noise_std)
    """
    nch, nx, nt = data.shape

    snrs, signals, noises = [], [], []
    for ch, phase_time in picks:
        ch = int(ch)
        phase_time = int(phase_time)
        if 0 <= ch < nx and noise_window < phase_time < nt - signal_window:
            noise = np.std(data[:, ch, max(0, phase_time - noise_window):phase_time])
            signal = np.std(data[:, ch, phase_time:phase_time + signal_window])
            if noise > 0 and signal > 0:
                snrs.append(signal / noise)
                signals.append(signal)
                noises.append(noise)

    if not snrs:
        return 0.0, 0.0, 0.0

    idx = np.argmax(snrs)
    return float(snrs[idx]), float(signals[idx]), float(noises[idx])


def read_PASSCAL_segy(fid, nTraces=1250, nSample=900000, TraceOff=0, strain_rate=True):
    """Read PASSCAL segy raw data.

    For Ridgecrest data: 1250 channels, sampling rate 250 Hz.
    Author: Jiuxun Yin
    Source: https://github.com/SCEDC/cloud/blob/master/pds_ridgecrest_das.ipynb
    """
    fs = nSample / 3600
    data = np.zeros((nTraces, nSample), dtype=np.float32)

    fid.seek(3600)
    fid.seek(TraceOff * (240 + nSample * 4), 1)
    for ii in range(nTraces):
        fid.seek(240, 1)
        bytes = fid.read(nSample * 4)
        data[ii, :] = np.frombuffer(bytes, dtype=np.float32)

    fid.close()

    Ridgecrest_conversion_factor = 1550.12 / (0.78 * 4 * np.pi * 1.46 * 8)
    data = data * Ridgecrest_conversion_factor

    if strain_rate:
        data = np.gradient(data, axis=1) * fs

    return data


def padding(data: torch.Tensor, min_nx: int = 256, min_nt: int = 256) -> torch.Tensor:
    """Pad data to multiples of min_nx and min_nt.

    Args:
        data: Tensor of shape (nch, nx, nt)
        min_nx: Minimum space samples (pads to multiple)
        min_nt: Minimum time samples (pads to multiple)

    Returns:
        Padded tensor
    """
    nch, nx, nt = data.shape
    pad_nx = (min_nx - nx % min_nx) % min_nx
    pad_nt = (min_nt - nt % min_nt) % min_nt

    if pad_nt > 0 or pad_nx > 0:
        with torch.no_grad():
            # F.pad for 3D: (last_right, last_left, second_last_right, second_last_left)
            data = F.pad(data, (0, pad_nt, 0, pad_nx), mode="constant")

    return data


def load_sample_from_h5(
    file_path: str,
    picks_df: pd.DataFrame | None = None,
    phases: list[str] = ["P", "S"],
) -> Sample:
    """Load a Sample from an HDF5 file.

    Args:
        file_path: Path to H5 file (local or GCS)
        picks_df: DataFrame with picks (channel_index, phase_index, phase_type)
        phases: Phase types to extract

    Returns:
        Sample instance with waveform shape (1, nx, nt)
    """
    with open_file(file_path, "rb") as f:
        with h5py.File(f, "r") as fp:
            data = fp["data"][:, :]  # (nx, nt) — keep native ordering
            dt_s = fp["data"].attrs.get("dt_s", 0.01)
            dx_m = fp["data"].attrs.get("dx_m", 10.0)

    data = data[np.newaxis, :, :]  # (1, nx, nt)
    data = data / (np.std(data) + 1e-10)
    data = data - np.mean(data, axis=-1, keepdims=True)  # demean along time

    p_picks = []
    s_picks = []
    event_centers = []
    ps_intervals = []

    if picks_df is not None:
        if "channel_index" not in picks_df.columns and "station_id" in picks_df.columns:
            picks_df = picks_df.rename(columns={"station_id": "channel_index"})

        for phase in phases:
            phase_picks = picks_df[picks_df["phase_type"] == phase][
                ["channel_index", "phase_index"]
            ].values.tolist()
            phase_picks = [(int(ch), float(t)) for ch, t in phase_picks]

            if phase == "P":
                p_picks = phase_picks
            elif phase == "S":
                s_picks = phase_picks

        # Compute event centers from P and S picks
        if p_picks and s_picks:
            p_dict = {ch: t for ch, t in p_picks}
            s_dict = {ch: t for ch, t in s_picks}
            for ch in set(p_dict.keys()) & set(s_dict.keys()):
                center = (p_dict[ch] + s_dict[ch]) / 2
                ps_int = s_dict[ch] - p_dict[ch]
                event_centers.append((ch, center))
                ps_intervals.append((ch, ps_int))

    target = Target(
        p_picks=p_picks,
        s_picks=s_picks,
        event_centers=event_centers,
        ps_intervals=ps_intervals,
    )

    return Sample(
        waveform=data.astype(np.float32),
        targets=[target] if not target.is_empty else [],
        dt_s=dt_s,
        dx_m=dx_m,
        file_name=os.path.basename(file_path),
    )


# =============================================================================
# Sample Buffer for Stacking
# =============================================================================

class SampleBuffer:
    """Efficient buffer for random sample access during stacking.
    Uses reservoir sampling for streaming datasets.
    """

    def __init__(self, max_size: int = 100):
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


# =============================================================================
# Dataset Classes
# =============================================================================

class DASIterableDataset(IterableDataset):
    """DAS Iterable Dataset supporting local and GCS paths.

    For training, uses the transform pipeline with Sample dataclass.
    For inference, handles multiple data formats (h5, npz, npy, segy).

    Args:
        data_path: Local path or GCS path (gs://...) to data directory
        data_list: List of data files or path to file containing list
        format: Data format ("h5", "npz", "npy", "segy")
        prefix: File prefix filter
        suffix: File suffix filter
        nt: Number of time samples per patch
        nx: Number of space samples per patch
        min_nt: Minimum time samples for padding
        min_nx: Minimum space samples for padding
        training: Whether in training mode
        phases: Phase types to use (e.g., ["P", "S"])
        label_path: Local path or GCS path to labels directory
        subdir: Subdirectory depth for label->data path conversion
        label_list: List of label files or path to file containing list
        noise_list: List of noise files for augmentation
        transforms: Transform pipeline (overrides individual augmentation flags)
        label_config: Configuration for label generation
        buffer_size: Size of sample buffer for stacking
        skip_existing: Skip files with existing picks
        pick_path: Path to save/check picks
        folder_depth: Parent folder depth for pick_path
        num_patch: Number of patches per sample
        highpass_filter: Highpass filter frequency
        system: DAS system type ("optasense" or None)
        cut_patch: Whether to cut into patches
        event_feature_scale: Downsampling factor for event labels
        rank: Worker rank for distributed training
        world_size: Total workers for distributed training
    """

    def __init__(
        self,
        data_path="./",
        data_list=None,
        format="h5",
        prefix="",
        suffix="",
        nt=1024 * 3,
        nx=1024 * 5,
        min_nt=256,
        min_nx=256,
        # Training
        training=False,
        phases=["P", "S"],
        label_path="./",
        subdir=3,
        label_list=None,
        noise_list=None,
        transforms: Transform | None = None,
        label_config: LabelConfig = LabelConfig(),
        buffer_size: int = 100,
        # Overfit mode: "fixed" (same crop), "random" (different crops), or None
        overfit: str | None = None,
        # Inference
        skip_existing=False,
        pick_path="./",
        folder_depth=1,
        num_patch=2,
        highpass_filter=0.0,
        system=None,
        cut_patch=False,
        resample_time=False,
        event_feature_scale: int = 16,
        rank=0,
        world_size=1,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.data_path = data_path
        self.format = format
        self.prefix = prefix
        self.suffix = suffix
        self.subdir = subdir
        self.label_path = label_path
        self.use_gcs = data_path.startswith("gs://") if data_path else False

        # Load data list
        if data_list is not None:
            if isinstance(data_list, list):
                self.data_list = []
                for data_list_ in data_list:
                    self.data_list += self._read_text_file(data_list_).rstrip("\n").split("\n")
            else:
                self.data_list = self._read_text_file(data_list).rstrip("\n").split("\n")
        else:
            self.data_list = self._list_files(self.data_path, f"{prefix}*{suffix}.{format}")

        if not training:
            self.data_list = self.data_list[rank::world_size]

        # Continuous data / inference
        self.system = system
        self.cut_patch = cut_patch
        self.resample_time = resample_time
        self.dt = kwargs.get("dt", 0.01)
        self.dx = kwargs.get("dx", 10.0)
        self.nt = nt
        self.nx = nx
        self.min_nt = min_nt
        self.min_nx = min_nx
        assert self.nt % self.min_nt == 0
        assert self.nx % self.min_nx == 0

        # Training
        self.training = training
        self.phases = phases
        self.label_config = label_config
        self.num_patch = num_patch
        self.overfit = overfit
        self.event_feature_scale = event_feature_scale
        self.highpass_filter = highpass_filter
        self.skip_existing = skip_existing
        self.pick_path = pick_path
        self.folder_depth = folder_depth

        # Load label list
        if label_list is not None:
            if isinstance(label_list, list):
                self.label_list = []
                for label_list_ in label_list:
                    self.label_list += self._read_text_file(label_list_).rstrip("\n").split("\n")
            else:
                self.label_list = self._read_text_file(label_list).rstrip("\n").split("\n")
            if training:
                self.label_list = self.label_list[: len(self.label_list) // world_size * world_size]
            self.label_list = self.label_list[rank::world_size]
        else:
            self.label_list = self._list_files(self.label_path, "*.csv")

        # Noise list for stacking
        self.noise_list = None
        if noise_list is not None:
            if isinstance(noise_list, list):
                self.noise_list = []
                for noise_list_ in noise_list:
                    self.noise_list += self._read_text_file(noise_list_).rstrip("\n").split("\n")
            else:
                self.noise_list = self._read_text_file(noise_list).rstrip("\n").split("\n")

        # Transforms
        if transforms is not None:
            self.transforms = transforms
        elif training:
            self.transforms = default_train_transforms(nt=nt, nx=nx)
        else:
            self.transforms = default_eval_transforms()

        # Sample buffer for stacking
        self.sample_buffer = SampleBuffer(buffer_size)
        self._setup_stacking_transforms()

        if self.training:
            print(f"DASIterableDataset: {len(self.label_list)} label files")
        else:
            print(f"DASIterableDataset: {len(self.data_list)} data files")

        # Pre-calculate length
        self._data_len = self._count()

    def _setup_stacking_transforms(self):
        """Connect stacking transforms to sample buffer and noise loader."""
        if not isinstance(self.transforms, Compose):
            return
        for t in self.transforms.transforms:
            if isinstance(t, StackEvents):
                t.set_sample_fn(self.sample_buffer.get_random)
            elif isinstance(t, StackNoise):
                if self.noise_list:
                    t.set_noise_fn(self._load_random_noise)
                t.set_sample_fn(self.sample_buffer.get_random)

    def _load_random_noise(self) -> np.ndarray | None:
        """Load a random noise file for stacking."""
        if not self.noise_list:
            return None
        noise_file = self.noise_list[random.randint(0, len(self.noise_list) - 1)]
        noise_path = self._construct_file_path(self.data_path, noise_file)
        try:
            with open_file(noise_path, "rb") as f:
                with h5py.File(f, "r") as fp:
                    noise = fp["data"][:, :]  # (nx, nt) — keep native ordering
                noise = np.roll(noise, max(0, self.nt - 3000), axis=-1)  # roll time
                noise = noise[np.newaxis, :, :self.nt]  # (1, nx, nt)
                return noise.astype(np.float32)
        except Exception as e:
            print(f"Error reading noise file {noise_path}: {e}")
            return None

    def _list_files(self, path: str, pattern: str) -> list[str]:
        """List files matching pattern from local or GCS path."""
        if path.startswith("gs://"):
            fs, path_clean = get_filesystem(path)
            files = fs.glob(f"{path_clean}/{pattern}")
            return [f"gs://{f}" for f in files]
        else:
            return glob(os.path.join(path, pattern))

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        """Read text file from local or GCS."""
        if file_path.startswith("gs://"):
            fs = fsspec.filesystem("gcs", **get_gcs_storage_options())
            with fs.open(file_path, "r") as f:
                return f.read()
        else:
            with open(file_path, "r") as f:
                return f.read()

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file from local or GCS."""
        if file_path.startswith("gs://"):
            storage_options = get_gcs_storage_options()
            return pd.read_csv(file_path, storage_options=storage_options)
        else:
            return pd.read_csv(file_path)

    def _construct_file_path(self, base_path: str, relative_path: str) -> str:
        """Construct full file path preserving GCS prefix."""
        if base_path.startswith("gs://"):
            return f"{base_path.rstrip('/')}/{relative_path.lstrip('/')}"
        else:
            return os.path.join(base_path, relative_path)

    def _load_sample(self, label_file: str) -> Sample | None:
        """Load a training sample from label file."""
        label_path_full = self._construct_file_path(self.label_path, label_file)
        picks_df = self._read_csv(label_path_full)
        if "channel_index" not in picks_df.columns:
            picks_df = picks_df.rename(columns={"station_id": "channel_index"})

        # Construct data file path
        data_file = "/".join(
            label_file.replace("labels", "data").replace(".csv", ".h5").split("/")[-self.subdir:]
        )
        data_path_full = self._construct_file_path(self.data_path, data_file)

        try:
            sample = load_sample_from_h5(data_path_full, picks_df, self.phases)
        except Exception as e:
            print(f"Error reading {data_path_full}: {e}")
            return None

        # Calculate SNR for the target
        if sample.targets:
            target = sample.targets[0]
            if target.p_picks:
                target.snr, target.amp_signal, target.amp_noise = calc_snr(
                    sample.waveform, target.p_picks
                )

        sample.file_name = os.path.splitext(label_file.split("/")[-1])[0]
        return sample

    def _sample_to_output(self, sample: Sample) -> dict[str, torch.Tensor]:
        return _to_output(sample, self.label_config, self.phases, self.event_feature_scale)

    def __len__(self):
        return self._data_len

    def _count(self):
        if self.training:
            return len(self.label_list) * self.num_patch

        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.format == "h5":
                with open_file(self.data_list[0], "rb") as fs:
                    with h5py.File(fs, "r") as meta:
                        if self.system == "optasense":
                            attrs = {}
                            if "Data" in meta:
                                nx, nt = meta["Data"].shape
                                attrs["dt_s"] = meta["Data"].attrs["dt"]
                                attrs["dx_m"] = meta["Data"].attrs["dCh"]
                            else:
                                nx, nt = meta["Acquisition/Raw[0]/RawData"].shape
                                dx = meta["Acquisition"].attrs["SpatialSamplingInterval"]
                                fs_rate = meta["Acquisition/Raw[0]"].attrs["OutputDataRate"]
                                attrs["dx_m"] = dx
                                attrs["dt_s"] = 1.0 / fs_rate
                        else:
                            nx, nt = meta["data"].shape
                            attrs = dict(meta["data"].attrs)
                if self.resample_time and ("dt_s" in attrs):
                    if (attrs["dt_s"] != 0.01) and (int(round(1.0 / attrs["dt_s"])) % 100 == 0):
                        nt = int(nt / round(0.01 / attrs["dt_s"]))

            elif self.format == "segy":
                print("Start reading segy file")
                with open_file(self.data_list[0], "rb") as fs:
                    nx, nt = read_PASSCAL_segy(fs).shape
                print("End reading segy file")
            else:
                raise ValueError("Unknown dataset")

            return len(self.data_list) * ((nt - 1) // self.nt + 1) * ((nx - 1) // self.nx + 1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if self.training:
            return iter(self._sample_training(self.label_list[worker_id::num_workers]))
        else:
            return iter(self._sample_inference(self.data_list[worker_id::num_workers]))

    def _sample_training(self, file_list):
        """Training sample generator using transform pipeline."""
        if self.overfit:
            yield from self._sample_training_overfit(file_list)
            return

        while True:
            file_list = np.random.permutation(file_list)
            for label_file in file_list:
                sample = self._load_sample(label_file)
                if sample is None:
                    continue

                # Add to buffer for stacking transforms
                self.sample_buffer.add(sample)

                for ii in range(self.num_patch):
                    s = sample.copy()
                    s = self.transforms(s)
                    output = self._sample_to_output(s)
                    output["file_name"] = sample.file_name + f"_{ii:02d}"
                    yield output

    def _sample_training_overfit(self, file_list):
        """Overfit mode: cache samples and replay them.

        Caches are stored as instance attributes so they persist across
        __iter__ calls (each epoch creates a new generator).

        overfit="fixed":  cache output after transforms — same crop every step.
        overfit="random": cache raw samples, re-apply transforms — different crops.
        """
        # Load samples once (first call only)
        if not hasattr(self, "_overfit_samples"):
            self._overfit_samples = []
            for label_file in file_list:
                sample = self._load_sample(label_file)
                if sample is not None:
                    self.sample_buffer.add(sample)
                    self._overfit_samples.append(sample)
            print(f"Overfit mode ({self.overfit}): cached {len(self._overfit_samples)} samples")

            if self.overfit == "fixed":
                self._overfit_outputs = []
                for sample in self._overfit_samples:
                    s = sample.copy()
                    s = self.transforms(s)
                    self._overfit_outputs.append(self._sample_to_output(s))

        if self.overfit == "fixed":
            while True:
                for output in self._overfit_outputs:
                    yield output
        else:
            while True:
                for sample in self._overfit_samples:
                    s = sample.copy()
                    s = self.transforms(s)
                    yield self._sample_to_output(s)

    def _sample_inference(self, file_list):
        """Inference sample generator supporting multiple formats."""
        for file in file_list:
            if not self.cut_patch:
                existing = self._check_existing(file)
                if self.skip_existing and existing:
                    print(f"Skip existing file {file}")
                    continue

            meta = {}

            if self.format == "npz":
                with open_file(file, "rb") as f:
                    data = np.load(f)["data"]
            elif self.format == "npy":
                with open_file(file, "rb") as f:
                    data = np.load(f)  # (nx, nt)
                meta["begin_time"] = datetime.fromisoformat("1970-01-01 00:00:00")
                meta["dt_s"] = 0.01
                meta["dx_m"] = 10.0
            elif self.format == "h5" and self.system is None:
                with open_file(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        dataset = fp["data"]
                        data = dataset[()]
                        if "begin_time" in dataset.attrs:
                            meta["begin_time"] = datetime.fromisoformat(
                                dataset.attrs["begin_time"].rstrip("Z")
                            )
                        meta["dt_s"] = dataset.attrs.get("dt_s", self.dt)
                        meta["dx_m"] = dataset.attrs.get("dx_m", self.dx)
            elif self.format == "h5" and self.system == "optasense":
                with open_file(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        if "Data" in fp:
                            dataset = fp["Data"]
                            meta["begin_time"] = datetime.fromisoformat(
                                dataset.attrs["startTime"].rstrip("Z")
                            )
                            meta["dt_s"] = dataset.attrs["dt"]
                            meta["dx_m"] = dataset.attrs["dCh"]
                        else:
                            dataset = fp["Acquisition/Raw[0]/RawData"]
                            dx = fp["Acquisition"].attrs["SpatialSamplingInterval"]
                            fs_rate = fp["Acquisition/Raw[0]"].attrs["OutputDataRate"]
                            begin_time = dataset.attrs["PartStartTime"].decode()
                            meta["dx_m"] = dx
                            meta["dt_s"] = 1.0 / fs_rate
                            meta["begin_time"] = datetime.fromisoformat(begin_time.rstrip("Z"))

                        nx, nt = dataset.shape
                        meta["nx"] = nx
                        meta["nt"] = nt

                        existing = self._check_existing(file, meta)
                        if self.skip_existing and existing:
                            print(f"Skip existing file {file}")
                            continue

                        data = dataset[()]
                        data = np.gradient(data, axis=-1, edge_order=2) / meta["dt_s"]
            elif self.format == "segy":
                with open_file(file, "rb") as fs:
                    data = read_PASSCAL_segy(fs)
                meta["begin_time"] = datetime.strptime(
                    file.split("/")[-1].rstrip(".segy"), "%Y%m%d%H"
                )
                meta["dt_s"] = 1.0 / 250.0
                meta["dx_m"] = 8.0
            else:
                raise ValueError(f"Unsupported format: {self.format}")

            # Resample time if needed
            if self.resample_time:
                dt_s = meta.get("dt_s", self.dt)
                if (dt_s != 0.01) and (int(round(1.0 / dt_s)) % 100 == 0):
                    print(f"Resample {file} from time interval {dt_s} to 0.01")
                    data = data[..., :: int(0.01 / dt_s)]
                    meta["dt_s"] = 0.01

            # Preprocessing — data is (nx, nt)
            data = data - np.mean(data, axis=-1, keepdims=True)  # demean along time
            data = data - np.median(data, axis=-2, keepdims=True)  # common-mode rejection
            if self.highpass_filter:
                b, a = scipy.signal.butter(2, self.highpass_filter, "hp", fs=100)
                data = scipy.signal.filtfilt(b, a, data, axis=-1)

            # (nx, nt) -> (1, nx, nt) — no transpose needed
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data.astype(np.float32))

            if not self.cut_patch:
                nx, nt = data.shape[1:]
                data = padding(data, self.min_nx, self.min_nt)

                yield {
                    "data": data,
                    "nt": nt,
                    "nx": nx,
                    "file_name": file,
                    "begin_time": meta["begin_time"].isoformat(timespec="milliseconds"),
                    "begin_time_index": 0,
                    "begin_channel_index": 0,
                    "dt_s": meta.get("dt_s", self.dt),
                    "dx_m": meta.get("dx_m", self.dx),
                }
            else:
                _, nx, nt = data.shape
                for it in range(0, nt, self.nt):
                    for ix in range(0, nx, self.nx):
                        if self.skip_existing:
                            patch_name = (
                                os.path.splitext(file.split("/")[-1])[0] + f"_{it:04d}_{ix:04d}.csv"
                            )
                            if os.path.exists(os.path.join(self.pick_path, patch_name)):
                                print(f"Skip existing file {patch_name}")
                                continue

                        data_patch = data[:, ix:ix + self.nx, it:it + self.nt]
                        _, nx_, nt_ = data_patch.shape
                        data_patch = padding(data_patch, self.min_nx, self.min_nt)

                        yield {
                            "data": data_patch,
                            "nt": nt_,
                            "nx": nx_,
                            "file_name": os.path.splitext(file)[0] + f"_{it:04d}_{ix:04d}",
                            "begin_time": (
                                meta["begin_time"] + timedelta(seconds=it * float(meta["dt_s"]))
                            ).isoformat(timespec="milliseconds"),
                            "begin_time_index": it,
                            "begin_channel_index": ix,
                            "dt_s": meta.get("dt_s", self.dt),
                            "dx_m": meta.get("dx_m", self.dx),
                        }

    def _check_existing(self, file, meta=None):
        """Check if picks already exist for a file."""
        parent_dir = "/".join(file.split("/")[-self.folder_depth:-1])
        existing = True
        if not self.cut_patch:
            path = os.path.join(
                self.pick_path, parent_dir, os.path.splitext(file.split("/")[-1])[0] + ".csv"
            )
            if not os.path.exists(path):
                existing = False
        else:
            nx, nt = meta["nx"], meta["nt"]
            if self.resample_time:
                dt_s = meta.get("dt_s", self.dt)
                if (dt_s != 0.01) and (int(round(1.0 / dt_s)) % 100 == 0):
                    nt = int(nt / round(0.01 / dt_s))
            for i in range(0, nt, self.nt):
                for j in range(0, nx, self.nx):
                    path = os.path.join(
                        self.pick_path, parent_dir,
                        os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv",
                    )
                    if not os.path.exists(path):
                        existing = False

        return existing


class DASDataset(Dataset):
    """DAS Map-style Dataset.

    For use with DataLoader when data fits in memory or is pre-downloaded.

    Args:
        data_path: Path to data files (local or GCS)
        label_path: Path to label CSV files
        noise_list: List of noise files for augmentation
        format: Data format ("h5")
        transforms: Transform pipeline
        label_config: Label generation config
        phases: Phase types
        buffer_size: Size of sample buffer for stacking
        event_feature_scale: Downsampling factor for event labels
    """

    def __init__(
        self,
        data_path: str = "./",
        label_path: str | None = None,
        noise_list: list[str] | None = None,
        format: str = "h5",
        prefix: str = "",
        suffix: str = "",
        transforms: Transform | None = None,
        label_config: LabelConfig = LabelConfig(),
        phases: list[str] = ["P", "S"],
        buffer_size: int = 100,
        event_feature_scale: int = 16,
        training: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.format = format
        self.phases = phases
        self.training = training
        self.label_config = label_config
        self.event_feature_scale = event_feature_scale

        self.transforms = transforms or (default_train_transforms() if training else minimal_transforms())

        # Build file lists
        self.data_list = sorted(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}")))
        self.label_list = []
        if label_path is not None:
            if isinstance(label_path, list):
                for lp in label_path:
                    self.label_list += sorted(glob(os.path.join(lp, f"{prefix}*{suffix}.csv")))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))

        # Noise list
        self.noise_list = noise_list

        # Sample buffer for stacking
        self.sample_buffer = SampleBuffer(buffer_size)
        self._setup_stacking_transforms()

        print(f"DASDataset: {len(self.label_list or self.data_list)} samples")

    def _setup_stacking_transforms(self):
        """Connect stacking transforms to sample buffer."""
        if not isinstance(self.transforms, Compose):
            return
        for t in self.transforms.transforms:
            if isinstance(t, StackEvents):
                t.set_sample_fn(self.sample_buffer.get_random)
            elif isinstance(t, StackNoise):
                t.set_sample_fn(self.sample_buffer.get_random)

    def __len__(self) -> int:
        if self.label_list:
            return len(self.label_list)
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.training and self.label_list:
            return self._get_training_item(idx)
        else:
            return self._get_inference_item(idx)

    def _get_training_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load and process a training sample."""
        label_file = self.label_list[idx]
        picks_df = pd.read_csv(label_file)
        if "channel_index" not in picks_df.columns:
            picks_df = picks_df.rename(columns={"station_id": "channel_index"})

        # Construct data path from label path
        parts = label_file.split("/")
        parts[-2] = "data"
        parts[-1] = parts[-1][:-4] + ".h5"
        data_file = "/".join(parts)

        sample = load_sample_from_h5(data_file, picks_df, self.phases)

        # Calculate SNR
        if sample.targets:
            target = sample.targets[0]
            if target.p_picks:
                target.snr, target.amp_signal, target.amp_noise = calc_snr(
                    sample.waveform, target.p_picks
                )

        # Add to buffer for stacking
        self.sample_buffer.add(sample)

        # Apply transforms
        sample = self.transforms(sample)
        sample.file_name = os.path.splitext(os.path.basename(label_file))[0]
        return _to_output(sample, self.label_config, self.phases, self.event_feature_scale)

    def _get_inference_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load an inference sample."""
        file = self.data_list[idx]
        meta = {}

        if self.format == "h5":
            with h5py.File(file, "r") as f:
                data = f["data"][()]  # (nx, nt) from H5
                data = data[np.newaxis, :, :]  # (1, nx, nt)
                if "begin_time" in f["data"].attrs:
                    meta["begin_time"] = f["data"].attrs["begin_time"]
                meta["dt_s"] = f["data"].attrs.get("dt_s", 0.01)
                meta["dx_m"] = f["data"].attrs.get("dx_m", 10.0)
                data = torch.from_numpy(data.astype(np.float32))
        elif self.format == "npz":
            data = np.load(file)["data"]
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data.astype(np.float32))
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Normalize — data is (1, nx, nt)
        data = data - torch.median(data, dim=-2, keepdim=True)[0]  # common-mode rejection
        std = torch.std(data, dim=-1, keepdim=True)  # per-channel std
        std[std == 0.0] = 1.0
        data = data / std

        return {
            "data": data,
            "file_name": os.path.splitext(os.path.basename(file))[0],
            "height": data.shape[-2],
            "width": data.shape[-1],
            **meta,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_train_dataset(
    data_path: str,
    label_path: str,
    label_list: str | list[str] | None = None,
    noise_list: str | list[str] | None = None,
    nt: int = 3072,
    nx: int = 5120,
    phases: list[str] = ["P", "S"],
    transforms: Transform | None = None,
    enable_stacking: bool = True,
    enable_noise_stacking: bool = True,
    enable_resample_time: bool = False,
    enable_resample_space: bool = False,
    enable_masking: bool = True,
    **kwargs,
) -> DASIterableDataset:
    """Create a DAS training dataset with default augmentation."""
    if transforms is None:
        transforms = default_train_transforms(
            nt=nt, nx=nx,
            enable_stacking=enable_stacking,
            enable_noise_stacking=enable_noise_stacking,
            enable_resample_time=enable_resample_time,
            enable_resample_space=enable_resample_space,
            enable_masking=enable_masking,
        )

    return DASIterableDataset(
        data_path=data_path,
        label_path=label_path,
        label_list=label_list,
        noise_list=noise_list,
        nt=nt,
        nx=nx,
        training=True,
        phases=phases,
        transforms=transforms,
        **kwargs,
    )


def create_eval_dataset(
    data_path: str,
    nt: int = 3072,
    nx: int = 5120,
    format: str = "h5",
    system: str | None = None,
    cut_patch: bool = False,
    highpass_filter: float | None = None,
    **kwargs,
) -> DASIterableDataset:
    """Create a DAS evaluation dataset without augmentation."""
    return DASIterableDataset(
        data_path=data_path,
        format=format,
        nt=nt,
        nx=nx,
        training=False,
        system=system,
        cut_patch=cut_patch,
        highpass_filter=highpass_filter,
        transforms=default_eval_transforms(),
        **kwargs,
    )


# =============================================================================
# Main - Demo and Testing
# =============================================================================

def plot_trace(
    sample: Sample,
    labels: dict[str, np.ndarray],
    ch: int,
    title: str = "",
    save_path: str = "das_trace.png",
):
    """Plot a single DAS channel with labels (4-panel vertical stack).

    Layout:
        [1] Waveform (strain rate) with P/S pick lines
        [2] P/S zoom-in (1s waveform at P and S picks, side by side)
        [3] Phase labels + phase_mask
        [4] Event center + event_time + masks
    """
    import matplotlib.pyplot as plt

    nt = sample.nt
    t = np.arange(nt)
    waveform = sample.waveform[0, ch, :]  # (nt,) — (nch, nx, nt)
    one_sec = int(1.0 / sample.dt_s)

    p_times = [ti for tgt in sample.targets for c, ti in tgt.p_picks if int(c) == ch]
    s_times = [ti for tgt in sample.targets for c, ti in tgt.s_picks if int(c) == ch]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1, 1, 1]})

    # [1] Waveform
    ax = axes[0]
    ax.plot(t, waveform, "k", linewidth=0.5)
    for ti in p_times:
        ax.axvline(ti, color="red", alpha=0.5, linewidth=0.8)
    for ti in s_times:
        ax.axvline(ti, color="blue", alpha=0.5, linewidth=0.8)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, nt)

    # [2] P/S zoom-in (side by side subplots via insets)
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
        inset.plot(t[t0:t1], waveform[t0:t1], "k", linewidth=0.8)
        inset.axvline(pick_t, color=color, linewidth=1.0, alpha=0.8)
        inset.set_title(f"{label} zoom (t={pick_t:.0f})", fontsize=9)
        inset.tick_params(labelsize=7)

    # [3] Phase labels + mask — labels are (label_ch, nx, nt)
    ax = axes[2]
    ax.plot(t, labels["phase_pick"][1, ch, :], label="P", color="red")
    ax.plot(t, labels["phase_pick"][2, ch, :], label="S", color="blue")
    ax.fill_between(t, labels["phase_mask"][0, ch, :], alpha=0.15, color="green", label="mask")
    ax.set_ylabel("Phase Labels")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)

    # [4] Event center + event_time + masks
    ax = axes[3]
    ax.plot(t, labels["event_center"][0, ch, :], label="Event Center", color="purple")
    ax.fill_between(t, labels["event_center_mask"][0, ch, :], alpha=0.1, color="green", label="Center Mask")
    ax.fill_between(t, labels["event_time_mask"][0, ch, :], alpha=0.2, color="green", label="Time Mask")
    ax.set_ylabel("Event")
    ax.set_xlabel("Time Sample")
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    event_time_ch = labels["event_time"][0, ch, :]
    mask = labels["event_time_mask"][0, ch, :] > 0
    if mask.any():
        ax2.plot(t[mask], event_time_ch[mask], color="orange", linewidth=1.0, alpha=0.7)
        ax2.set_ylabel("Event Time", color="orange")
        ax2.set_ylim(min(0, event_time_ch[mask].min()), None)
    ax2.tick_params(axis="y", labelcolor="orange")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overview(sample: Sample, labels: dict[str, np.ndarray], title: str = "", save_path: str = "das_overview.png"):
    """Plot a DAS sample with labels in a 2x2 grid.

    Layout:
        [1,1] DAS waveform
        [1,2] DAS waveform + P/S picks + event center picks + event mask
        [2,1] P/S labels + phase mask
        [2,2] Event time + event center + event mask

    Data is (nch, nx, nt). Display as (nt, nx) — time on y-axis, channel on x-axis.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    imshow_kwargs = dict(aspect="auto", interpolation="nearest")

    # Transpose (nx, nt) -> (nt, nx) for display: time on y-axis, channel on x-axis
    waveform_display = sample.waveform[0].T  # (nt, nx)

    # [1,1] DAS waveform
    ax = axes[0, 0]
    vmax = np.percentile(np.abs(waveform_display), 95)
    ax.imshow(waveform_display, cmap="seismic", vmin=-vmax, vmax=vmax, **imshow_kwargs)
    ax.set_title("DAS Waveform")
    ax.set_ylabel("Time Sample")

    # [1,2] DAS waveform + P/S picks — scatter(x=ch, y=t) on (nt, nx) display
    ax = axes[0, 1]
    ax.imshow(waveform_display, cmap="seismic", vmin=-vmax, vmax=vmax, **imshow_kwargs)
    all_p = [(ch, ti) for tgt in sample.targets for ch, ti in tgt.p_picks]
    all_s = [(ch, ti) for tgt in sample.targets for ch, ti in tgt.s_picks]
    if all_p:
        p_ch, p_t = zip(*all_p)
        ax.scatter(p_ch, p_t, c="red", s=0.3, alpha=0.5, label="P")
    if all_s:
        s_ch, s_t = zip(*all_s)
        ax.scatter(s_ch, s_t, c="blue", s=0.3, alpha=0.5, label="S")
    ax.legend(loc="upper right", fontsize=8, markerscale=10)
    ax.set_title("Waveform + Picks")

    # [2,1] P/S phase labels + phase mask — transpose for display
    ax = axes[1, 0]
    mask = labels["phase_mask"][0].T  # (nt, nx)
    p_disp = labels["phase_pick"][1].T  # (nt, nx)
    s_disp = labels["phase_pick"][2].T  # (nt, nx)
    rgb = np.ones((*mask.shape, 3))  # white background
    rgb[:, :, 1] = np.clip(1.0 - p_disp * 0.7, 0, 1)  # P -> red
    rgb[:, :, 2] = np.clip(1.0 - p_disp * 0.7, 0, 1)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] - s_disp * 0.7, 0, 1)  # S -> blue
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] - s_disp * 0.7, 0, 1)
    rgb[:, :, 0] = np.where(mask > 0, rgb[:, :, 0] * 0.85, rgb[:, :, 0])
    rgb[:, :, 2] = np.where(mask > 0, rgb[:, :, 2] * 0.85, rgb[:, :, 2])
    ax.imshow(rgb, **imshow_kwargs)
    # Overlay event center as yellow
    ec_disp = labels["event_center"][0].T  # (nt, nx)
    event_rgba = np.zeros((*ec_disp.shape, 4))
    event_rgba[:, :, 0] = 1.0
    event_rgba[:, :, 1] = 1.0
    event_rgba[:, :, 3] = ec_disp * 0.8
    ax.imshow(event_rgba, **imshow_kwargs)
    ax.set_title("Phase Labels + Event Center")
    ax.set_ylabel("Time Sample")
    ax.set_xlabel("Channel")

    # [2,2] Event time + event center + event mask — transpose for display
    ax = axes[1, 1]
    center_mask_disp = labels["event_center_mask"][0].T  # (nt, nx)
    time_mask_disp = labels["event_time_mask"][0].T  # (nt, nx)
    # Recompute event_time for full range visualization
    vp, vs = 6.0, 6.0 / 1.73
    nx_plot, nt_plot = sample.nx, sample.nt
    t = np.arange(nt_plot)
    event_time_full = np.zeros((nx_plot, nt_plot), dtype=np.float32)
    ps_dict: dict[int, float] = {}
    for tgt in sample.targets:
        ps_dict.update({int(ch): d for ch, d in tgt.ps_intervals})
    for tgt in sample.targets:
        for ch_c, center in tgt.event_centers:
            ch = int(ch_c)
            if 0 <= ch < nx_plot:
                ps_int = ps_dict.get(ch, 0.0)
                ps_seconds = ps_int * sample.dt_s
                distance = ps_seconds * vp * vs / (vp - vs)
                shift = distance * (1 / vp + 1 / vs) / (2 * sample.dt_s)
                event_time_full[ch, :] = (t - center) + shift
    event_time_disp = event_time_full.T  # (nt, nx)
    # Green background for center mask
    bg = np.ones((nt_plot, nx_plot, 3))
    bg[:, :, 0] = np.where(center_mask_disp > 0, 0.85, 1.0)
    bg[:, :, 2] = np.where(center_mask_disp > 0, 0.85, 1.0)
    ax.imshow(bg, **imshow_kwargs)
    # Event time heatmap
    event_time_display = np.where(center_mask_disp > 0, event_time_disp, np.nan)
    vabs = np.nanmax(np.abs(event_time_display)) or 1.0
    im = ax.imshow(event_time_display, cmap="seismic", vmin=-vabs, vmax=vabs, **imshow_kwargs)
    # Time mask overlay
    mask_rgba = np.zeros((nt_plot, nx_plot, 4))
    mask_rgba[:, :, 1] = 1.0
    mask_rgba[:, :, 3] = time_mask_disp * 0.3
    ax.imshow(mask_rgba, **imshow_kwargs)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Event Time + Center")
    ax.set_xlabel("Channel")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {save_path}")
    plt.close(fig)


def plot_demo(
    sample: Sample,
    transforms: Compose,
    output_dir: str = "figures",
    event_id: str = "das_event",
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

    # Raw overview
    print(f"\nRaw data: {sample.waveform.shape}")
    raw_labels = generate_labels(sample, config)
    plot_overview(sample, raw_labels, title=event_id, save_path=os.path.join(event_dir, "overview.png"))

    # Individual trace details (channels with picks, evenly sampled)
    all_p_chs = {int(c) for tgt in sample.targets for c, _ in tgt.p_picks}
    all_s_chs = {int(c) for tgt in sample.targets for c, _ in tgt.s_picks}
    labeled_chs = sorted(all_p_chs & all_s_chs)
    if labeled_chs:
        n = min(n_traces, len(labeled_chs))
        channels = [labeled_chs[i] for i in np.linspace(0, len(labeled_chs) - 1, n, dtype=int)]
        for j, ch in enumerate(channels):
            plot_trace(
                sample, raw_labels, ch,
                title=f"{event_id} | ch={ch}",
                save_path=os.path.join(trace_dir, f"{j:03d}.png"),
            )
        print(f"  Saved {n} trace plots to {trace_dir}/")

    # N augmented overviews
    print(f"\nGenerating {n_augmented} augmented views...")
    seed_offset = 0
    for i in range(n_augmented):
        for seed in range(seed_offset, seed_offset + 100):
            random.seed(seed)
            np.random.seed(seed)
            transformed = transforms(sample.copy())
            if any(not t.is_empty for t in transformed.targets):
                seed_offset = seed + 1
                break
        aug_labels = generate_labels(transformed, config)
        plot_overview(
            transformed, aug_labels,
            title=f"{event_id} | augmented #{i}",
            save_path=os.path.join(aug_dir, f"{i:03d}.png"),
        )
    print(f"  Saved {n_augmented} augmented overviews to {aug_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAS Dataset Demo")
    parser.add_argument("--data-file", default="debug_data/mammoth_north/data/nc71121689.h5")
    parser.add_argument("--label-file", default="debug_data/mammoth_north/labels/nc71121689.csv")
    parser.add_argument("--nt", type=int, default=3072)
    parser.add_argument("--nx", type=int, default=5120)
    parser.add_argument("--n-augmented", type=int, default=5)
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    print("=" * 60)
    print("DAS Dataset Demo")
    print("=" * 60)

    # Load data
    if os.path.exists(args.label_file):
        picks_df = pd.read_csv(args.label_file)
        print(f"Labels: {args.label_file} ({len(picks_df)} picks)")
    else:
        picks_df = None
        print(f"No label file at {args.label_file}")

    sample = load_sample_from_h5(args.data_file, picks_df)
    print(f"Waveform: {sample.waveform.shape}, targets: {len(sample.targets)}")

    if sample.targets:
        target = sample.targets[0]
        if target.p_picks:
            target.snr, target.amp_signal, target.amp_noise = calc_snr(
                sample.waveform, target.p_picks
            )

    transforms = default_train_transforms(
        nt=args.nt, nx=args.nx, enable_stacking=False, enable_noise_stacking=False,
    )

    event_id = os.path.splitext(os.path.basename(args.data_file))[0]
    plot_demo(sample, transforms, output_dir=args.output_dir, event_id=event_id, n_augmented=args.n_augmented)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
