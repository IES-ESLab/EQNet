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
1. Transform-based augmentation operating on Sample dataclass
2. Compose pattern for chaining transforms
3. Clean separation: loading -> transforms -> label generation
4. Support for both local filesystem and Google Cloud Storage (GCS)

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
# Sample Dataclass
# =============================================================================

@dataclass
class Sample:
    """A DAS sample with waveform and phase annotations.

    This is the core data structure passed through transforms.
    Phase picks are stored as lists of (channel_index, time_index) tuples
    until final label generation.

    Attributes:
        waveform: DAS data array of shape (nt, nx) or (1, nt, nx)
        p_picks: List of (channel_index, time_index) for P-wave picks
        s_picks: List of (channel_index, time_index) for S-wave picks
        event_centers: List of (channel_index, center_time) for events
        ps_intervals: List of (channel_index, S-P interval in samples) for events
        snr: Signal-to-noise ratio
        amp_signal: Signal amplitude
        amp_noise: Noise amplitude
        dt_s: Temporal sampling interval (seconds)
        dx_m: Spatial sampling interval (meters)
        file_name: Source file name
        begin_time: Start time of the waveform
    """
    waveform: np.ndarray  # (nt, nx) or (1, nt, nx)
    p_picks: list[tuple[int, float]] = field(default_factory=list)
    s_picks: list[tuple[int, float]] = field(default_factory=list)
    event_centers: list[tuple[int, float]] = field(default_factory=list)
    ps_intervals: list[tuple[int, float]] = field(default_factory=list)

    # Metadata
    snr: float = 0.0
    amp_signal: float = 0.0
    amp_noise: float = 0.0
    dt_s: float = 0.01
    dx_m: float = 10.0
    file_name: str = ""
    begin_time: datetime | None = None

    @property
    def nt(self) -> int:
        """Number of time samples."""
        return self.waveform.shape[-2] if self.waveform.ndim == 3 else self.waveform.shape[0]

    @property
    def nx(self) -> int:
        """Number of spatial channels."""
        return self.waveform.shape[-1]

    @property
    def nch(self) -> int:
        """Number of data channels (always 1 for DAS)."""
        return 1 if self.waveform.ndim == 2 else self.waveform.shape[0]

    def copy(self) -> "Sample":
        """Create a deep copy of the sample."""
        return Sample(
            waveform=self.waveform.copy(),
            p_picks=self.p_picks.copy(),
            s_picks=self.s_picks.copy(),
            event_centers=self.event_centers.copy(),
            ps_intervals=self.ps_intervals.copy(),
            snr=self.snr,
            amp_signal=self.amp_signal,
            amp_noise=self.amp_noise,
            dt_s=self.dt_s,
            dx_m=self.dx_m,
            file_name=self.file_name,
            begin_time=self.begin_time,
        )

    def ensure_3d(self) -> "Sample":
        """Ensure waveform is 3D (nch, nt, nx)."""
        if self.waveform.ndim == 2:
            self.waveform = self.waveform[np.newaxis, :, :]
        return self

    def to_tensor(self) -> torch.Tensor:
        """Convert waveform to torch tensor."""
        self.ensure_3d()
        return torch.from_numpy(self.waveform.astype(np.float32))


# =============================================================================
# Transforms - Following CV Best Practices (torchvision/albumentations pattern)
# =============================================================================

class Transform(ABC):
    """Base class for all DAS transforms.

    Transforms operate on Sample objects, modifying both waveform and
    phase picks. This is similar to how object detection transforms
    operate on both images and bounding boxes.
    """

    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """Compose multiple transforms together.

    Example:
        >>> transforms = Compose([
        ...     Normalize(),
        ...     RandomCrop(nt=3072, nx=5120),
        ...     FlipLR(p=0.5),
        ... ])
    """

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
    """Identity transform - returns sample unchanged."""

    def __call__(self, sample: Sample) -> Sample:
        return sample


class RandomApply(Transform):
    """Randomly apply a transform with given probability.

    Args:
        transform: Transform to apply
        p: Probability of applying
    """

    def __init__(self, transform: Transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            return self.transform(sample)
        return sample

    def __repr__(self) -> str:
        return f"RandomApply({self.transform}, p={self.p})"


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
        sample.ensure_3d()
        data = np.nan_to_num(sample.waveform)
        data = data - data.mean(axis=-2, keepdims=True)

        if self.mode == "global":
            std = data.std()
        else:
            std = data.std(axis=-2, keepdims=True)

        if np.any(std > self.eps):
            data = data / np.maximum(std, self.eps)

        sample.waveform = data.astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"Normalize(mode='{self.mode}')"


class MedianFilter(Transform):
    """Remove median along spatial axis (common-mode rejection)."""

    def __call__(self, sample: Sample) -> Sample:
        sample.ensure_3d()
        sample.waveform = sample.waveform - np.median(sample.waveform, axis=-1, keepdims=True)
        return sample


class HighpassFilter(Transform):
    """Apply highpass filter to waveform."""

    def __init__(self, freq: float = 1.0, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        from scipy import signal
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.b, self.a = signal.butter(2, freq, "hp", fs=sampling_rate)

    def __call__(self, sample: Sample) -> Sample:
        from scipy import signal
        sample.ensure_3d()
        sample.waveform = signal.filtfilt(self.b, self.a, sample.waveform, axis=-2).astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"HighpassFilter(freq={self.freq})"


# -----------------------------------------------------------------------------
# Spatial-Temporal Transforms
# -----------------------------------------------------------------------------

class RandomCrop(Transform):
    """Randomly crop waveform to fixed size, adjusting phase picks.

    If the data is smaller than the target size, it is padded with reflection.

    Args:
        nt: Target time samples
        nx: Target spatial samples
        min_label_sum: Minimum sum of phase labels to accept crop
        max_tries: Maximum attempts to find valid crop
    """

    def __init__(
        self,
        nt: int = 3072,
        nx: int = 5120,
        min_label_sum: float = 0.0,
        max_tries: int = 100,
    ):
        self.nt = nt
        self.nx = nx
        self.min_label_sum = min_label_sum
        self.max_tries = max_tries

    def _pad_if_needed(self, sample: Sample) -> Sample:
        """Pad waveform with reflection if smaller than target."""
        sample.ensure_3d()
        _, nt_orig, nx_orig = sample.waveform.shape

        if nt_orig >= self.nt and nx_orig >= self.nx:
            return sample

        # Pad using torch reflect mode
        data_tensor = torch.from_numpy(sample.waveform).unsqueeze(0)  # (1, 1, nt, nx)

        pad_nt = max(0, self.nt - nt_orig)
        pad_nx = max(0, self.nx - nx_orig)
        # F.pad: (left, right, top, bottom) for 4D input
        if pad_nt > 0 or pad_nx > 0:
            data_tensor = F.pad(data_tensor, (0, pad_nx, 0, pad_nt), mode="reflect")
            sample.waveform = data_tensor.squeeze(0).numpy().astype(np.float32)

        return sample

    def __call__(self, sample: Sample) -> Sample:
        sample = self._pad_if_needed(sample)
        sample.ensure_3d()
        _, nt_orig, nx_orig = sample.waveform.shape

        if nt_orig <= self.nt and nx_orig <= self.nx:
            return sample

        best_t0, best_x0 = 0, 0
        best_sum = 0

        for _ in range(self.max_tries):
            t0 = random.randint(0, max(0, nt_orig - self.nt))
            x0 = random.randint(0, max(0, nx_orig - self.nx))

            # Count picks in this crop
            pick_sum = sum(
                1 for ch, t in sample.p_picks + sample.s_picks
                if x0 <= ch < x0 + self.nx and t0 <= t < t0 + self.nt
            )

            if pick_sum > best_sum:
                best_sum = pick_sum
                best_t0, best_x0 = t0, x0

            if pick_sum >= self.min_label_sum:
                break

        # Apply crop
        sample.waveform = sample.waveform[:, best_t0:best_t0 + self.nt, best_x0:best_x0 + self.nx].copy()

        # Adjust picks
        sample.p_picks = [
            (ch - best_x0, t - best_t0)
            for ch, t in sample.p_picks
            if best_x0 <= ch < best_x0 + self.nx and best_t0 <= t < best_t0 + self.nt
        ]
        sample.s_picks = [
            (ch - best_x0, t - best_t0)
            for ch, t in sample.s_picks
            if best_x0 <= ch < best_x0 + self.nx and best_t0 <= t < best_t0 + self.nt
        ]
        sample.event_centers = [
            (ch - best_x0, t - best_t0)
            for ch, t in sample.event_centers
            if best_x0 <= ch < best_x0 + self.nx and best_t0 <= t < best_t0 + self.nt
        ]
        sample.ps_intervals = [
            (ch - best_x0, d)
            for ch, d in sample.ps_intervals
            if best_x0 <= ch < best_x0 + self.nx
        ]

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
            sample.ensure_3d()
            nx = sample.nx
            sample.waveform = np.flip(sample.waveform, axis=-1).copy()

            # Adjust channel indices
            sample.p_picks = [(nx - 1 - ch, t) for ch, t in sample.p_picks]
            sample.s_picks = [(nx - 1 - ch, t) for ch, t in sample.s_picks]
            sample.event_centers = [(nx - 1 - ch, t) for ch, t in sample.event_centers]
            sample.ps_intervals = [(nx - 1 - ch, d) for ch, d in sample.ps_intervals]

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
        sample.ensure_3d()
        factor = random.uniform(self.min_factor, self.max_factor)
        if abs(factor - 1.0) < 0.01:
            return sample

        # Resample waveform
        data_tensor = torch.from_numpy(sample.waveform).unsqueeze(0)
        data_resampled = F.interpolate(
            data_tensor, scale_factor=(factor, 1), mode="bilinear", align_corners=False
        ).squeeze(0).numpy()
        sample.waveform = data_resampled.astype(np.float32)

        # Scale time indices
        sample.p_picks = [(ch, t * factor) for ch, t in sample.p_picks]
        sample.s_picks = [(ch, t * factor) for ch, t in sample.s_picks]
        sample.event_centers = [(ch, t * factor) for ch, t in sample.event_centers]

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
        sample.ensure_3d()
        factor = random.uniform(self.min_factor, self.max_factor)
        if abs(factor - 1.0) < 0.01:
            return sample

        # Resample waveform
        data_tensor = torch.from_numpy(sample.waveform)
        data_resampled = F.interpolate(data_tensor, scale_factor=factor, mode="nearest")
        sample.waveform = data_resampled.numpy().astype(np.float32)

        # Scale channel indices
        sample.p_picks = [(int(ch * factor), t) for ch, t in sample.p_picks]
        sample.s_picks = [(int(ch * factor), t) for ch, t in sample.s_picks]
        sample.event_centers = [(int(ch * factor), t) for ch, t in sample.event_centers]
        sample.ps_intervals = [(int(ch * factor), d) for ch, d in sample.ps_intervals]

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

        sample.ensure_3d()
        _, nt, nx = sample.waveform.shape

        mask_nt = random.randint(32, min(self.max_mask_nt, nt // 2))
        t0 = random.randint(0, nt - mask_nt)

        sample.waveform[:, t0:t0 + mask_nt, :] = 0.0

        # Remove picks in masked region
        sample.p_picks = [(ch, t) for ch, t in sample.p_picks if not (t0 <= t < t0 + mask_nt)]
        sample.s_picks = [(ch, t) for ch, t in sample.s_picks if not (t0 <= t < t0 + mask_nt)]

        return sample

    def __repr__(self) -> str:
        return f"Masking(max_mask_nt={self.max_mask_nt}, p={self.p})"


# -----------------------------------------------------------------------------
# Stacking Transforms
# -----------------------------------------------------------------------------

class StackEvents(Transform):
    """Stack multiple events onto a single waveform.

    This is one of the most important augmentations for DAS data.
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

        sample.ensure_3d()
        sample2.ensure_3d()

        if sample.waveform.shape != sample2.waveform.shape:
            return sample

        # Random shift and amplitude scaling
        shift = random.randint(-self.max_shift, self.max_shift)
        scale = 1 + random.random() * 2

        # Stack waveforms
        waveform2 = np.roll(sample2.waveform, shift, axis=-2)
        sample.waveform = sample.waveform * scale + waveform2 * scale

        # Merge picks with shift
        nt = sample.nt
        sample.p_picks += [(ch, (t + shift) % nt) for ch, t in sample2.p_picks]
        sample.s_picks += [(ch, (t + shift) % nt) for ch, t in sample2.s_picks]
        sample.event_centers += [(ch, (t + shift) % nt) for ch, t in sample2.event_centers]
        sample.ps_intervals += sample2.ps_intervals

        return sample

    def __repr__(self) -> str:
        return f"StackEvents(p={self.p}, min_snr={self.min_snr})"


class StackNoise(Transform):
    """Stack noise onto the waveform.

    Args:
        max_ratio: Maximum noise amplitude ratio relative to signal
        p: Probability of applying
    """

    def __init__(self, max_ratio: float = 2.0, p: float = 0.5):
        self.max_ratio = max_ratio
        self.p = p
        self._noise_fn: Callable[[], np.ndarray | None] | None = None

    def set_noise_fn(self, fn: Callable[[], np.ndarray | None]):
        """Set function to get random noise samples."""
        self._noise_fn = fn

    def __call__(self, sample: Sample) -> Sample:
        if random.random() > self.p or self._noise_fn is None:
            return sample

        noise = self._noise_fn()
        if noise is None:
            return sample

        sample.ensure_3d()
        if noise.shape != sample.waveform.shape:
            return sample

        ratio = random.uniform(0, self.max_ratio) * max(0, sample.snr - 2)
        sample.waveform = sample.waveform + noise * ratio

        return sample

    def __repr__(self) -> str:
        return f"StackNoise(max_ratio={self.max_ratio}, p={self.p})"


# =============================================================================
# Label Generation
# =============================================================================

def generate_phase_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
    phases: list[str] = ["P", "S"],
) -> dict[str, np.ndarray]:
    """Generate phase labels from a Sample.

    Args:
        sample: Sample with picks
        config: Label configuration
        phases: Phase types to generate labels for

    Returns:
        Dictionary with phase_pick, phase_mask arrays
    """
    sample.ensure_3d()
    _, nt, nx = sample.waveform.shape

    n_phases = len(phases)
    target = np.zeros([n_phases + 1, nt, nx], dtype=np.float32)
    phase_mask = np.zeros([1, nt, nx], dtype=np.float32)

    picks_by_phase = {"P": sample.p_picks, "S": sample.s_picks}

    sigma = config.phase_width / 6
    mask_width = int(config.phase_width * config.mask_width_factor)
    t = np.arange(nt)

    space_mask = np.zeros((n_phases, nx), dtype=bool)
    picks_per_channel = {}  # ch -> list of (phase_idx, time)

    for i, phase in enumerate(phases):
        picks = picks_by_phase.get(phase, [])
        for ch, phase_time in picks:
            ch = int(ch)
            if 0 <= ch < nx:
                gaussian = np.exp(-((t - phase_time) ** 2) / (2 * sigma ** 2))
                gaussian[gaussian < config.gaussian_threshold] = 0.0
                target[i + 1, :, ch] += gaussian
                space_mask[i, ch] = True
                picks_per_channel.setdefault(ch, []).append(phase_time)

    # Phase mask: entire channel if all phases present, narrow window otherwise
    for ch in picks_per_channel:
        if np.all(space_mask[:, ch]):
            phase_mask[0, :, ch] = 1.0
        else:
            for pt in picks_per_channel[ch]:
                t0 = max(0, int(pt) - mask_width)
                t1 = min(nt, int(pt) + mask_width)
                phase_mask[0, t0:t1, ch] = 1.0

    # Compute noise channel (1 - sum of phases)
    valid_mask = np.all(space_mask, axis=0)  # (nx,) - channels with all phases
    phase_sum = np.sum(target[1:, :, :], axis=0)  # (nt, nx)
    target[0, :, :] = np.maximum(0, 1 - phase_sum)
    target[:, :, ~valid_mask] = 0

    return {
        "phase_pick": target,
        "phase_mask": phase_mask,
    }


def generate_event_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
) -> dict[str, np.ndarray]:
    """Generate event labels from a Sample.

    Args:
        sample: Sample with event centers and ps_intervals
        config: Label configuration (includes vp, vp_vs_ratio)

    Returns:
        Dictionary with:
            event_center: Gaussian peak at event center
            event_time: (t - center) + shift, in samples. The shift is the
                estimated travel time to event_center, computed from ps_interval
                assuming vp and vp/vs ratio. shift = ps_interval * (vp + vs) / (2 * (vp - vs)).
            event_center_mask: Channels with events
            event_time_mask: Time window around event center
    """
    sample.ensure_3d()
    _, nt, nx = sample.waveform.shape

    target_center = np.zeros([1, nt, nx], dtype=np.float32)
    target_time = np.zeros([1, nt, nx], dtype=np.float32)
    center_mask = np.zeros([1, nt, nx], dtype=np.float32)
    time_mask = np.zeros([1, nt, nx], dtype=np.float32)

    ps_dict = dict(sample.ps_intervals) if sample.ps_intervals else {}
    vp = config.vp
    vs = vp / config.vp_vs_ratio
    sigma = config.event_width / 6
    mask_width = int(config.event_width * config.mask_width_factor)
    t = np.arange(nt)

    for ch_c, center in sample.event_centers:
        ch = int(ch_c)
        if 0 <= ch < nx:
            gaussian = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target_center[0, :, ch] += gaussian

            # Estimate travel time to event center from ps_interval
            ps_int = ps_dict.get(ch, 0.0)
            ps_seconds = ps_int * sample.dt_s
            distance = ps_seconds * vp * vs / (vp - vs)
            center_travel = distance * (1 / vp + 1 / vs) / 2  # average of P and S travel times
            shift = center_travel / sample.dt_s  # convert back to samples

            center_mask[0, :, ch] = 1.0

            t0 = max(0, int(center) - mask_width)
            t1 = min(nt, int(center) + mask_width)
            time_mask[0, t0:t1, ch] = 1.0
            target_time[0, t0:t1, ch] = (t[t0:t1] - center) + shift

    return {
        "event_center": target_center,
        "event_time": target_time,
        "event_center_mask": center_mask,
        "event_time_mask": time_mask,
    }


def generate_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
    phases: list[str] = ["P", "S"],
) -> dict[str, np.ndarray]:
    """Generate all labels from a Sample.

    Args:
        sample: Sample with picks and event info
        config: Label configuration
        phases: Phase types

    Returns:
        Combined dictionary of all labels
    """
    labels = generate_phase_labels(sample, config, phases)
    labels.update(generate_event_labels(sample, config))
    return labels


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
) -> Compose:
    """Default transforms for DAS training.

    Args:
        nt: Target time samples
        nx: Target spatial samples
        enable_stacking: Enable event stacking
        enable_noise_stacking: Enable noise stacking
        enable_resample_time: Enable time resampling
        enable_resample_space: Enable space resampling
        enable_masking: Enable masking augmentation
    """
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

    transforms.append(FlipLR(p=0.5))

    if enable_masking:
        transforms.append(Masking(max_mask_nt=256, p=0.2))

    transforms.extend([
        RandomApply(MedianFilter(), p=0.5),
        Normalize(),  # Final normalization
    ])

    return Compose(transforms)


def default_eval_transforms() -> Compose:
    """Default transforms for DAS evaluation."""
    return Compose([Normalize()])


def minimal_transforms() -> Compose:
    """Minimal transforms - just normalize."""
    return Compose([Normalize()])


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
    """Get appropriate filesystem for the given path.

    Args:
        path: Local path or GCS path (gs://...)

    Returns:
        (filesystem, path_without_protocol) tuple
    """
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
        data: Waveform array (nch, nt, nx) or (nt, nx)
        picks: List of (channel_index, time_index) tuples
        noise_window: Number of samples before pick for noise
        signal_window: Number of samples after pick for signal

    Returns:
        (snr, signal_std, noise_std)
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    snrs, signals, noises = [], [], []
    nch, nt, nx = data.shape
    for trace, phase_time in picks:
        trace = int(trace)
        phase_time = int(phase_time)
        if 0 <= trace < nx and noise_window < phase_time < nt - signal_window:
            noise = np.std(data[:, max(0, phase_time - noise_window):phase_time, trace])
            signal = np.std(data[:, phase_time:phase_time + signal_window, trace])
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


def padding(data: torch.Tensor, min_nt: int = 1024, min_nx: int = 1024) -> torch.Tensor:
    """Pad data to multiples of min_nt and min_nx.

    Args:
        data: Tensor of shape (nch, nt, nx)
        min_nt: Minimum time samples (pads to multiple)
        min_nx: Minimum space samples (pads to multiple)

    Returns:
        Padded tensor
    """
    nch, nt, nx = data.shape
    pad_nt = (min_nt - nt % min_nt) % min_nt
    pad_nx = (min_nx - nx % min_nx) % min_nx

    if pad_nt > 0 or pad_nx > 0:
        with torch.no_grad():
            data = F.pad(data, (0, pad_nx, 0, pad_nt), mode="constant")

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
        Sample instance
    """
    with open_file(file_path, "rb") as f:
        with h5py.File(f, "r") as fp:
            data = fp["data"][:, :].T  # (nx, nt) -> (nt, nx)
            dt_s = fp["data"].attrs.get("dt_s", 0.01)
            dx_m = fp["data"].attrs.get("dx_m", 10.0)

    data = data[np.newaxis, :, :]  # (1, nt, nx)
    data = data / (np.std(data) + 1e-10)
    data = data - np.mean(data, axis=1, keepdims=True)

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

    return Sample(
        waveform=data.astype(np.float32),
        p_picks=p_picks,
        s_picks=s_picks,
        event_centers=event_centers,
        ps_intervals=ps_intervals,
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
        """Add a sample to the buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample.copy())
        else:
            idx = random.randint(0, self.count)
            if idx < self.max_size:
                self.buffer[idx] = sample.copy()
        self.count += 1

    def get_random(self) -> Sample | None:
        """Get a random sample from the buffer."""
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
        min_nt=1024,
        min_nx=1024,
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
            elif isinstance(t, StackNoise) and self.noise_list:
                t.set_noise_fn(self._load_random_noise)

    def _load_random_noise(self) -> np.ndarray | None:
        """Load a random noise file for stacking."""
        if not self.noise_list:
            return None
        noise_file = self.noise_list[random.randint(0, len(self.noise_list) - 1)]
        noise_path = self._construct_file_path(self.data_path, noise_file)
        try:
            with open_file(noise_path, "rb") as f:
                with h5py.File(f, "r") as fp:
                    noise = fp["data"][:, :].T  # (nx, nt) -> (nt, nx)
                # Roll to use noise portion
                noise = np.roll(noise, max(0, self.nt - 3000), axis=0)
                noise = noise[np.newaxis, :self.nt, :]  # (1, nt, nx)
                noise = noise / (np.std(noise) + 1e-10)
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
        """Load a training sample from label file.

        Reads CSV labels and corresponding H5 data, computes SNR.

        Args:
            label_file: Relative path to label CSV file

        Returns:
            Sample instance or None if loading fails
        """
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

        # Calculate SNR
        if sample.p_picks:
            sample.snr, sample.amp_signal, sample.amp_noise = calc_snr(
                sample.waveform, sample.p_picks
            )

        sample.file_name = os.path.splitext(label_file.split("/")[-1])[0]
        return sample

    def _to_output(self, sample: Sample) -> dict[str, torch.Tensor]:
        """Convert transformed Sample to output dict with labels.

        Generates labels, converts to tensors, permutes (nt, nx) -> (nx, nt).
        """
        labels = generate_labels(sample, self.label_config, self.phases)

        # Convert to tensors
        data = torch.from_numpy(sample.waveform).float()
        phase_pick = torch.from_numpy(labels["phase_pick"]).float()
        phase_mask = torch.from_numpy(labels["phase_mask"]).float()
        event_center = torch.from_numpy(labels["event_center"]).float()
        event_time = torch.from_numpy(labels["event_time"]).float()
        event_center_mask = torch.from_numpy(labels["event_center_mask"]).float()
        event_time_mask = torch.from_numpy(labels["event_time_mask"]).float()

        # Permute (nch, nt, nx) -> (nch, nx, nt) for the model
        data = data.permute(0, 2, 1)
        phase_pick = phase_pick.permute(0, 2, 1)
        phase_mask = phase_mask.permute(0, 2, 1)
        event_center = event_center.permute(0, 2, 1)
        event_time = event_time.permute(0, 2, 1)
        event_center_mask = event_center_mask.permute(0, 2, 1)
        event_time_mask = event_time_mask.permute(0, 2, 1)

        # Downsample event labels along spatial dimension
        s = self.event_feature_scale
        event_center = event_center[:, :, ::s]
        event_time = event_time[:, :, ::s]
        event_center_mask = event_center_mask[:, :, ::s]
        event_time_mask = event_time_mask[:, :, ::s]

        return {
            "data": torch.nan_to_num(data),
            "phase_pick": phase_pick,
            "phase_mask": phase_mask,
            "event_center": event_center,
            "event_time": event_time,
            "event_time_mask": event_time_mask,
            "event_center_mask": event_center_mask,
            "file_name": sample.file_name,
            "height": data.shape[-2],
            "width": data.shape[-1],
            "dt_s": sample.dt_s,
            "dx_m": sample.dx_m,
        }

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
                    output = self._to_output(s)
                    output["file_name"] = sample.file_name + f"_{ii:02d}"
                    yield output

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

            # Preprocessing
            data = data - np.mean(data, axis=-1, keepdims=True)  # (nx, nt)
            data = data - np.median(data, axis=-2, keepdims=True)
            if self.highpass_filter:
                b, a = scipy.signal.butter(2, self.highpass_filter, "hp", fs=100)
                data = scipy.signal.filtfilt(b, a, data, axis=-1)

            data = data.T  # (nx, nt) -> (nt, nx)
            data = data[np.newaxis, :, :]  # (1, nt, nx)
            data = torch.from_numpy(data.astype(np.float32))

            if not self.cut_patch:
                nt, nx = data.shape[1:]
                data = padding(data, self.min_nt, self.min_nx)
                data = data.permute(0, 2, 1)  # (1, nt, nx) -> (1, nx, nt)

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
                _, nt, nx = data.shape
                for i in range(0, nt, self.nt):
                    for j in range(0, nx, self.nx):
                        if self.skip_existing:
                            patch_name = (
                                os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv"
                            )
                            if os.path.exists(os.path.join(self.pick_path, patch_name)):
                                print(f"Skip existing file {patch_name}")
                                continue

                        data_patch = data[:, i:i + self.nt, j:j + self.nx]
                        _, nt_, nx_ = data_patch.shape
                        data_patch = padding(data_patch, self.min_nt, self.min_nx)
                        data_patch = data_patch.permute(0, 2, 1)

                        yield {
                            "data": data_patch,
                            "nt": nt_,
                            "nx": nx_,
                            "file_name": os.path.splitext(file)[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (
                                meta["begin_time"] + timedelta(seconds=i * float(meta["dt_s"]))
                            ).isoformat(timespec="milliseconds"),
                            "begin_time_index": i,
                            "begin_channel_index": j,
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
        if sample.p_picks:
            sample.snr, sample.amp_signal, sample.amp_noise = calc_snr(
                sample.waveform, sample.p_picks
            )

        # Add to buffer for stacking
        self.sample_buffer.add(sample)

        # Apply transforms
        sample = self.transforms(sample)

        # Generate labels and convert to output
        labels = generate_labels(sample, self.label_config, self.phases)

        data = torch.from_numpy(sample.waveform).float()
        phase_pick = torch.from_numpy(labels["phase_pick"]).float()
        phase_mask = torch.from_numpy(labels["phase_mask"]).float()
        event_center = torch.from_numpy(labels["event_center"]).float()
        event_time = torch.from_numpy(labels["event_time"]).float()
        event_center_mask = torch.from_numpy(labels["event_center_mask"]).float()
        event_time_mask = torch.from_numpy(labels["event_time_mask"]).float()

        # Permute (nch, nt, nx) -> (nch, nx, nt)
        data = data.permute(0, 2, 1)
        phase_pick = phase_pick.permute(0, 2, 1)
        phase_mask = phase_mask.permute(0, 2, 1)
        event_center = event_center.permute(0, 2, 1)
        event_time = event_time.permute(0, 2, 1)
        event_center_mask = event_center_mask.permute(0, 2, 1)
        event_time_mask = event_time_mask.permute(0, 2, 1)

        # Downsample event labels
        s = self.event_feature_scale
        event_center = event_center[:, :, ::s]
        event_time = event_time[:, :, ::s]
        event_center_mask = event_center_mask[:, :, ::s]
        event_time_mask = event_time_mask[:, :, ::s]

        return {
            "data": torch.nan_to_num(data),
            "phase_pick": phase_pick,
            "phase_mask": phase_mask,
            "event_center": event_center,
            "event_time": event_time,
            "event_time_mask": event_time_mask,
            "event_center_mask": event_center_mask,
            "file_name": os.path.splitext(os.path.basename(label_file))[0],
            "height": data.shape[-2],
            "width": data.shape[-1],
        }

    def _get_inference_item(self, idx: int) -> dict[str, torch.Tensor]:
        """Load an inference sample."""
        file = self.data_list[idx]
        meta = {}

        if self.format == "h5":
            with h5py.File(file, "r") as f:
                data = f["data"][()]
                data = data[np.newaxis, :, :]  # (1, nt, nx)
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

        # Normalize
        data = data - torch.median(data, dim=2, keepdim=True)[0]
        std = torch.std(data, dim=1, keepdim=True)
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
    """Create a DAS training dataset with default augmentation.

    Args:
        data_path: Path to data files (local or GCS)
        label_path: Path to label files (local or GCS)
        label_list: List of label files or path to file containing list
        noise_list: List of noise files for augmentation
        nt: Target time samples
        nx: Target spatial samples
        phases: Phase types to use
        transforms: Custom transform pipeline (overrides enable_* flags)
        enable_stacking: Enable event stacking
        enable_noise_stacking: Enable noise stacking
        enable_resample_time: Enable time resampling
        enable_resample_space: Enable space resampling
        enable_masking: Enable masking augmentation
        **kwargs: Additional arguments for DASIterableDataset

    Returns:
        DASIterableDataset instance
    """
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
    """Create a DAS evaluation dataset without augmentation.

    Args:
        data_path: Path to data files (local or GCS)
        nt: Target time samples
        nx: Target spatial samples
        format: Data format (h5, npz, npy, segy)
        system: DAS system type (optasense or None)
        cut_patch: Whether to cut into patches
        highpass_filter: Highpass filter frequency
        **kwargs: Additional arguments

    Returns:
        DASIterableDataset instance
    """
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

    nt = sample.waveform.shape[-2]
    t = np.arange(nt)
    waveform = sample.waveform[0, :, ch]
    one_sec = int(1.0 / sample.dt_s)

    p_times = [ti for c, ti in sample.p_picks if int(c) == ch]
    s_times = [ti for c, ti in sample.s_picks if int(c) == ch]

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

    # [3] Phase labels + mask
    ax = axes[2]
    ax.plot(t, labels["phase_pick"][1, :, ch], label="P", color="red")
    ax.plot(t, labels["phase_pick"][2, :, ch], label="S", color="blue")
    ax.fill_between(t, labels["phase_mask"][0, :, ch], alpha=0.15, color="green", label="mask")
    ax.set_ylabel("Phase Labels")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)

    # [4] Event center + event_time + masks
    ax = axes[3]
    ax.plot(t, labels["event_center"][0, :, ch], label="Event Center", color="purple")
    ax.fill_between(t, labels["event_center_mask"][0, :, ch], alpha=0.1, color="green", label="Center Mask")
    ax.fill_between(t, labels["event_time_mask"][0, :, ch], alpha=0.2, color="green", label="Time Mask")
    ax.set_ylabel("Event")
    ax.set_xlabel("Time Sample")
    ax.set_xlim(0, nt)
    ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx()
    event_time_ch = labels["event_time"][0, :, ch]
    mask = labels["event_time_mask"][0, :, ch] > 0
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
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    imshow_kwargs = dict(aspect="auto", interpolation="nearest")

    # Data is (nch, nt, nx). Show as (nt, nx) — time on y-axis, channel on x-axis.
    # [1,1] DAS waveform
    ax = axes[0, 0]
    vmax = np.percentile(np.abs(sample.waveform[0]), 95)
    ax.imshow(sample.waveform[0], cmap="seismic", vmin=-vmax, vmax=vmax, **imshow_kwargs)
    ax.set_title("DAS Waveform")
    ax.set_ylabel("Time Sample")

    # [1,2] DAS waveform + P/S picks
    ax = axes[0, 1]
    ax.imshow(sample.waveform[0], cmap="seismic", vmin=-vmax, vmax=vmax, **imshow_kwargs)
    if sample.p_picks:
        p_ch, p_t = zip(*sample.p_picks)
        ax.scatter(p_ch, p_t, c="red", s=0.3, alpha=0.5, label="P")
    if sample.s_picks:
        s_ch, s_t = zip(*sample.s_picks)
        ax.scatter(s_ch, s_t, c="blue", s=0.3, alpha=0.5, label="S")
    if sample.event_centers:
        e_ch, e_t = zip(*sample.event_centers)
        ax.scatter(e_ch, e_t, c="yellow", s=0.3, alpha=0.5, label="Event")
    ax.legend(loc="upper right", fontsize=8, markerscale=10)
    ax.set_title("Waveform + Picks")

    # [2,1] P/S phase labels + phase mask
    ax = axes[1, 0]
    mask = labels["phase_mask"][0]  # (nt, nx)
    rgb = np.ones((*mask.shape, 3))  # white background
    rgb[:, :, 1] = np.clip(1.0 - labels["phase_pick"][1] * 0.7, 0, 1)  # P -> red (reduce G)
    rgb[:, :, 2] = np.clip(1.0 - labels["phase_pick"][1] * 0.7, 0, 1)  # P -> red (reduce B)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] - labels["phase_pick"][2] * 0.7, 0, 1)  # S -> blue (reduce R)
    rgb[:, :, 1] = np.clip(rgb[:, :, 1] - labels["phase_pick"][2] * 0.7, 0, 1)  # S -> blue (reduce G)
    # Tint mask region green (darken non-green channels slightly)
    rgb[:, :, 0] = np.where(mask > 0, rgb[:, :, 0] * 0.85, rgb[:, :, 0])
    rgb[:, :, 2] = np.where(mask > 0, rgb[:, :, 2] * 0.85, rgb[:, :, 2])
    ax.imshow(rgb, **imshow_kwargs)
    # Overlay event center Gaussian as yellow
    event_center = labels["event_center"][0]
    event_rgba = np.zeros((*event_center.shape, 4))
    event_rgba[:, :, 0] = 1.0  # yellow R
    event_rgba[:, :, 1] = 1.0  # yellow G
    event_rgba[:, :, 3] = event_center * 0.8  # alpha from Gaussian
    ax.imshow(event_rgba, **imshow_kwargs)
    ax.set_title("Phase Labels + Event Center")
    ax.set_ylabel("Time Sample")
    ax.set_xlabel("Channel")

    # [2,2] Event time + event center + event mask
    ax = axes[1, 1]
    center_mask = labels["event_center_mask"][0]
    time_mask = labels["event_time_mask"][0]
    # FIXME: Recompute event_time for full time range for visualization only
    vp, vs = 6.0, 6.0 / 1.73
    ps_dict = dict(sample.ps_intervals)
    _, nt_plot, nx_plot = sample.waveform.shape
    t = np.arange(nt_plot)
    event_time_full = np.zeros((nt_plot, nx_plot), dtype=np.float32)
    for ch_c, center in sample.event_centers:
        ch = int(ch_c)
        if 0 <= ch < nx_plot:
            ps_int = ps_dict.get(ch, 0.0)
            ps_seconds = ps_int * sample.dt_s
            distance = ps_seconds * vp * vs / (vp - vs)
            shift = distance * (1 / vp + 1 / vs) / (2 * sample.dt_s)
            event_time_full[:, ch] = (t - center) + shift
    # Event mask as green background (same tinting as [2,1])
    bg = np.ones((nt_plot, nx_plot, 3))
    bg[:, :, 0] = np.where(center_mask > 0, 0.85, 1.0)
    bg[:, :, 2] = np.where(center_mask > 0, 0.85, 1.0)
    ax.imshow(bg, **imshow_kwargs)
    # Event time heatmap (centered at 0: white=0)
    event_time_display = np.where(center_mask > 0, event_time_full, np.nan)
    vabs = np.nanmax(np.abs(event_time_display)) or 1.0
    im = ax.imshow(event_time_display, cmap="seismic", vmin=-vabs, vmax=vabs, **imshow_kwargs)
    # Overlay event_time_mask as green tint
    mask_rgba = np.zeros((nt_plot, nx_plot, 4))
    mask_rgba[:, :, 1] = 1.0  # green
    mask_rgba[:, :, 3] = time_mask * 0.3  # alpha from mask
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
    """Generate demo plots: overview, individual traces, and augmented overviews.

    Output structure:
        {output_dir}/{event_id}/overview.png           # raw data overview
        {output_dir}/{event_id}/traces/000.png         # individual channel details
        {output_dir}/{event_id}/augmented/000.png      # augmented overview #0
        ...
    """
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
    labeled_chs = sorted({int(c) for c, _ in sample.p_picks} & {int(c) for c, _ in sample.s_picks})
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

    # N augmented overviews (different seeds)
    print(f"\nGenerating {n_augmented} augmented views...")
    seed_offset = 0
    for i in range(n_augmented):
        for seed in range(seed_offset, seed_offset + 100):
            random.seed(seed)
            np.random.seed(seed)
            transformed = transforms(sample.copy())
            if transformed.p_picks and transformed.s_picks:
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
    print(f"Waveform: {sample.waveform.shape}, P: {len(sample.p_picks)}, S: {len(sample.s_picks)}")

    if sample.p_picks:
        sample.snr, sample.amp_signal, sample.amp_noise = calc_snr(sample.waveform, sample.p_picks)

    transforms = default_train_transforms(
        nt=args.nt, nx=args.nx, enable_stacking=False, enable_noise_stacking=False,
    )

    event_id = os.path.splitext(os.path.basename(args.data_file))[0]
    plot_demo(sample, transforms, output_dir=args.output_dir, event_id=event_id, n_augmented=args.n_augmented)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)