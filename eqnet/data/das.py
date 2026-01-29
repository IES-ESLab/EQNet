"""
DAS (Distributed Acoustic Sensing) Data Loading Module

A modern, efficient dataset implementation for DAS phase picking following
best practices from computer vision (torchvision, timm, albumentations).

Design Principles:
1. Transform-based augmentation operating on DASSample dataclass
2. Compose pattern for chaining transforms
3. Clean separation: loading -> transforms -> label generation
4. Support for both local filesystem and Google Cloud Storage (GCS)

Example:
    >>> from eqnet.data.das import DASSample, default_train_transforms, DASIterableDataset
    >>> dataset = DASIterableDataset(
    ...     data_path="gs://quakeflow_das/ridgecrest_north",
    ...     label_path="gs://quakeflow_das/ridgecrest_north",
    ...     training=True,
    ...     transforms=default_train_transforms(),
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
from typing import Any, Callable, Sequence

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
# GCS Configuration
# =============================================================================

BUCKET_DAS = "gs://quakeflow_das"
GCS_CREDENTIALS_PATH = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
DEFAULT_SAMPLING_RATE = 100  # Hz
DEFAULT_SPATIAL_INTERVAL = 10.0  # meters


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DASLabelConfig:
    """Configuration for DAS label generation."""
    phase_width: int = 150  # Gaussian width for phase picks (samples)
    event_width: int = 150  # Gaussian width for event center
    mask_width_factor: float = 1.5  # mask_width = width * factor
    gaussian_threshold: float = 0.1  # Values below this are zeroed


# =============================================================================
# DASSample Dataclass
# =============================================================================

@dataclass
class DASSample:
    """A DAS sample with waveform and phase annotations.

    This is the core data structure passed through transforms.
    Phase picks are stored as lists of (channel_index, time_index) tuples
    until final label generation.

    Attributes:
        waveform: DAS data array of shape (nt, nx) or (1, nt, nx)
        p_picks: List of (channel_index, time_index) for P-wave picks
        s_picks: List of (channel_index, time_index) for S-wave picks
        event_centers: List of (channel_index, center_time) for events
        event_durations: List of (channel_index, duration) for events
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
    event_durations: list[tuple[int, float]] = field(default_factory=list)

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

    def copy(self) -> "DASSample":
        """Create a deep copy of the sample."""
        return DASSample(
            waveform=self.waveform.copy(),
            p_picks=self.p_picks.copy(),
            s_picks=self.s_picks.copy(),
            event_centers=self.event_centers.copy(),
            event_durations=self.event_durations.copy(),
            snr=self.snr,
            amp_signal=self.amp_signal,
            amp_noise=self.amp_noise,
            dt_s=self.dt_s,
            dx_m=self.dx_m,
            file_name=self.file_name,
            begin_time=self.begin_time,
        )

    def ensure_3d(self) -> "DASSample":
        """Ensure waveform is 3D (nch, nt, nx)."""
        if self.waveform.ndim == 2:
            self.waveform = self.waveform[np.newaxis, :, :]
        return self

    def to_tensor(self) -> torch.Tensor:
        """Convert waveform to torch tensor."""
        self.ensure_3d()
        return torch.from_numpy(self.waveform.astype(np.float32))


# =============================================================================
# Transform Base Classes - Following CV Best Practices
# =============================================================================

class DASTransform(ABC):
    """Base class for all DAS transforms.

    Transforms operate on DASSample objects, modifying both waveform and
    phase picks. This is similar to how object detection transforms
    operate on both images and bounding boxes.
    """

    @abstractmethod
    def __call__(self, sample: DASSample) -> DASSample:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DASCompose(DASTransform):
    """Compose multiple transforms together.

    Example:
        >>> transforms = DASCompose([
        ...     DASNormalize(),
        ...     DASRandomCrop(nt=3072, nx=5120),
        ...     DASFlipLR(p=0.5),
        ... ])
    """

    def __init__(self, transforms: Sequence[DASTransform]):
        self.transforms = list(transforms)

    def __call__(self, sample: DASSample) -> DASSample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


class DASIdentity(DASTransform):
    """Identity transform - returns sample unchanged."""

    def __call__(self, sample: DASSample) -> DASSample:
        return sample


# =============================================================================
# Basic Waveform Transforms
# =============================================================================

class DASNormalize(DASTransform):
    """Normalize DAS waveform.

    Args:
        mode: "global" for global normalization, "channel" for per-channel
        eps: Small value for numerical stability
    """

    def __init__(self, mode: str = "global", eps: float = 1e-10):
        self.mode = mode
        self.eps = eps

    def __call__(self, sample: DASSample) -> DASSample:
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
        return f"DASNormalize(mode='{self.mode}')"


class DASMedianFilter(DASTransform):
    """Remove median along spatial axis (common-mode rejection)."""

    def __call__(self, sample: DASSample) -> DASSample:
        sample.ensure_3d()
        sample.waveform = sample.waveform - np.median(sample.waveform, axis=-1, keepdims=True)
        return sample


class DASHighpassFilter(DASTransform):
    """Apply highpass filter to waveform."""

    def __init__(self, freq: float = 1.0, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        from scipy import signal
        self.freq = freq
        self.sampling_rate = sampling_rate
        self.b, self.a = signal.butter(2, freq, "hp", fs=sampling_rate)

    def __call__(self, sample: DASSample) -> DASSample:
        from scipy import signal
        sample.ensure_3d()
        sample.waveform = signal.filtfilt(self.b, self.a, sample.waveform, axis=-2).astype(np.float32)
        return sample

    def __repr__(self) -> str:
        return f"DASHighpassFilter(freq={self.freq})"


# =============================================================================
# Spatial-Temporal Transforms
# =============================================================================

class DASRandomCrop(DASTransform):
    """Randomly crop waveform to fixed size, adjusting phase picks.

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

    def __call__(self, sample: DASSample) -> DASSample:
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
        sample.event_durations = [
            (ch - best_x0, d)
            for ch, d in sample.event_durations
            if best_x0 <= ch < best_x0 + self.nx
        ]

        return sample

    def __repr__(self) -> str:
        return f"DASRandomCrop(nt={self.nt}, nx={self.nx})"


class DASFlipLR(DASTransform):
    """Randomly flip waveform horizontally (spatial axis).

    Args:
        p: Probability of flipping
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: DASSample) -> DASSample:
        if random.random() < self.p:
            sample.ensure_3d()
            nx = sample.nx
            sample.waveform = np.flip(sample.waveform, axis=-1).copy()

            # Adjust channel indices
            sample.p_picks = [(nx - 1 - ch, t) for ch, t in sample.p_picks]
            sample.s_picks = [(nx - 1 - ch, t) for ch, t in sample.s_picks]
            sample.event_centers = [(nx - 1 - ch, t) for ch, t in sample.event_centers]
            sample.event_durations = [(nx - 1 - ch, d) for ch, d in sample.event_durations]

        return sample

    def __repr__(self) -> str:
        return f"DASFlipLR(p={self.p})"


class DASResampleTime(DASTransform):
    """Resample time axis by a random factor.

    Args:
        min_factor: Minimum scale factor
        max_factor: Maximum scale factor
    """

    def __init__(self, min_factor: float = 0.5, max_factor: float = 3.0):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: DASSample) -> DASSample:
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
        return f"DASResampleTime(min_factor={self.min_factor}, max_factor={self.max_factor})"


class DASResampleSpace(DASTransform):
    """Resample spatial axis by a random factor.

    Args:
        min_factor: Minimum scale factor
        max_factor: Maximum scale factor
    """

    def __init__(self, min_factor: float = 0.5, max_factor: float = 5.0):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, sample: DASSample) -> DASSample:
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
        sample.event_durations = [(int(ch * factor), d) for ch, d in sample.event_durations]

        return sample

    def __repr__(self) -> str:
        return f"DASResampleSpace(min_factor={self.min_factor}, max_factor={self.max_factor})"


# =============================================================================
# Masking Transforms
# =============================================================================

class DASMasking(DASTransform):
    """Randomly mask a time window in the waveform.

    Args:
        max_mask_nt: Maximum mask width in time samples
        p: Probability of applying mask
    """

    def __init__(self, max_mask_nt: int = 256, p: float = 0.2):
        self.max_mask_nt = max_mask_nt
        self.p = p

    def __call__(self, sample: DASSample) -> DASSample:
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
        return f"DASMasking(max_mask_nt={self.max_mask_nt}, p={self.p})"


# =============================================================================
# Stacking Transforms
# =============================================================================

class DASStackEvents(DASTransform):
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
        self._sample_fn: Callable[[], DASSample | None] | None = None

    def set_sample_fn(self, fn: Callable[[], DASSample | None]):
        """Set function to get random samples for stacking."""
        self._sample_fn = fn

    def __call__(self, sample: DASSample) -> DASSample:
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
        sample.event_durations += sample2.event_durations

        return sample

    def __repr__(self) -> str:
        return f"DASStackEvents(p={self.p}, min_snr={self.min_snr})"


class DASStackNoise(DASTransform):
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

    def __call__(self, sample: DASSample) -> DASSample:
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
        return f"DASStackNoise(max_ratio={self.max_ratio}, p={self.p})"


# =============================================================================
# Label Generation
# =============================================================================

def generate_das_phase_labels(
    sample: DASSample,
    config: DASLabelConfig = DASLabelConfig(),
    phases: list[str] = ["P", "S"],
) -> dict[str, np.ndarray]:
    """Generate phase labels from a DASSample.

    Args:
        sample: DASSample with picks
        config: Label configuration
        phases: Phase types to generate labels for

    Returns:
        Dictionary with phase_pick, phase_mask arrays
    """
    sample.ensure_3d()
    _, nt, nx = sample.waveform.shape

    # Create label arrays
    n_phases = len(phases)
    target = np.zeros([n_phases + 1, nt, nx], dtype=np.float32)
    phase_mask = np.zeros([1, nt, nx], dtype=np.float32)

    # Get picks for each phase
    picks_by_phase = {"P": sample.p_picks, "S": sample.s_picks}

    sigma = config.phase_width / 6
    t = np.arange(nt)

    # Track which channels have all phases
    space_mask = np.zeros((n_phases, nx), dtype=bool)

    for i, phase in enumerate(phases):
        picks = picks_by_phase.get(phase, [])
        for ch, phase_time in picks:
            ch = int(ch)
            if 0 <= ch < nx:
                gaussian = np.exp(-((t - phase_time) ** 2) / (2 * sigma ** 2))
                gaussian[gaussian < config.gaussian_threshold] = 0.0
                target[i + 1, :, ch] += gaussian
                space_mask[i, ch] = True
                phase_mask[0, :, ch] = 1.0

    # Compute noise channel (1 - sum of phases)
    valid_mask = np.all(space_mask, axis=0)  # (nx,) - channels with all phases
    # Sum phase labels along phase axis (axis=0 of target[1:])
    phase_sum = np.sum(target[1:, :, :], axis=0)  # (nt, nx)
    target[0, :, :] = np.maximum(0, 1 - phase_sum)
    # Zero out channels without all phases
    target[:, :, ~valid_mask] = 0

    return {
        "phase_pick": target,
        "phase_mask": phase_mask,
    }


def generate_das_event_labels(
    sample: DASSample,
    config: DASLabelConfig = DASLabelConfig(),
) -> dict[str, np.ndarray]:
    """Generate event labels from a DASSample.

    Args:
        sample: DASSample with event centers/durations
        config: Label configuration

    Returns:
        Dictionary with event_center, event_time, event_center_mask, event_time_mask
    """
    sample.ensure_3d()
    _, nt, nx = sample.waveform.shape

    target_center = np.zeros([1, nt, nx], dtype=np.float32)
    target_time = np.zeros([1, nt, nx], dtype=np.float32)
    center_mask = np.zeros([1, nt, nx], dtype=np.float32)
    time_mask = np.zeros([1, nt, nx], dtype=np.float32)

    sigma = config.event_width / 6
    mask_width = int(config.event_width * config.mask_width_factor)
    t = np.arange(nt)

    for (ch_c, center), (ch_d, duration) in zip(sample.event_centers, sample.event_durations):
        ch = int(ch_c)
        if 0 <= ch < nx:
            gaussian = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
            gaussian[gaussian < 0.05] = 0.0
            target_center[0, :, ch] += gaussian
            target_time[0, :, ch] = t - center + duration
            center_mask[0, :, ch] = 1.0

            t0 = max(0, int(center) - mask_width)
            t1 = min(nt, int(center) + mask_width)
            time_mask[0, t0:t1, ch] = 1.0

    return {
        "event_center": target_center,
        "event_time": target_time,
        "event_center_mask": center_mask,
        "event_time_mask": time_mask,
    }


def generate_das_labels(
    sample: DASSample,
    config: DASLabelConfig = DASLabelConfig(),
    phases: list[str] = ["P", "S"],
) -> dict[str, np.ndarray]:
    """Generate all labels from a DASSample.

    Args:
        sample: DASSample with picks and event info
        config: Label configuration
        phases: Phase types

    Returns:
        Combined dictionary of all labels
    """
    labels = generate_das_phase_labels(sample, config, phases)
    labels.update(generate_das_event_labels(sample, config))
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
) -> DASCompose:
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
    transforms = [DASNormalize()]

    if enable_resample_time:
        transforms.append(DASResampleTime(0.5, 3.0))

    if enable_resample_space:
        transforms.append(DASResampleSpace(0.5, 5.0))

    if enable_stacking:
        transforms.append(DASStackEvents(p=0.3, min_snr=10.0))

    transforms.append(DASRandomCrop(nt=nt, nx=nx))

    if enable_noise_stacking:
        transforms.append(DASStackNoise(max_ratio=2.0, p=0.5))

    transforms.extend([
        DASFlipLR(p=0.5),
    ])

    if enable_masking:
        transforms.append(DASMasking(max_mask_nt=256, p=0.2))

    transforms.append(DASNormalize())

    return DASCompose(transforms)


def default_eval_transforms(min_nt: int = 1024, min_nx: int = 1024) -> DASCompose:
    """Default transforms for DAS evaluation."""
    return DASCompose([
        DASNormalize(),
    ])


def minimal_transforms() -> DASCompose:
    """Minimal transforms - just normalize."""
    return DASCompose([DASNormalize()])


# =============================================================================
# GCS Utilities
# =============================================================================

def get_gcs_storage_options() -> dict:
    """Load GCS credentials for authenticated access.

    Checks for application default credentials at the standard path.
    Returns empty dict if credentials not found (for anonymous access).
    """
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
        # Remove gs:// prefix for gcs filesystem
        path_clean = path[5:]  # Remove "gs://"
        return fs, path_clean
    else:
        fs = fsspec.filesystem("file")
        return fs, path


def open_file(path: str, mode: str = "rb"):
    """Open a file from local or GCS path.

    Args:
        path: Local path or GCS path (gs://...)
        mode: File open mode

    Returns:
        File-like object
    """
    if path.startswith("gs://"):
        storage_options = get_gcs_storage_options()
        return fsspec.open(path, mode, **storage_options)
    else:
        return fsspec.open(path, mode)


# =============================================================================
# Legacy Helper Functions (used by DASIterableDataset)
# =============================================================================

def normalize(data: torch.Tensor):
    """Channel-wise normalization for torch tensors.

    Args:
        data (tensor): [nch, nt, nx]

    Returns:
        tensor: [nch, nt, nx]
    """
    nch, nt, nx = data.shape
    data = data.double()
    mean = torch.mean(data, dim=(1), keepdims=True)
    std = torch.std(data, dim=(1), keepdims=True)
    std[std == 0.0] = 1.0
    data = data / std
    return data.float()


def generate_phase_label(
    data: torch.Tensor,
    phase_list: list,
    label_width: list = [150],
    mask_width: list = None,
    label_shape: str = "gaussian",
):
    """generate gaussian-shape label for phase picks

    Args:
        data (tensor): [nch, nt, nx]
        phase_list (list): [[p_channel, p_index], [s_channel, s_index], [other phases]]
        label_width (list, optional): [150, 150] samples.
        label_shape (str, optional): Defaults to "gaussian".
        space_mask (tensor, optional): [nch, nt, nx], 1 for valid, 0 for invalid.
        return_time_mask (bool, optional): Use to prevent stacking phases too closely in in time. Defaults to True.

    Returns:
        phase label: [nch, nt, nx]
    """
    nch, nt, nx = data.shape

    if mask_width is None:
        mask_width = [label_width] * len(phase_list)

    target = np.zeros([len(phase_list) + 1, nt, nx], dtype=np.float32)
    ## mask for window near the phase arrival
    time_mask = np.zeros([nt, nx], dtype=np.float32)

    if len(label_width) == 1:
        label_width = label_width * len(phase_list)

    space_mask = np.zeros((len(phase_list), nx), dtype=bool)
    for i, (picks, w) in enumerate(zip(phase_list, label_width)):
        for trace, phase_time in picks:
            trace = int(trace)
            t = np.arange(nt) - phase_time
            gaussian = np.exp(-(t**2) / (2 * (w / 6) ** 2))
            gaussian[gaussian < 0.1] = 0.0
            target[i + 1, :, trace] += gaussian
            space_mask[i, trace] = True
            # time_mask[int(phase_time) - w : int(phase_time) + w, trace] = 1
            time_mask[:, trace] = 1

    space_mask = np.all(space_mask, axis=0)  ## traces with all picks
    target[0:1, :, space_mask] = np.maximum(0, 1 - np.sum(target[1:, :, space_mask], axis=0, keepdims=True))
    target[:, :, ~space_mask] = 0
    time_mask = time_mask[np.newaxis, :, :]

    return target, time_mask

def generate_event_label(
    data,
    center,
    duration,
    label_width=150,
    mask_width=None,
):

    nch, nt, nx = data.shape
    target_center = np.zeros([nt, nx], dtype=np.float32)
    target_time = np.zeros([nt, nx], dtype=np.float32)
    center_mask = np.zeros([nt, nx], dtype=np.float32)
    time_mask = np.zeros([nt, nx], dtype=np.float32)

    if mask_width is None:
        mask_width = int(label_width * 1.5)

    for c0, d0 in zip(center, duration):
        ich = c0[0]
        assert ich == d0[0]
        c0 = c0[1]
        d0 = d0[1]
        t = np.arange(nt) - c0
        gaussian = np.exp(-(t**2) / (2 * (label_width / 6) ** 2))
        gaussian[gaussian < 0.05] = 0.0
        target_center[:, ich] += gaussian
        target_time[:, ich] = t + d0
        time_mask[int(c0) - mask_width : int(c0) + mask_width, ich] = 1.0
        center_mask[:, ich] = 1.0

    target_center = target_center[np.newaxis, :, :]
    target_time = target_time[np.newaxis, :, :]
    center_mask = center_mask[np.newaxis, :, :]
    time_mask = time_mask[np.newaxis, :, :]
    return target_center, target_time, center_mask, time_mask

def stack_event(
    data1,
    targets1,
    masks1,
    snrs1,
    data2,
    targets2,
    masks2,
    snrs2,
    min_shift=0,
    max_shift=1024 * 2,
):
    """targets[0] is the phase label"""
    tries = 0
    max_tries = 100
    nch, nt, nx = data2.shape
    success = False
    while tries < max_tries:
        # shift = random.randint(-nt, nt)
        shift = random.randint(-max_shift, max_shift)
        # if masks2 is not None:
        #     masks2_ = {k: torch.clone(v) for k, v in masks2.items()}
        #     masks2_ = {k: torch.roll(v, shift, dims=-2) for k, v in masks2_.items()}
        #     if any(torch.max(masks1[k] + masks2_[k]) >= 2 for k in ["event_time_mask"]):
        #         tries += 1
        #         continue

        data2_ = torch.clone(data2)
        data2_ = torch.roll(data2_, shift, dims=-2)
        masks2_ = {k: torch.clone(v) for k, v in masks2.items()}
        targets2_ = {k: torch.clone(v) for k, v in targets2.items()}
        for k, v in masks2_.items():
            masks2_[k] = torch.roll(v, shift, dims=-2)
        for k, v in targets2_.items():
            targets2_[k] = torch.roll(v, shift, dims=-2)

        ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr
        # data = data1 + data2_ * (1 + max(0, snr1 - 1.0) * torch.rand(1) * 0.5)
        data = data1 * (1 + torch.rand(1) * 2) + data2_ * (1 + torch.rand(1) * 2)

        targets = {k: torch.zeros_like(v) for k, v in targets1.items()}
        for k, v in targets.items():
            if k in ["phase_pick"]:
                targets[k][1:, :, :] = targets1[k][1:, :, :] + targets2_[k][1:, :, :]
                targets[k][0, :, :] = torch.maximum(torch.tensor(0.0), 1.0 - torch.sum(targets[k][1:, :, :], axis=0))
            else:
                targets[k][:, :, :] = torch.maximum(targets1[k][:, :, :], targets2_[k][:, :, :])

        masks = {k: torch.zeros_like(v) for k, v in masks1.items()}
        for k, v in masks.items():
            masks[k][:, :, :] = masks1[k][:, :, :] + masks2_[k][:, :, :]
            if k in ["phase_mask", "event_center_mask"]:
                masks[k][masks[k] < 2.0] = 0.0
                masks[k][masks[k] >= 2.0] = 1.0
            else:
                masks[k][masks[k] >= 1.0] = 1.0

        targets["phase_pick"][:, :, (masks["phase_mask"] == 0).all(dim=(0, 1))] = 0
        targets["event_center"][:, :, (masks["event_center_mask"] == 0).all(dim=(0, 1))] = 0
        targets["event_time"][:, :, (masks["event_time_mask"] == 0).all(dim=(0, 1))] = 0

        success = True
        break

    if tries >= max_tries:
        data = data1
        targets = targets1
        masks = masks1
        print(f"stack event failed, tries={tries}")

    return data, targets, masks, success


def pad_data(data, targets, masks, nt=1024 * 4, nx=1024 * 6):
    """pad data to the same size as required nt and nx"""
    nch, w, h = data.shape
    if h < nx:
        with torch.no_grad():
            data_ = data.unsqueeze(0)
            targets_ = {k: v.unsqueeze(0) for k, v in targets.items()}
            masks_ = {k: v.unsqueeze(0) for k, v in masks.items()}
            if (nx // h - 1) > 0:
                for i in range(nx // h - 1):
                    data_ = F.pad(data_, (0, h - 1, 0, 0), mode="reflect")
                    targets_ = {k: F.pad(v, (0, h - 1, 0, 0), mode="reflect") for k, v in targets_.items()}
                    masks_ = {k: F.pad(v, (0, h - 1, 0, 0), mode="reflect") for k, v in masks_.items()}
                data_ = F.pad(data_, (0, nx // h - 1, 0, 0), mode="reflect")
                targets_ = {k: F.pad(v, (0, nx // h - 1, 0, 0), mode="reflect") for k, v in targets_.items()}
                masks_ = {k: F.pad(v, (0, nx // h - 1, 0, 0), mode="reflect") for k, v in masks_.items()}
            data_ = F.pad(data_, (0, nx % h, 0, 0), mode="reflect").squeeze(0)
            targets_ = {k: F.pad(v, (0, nx % h, 0, 0), mode="reflect").squeeze(0) for k, v in targets_.items()}
            masks_ = {k: F.pad(v, (0, nx % h, 0, 0), mode="reflect").squeeze(0) for k, v in masks_.items()}
    else:
        data_ = data
        targets_ = targets
        masks_ = masks
    return data_, targets_, masks_

def cut_data(
    data: torch.Tensor,
    targets = [],
    masks = [],
    label_width = 150,
    nt: int = 1024 * 3,
    nx: int = 1024 * 5,
):
    """cut data window for training"""

    tmp_sum = 0
    max_sum = 0
    tmp_tries = 0
    max_tries = 100
    max_w0 = 0
    max_h0 = 0
    w, h = data.shape[-2:]
    while tmp_sum < label_width / 2 * nx * 0.1:
        w0 = np.random.randint(0, max(1, w - nt))
        h0 = np.random.randint(0, max(1, h - nx))
        if len(targets) > 0:
            tmp_sum = torch.sum(targets["phase_pick"][1:, w0 : w0 + nt, h0 : h0 + nx])  # nch, nt, nx
        else:
            tmp_sum = nx * nt
        if tmp_sum > max_sum:
            max_sum = tmp_sum
            max_w0 = w0
            max_h0 = h0
        tmp_tries += 1
        if tmp_tries >= max_tries:
            break
    w0 = max_w0
    h0 = max_h0

    data_ = data[:, w0 : w0 + nt, h0 : h0 + nx].clone()
    targets_ = {k: v[..., w0 : w0 + nt, h0 : h0 + nx].clone() for k, v in targets.items()}
    masks_ = {k: v[..., w0 : w0 + nt, h0 : h0 + nx].clone() for k, v in masks.items()}
    return data_, targets_, masks_


def cut_noise(noise: torch.Tensor, nt: int = 1024 * 3, nx: int = 1024 * 5):
    nch, w, h = noise.shape
    w0 = np.random.randint(0, max(1, w - nt))
    h0 = np.random.randint(0, max(1, h - nx))
    return noise[:, w0 : w0 + nt, h0 : h0 + nx]


def pad_noise(noise: torch.Tensor, nt: int = 1024 * 3, nx: int = 1024 * 5):
    """pad noise to the same size as required nt and nx"""

    nch, w, h = noise.shape
    if w < nt:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nt // w - 1) > 0:
                for i in range(nt // w - 1):
                    noise = F.pad(noise, (0, 0, 0, w - 1), mode="reflect")
                noise = F.pad(noise, (0, 0, 0, nt // w - 1), mode="reflect")
            noise = F.pad(noise, (0, 0, 0, nt % w), mode="reflect").squeeze(0)
    if h < nx:
        with torch.no_grad():
            noise = noise.unsqueeze(0)
            if (nx // h - 1) > 0:
                for i in range(nx // h - 1):
                    noise = F.pad(noise, (0, h - 1, 0, 0), mode="reflect")
                noise = F.pad(noise, (0, nx // h - 1, 0, 0), mode="reflect")
            noise = F.pad(noise, (0, nx % h, 0, 0), mode="reflect").squeeze(0)
    return noise


def calc_snr(data: torch.Tensor, picks: list, noise_window: int = 200, signal_window: int = 200):
    SNR = []
    S = []
    N = []
    for trace, phase_time in picks:
        trace = int(trace)
        phase_time = int(phase_time)
        noise = torch.std(data[:, max(0, phase_time - noise_window) : phase_time, trace])
        signal = torch.std(data[:, phase_time : phase_time + signal_window, trace])
        S.append(signal)
        N.append(noise)
        SNR.append(signal / noise)

    return np.median(SNR), np.median(S), np.median(N)


def stack_noise(data, noise, snr):
    ## approximately after normalization, noise=1, signal=snr, so signal ~ noise * snr
    return data + noise * max(0, snr - 2) * torch.rand(1)

def flip_lr(data, targets=[], masks=[]):
    data = data.flip(-1)
    targets = {k: v.flip(-1) for k, v in targets.items()}
    masks = {k: v.flip(-1) for k, v in masks.items()}
    return data, targets, masks

def masking(data, targets, masks,nt=256, nx=256):
    nc0, nt0, nx0 = data.shape
    nt_ = random.randint(32, nt)
    nt0_ = random.randint(0, nt0 - nt_)
    data_ = data.clone()
    targets_ = {k: v.clone() for k, v in targets.items()}
    masks_ = {k: v.clone() for k, v in masks.items()}

    data_[:, nt0_ : nt0_ + nt_, :] = 0.0
    for k, v in targets_.items():
        if k == "phase_pick":
            targets_[k][0, nt0_ : nt0_ + nt_, :] = 1.0
            targets_[k][1:, nt0_ : nt0_ + nt_, :] = 0.0
        # else:
        #     targets_[k][:, nt0_ : nt0_ + nt_, :] = 0.0 # event center could still be predicted 

    return data_, targets_, masks_


def masking_edge(data, targets, masks, nt=1024, nx=1024):
    """masking edges to prevent edge effects"""

    crop_nt = random.randint(1, nt)
    crop_nx = random.randint(1, nx)

    data_ = data.clone()
    targets_ = {k: v.clone() for k, v in targets.items()}
    masks_ = {k: v.clone() for k, v in masks.items()}

    data_[:, -crop_nt:, :] = 0.0

    for k, v in targets_.items():
        if k in ["phase_pick"]:
            targets_[k][0, -crop_nt:, :] = 1.0
            targets_[k][1:, -crop_nt:, :] = 0.0
        # if k in ["event_center"]:
        else:
            targets_[k][:, -crop_nt:, :] = 0.0
    for k, v in masks_.items():
        if k in ["event_time_mask"]:
            masks_[k][:,-crop_nt:, :] = 0.0

    return data_, targets_, masks_


def resample_space(data, targets, masks, noise=None, factor=1):
    """resample space by factor to adjust the spatial resolution"""
    nch, nt, nx = data.shape
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data, scale_factor=scale_factor, mode="nearest")
        targets_ = {k: F.interpolate(v, scale_factor=scale_factor, mode="nearest") for k, v in targets.items()}
        masks_ = {k: F.interpolate(v, scale_factor=scale_factor, mode="nearest") for k, v in masks.items()}
        if noise is not None:
            noise_ = F.interpolate(noise, scale_factor=scale_factor, mode="nearest")
        else:
            noise_ = None
    return data_, targets_, masks_, noise_


def resample_time(data, picks, noise=None, factor=1):
    """resample time by factor to adjust the temporal resolution

    Args:
        picks (list): [[[channel_index, time_index], ..], [[channel_index, time_index], ], ...]
    """
    nch, nt, nx = data.shape
    scale_factor = random.uniform(min(1, factor), max(1, factor))
    with torch.no_grad():
        data_ = F.interpolate(data.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
        if noise is not None:
            noise_ = F.interpolate(noise.unsqueeze(0), scale_factor=(scale_factor, 1), mode="bilinear").squeeze(0)
        else:
            noise_ = None
    picks_ = []
    for phase in picks:
        tmp = []
        for p in phase:
            tmp.append([p[0], p[1] * scale_factor])
        picks_.append(tmp)
    return data_, picks_, noise_


def read_PASSCAL_segy(fid, nTraces=1250, nSample=900000, TraceOff=0, strain_rate=True):
    """Function to read PASSCAL segy raw data
    For Ridgecrest data, there are 1250 channels in total,
    Sampling rate is 250 Hz so for one hour data: 250 * 3600 samples
    author: Jiuxun Yin
    source: https://github.com/SCEDC/cloud/blob/master/pds_ridgecrest_das.ipynb
    """
    fs = nSample / 3600  # sampling rate
    data = np.zeros((nTraces, nSample), dtype=np.float32)

    fid.seek(3600)
    # Skipping traces if necessary
    fid.seek(TraceOff * (240 + nSample * 4), 1)
    # Looping over traces
    for ii in range(nTraces):
        fid.seek(240, 1)
        bytes = fid.read(nSample * 4)
        data[ii, :] = np.frombuffer(bytes, dtype=np.float32)

    fid.close()

    # Convert the phase-shift to strain (in nanostrain)
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
            # F.pad format: (left, right, top, bottom) for 3D input
            data = F.pad(data, (0, pad_nx, 0, pad_nt), mode="constant")

    return data


# =============================================================================
# Dataset Classes
# =============================================================================

class DASIterableDataset(IterableDataset):
    """DAS Iterable Dataset supporting local and GCS paths.

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
        stack_noise: Whether to stack noise during training
        stack_event: Whether to stack events during training
        resample_time: Whether to resample time axis
        resample_space: Whether to resample space axis
        skip_existing: Skip files with existing picks
        pick_path: Path to save/check picks
        folder_depth: Parent folder depth for pick_path
        num_patch: Number of patches per sample
        masking: Apply masking augmentation
        highpass_filter: Highpass filter frequency
        system: DAS system type ("optasense" or None)
        cut_patch: Whether to cut into patches
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
        ## training
        training=False,
        phases=["P", "S"],
        label_path="./",
        subdir=3,
        label_list=None,
        noise_list=None,
        stack_noise=False,
        stack_event=False,
        resample_time=False,
        resample_space=False,
        skip_existing=False,
        pick_path="./",
        folder_depth=1,  # parent folder depth of pick_path
        num_patch=2,
        masking=False,
        highpass_filter=0.0,
        filter_params={
            "freqmin": 0.1,
            "freqmax": 10.0,
            "corners": 4,
            "zerophase": True,
        },
        ## continuous data
        system=None,  # "eqnet" or "optasense" or None
        cut_patch=False,
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

        # Determine if using GCS
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
            # List files from local or GCS
            self.data_list = self._list_files(
                self.data_path, f"{prefix}*{suffix}.{format}"
            )

        if not training:
            self.data_list = self.data_list[rank::world_size]

        ## continuous data
        self.system = system
        self.cut_patch = cut_patch
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01  # s
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m
        self.nt = nt
        self.nx = nx
        self.min_nt = min_nt
        self.min_nx = min_nx
        assert self.nt % self.min_nt == 0
        assert self.nx % self.min_nx == 0

        ## training and data augmentation
        self.training = training
        self.phases = phases

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
            # List label files from local or GCS
            self.label_list = self._list_files(self.label_path, "*.csv")
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        self.noise_list = None
        if noise_list is not None:
            if isinstance(noise_list, list):
                self.noise_list = []
                for noise_list_ in noise_list:
                    self.noise_list += self._read_text_file(noise_list_).rstrip("\n").split("\n")
            else:
                self.noise_list = self._read_text_file(noise_list).rstrip("\n").split("\n")
        self.stack_noise = stack_noise
        self.stack_event = stack_event
        self.resample_space = resample_space
        self.resample_time = resample_time
        self.skip_existing = skip_existing
        self.pick_path = pick_path
        self.folder_depth = folder_depth
        self.num_patch = num_patch
        self.masking = masking
        self.highpass_filter = highpass_filter

        if self.training:
            print(f"Total samples: {len(self.label_list)} files")
        else:
            print(f"Total samples: {len(self.data_list)} files")

        ## pre-calcuate length
        self._data_len = self._count()

    def _list_files(self, path: str, pattern: str) -> list[str]:
        """List files matching pattern from local or GCS path.

        Args:
            path: Local path or GCS path (gs://...)
            pattern: Glob pattern to match

        Returns:
            List of file paths
        """
        if path.startswith("gs://"):
            fs, path_clean = get_filesystem(path)
            # Use fsspec glob
            full_pattern = f"{path_clean}/{pattern}"
            files = fs.glob(full_pattern)
            # Return with gs:// prefix
            return [f"gs://{f}" for f in files]
        else:
            return glob(os.path.join(path, pattern))

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        """Read text file from local or GCS.

        Args:
            file_path: Local path or GCS path

        Returns:
            File contents as string
        """
        if file_path.startswith("gs://"):
            fs = fsspec.filesystem("gcs", **get_gcs_storage_options())
            with fs.open(file_path, "r") as f:
                return f.read()
        else:
            with open(file_path, "r") as f:
                return f.read()

    def _open_h5_file(self, file_path: str):
        """Open an HDF5 file from local or GCS.

        Args:
            file_path: Local path or GCS path

        Returns:
            Context manager for h5py.File
        """
        return open_file(file_path, "rb")

    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file from local or GCS.

        Args:
            file_path: Local path or GCS path

        Returns:
            pandas DataFrame
        """
        if file_path.startswith("gs://"):
            storage_options = get_gcs_storage_options()
            return pd.read_csv(file_path, storage_options=storage_options)
        else:
            return pd.read_csv(file_path)

    def _construct_data_path(self, label_file: str) -> str:
        """Construct data file path from label file path.

        Args:
            label_file: Path to label file (CSV)

        Returns:
            Path to corresponding data file (H5)
        """
        # Convert label path to data path
        data_file = "/".join(
            label_file.replace("labels", "data").replace(".csv", ".h5").split("/")[-self.subdir:]
        )
        # Join with data_path preserving GCS prefix if needed
        if self.data_path.startswith("gs://"):
            return f"{self.data_path.rstrip('/')}/{data_file}"
        else:
            return os.path.join(self.data_path, data_file)

    def __len__(self):
        return self._data_len

    def _count(self):
        if self.training:
            return len(self.label_list) * self.num_patch

        if not self.cut_patch:
            return len(self.data_list)
        else:
            if self.format == "h5":
                with self._open_h5_file(self.data_list[0]) as fs:
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
                with self._open_h5_file(self.data_list[0]) as fs:
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
            return iter(self.sample_training(self.label_list[worker_id::num_workers]))
        else:
            return iter(self.sample(self.data_list[worker_id::num_workers]))

    def _construct_file_path(self, base_path: str, relative_path: str) -> str:
        """Construct full file path preserving GCS prefix.

        Args:
            base_path: Base path (local or GCS)
            relative_path: Relative path to append

        Returns:
            Full path with proper prefix
        """
        if base_path.startswith("gs://"):
            return f"{base_path.rstrip('/')}/{relative_path.lstrip('/')}"
        else:
            return os.path.join(base_path, relative_path)

    def sample_training(self, file_list):
        while True:
            ## load picks
            file_list = np.random.permutation(file_list)
            for label_file in file_list:
                # Construct full label path
                label_path_full = self._construct_file_path(self.label_path, label_file)
                picks = self._read_csv(label_path_full)
                if "channel_index" not in picks.columns:
                    picks = picks.rename(columns={"station_id": "channel_index"})

                meta = {}
                for pick_type in self.phases:
                    meta[pick_type] = picks[picks["phase_type"] == pick_type][
                        ["channel_index", "phase_index"]
                    ].to_numpy()

                ## load waveform data
                data_file = "/".join(
                    label_file.replace("labels", "data").replace(".csv", ".h5").split("/")[-self.subdir:]
                )  # folder/data/event_id
                data_path_full = self._construct_file_path(self.data_path, data_file)

                try:
                    with open_file(data_path_full, "rb") as f:
                        with h5py.File(f, "r") as fp:
                            data = fp["data"][:, :].T
                            dt = fp["data"].attrs["dt_s"]
                            dx = fp["data"].attrs["dx_m"]
                        data = data[np.newaxis, :, :]  # nchn, nt, nx
                        data = data / np.std(data)
                        data = torch.from_numpy(data.astype(np.float32))

                except Exception as e:
                    print(f"Error reading {data_path_full}: {e}")
                    continue

                ## basic normalize
                data = data - torch.mean(data, dim=1, keepdim=True)

                # load noise
                noise = None
                if self.stack_noise and (self.noise_list is not None):
                    tmp = self.noise_list[np.random.randint(0, len(self.noise_list))]
                    noise_path_full = self._construct_file_path(self.data_path, tmp)
                    try:
                        with open_file(noise_path_full, "rb") as f:
                            with h5py.File(f, "r") as fp:
                                noise = fp["data"][:, :].T
                            ## The first 30s are noise in the training data
                            noise = np.roll(noise, max(0, self.nt - 3000), axis=0)  # nt, nx
                            noise = noise[np.newaxis, : self.nt, :]  # nchn, nt, nx
                            noise = noise / np.std(noise)
                            noise = torch.from_numpy(noise.astype(np.float32))

                        noise = noise - torch.mean(noise, dim=1, keepdim=True)
                    except Exception as e:
                        print(f"Error reading noise file {noise_path_full}: {e}")
                        noise = torch.zeros([1, self.nt, self.nx], dtype=torch.float32)

                ## snr
                if "P" in meta:
                    snr, S, N = calc_snr(data, meta["P"])
                else:
                    snr, S, N = 0, 0, 0


                ## generate training labels
                picks = [meta[x] for x in self.phases]

                ## augmentation
                rand_i = np.random.rand()
                if self.resample_time:
                    if rand_i < 0.2:
                        data, picks, noise = resample_time(data, picks, noise, 3)
                    elif rand_i < 0.4:
                        data, picks, noise = resample_time(data, picks, noise, 0.5)

                ## generate training labels
                phase_pick, phase_mask = generate_phase_label(data, picks)
                phase_pick = torch.from_numpy(phase_pick)
                phase_mask = torch.from_numpy(phase_mask)

                c0 = [[x1[0], (x1[1] + x2[1]) / 2] for x1, x2 in zip(picks[0], picks[1])]
                t0 = [[x1[0], x2[1] - x1[1]] for x1, x2 in zip(picks[0], picks[1])]

                event_center, event_time, event_center_mask, event_time_mask = generate_event_label(data, c0, t0)
                event_center = torch.from_numpy(event_center)
                event_time = torch.from_numpy(event_time)
                event_center_mask = torch.from_numpy(event_center_mask)
                event_time_mask = torch.from_numpy(event_time_mask)


                targets = {
                    "phase_pick": phase_pick,
                    "event_center": event_center,
                    "event_time": event_time,
                }
                masks = {
                    "phase_mask": phase_mask,
                    "event_center_mask": event_center_mask,
                    "event_time_mask": event_time_mask,
                }

                ## augmentation
                status_stack_event = False
                if self.stack_event and (snr > 10) and (np.random.rand() < 0.3):
                    data, targets, masks, status_stack_event = stack_event(
                        data, targets, masks, snr, data, targets, masks, snr
                    )

                ## augmentation
                if self.resample_space:
                    if rand_i < 0.2:
                        data, targets, masks, noise = resample_space(data, targets, masks, noise, 5)
                    elif (rand_i < 0.4) and (data.shape[-1] > 2000):
                        data, targets, masks, noise = resample_space(data, targets, masks, noise, 0.5)

                ## pad data
                data, targets, masks = pad_data(data, targets, masks, nx=self.nx + self.nx // 2)
                if self.stack_noise:
                    noise = pad_noise(noise, self.nt, self.nx + self.nx // 2)

                for ii in range(self.num_patch):
                    data_, targets_, masks_ = cut_data(data, targets, masks, nt=self.nt, nx=self.nx)

                    ## augmentation
                    if self.stack_noise and (not status_stack_event) and (np.random.rand() < 0.5):
                        noise_ = cut_noise(noise, self.nt, self.nx)
                        data_ = stack_noise(data_, noise_, snr)

                    ## augmentation
                    if np.random.rand() < 0.5:
                        data_, targets_, masks_ = flip_lr(data_, targets_, masks_)

                    ## augmentation
                    if self.masking and (np.random.rand() < 0.2):
                        data_, targets_, masks_ = masking(data_, targets_, masks_)


                    # ## prevent edge effect on the right and bottom
                    # if np.random.rand() < 0.05:
                    #     data_, targets_, masks_ = masking_edge(data_, targets_, masks_)


                    # data_ = normalize(data_)
                    if np.random.rand() < 0.5:
                        data_ = data_ - torch.median(data_, dim=-2, keepdims=True)[0]

                    phase_pick_, event_center_, event_time_ = targets_["phase_pick"], targets_["event_center"], targets_["event_time"]
                    phase_mask_, event_center_mask_, event_time_mask_ = masks_["phase_mask"], masks_["event_center_mask"], masks_["event_time_mask"]

                    ## FIXME: shift (nt, nx) to (nx, nt)
                    data_ = data_.permute(0, 2, 1)
                    phase_pick_ = phase_pick_.permute(0, 2, 1)
                    phase_mask_ = phase_mask_.permute(0, 2, 1)
                    event_center_ = event_center_.permute(0, 2, 1)
                    event_time_ = event_time_.permute(0, 2, 1)
                    event_center_mask_ = event_center_mask_.permute(0, 2, 1)
                    event_time_mask_ = event_time_mask_.permute(0, 2, 1)
                    event_feature_scale = 16
                    event_center_ = event_center_[:, ::, ::event_feature_scale]
                    event_time_ = event_time_[:, ::, ::event_feature_scale]
                    event_center_mask_ = event_center_mask_[:, ::, ::event_feature_scale]
                    event_time_mask_ = event_time_mask_[:, ::, ::event_feature_scale]

                    yield {
                        "data": torch.nan_to_num(data_),
                        "phase_pick": phase_pick_,
                        "phase_mask": phase_mask_,
                        "event_center": event_center_,
                        "event_time": event_time_,
                        "event_time_mask": event_time_mask_,
                        "event_center_mask": event_center_mask_,
                        "file_name": os.path.splitext(label_file.split("/")[-1])[0] + f"_{ii:02d}",
                        "height": data_.shape[-2],
                        "width": data_.shape[-1],
                        "dt_s": dt,
                        "dx_m": dx,
                    }

    def sample(self, file_list):
        for file in file_list:
            if not self.cut_patch:
                existing = self.check_existing(file)
                if self.skip_existing and existing:
                    print(f"Skip existing file {file}")
                    continue

            sample = {}

            if self.format == "npz":
                with open_file(file, "rb") as f:
                    data = np.load(f)["data"]

            elif self.format == "npy":
                with open_file(file, "rb") as f:
                    data = np.load(f)  # (nx, nt)
                sample["begin_time"] = datetime.fromisoformat("1970-01-01 00:00:00")
                sample["dt_s"] = 0.01
                sample["dx_m"] = 10.0

            elif self.format == "h5" and (self.system is None):
                with open_file(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        dataset = fp["data"]  # nt x nx
                        data = dataset[()]
                        if "begin_time" in dataset.attrs:
                            sample["begin_time"] = datetime.fromisoformat(dataset.attrs["begin_time"].rstrip("Z"))
                        if "dt_s" in dataset.attrs:
                            sample["dt_s"] = dataset.attrs["dt_s"]
                        else:
                            sample["dt_s"] = self.dt
                        if "dx_m" in dataset.attrs:
                            sample["dx_m"] = dataset.attrs["dx_m"]
                        else:
                            sample["dx_m"] = self.dx
            elif (self.format == "h5") and (self.system == "optasense"):
                with open_file(file, "rb") as fs:
                    with h5py.File(fs, "r") as fp:
                        # dataset = fp["Data"]
                        if "Data" in fp:  # converted format by Ettore Biondi
                            dataset = fp["Data"]
                            sample["begin_time"] = datetime.fromisoformat(dataset.attrs["startTime"].rstrip("Z"))
                            sample["dt_s"] = dataset.attrs["dt"]
                            sample["dx_m"] = dataset.attrs["dCh"]
                        else:
                            dataset = fp["Acquisition/Raw[0]/RawData"]
                            dx = fp["Acquisition"].attrs["SpatialSamplingInterval"]
                            fs_rate = fp["Acquisition/Raw[0]"].attrs["OutputDataRate"]
                            begin_time = dataset.attrs["PartStartTime"].decode()

                            sample["dx_m"] = dx
                            sample["dt_s"] = 1.0 / fs_rate
                            sample["begin_time"] = datetime.fromisoformat(begin_time.rstrip("Z"))

                        nx, nt = dataset.shape
                        sample["nx"] = nx
                        sample["nt"] = nt

                        ## check existing
                        existing = self.check_existing(file, sample)
                        if self.skip_existing and existing:
                            print(f"Skip existing file {file}")
                            continue

                        data = dataset[()]  # (nx, nt)
                        data = np.gradient(data, axis=-1, edge_order=2) / sample["dt_s"]

            elif self.format == "segy":
                meta = {}
                with open_file(file, "rb") as fs:
                    data = read_PASSCAL_segy(fs)

                ## FIXME: hard code for Ridgecrest DAS
                sample["begin_time"] = datetime.strptime(file.split("/")[-1].rstrip(".segy"), "%Y%m%d%H")
                sample["dt_s"] = 1.0 / 250.0
                sample["dx_m"] = 8.0
            else:
                raise (f"Unsupported format: {self.format}")

            if self.resample_time:
                if (sample["dt_s"] != 0.01) and (int(round(1.0 / sample["dt_s"])) % 100 == 0):
                    print(f"Resample {file} from time interval {sample['dt_s']} to 0.01")
                    data = data[..., :: int(0.01 / sample["dt_s"])]
                    sample["dt_s"] = 0.01

            data = data - np.mean(data, axis=-1, keepdims=True)  # (nx, nt)
            data = data - np.median(data, axis=-2, keepdims=True)
            if (self.highpass_filter is not None):
                b, a = scipy.signal.butter(2, self.highpass_filter, "hp", fs=100)
                data = scipy.signal.filtfilt(b, a, data, axis=-1)  # (nt, nx)

            data = data.T  # (nx, nt) -> (nt, nx)
            data = data[np.newaxis, :, :]  # (nchn, nt, nx)
            data = torch.from_numpy(data.astype(np.float32))

            # data = torch.from_numpy(data).float()
            # data = data - torch.mean(data, axis=-1, keepdims=True)  # (nx, nt)
            # data = data - torch.median(data, axis=-2, keepdims=True).values
            # data = data.T  # (nx, nt) -> (nt, nx)
            # data = data.unsqueeze(0)  # (nchn, nt, nx)

            if not self.cut_patch:
                nt, nx = data.shape[1:]
                data = padding(data, self.min_nt, self.min_nx)

                ## FIXME: (nt, nx) -> (nx, nt)
                data = data.permute(0, 2, 1)

                yield {
                    "data": data,
                    "nt": nt,
                    "nx": nx,
                    # "file_name": os.path.splitext(file.split("/")[-1])[0],
                    "file_name": file,
                    "begin_time": sample["begin_time"].isoformat(timespec="milliseconds"),
                    "begin_time_index": 0,
                    "begin_channel_index": 0,
                    "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                    "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                }
            else:
                _, nt, nx = data.shape
                for i in list(range(0, nt, self.nt)):
                    for j in list(range(0, nx, self.nx)):
                        if self.skip_existing:
                            if os.path.exists(
                                os.path.join(
                                    self.pick_path, os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv"
                                )
                            ):
                                print(
                                    f"Skip existing file",
                                    os.path.join(
                                        self.pick_path,
                                        os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv",
                                    ),
                                )
                                continue
                        data_patch = data[:, i : i + self.nt, j : j + self.nx]
                        _, nt_, nx_ = data_patch.shape
                        data_patch = padding(data_patch, self.min_nt, self.min_nx)

                        ## FIXME: (nt, nx) -> (nx, nt)
                        data_patch = data_patch.permute(0, 2, 1)
                        yield {
                            "data": data_patch,
                            "nt": nt_,
                            "nx": nx_,
                            # "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "file_name": os.path.splitext(file)[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(seconds=i * float(sample["dt_s"]))).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }

    def check_existing(self, file, sample=None):
        parent_dir = "/".join(file.split("/")[-self.folder_depth : -1])
        existing = True
        if not self.cut_patch:
            if not os.path.exists(
                os.path.join(
                    os.path.join(self.pick_path, parent_dir, os.path.splitext(file.split("/")[-1])[0] + ".csv")
                )
            ):
                existing = False
        else:
            nx, nt = sample["nx"], sample["nt"]
            if self.resample_time:
                if (sample["dt_s"] != 0.01) and (int(round(1.0 / sample["dt_s"])) % 100 == 0):
                    nt = int(nt / round(0.01 / sample["dt_s"]))
            for i in list(range(0, nt, self.nt)):
                for j in list(range(0, nx, self.nx)):
                    if not os.path.exists(
                        os.path.join(
                            self.pick_path,
                            parent_dir,
                            os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}.csv",
                        )
                    ):
                        existing = False

        return existing


class AutoEncoderIterableDataset(DASIterableDataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=False,
        stack_noise=False,
        highpass_filter=0.0,
        **kwargs,
    ):
        super().__init__(data_path, noise_path, format=format, training=training)

    def sample(self, file_list):
        sample = {}
        # for file in file_list:
        idx = 0
        while True:
            if self.training:
                file = file_list[np.random.randint(0, len(file_list))]
            else:
                if idx >= len(file_list):
                    break
                file = file_list[idx]
                idx += 1

            if self.training and (self.format == "h5"):
                with h5py.File(file, "r") as f:
                    data = f["data"][()]
                    data = data[np.newaxis, :, :]  # nchn, nt, nx
                    data = torch.from_numpy(data.astype(np.float32))
            else:
                raise (f"Unsupported format: {self.format}")

            data = data - np.median(data, axis=2, keepdims=True)
            data = normalize(data)  # nch, nt, nx

            if self.training:
                for ii in range(10):
                    pre_nt = 255
                    data_ = cut_data(data, pre_nt=pre_nt)
                    if data_ is None:
                        continue
                    if np.random.rand() < 0.5:
                        data_ = add_moveout(data_)
                    data_ = data_[:, pre_nt:, :]
                    if np.random.rand() < 0.5:
                        data_ = flip_lr(data_)
                    data_ = data_ - np.median(data_, axis=2, keepdims=True)

                    yield {
                        "data": data_,
                        "phase_pick": data_,
                        "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{ii:02d}",
                        "height": data_.shape[-2],
                        "width": data_.shape[-1],
                    }
            else:
                sample["data"] = data
                if self.nt is None:
                    self.nt = data.shape[1]
                if self.nx is None:
                    self.nx = data.shape[2]
                for i in list(range(0, data.shape[1], self.nt)):
                    if self.nt + i + 512 >= data.shape[1]:
                        tn = data.shape[1]
                    else:
                        tn = i + self.nt
                    for j in list(range(0, data.shape[2], self.nx)):
                        if self.nx + j + 512 >= data.shape[2]:
                            xn = data.shape[2]
                        else:
                            xn = j + self.nx
                        yield {
                            "data": data[:, i:tn, j:xn],
                            "file_name": os.path.splitext(file.split("/")[-1])[0] + f"_{i:04d}_{j:04d}",
                            "begin_time": (sample["begin_time"] + timedelta(i * sample["dt_s"])).isoformat(
                                timespec="milliseconds"
                            ),
                            "begin_time_index": i,
                            "begin_channel_index": j,
                            "dt_s": sample["dt_s"] if "dt_s" in sample else self.dt,
                            "dx_m": sample["dx_m"] if "dx_m" in sample else self.dx,
                        }
                        if xn == data.shape[2]:
                            break
                    if tn == data.shape[1]:
                        break


class DASDataset(Dataset):
    def __init__(
        self,
        data_path="./",
        noise_path=None,
        label_path=None,
        format="npz",
        prefix="",
        suffix="",
        training=True,
        stack_noise=True,
        phases=["P", "S"],
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.noise_path = noise_path
        self.label_path = label_path
        self.format = format
        self.training = training
        self.prefix = prefix
        self.suffix = suffix
        self.phases = phases
        self.data_list = sorted(glob(os.path.join(data_path, f"{prefix}*{suffix}.{format}")))
        if label_path is not None:
            if type(label_path) is list:
                self.label_list = []
                for i in range(len(label_path)):
                    self.label_list += list(sorted(glob(os.path.join(label_path[i], f"{prefix}*{suffix}.csv"))))
            else:
                self.label_list = sorted(glob(os.path.join(label_path, f"{prefix}*{suffix}.csv")))
        print(os.path.join(data_path, f"{prefix}*{suffix}.{format}"), len(self.data_list))
        if self.noise_path is not None:
            self.noise_list = glob(os.path.join(noise_path, f"*.{format}"))
        self.num_data = len(self.data_list)
        self.min_picks = kwargs["min_picks"] if "min_picks" in kwargs else 500
        self.dt = kwargs["dt"] if "dt" in kwargs else 0.01
        self.dx = kwargs["dx"] if "dx" in kwargs else 10.0  # m

    def __len__(self):
        if self.label_path is not None:
            return len(self.label_list)
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = {}
        if self.training and (self.format == "npz"):
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            data = torch.from_numpy(data.astype(np.float32))

        elif self.training and (self.format == "h5"):
            file = self.label_list[idx]
            picks = pd.read_csv(file)
            meta = {}
            for pick_type in self.phases:
                meta[pick_type] = picks[picks["phase_type"] == pick_type][["channel_index", "phase_index"]].to_numpy()
            # if (len(meta["p_picks"]) < 500) or (len(meta["s_picks"]) < 500):
            #     continue
            tmp = file.split("/")
            tmp[-2] = "data"
            tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
            with h5py.File("/".join(tmp), "r") as f:
                data = f["data"][()]
                data = data[np.newaxis, :, :]  # nchn, nt, nx
                data = torch.from_numpy(data.astype(np.float32))

            if self.stack_noise and (not self.noise_path):
                tries = 0
                max_tries = 10
                while tries < max_tries:
                    tmp_file = self.label_list[np.random.randint(0, len(self.label_list))]
                    tmp_picks = pd.read_csv(tmp_file)
                    if tmp_picks["phase_index"].min() < 3000:
                        tries += 1
                        continue
                    tmp = tmp_file.split("/")
                    tmp[-2] = "data"
                    tmp[-1] = tmp[-1][:-4] + ".h5"  ## remove .csv
                    with h5py.File("/".join(tmp), "r") as f:
                        noise = f["data"][()]
                        noise = noise[np.newaxis, :, :]  # nchn, nt, nx
                        noise = torch.from_numpy(noise.astype(np.float32))
                    break
                if tries >= max_tries:
                    print(f"Failed to find noise file for {file}")
                    noise = torch.zeros_like(data)

        elif self.format == "npz":
            meta = np.load(self.data_list[idx])
            data = meta["data"]
            data = data[np.newaxis, :, :]
            # data = np.diff(data, axis=-2)
            # b, a = scipy.signal.butter(2, 4, 'hp', fs=100)
            # b, a = scipy.signal.butter(2, [0.5, 2.5], 'bandpass', fs=100)
            # data = scipy.signal.filtfilt(b, a, data, axis=-2)
            data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "h5":
            begin_time_index = 0
            begin_channel_index = 0
            with h5py.File(self.data_list[idx], "r") as f:
                data = f["data"][()]
                # data = data[np.newaxis, :, :]
                data = data[np.newaxis, begin_time_index:, begin_channel_index:]
                if "begin_time" in f["data"].attrs:
                    if begin_time_index == 0:
                        sample["begin_time"] = datetime.fromisoformat(
                            f["data"].attrs["begin_time"].rstrip("Z")
                        ).isoformat(timespec="milliseconds")
                    else:
                        sample["begin_time_index"] = begin_time_index
                        sample["begin_time"] = (
                            datetime.fromisoformat(f["data"].attrs["begin_time"].rstrip("Z"))
                            + timedelta(seconds=begin_time_index * f["data"].attrs["dt_s"])
                        ).isoformat(timespec="milliseconds")
                if "dt_s" in f["data"].attrs:
                    sample["dt_s"] = f["data"].attrs["dt_s"]
                if "dx_m" in f["data"].attrs:
                    sample["dx_m"] = f["data"].attrs["dx_m"]
                data = torch.from_numpy(data.astype(np.float32))

        elif self.format == "segy":
            data = load_segy(os.path.join(self.data_path, self.data_list[idx]), nTrace=self.nTrace)
            data = torch.from_numpy(data)
            with torch.no_grad():
                data = torch.diff(data, n=1, dim=-1)
                data = F.interpolate(
                    data.unsqueeze(dim=0),
                    scale_factor=self.raw_dt / self.dt,
                    mode="linear",
                    align_corners=False,
                )
                data = data.permute(0, 2, 1)
        else:
            raise (f"Unsupported format: {self.format}")

        # data = normalize_local_1d(data)
        data = data - np.median(data, axis=2, keepdims=True)
        data = normalize(data)

        if self.training:
            if self.stack_noise:
                if torch.max(torch.abs(noise)) > 0:
                    noise = normalize(noise)
            picks = [meta[x] for x in self.phases]
            phase_pick = generate_phase_label(data, picks)
            phase_pick = torch.from_numpy(phase_pick)
            snr = calc_snr(data, meta["p_picks"])
            with_event = False
            if (snr > 3) and (np.random.rand() < 0.3):
                data, phase_pick = stack_event(data, phase_pick, data, phase_pick, snr)
                with_event = True
            pre_nt = 255
            data, phase_pick = cut_data(data, phase_pick, pre_nt=pre_nt)
            if np.random.rand() < 0.5:
                data, phase_pick = add_moveout(data, phase_pick)
            data = data[:, pre_nt:, :]
            phase_pick_ = phase_pick[:, pre_nt:, :]
            # if (snr > 10) and (np.random.rand() < 0.5):
            if not with_event:
                noise = cut_noise(noise)
                data = stack_noise(data, noise, snr)
            if np.random.rand() < 0.5:
                data, phase_pick = flip_lr(data, phase_pick)

            data = data - np.median(data, axis=2, keepdims=True)
            sample["phase_pick"] = phase_pick

        sample["data"] = data
        sample["file_name"] = os.path.splitext(self.data_list[idx].split("/")[-1])[0]
        sample["height"], sample["width"] = sample["data"].shape[-2:]

        return sample


# =============================================================================
# Sample Buffer for Stacking
# =============================================================================

class DASSampleBuffer:
    """Efficient buffer for random sample access during stacking.

    Uses reservoir sampling for streaming datasets.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: list[DASSample] = []
        self.count = 0

    def add(self, sample: DASSample):
        """Add a sample to the buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample.copy())
        else:
            # Reservoir sampling
            idx = random.randint(0, self.count)
            if idx < self.max_size:
                self.buffer[idx] = sample.copy()
        self.count += 1

    def get_random(self) -> DASSample | None:
        """Get a random sample from the buffer."""
        if not self.buffer:
            return None
        return random.choice(self.buffer).copy()

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Factory Functions
# =============================================================================

def create_das_train_dataset(
    data_path: str,
    label_path: str,
    label_list: str | list[str] | None = None,
    noise_list: str | list[str] | None = None,
    nt: int = 3072,
    nx: int = 5120,
    min_nt: int = 1024,
    min_nx: int = 1024,
    phases: list[str] = ["P", "S"],
    stack_noise: bool = True,
    stack_event: bool = True,
    resample_time: bool = False,
    resample_space: bool = False,
    masking: bool = True,
    num_patch: int = 2,
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
        min_nt: Minimum time samples for padding
        min_nx: Minimum space samples for padding
        phases: Phase types to use
        stack_noise: Enable noise stacking
        stack_event: Enable event stacking
        resample_time: Enable time resampling
        resample_space: Enable space resampling
        masking: Enable masking augmentation
        num_patch: Number of patches per sample
        **kwargs: Additional arguments

    Returns:
        DASIterableDataset instance
    """
    return DASIterableDataset(
        data_path=data_path,
        label_path=label_path,
        label_list=label_list,
        noise_list=noise_list,
        nt=nt,
        nx=nx,
        min_nt=min_nt,
        min_nx=min_nx,
        training=True,
        phases=phases,
        stack_noise=stack_noise,
        stack_event=stack_event,
        resample_time=resample_time,
        resample_space=resample_space,
        masking=masking,
        num_patch=num_patch,
        **kwargs,
    )


def create_das_eval_dataset(
    data_path: str,
    nt: int = 3072,
    nx: int = 5120,
    min_nt: int = 1024,
    min_nx: int = 1024,
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
        min_nt: Minimum time samples for padding
        min_nx: Minimum space samples for padding
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
        min_nt=min_nt,
        min_nx=min_nx,
        training=False,
        system=system,
        cut_patch=cut_patch,
        highpass_filter=highpass_filter,
        **kwargs,
    )


def load_das_sample_from_h5(
    file_path: str,
    picks_df: pd.DataFrame | None = None,
    phases: list[str] = ["P", "S"],
) -> DASSample:
    """Load a DASSample from an HDF5 file.

    Args:
        file_path: Path to H5 file (local or GCS)
        picks_df: DataFrame with picks (channel_index, phase_index, phase_type)
        phases: Phase types to extract

    Returns:
        DASSample instance
    """
    with open_file(file_path, "rb") as f:
        with h5py.File(f, "r") as fp:
            data = fp["data"][:, :].T  # (nx, nt) -> (nt, nx)
            dt_s = fp["data"].attrs.get("dt_s", 0.01)
            dx_m = fp["data"].attrs.get("dx_m", 10.0)

    data = data[np.newaxis, :, :]  # (1, nt, nx)
    data = data / np.std(data)
    data = data - np.mean(data, axis=1, keepdims=True)

    p_picks = []
    s_picks = []
    event_centers = []
    event_durations = []

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
                duration = s_dict[ch] - p_dict[ch]
                event_centers.append((ch, center))
                event_durations.append((ch, duration))

    return DASSample(
        waveform=data.astype(np.float32),
        p_picks=p_picks,
        s_picks=s_picks,
        event_centers=event_centers,
        event_durations=event_durations,
        dt_s=dt_s,
        dx_m=dx_m,
        file_name=os.path.basename(file_path),
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "DASLabelConfig",
    "BUCKET_DAS",
    # Sample
    "DASSample",
    "DASSampleBuffer",
    # Transforms
    "DASTransform",
    "DASCompose",
    "DASIdentity",
    "DASNormalize",
    "DASMedianFilter",
    "DASHighpassFilter",
    "DASRandomCrop",
    "DASFlipLR",
    "DASResampleTime",
    "DASResampleSpace",
    "DASMasking",
    "DASStackEvents",
    "DASStackNoise",
    # Label generation
    "generate_das_phase_labels",
    "generate_das_event_labels",
    "generate_das_labels",
    # Transform presets
    "default_train_transforms",
    "default_eval_transforms",
    "minimal_transforms",
    # Datasets
    "DASIterableDataset",
    "DASDataset",
    "AutoEncoderIterableDataset",
    # Factory functions
    "create_das_train_dataset",
    "create_das_eval_dataset",
    "load_das_sample_from_h5",
    # GCS utilities
    "get_gcs_storage_options",
    "get_filesystem",
    "open_file",
]
