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
1. Transform-based augmentation operating on raw labels (phase indices)
   before Gaussian label generation - similar to object detection transforms
2. Compose pattern for chaining transforms
3. Efficient stacking via pre-built indices and reservoir sampling
4. Streaming support via HuggingFace datasets
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
    event_width: int = 150  # Gaussian width for event center
    mask_width_factor: float = 1.5  # mask_width = phase_width * factor
    gaussian_threshold: float = 0.1  # Values below this are zeroed


@dataclass
class Sample:
    """A seismic sample with waveform and phase annotations.

    This is the core data structure passed through transforms.
    Phase indices are kept as lists until final label generation,
    making augmentation simple (just shift/scale indices).
    """
    waveform: np.ndarray  # (3, nt) or (3, nx, nt)
    p_indices: list[int] = field(default_factory=list)  # P-phase sample indices
    s_indices: list[int] = field(default_factory=list)  # S-phase sample indices
    polarity_up: list[int] = field(default_factory=list)  # Up polarity indices
    polarity_down: list[int] = field(default_factory=list)  # Down polarity indices
    event_center: list[float] = field(default_factory=list)  # Event center indices
    event_time: list[float] = field(default_factory=list)  # Event time indices

    # Metadata (not modified by most transforms)
    snr: float = 0.0
    amp_signal: float = 0.0
    amp_noise: float = 0.0
    station_location: np.ndarray | None = None
    event_location: np.ndarray | None = None
    distance_km: float = 0.0  # Event-station distance
    trace_id: str = ""
    sensor: str = ""
    sampling_rate: float = DEFAULT_SAMPLING_RATE

    @property
    def nt(self) -> int:
        return self.waveform.shape[-1]

    @property
    def nch(self) -> int:
        return self.waveform.shape[0]

    def copy(self) -> "Sample":
        """Create a deep copy of the sample."""
        return Sample(
            waveform=self.waveform.copy(),
            p_indices=self.p_indices.copy(),
            s_indices=self.s_indices.copy(),
            polarity_up=self.polarity_up.copy(),
            polarity_down=self.polarity_down.copy(),
            event_center=self.event_center.copy(),
            event_time=self.event_time.copy(),
            snr=self.snr,
            amp_signal=self.amp_signal,
            amp_noise=self.amp_noise,
            station_location=self.station_location.copy() if self.station_location is not None else None,
            distance_km=self.distance_km,
            event_location=self.event_location.copy() if self.event_location is not None else None,
            trace_id=self.trace_id,
            sensor=self.sensor,
            sampling_rate=self.sampling_rate,
        )


# =============================================================================
# Transforms - Following CV Best Practices (torchvision/albumentations pattern)
# =============================================================================

class Transform(ABC):
    """Base class for all transforms.

    Transforms operate on Sample objects, modifying both waveform and
    phase indices. This is similar to how object detection transforms
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
        ...     RandomCrop(4096),
        ...     FlipPolarity(p=0.5),
        ...     Normalize(),
        ... ])
    """

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
    """Identity transform - returns sample unchanged."""

    def __call__(self, sample: Sample) -> Sample:
        return sample


# -----------------------------------------------------------------------------
# Basic Waveform Transforms
# -----------------------------------------------------------------------------

class Normalize(Transform):
    """Normalize waveform to zero mean and unit std.

    Handles NaN values and zero-std edge cases.
    """

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


class HighpassFilter(Transform):
    """Apply highpass filter to waveform."""

    def __init__(self, freq: float = 1.0, sampling_rate: float = DEFAULT_SAMPLING_RATE):
        from scipy import signal
        self.freq = freq
        self.sampling_rate = sampling_rate
        # Design filter once
        self.sos = signal.butter(4, freq, btype='high', fs=sampling_rate, output='sos')

    def __call__(self, sample: Sample) -> Sample:
        from scipy import signal
        sample.waveform = signal.sosfilt(self.sos, sample.waveform, axis=-1).astype(np.float32)
        return sample


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


# -----------------------------------------------------------------------------
# Temporal Transforms (modify both waveform and phase indices)
# -----------------------------------------------------------------------------

class RandomCrop(Transform):
    """Randomly crop waveform to fixed length, adjusting phase indices.

    This is the seismic equivalent of RandomCrop in image augmentation.
    Phases that fall outside the crop are removed.

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

        # Find valid crop that contains phases
        all_phases = sample.p_indices + sample.s_indices
        for _ in range(self.max_tries):
            start = random.randint(0, nt - self.length)
            end = start + self.length

            # Count phases in crop
            phases_in_crop = sum(1 for p in all_phases if start <= p < end)
            if phases_in_crop >= self.min_phases:
                break

        # Apply crop
        sample.waveform = sample.waveform[..., start:end]

        # Adjust indices (remove out-of-bounds, shift remaining)
        sample.p_indices = [p - start for p in sample.p_indices if start <= p < end]
        sample.s_indices = [p - start for p in sample.s_indices if start <= p < end]
        sample.polarity_up = [p - start for p in sample.polarity_up if start <= p < end]
        sample.polarity_down = [p - start for p in sample.polarity_down if start <= p < end]
        sample.event_center = [p - start for p in sample.event_center if start <= p < end]
        sample.event_time = [p - start for p in sample.event_time if start <= p < end]

        return sample


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

        sample.waveform = sample.waveform[..., start:end]
        sample.p_indices = [p - start for p in sample.p_indices if start <= p < end]
        sample.s_indices = [p - start for p in sample.s_indices if start <= p < end]
        sample.polarity_up = [p - start for p in sample.polarity_up if start <= p < end]
        sample.polarity_down = [p - start for p in sample.polarity_down if start <= p < end]
        sample.event_center = [p - start for p in sample.event_center if start <= p < end]
        sample.event_time = [p - start for p in sample.event_time if start <= p < end]

        return sample


class RandomShift(Transform):
    """Randomly shift waveform and indices (circular or zero-pad).

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

        if self.mode == "circular":
            sample.waveform = np.roll(sample.waveform, shift, axis=-1)
            # Adjust indices with wrapping
            sample.p_indices = [(p + shift) % nt for p in sample.p_indices]
            sample.s_indices = [(p + shift) % nt for p in sample.s_indices]
            sample.polarity_up = [(p + shift) % nt for p in sample.polarity_up]
            sample.polarity_down = [(p + shift) % nt for p in sample.polarity_down]
            sample.event_center = [(p + shift) % nt for p in sample.event_center]
            sample.event_time = [(p + shift) % nt for p in sample.event_time]
        else:
            # Zero-pad mode - remove phases that shift out of bounds
            if shift > 0:
                sample.waveform = np.concatenate([
                    np.zeros_like(sample.waveform[..., :shift]),
                    sample.waveform[..., :-shift]
                ], axis=-1)
            else:
                sample.waveform = np.concatenate([
                    sample.waveform[..., -shift:],
                    np.zeros_like(sample.waveform[..., :(-shift)])
                ], axis=-1)

            sample.p_indices = [p + shift for p in sample.p_indices if 0 <= p + shift < nt]
            sample.s_indices = [p + shift for p in sample.s_indices if 0 <= p + shift < nt]
            sample.polarity_up = [p + shift for p in sample.polarity_up if 0 <= p + shift < nt]
            sample.polarity_down = [p + shift for p in sample.polarity_down if 0 <= p + shift < nt]
            sample.event_center = [p + shift for p in sample.event_center if 0 <= p + shift < nt]
            sample.event_time = [p + shift for p in sample.event_time if 0 <= p + shift < nt]

        return sample


class TimeStretch(Transform):
    """Randomly stretch/compress time axis.

    Similar to pitch shifting in audio or scaling in images.
    Phase indices are scaled accordingly.

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

        nt_orig = sample.nt
        nt_new = int(nt_orig * factor)

        # Resample waveform
        zoom_factors = [1.0] * (sample.waveform.ndim - 1) + [factor]
        sample.waveform = ndimage.zoom(sample.waveform, zoom_factors, order=1).astype(np.float32)

        # Scale indices
        sample.p_indices = [int(p * factor) for p in sample.p_indices if int(p * factor) < nt_new]
        sample.s_indices = [int(p * factor) for p in sample.s_indices if int(p * factor) < nt_new]
        sample.polarity_up = [int(p * factor) for p in sample.polarity_up if int(p * factor) < nt_new]
        sample.polarity_down = [int(p * factor) for p in sample.polarity_down if int(p * factor) < nt_new]
        sample.event_center = [p * factor for p in sample.event_center if p * factor < nt_new]
        sample.event_time = [p * factor for p in sample.event_time if p * factor < nt_new]

        return sample


# -----------------------------------------------------------------------------
# Amplitude/Polarity Transforms
# -----------------------------------------------------------------------------

class FlipPolarity(Transform):
    """Randomly flip waveform polarity (multiply by -1).

    Also swaps Up/Down polarity labels.

    Args:
        p: Probability of flipping
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample.waveform = -sample.waveform
            # Swap polarity labels
            sample.polarity_up, sample.polarity_down = sample.polarity_down, sample.polarity_up
        return sample


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
        sample.amp_signal *= scale
        sample.amp_noise *= scale
        return sample


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
        # Choose a drop pattern (always drops something when applied)
        r = random.random()
        if r < 0.15:
            sample.waveform[0] = 0  # Drop E
        elif r < 0.30:
            sample.waveform[1] = 0  # Drop N
        elif r < 0.45:
            sample.waveform[:2] = 0  # Drop E and N
        elif r < 0.60:
            sample.waveform[2] = 0  # Drop Z only
            drop_z = True
        elif r < 0.75:
            sample.waveform[0] = 0  # Drop E
            sample.waveform[2] = 0  # Drop Z
            drop_z = True
        elif r < 0.90:
            sample.waveform[1] = 0  # Drop N
            sample.waveform[2] = 0  # Drop Z
            drop_z = True
        else:
            # Drop random single channel
            ch = random.randint(0, 2)
            sample.waveform[ch] = 0
            if ch == 2:
                drop_z = True

        # If Z channel is dropped, polarity is not reliable
        if drop_z:
            sample.polarity_up = []
            sample.polarity_down = []

        return sample


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


# -----------------------------------------------------------------------------
# Stacking Transforms - Core Seismic Augmentation
# -----------------------------------------------------------------------------

# Default velocity model (km/s)
DEFAULT_VP = 6.0  # P-wave velocity
DEFAULT_VS = 3.5  # S-wave velocity


def predict_ps_times(
    distance_km: float,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    vp: float = DEFAULT_VP,
    vs: float = DEFAULT_VS,
) -> tuple[float, float]:
    """Predict P and S arrival times from distance using constant velocity model.

    Args:
        distance_km: Event-station distance in km
        sampling_rate: Samples per second
        vp: P-wave velocity in km/s
        vs: S-wave velocity in km/s

    Returns:
        (p_samples, s_samples): Predicted P and S travel times in samples
    """
    p_time_s = distance_km / vp
    s_time_s = distance_km / vs
    return p_time_s * sampling_rate, s_time_s * sampling_rate


def get_event_window(
    sample: Sample,
    vp: float = DEFAULT_VP,
    vs: float = DEFAULT_VS,
) -> tuple[int, int] | None:
    """Get event window (P to S) using picks or predicted times.

    If both P and S picks exist, use them directly.
    Otherwise, use distance and velocity model to predict.

    Args:
        sample: Sample with picks and/or distance
        vp: P-wave velocity for prediction
        vs: S-wave velocity for prediction

    Returns:
        (start, end) sample indices of event window, or None if cannot determine
    """
    # If we have both P and S, use them directly
    if sample.p_indices and sample.s_indices:
        p = min(sample.p_indices)
        s = max(sample.s_indices)  # Use max S for window end
        return (int(p), int(s)) if p < s else (int(s), int(p))

    # If we have distance, predict P-S difference
    if sample.distance_km > 0:
        p_travel, s_travel = predict_ps_times(
            sample.distance_km, sample.sampling_rate, vp, vs
        )
        ps_diff = s_travel - p_travel  # S-P time in samples

        if sample.p_indices:
            p = min(sample.p_indices)
            return (int(p), int(p + ps_diff))
        elif sample.s_indices:
            s = min(sample.s_indices)
            return (int(s - ps_diff), int(s))

    return None

class StackEvents(Transform):
    """Stack multiple events onto a single waveform (Mixup-inspired).

    This is one of the most important augmentations for seismic data.
    Similar to Mixup/CutMix in computer vision, but designed for seismic.

    Uses predicted P-S times from distance and velocity model to determine
    event windows, allowing stacking even when only P or S is available.

    Overlap Policy (allow_overlap):
        - "none": No picks can fall within another event's P-S window
        - "partial": P picks cannot fall in P-S range, S picks can overlap
        - "full": Allow any overlap (for training robustness)

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
        self._sample_buffer: list[Sample] = []
        self._sample_fn: Callable[[], Sample | None] | None = None

    def set_sample_fn(self, fn: Callable[[], Sample | None]):
        """Set function to get random samples for stacking."""
        self._sample_fn = fn

    def set_sample_buffer(self, buffer: list[Sample]):
        """Set buffer of samples for stacking."""
        self._sample_buffer = buffer

    def _get_random_sample(self) -> Sample | None:
        """Get a random sample for stacking."""
        if self._sample_fn is not None:
            return self._sample_fn()
        if self._sample_buffer:
            return random.choice(self._sample_buffer).copy()
        return None

    def _get_event_windows(self, sample: Sample) -> list[tuple[int, int]]:
        """Get all event windows (P to S ranges) for a sample."""
        windows = []

        # Get P-S difference from distance
        if sample.distance_km > 0:
            _, ps_diff = predict_ps_times(
                sample.distance_km, sample.sampling_rate, self.vp, self.vs
            )
            ps_diff = ps_diff - predict_ps_times(
                sample.distance_km, sample.sampling_rate, self.vp, self.vs
            )[0]
        else:
            ps_diff = 300  # Default ~3 seconds at 100Hz

        # Build windows from P picks
        for p in sample.p_indices:
            windows.append((int(p), int(p + ps_diff)))

        # Add windows from S picks (if no corresponding P)
        if len(sample.s_indices) > len(sample.p_indices):
            for s in sample.s_indices[len(sample.p_indices):]:
                windows.append((int(s - ps_diff), int(s)))

        return windows

    def _check_overlap(
        self,
        windows1: list[tuple[int, int]],
        windows2_shifted: list[tuple[int, int]],
        p1: list[int],
        s1: list[int],
        p2_shifted: list[int],
        s2_shifted: list[int],
    ) -> bool:
        """Check if stacking is valid based on overlap policy.

        Args:
            windows1: Event windows from sample1
            windows2_shifted: Event windows from sample2 (shifted)
            p1, s1: P and S picks from sample1
            p2_shifted, s2_shifted: Shifted picks from sample2

        Returns True if stacking is allowed.
        """
        if self.allow_overlap == "full":
            return True

        # Check if new P picks fall within existing windows
        for p2 in p2_shifted:
            for w1_start, w1_end in windows1:
                if w1_start <= p2 <= w1_end:
                    return False  # P pick in existing event window

        if self.allow_overlap == "none":
            # Also check S picks in existing windows
            for s2 in s2_shifted:
                for w1_start, w1_end in windows1:
                    if w1_start <= s2 <= w1_end:
                        return False

            # Check reverse: existing picks in new windows
            for p in p1:
                for w2_start, w2_end in windows2_shifted:
                    if w2_start <= p <= w2_end:
                        return False
            for s in s1:
                for w2_start, w2_end in windows2_shifted:
                    if w2_start <= s <= w2_end:
                        return False

        return True

    def __call__(self, sample: Sample) -> Sample:
        n_events = random.randint(1, self.max_events)

        # Get existing event windows
        windows1 = self._get_event_windows(sample)

        for _ in range(n_events):
            sample2 = self._get_random_sample()
            if sample2 is None:
                continue

            # Ensure same shape
            if sample2.waveform.shape != sample.waveform.shape:
                continue

            # Check SNR/amplitude compatibility
            if sample.amp_signal == 0 or sample.amp_noise == 0:
                continue
            if sample2.amp_signal == 0 or sample2.amp_noise == 0:
                continue

            # Get first arrival for alignment
            first_arrival1 = min(sample.p_indices + sample.s_indices) if (sample.p_indices or sample.s_indices) else sample.nt // 2
            first_arrival2 = min(sample2.p_indices + sample2.s_indices) if (sample2.p_indices or sample2.s_indices) else sample2.nt // 2

            for _ in range(self.max_tries):
                shift = random.randint(-self.max_shift, self.max_shift) + first_arrival1 - first_arrival2
                nt = sample.nt

                # Compute shifted indices
                p2_shifted = [(p + shift) % nt for p in sample2.p_indices]
                s2_shifted = [(s + shift) % nt for s in sample2.s_indices]

                # Get shifted windows
                windows2 = self._get_event_windows(sample2)
                windows2_shifted = [((w[0] + shift) % nt, (w[1] + shift) % nt) for w in windows2]

                if self._check_overlap(windows1, windows2_shifted, sample.p_indices, sample.s_indices, p2_shifted, s2_shifted):
                    # Found non-overlapping position
                    # Calculate amplitude ratio
                    min_r = max(self.min_ratio, np.log10(sample.amp_noise * 2 / sample2.amp_signal + 1e-10))
                    max_r = min(self.max_ratio, np.log10(sample.amp_signal / 2 / sample2.amp_noise + 1e-10))
                    if min_r > max_r:
                        continue

                    ratio = 10 ** random.uniform(min_r, max_r)
                    flip = random.choice([-1.0, 1.0])

                    # Stack waveforms
                    sample.waveform = sample.waveform + np.roll(sample2.waveform, shift, axis=-1) * ratio * flip

                    # Merge phase indices
                    nt = sample.nt
                    sample.p_indices += [(p + shift) % nt for p in sample2.p_indices]
                    sample.s_indices += [(p + shift) % nt for p in sample2.s_indices]

                    # Handle polarity with flip
                    if flip > 0:
                        sample.polarity_up += [(p + shift) % nt for p in sample2.polarity_up]
                        sample.polarity_down += [(p + shift) % nt for p in sample2.polarity_down]
                    else:
                        sample.polarity_up += [(p + shift) % nt for p in sample2.polarity_down]
                        sample.polarity_down += [(p + shift) % nt for p in sample2.polarity_up]

                    sample.event_center += [(p + shift) % nt for p in sample2.event_center]
                    sample.event_time += [(p + shift) % nt for p in sample2.event_time]

                    # Update SNR estimates
                    sample.amp_noise = max(sample.amp_noise, sample2.amp_noise * ratio)
                    sample.amp_signal = min(sample.amp_signal, sample2.amp_signal * ratio)
                    break

        return sample

    def __repr__(self) -> str:
        return f"StackEvents(max_events={self.max_events}, max_shift={self.max_shift})"


class StackNoise(Transform):
    """Stack noise from another sample onto the waveform.

    Extracts noise segment (before first arrival) from a random sample
    and adds it to the current waveform.

    Args:
        max_ratio: Maximum noise amplitude ratio relative to signal
    """

    def __init__(self, max_ratio: float = 2.0):
        self.max_ratio = max_ratio
        self._sample_fn: Callable[[], Sample | None] | None = None

    def set_sample_fn(self, fn: Callable[[], Sample | None]):
        """Set function to get random samples for noise extraction."""
        self._sample_fn = fn

    def _get_noise(self, sample: Sample, length: int) -> np.ndarray | None:
        """Extract noise segment from sample."""
        first_arrival = min(sample.p_indices + sample.s_indices) if (sample.p_indices or sample.s_indices) else 0

        if first_arrival < length + 10:
            return None

        # Extract noise from before first arrival
        for _ in range(10):
            shift = random.randint(10 - first_arrival, 0)
            noise_segment = np.roll(sample.waveform, shift, axis=-1)[..., -length:]

            # Check this is actually noise (no phases)
            shifted_arrivals = [p + shift for p in sample.p_indices + sample.s_indices]
            if not any(-length <= p < 0 for p in shifted_arrivals):
                return noise_segment
        return None

    def __call__(self, sample: Sample) -> Sample:
        if self._sample_fn is None:
            return sample

        noise_sample = self._sample_fn()
        if noise_sample is None:
            return sample

        noise = self._get_noise(noise_sample, sample.nt)
        if noise is None:
            return sample

        if sample.amp_signal == 0:
            return sample

        ratio = random.uniform(0, self.max_ratio) * sample.amp_signal
        noise_std = np.std(noise)
        if noise_std > 0:
            noise = noise / noise_std * ratio
            sample.waveform = sample.waveform + noise

        return sample

    def __repr__(self) -> str:
        return f"StackNoise(max_ratio={self.max_ratio})"


# =============================================================================
# Label Generation
# =============================================================================

def generate_gaussian_label(
    indices: list[int | float],
    length: int,
    width: int = 60,
    threshold: float = 0.1,
) -> np.ndarray:
    """Generate Gaussian labels from phase indices.

    Args:
        indices: List of phase arrival indices
        length: Output array length
        width: Gaussian width (samples)
        threshold: Values below this are zeroed

    Returns:
        Label array of shape (length,)
    """
    label = np.zeros(length, dtype=np.float32)
    sigma = width / 6  # width is ~6 sigma

    t = np.arange(length)
    for idx in indices:
        gaussian = np.exp(-((t - idx) ** 2) / (2 * sigma ** 2))
        gaussian[gaussian < threshold] = 0.0
        label += gaussian

    return label


def generate_phase_labels(
    sample: Sample,
    config: LabelConfig = LabelConfig(),
) -> dict[str, np.ndarray]:
    """Generate all labels from a Sample.

    This is called after all transforms are applied.

    Returns:
        Dictionary with keys:
        - phase_pick: (3, nt) - [noise, P, S]
        - phase_mask: (nt,) - mask around picks
        - polarity: (nt,) - polarity label (0.5 = unknown, 0 = down, 1 = up)
        - polarity_mask: (nt,) - mask for polarity
        - event_center: (nt,) - event center label (Gaussian at event locations)
        - event_time: (nt,) - event time regression target
        - event_center_mask: (nt,) - mask for center detection loss (where Gaussian > 0)
        - event_time_mask: (nt,) - mask for time regression loss (narrow window around centers)
    """
    nt = sample.nt

    # Phase labels
    p_label = generate_gaussian_label(sample.p_indices, nt, config.phase_width, config.gaussian_threshold)
    s_label = generate_gaussian_label(sample.s_indices, nt, config.phase_width, config.gaussian_threshold)
    noise_label = np.maximum(0, 1.0 - p_label - s_label)
    phase_pick = np.stack([noise_label, p_label, s_label], axis=0)

    # Phase mask
    mask_width = int(config.phase_width * config.mask_width_factor)
    phase_mask = np.zeros(nt, dtype=np.float32)
    for idx in sample.p_indices + sample.s_indices:
        start = max(0, int(idx) - mask_width)
        end = min(nt, int(idx) + mask_width)
        phase_mask[start:end] = 1.0

    # Polarity labels
    up_label = generate_gaussian_label(sample.polarity_up, nt, config.polarity_width, config.gaussian_threshold)
    down_label = generate_gaussian_label(sample.polarity_down, nt, config.polarity_width, config.gaussian_threshold)
    polarity = (up_label - down_label + 1.0) / 2.0  # Map to [0, 1]
    polarity_mask = np.zeros(nt, dtype=np.float32)
    for idx in sample.polarity_up + sample.polarity_down:
        start = max(0, int(idx) - mask_width)
        end = min(nt, int(idx) + mask_width)
        polarity_mask[start:end] = 1.0

    # Event center labels
    event_center = generate_gaussian_label(sample.event_center, nt, config.event_width, 0.05)
    event_time = np.zeros(nt, dtype=np.float32)
    # event_center_mask: where to compute center detection loss (where Gaussian is non-zero)
    event_center_mask = (event_center > 0).astype(np.float32)
    # event_time_mask: where to compute time regression loss (narrow window for accurate timing)
    event_time_mask = np.zeros(nt, dtype=np.float32)

    event_mask_width = int(config.event_width * config.mask_width_factor)
    for center, time in zip(sample.event_center, sample.event_time):
        start = max(0, int(center) - event_mask_width)
        end = min(nt, int(center) + event_mask_width)
        event_time_mask[start:end] = 1.0
        t = np.arange(nt) - time
        event_time[start:end] = t[start:end]

    return {
        "phase_pick": phase_pick,
        "phase_mask": phase_mask,
        "polarity": polarity,
        "polarity_mask": polarity_mask,
        "event_center": event_center,
        "event_time": event_time,
        "event_center_mask": event_center_mask,
        "event_time_mask": event_time_mask,
    }


# =============================================================================
# Default Transform Presets
# =============================================================================

def default_train_transforms(
    crop_length: int = 4096,
    enable_stacking: bool = True,
    enable_noise_stacking: bool = True,
) -> Compose:
    """Default transforms for training.

    Args:
        crop_length: Length to crop waveforms to
        enable_stacking: Enable event stacking augmentation
        enable_noise_stacking: Enable noise stacking augmentation
    """
    transforms = [
        Normalize(),
    ]

    if enable_stacking:
        transforms.append(StackEvents(max_events=2, max_shift=4096))

    transforms.extend([
        RandomCrop(crop_length),
    ])

    if enable_noise_stacking:
        transforms.append(StackNoise(max_ratio=2.0))

    transforms.extend([
        FlipPolarity(p=0.5),
        RandomAmplitudeScale(min_scale=0.5, max_scale=2.0),
        DropChannel(p=0.1),
        Normalize(),  # Final normalization
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
    """Minimal transforms - just normalize."""
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
        region: "SC" (Southern California) or "NC" (Northern California).
                Can be comma-separated for multiple regions, e.g., "NC,SC".
        years: List of years to load, e.g., [2025, 2026]. None = all available.
        days: List of days to load, e.g., [1, 2, 3]. None = all days.
        streaming: If True, stream data without downloading everything first.

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_dataset

    storage_options = get_gcs_storage_options()

    # Handle comma-separated regions
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
        "parquet",
        data_files=pattern,
        streaming=streaming,
        storage_options=storage_options,
    )

    return dataset["train"]


def record_to_sample(record: dict) -> Sample:
    """Convert a HuggingFace dataset record to a Sample."""
    waveform = np.array(record["waveform"], dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform.reshape(3, -1)

    # Extract phase indices
    p_indices = []
    s_indices = []
    polarity_up = []
    polarity_down = []

    if record.get("p_phase_index") is not None:
        p_indices = [int(record["p_phase_index"])]
    if record.get("s_phase_index") is not None:
        s_indices = [int(record["s_phase_index"])]

    # Handle polarity if available
    if record.get("p_phase_polarity") == "U":
        polarity_up = p_indices.copy()
    elif record.get("p_phase_polarity") == "D":
        polarity_down = p_indices.copy()

    # Compute event center (midpoint between P and S)
    event_center = []
    event_time = []
    if p_indices and s_indices:
        event_center = [(p_indices[0] + s_indices[0]) / 2]
        if record.get("event_time_index") is not None:
            event_time = [record["event_time_index"]]

    # Calculate SNR and get distance
    snr = record.get("snr", 0.0) or 0.0
    distance_km = record.get("distance_km", 0.0) or 0.0

    return Sample(
        waveform=waveform,
        p_indices=p_indices,
        s_indices=s_indices,
        polarity_up=polarity_up,
        polarity_down=polarity_down,
        event_center=event_center,
        event_time=event_time,
        snr=snr,
        distance_km=distance_km,
        trace_id=f"{record.get('event_id', '')}/{record.get('network', '')}.{record.get('station', '')}",
        sensor=record.get("instrument", ""),
        sampling_rate=DEFAULT_SAMPLING_RATE,
    )


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
        self.count = 0  # Total samples seen (for reservoir sampling)

    def add(self, sample: Sample):
        """Add a sample to the buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample.copy())
        else:
            # Reservoir sampling
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


class CEEDDataset(Dataset):
    """California Earthquake Event Dataset - Map-style Dataset.

    For use with DataLoader when data fits in memory or is pre-downloaded.

    Args:
        region: "SC" or "NC"
        years: List of years to load
        days: List of days to load
        transforms: Transform pipeline to apply
        label_config: Configuration for label generation
        min_snr: Minimum SNR to include sample
        buffer_size: Size of sample buffer for stacking
        preload: If True, load all data into memory
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
        preload: bool = False,
    ):
        self.transforms = transforms or minimal_transforms()
        self.label_config = label_config
        self.min_snr = min_snr

        # Load dataset
        hf_dataset = load_quakeflow_dataset(region, years, days, streaming=False)

        # Convert to samples and calculate SNR
        self.samples: list[Sample] = []
        for record in hf_dataset:
            sample = record_to_sample(record)
            if not sample.p_indices or not sample.s_indices:
                continue

            # Calculate SNR during loading (required for stacking)
            sample.snr, sample.amp_signal, sample.amp_noise = self._calc_snr(sample)

            if sample.snr >= min_snr and sample.snr > 0:
                self.samples.append(sample)

        # Setup sample buffer for stacking transforms
        self.sample_buffer = SampleBuffer(buffer_size)
        for sample in random.sample(self.samples, min(buffer_size, len(self.samples))):
            self.sample_buffer.add(sample)

        # Connect stacking transforms to buffer
        self._setup_stacking_transforms()

        print(f"CEEDDataset: loaded {len(self.samples)} samples")

    def _setup_stacking_transforms(self):
        """Connect stacking transforms to sample buffer."""
        if isinstance(self.transforms, Compose):
            for t in self.transforms.transforms:
                if isinstance(t, (StackEvents, StackNoise)):
                    t.set_sample_fn(self._get_random_sample)

    def _get_random_sample(self) -> Sample | None:
        """Get random sample for stacking."""
        return self.sample_buffer.get_random()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx].copy()

        # Apply transforms
        sample = self.transforms(sample)

        # Generate labels
        labels = generate_phase_labels(sample, self.label_config)

        # Convert to tensors
        return {
            "data": torch.from_numpy(sample.waveform[:, np.newaxis, :]).float(),
            "phase_pick": torch.from_numpy(labels["phase_pick"][:, np.newaxis, :]).float(),
            "phase_mask": torch.from_numpy(labels["phase_mask"][np.newaxis, np.newaxis, :]).float(),
            "polarity": torch.from_numpy(labels["polarity"][np.newaxis, np.newaxis, :]).float(),
            "polarity_mask": torch.from_numpy(labels["polarity_mask"][np.newaxis, np.newaxis, :]).float(),
            "event_center": torch.from_numpy(labels["event_center"][np.newaxis, np.newaxis, :]).float(),
            "event_time": torch.from_numpy(labels["event_time"][np.newaxis, np.newaxis, :]).float(),
            "event_center_mask": torch.from_numpy(labels["event_center_mask"][np.newaxis, np.newaxis, :]).float(),
            "event_time_mask": torch.from_numpy(labels["event_time_mask"][np.newaxis, np.newaxis, :]).float(),
        }

    def _calc_snr(
        self,
        sample: Sample,
        noise_window: int = 300,
        signal_window: int = 300,
        gap_window: int = 50,
    ) -> tuple[float, float, float]:
        """Calculate SNR from waveform."""
        waveform = sample.waveform
        picks = sample.p_indices

        snrs, signals, noises = [], [], []
        for i in range(waveform.shape[0]):
            for j in picks:
                if gap_window < j < waveform.shape[-1] - gap_window:
                    noise = np.std(waveform[i, max(0, j - noise_window):j - gap_window])
                    signal = np.std(waveform[i, j + gap_window:j + signal_window])
                    if noise > 0 and signal > 0:
                        snrs.append(signal / noise)
                        signals.append(signal)
                        noises.append(noise)

        if not snrs:
            return 0.0, 0.0, 0.0

        idx = np.argmax(snrs)
        return snrs[idx], signals[idx], noises[idx]


class CEEDIterableDataset(IterableDataset):
    """California Earthquake Event Dataset - Iterable/Streaming Dataset.

    For use with DataLoader when streaming data from GCS.
    Supports distributed training via worker_init_fn.

    Args:
        region: "SC" or "NC"
        years: List of years to load
        days: List of days to load
        transforms: Transform pipeline to apply
        label_config: Configuration for label generation
        min_snr: Minimum SNR to include sample
        buffer_size: Size of sample buffer for stacking
        shuffle_buffer_size: Size of shuffle buffer
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
    ):
        self.region = region
        self.years = years
        self.days = days
        self.transforms = transforms or minimal_transforms()
        self.label_config = label_config
        self.min_snr = min_snr
        self.buffer_size = buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Get worker info for distributed
        worker_info = torch.utils.data.get_worker_info()

        # Load streaming dataset
        hf_dataset = load_quakeflow_dataset(
            self.region, self.years, self.days, streaming=True
        )

        # Setup sample buffer
        sample_buffer = SampleBuffer(self.buffer_size)

        # Connect stacking transforms
        if isinstance(self.transforms, Compose):
            for t in self.transforms.transforms:
                if isinstance(t, (StackEvents, StackNoise)):
                    t.set_sample_fn(sample_buffer.get_random)

        # Shuffle buffer for randomization
        shuffle_buffer: list[Sample] = []

        for i, record in enumerate(hf_dataset):
            # Skip samples for other workers
            if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                continue

            sample = record_to_sample(record)

            # Filter by SNR and phase presence
            if sample.snr < self.min_snr or not sample.p_indices or not sample.s_indices:
                continue

            # Calculate amp for stacking
            sample.snr, sample.amp_signal, sample.amp_noise = self._calc_snr(sample)
            if sample.snr == 0:
                continue

            # Add to buffer for stacking
            sample_buffer.add(sample)

            # Add to shuffle buffer and yield when full
            shuffle_buffer.append(sample)
            if len(shuffle_buffer) >= self.shuffle_buffer_size:
                random.shuffle(shuffle_buffer)
                for s in shuffle_buffer:
                    yield self._process_sample(s.copy())
                shuffle_buffer.clear()

        # Yield remaining samples in buffer
        if shuffle_buffer:
            random.shuffle(shuffle_buffer)
            for s in shuffle_buffer:
                yield self._process_sample(s.copy())

    def _process_sample(self, sample: Sample) -> dict[str, torch.Tensor]:
        """Apply transforms and generate labels."""
        sample = self.transforms(sample)
        labels = generate_phase_labels(sample, self.label_config)

        return {
            "data": torch.from_numpy(sample.waveform[:, np.newaxis, :]).float(),
            "phase_pick": torch.from_numpy(labels["phase_pick"][:, np.newaxis, :]).float(),
            "phase_mask": torch.from_numpy(labels["phase_mask"][np.newaxis, np.newaxis, :]).float(),
            "polarity": torch.from_numpy(labels["polarity"][np.newaxis, np.newaxis, :]).float(),
            "polarity_mask": torch.from_numpy(labels["polarity_mask"][np.newaxis, np.newaxis, :]).float(),
            "event_center": torch.from_numpy(labels["event_center"][np.newaxis, np.newaxis, :]).float(),
            "event_time": torch.from_numpy(labels["event_time"][np.newaxis, np.newaxis, :]).float(),
            "event_center_mask": torch.from_numpy(labels["event_center_mask"][np.newaxis, np.newaxis, :]).float(),
            "event_time_mask": torch.from_numpy(labels["event_time_mask"][np.newaxis, np.newaxis, :]).float(),
        }

    def _calc_snr(
        self,
        sample: Sample,
        noise_window: int = 300,
        signal_window: int = 300,
        gap_window: int = 50,
    ) -> tuple[float, float, float]:
        """Calculate SNR from waveform."""
        waveform = sample.waveform
        picks = sample.p_indices

        snrs, signals, noises = [], [], []
        for i in range(waveform.shape[0]):
            for j in picks:
                if gap_window < j < waveform.shape[-1] - gap_window:
                    noise = np.std(waveform[i, max(0, j - noise_window):j - gap_window])
                    signal = np.std(waveform[i, j + gap_window:j + signal_window])
                    if noise > 0 and signal > 0:
                        snrs.append(signal / noise)
                        signals.append(signal)
                        noises.append(noise)

        if not snrs:
            return 0.0, 0.0, 0.0

        idx = np.argmax(snrs)
        return snrs[idx], signals[idx], noises[idx]


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
    """Create a training dataset with default augmentation.

    Args:
        region: "SC" or "NC"
        years: Years to load
        days: Days to load
        crop_length: Length to crop waveforms
        streaming: Use streaming dataset
        **kwargs: Additional arguments to dataset class

    Returns:
        Dataset instance
    """
    transforms = default_train_transforms(crop_length=crop_length)

    if streaming:
        return CEEDIterableDataset(
            region=region,
            years=years,
            days=days,
            transforms=transforms,
            **kwargs,
        )
    else:
        return CEEDDataset(
            region=region,
            years=years,
            days=days,
            transforms=transforms,
            **kwargs,
        )


def create_eval_dataset(
    region: str = "SC",
    years: list[int] | None = None,
    days: list[int] | None = None,
    crop_length: int = 4096,
    streaming: bool = False,
    **kwargs,
) -> Dataset | IterableDataset:
    """Create an evaluation dataset without augmentation.

    Args:
        region: "SC" or "NC"
        years: Years to load
        days: Days to load
        crop_length: Length to crop waveforms
        streaming: Use streaming dataset
        **kwargs: Additional arguments to dataset class

    Returns:
        Dataset instance
    """
    transforms = default_eval_transforms(crop_length=crop_length)

    if streaming:
        return CEEDIterableDataset(
            region=region,
            years=years,
            days=days,
            transforms=transforms,
            **kwargs,
        )
    else:
        return CEEDDataset(
            region=region,
            years=years,
            days=days,
            transforms=transforms,
            **kwargs,
        )


# =============================================================================
# Main - Demo and Testing
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("CEED Dataset Demo")
    print("=" * 60)

    # Demo transforms
    print("\n1. Transform Pipeline Demo")
    print("-" * 40)

    # Create a synthetic sample for testing transforms
    nt = 8192
    sample = Sample(
        waveform=np.random.randn(3, nt).astype(np.float32),
        p_indices=[2000],
        s_indices=[3000],
        polarity_up=[2000],
        amp_signal=1.0,
        amp_noise=0.1,
    )

    transforms = default_train_transforms(crop_length=4096, enable_stacking=False)
    print(f"Transforms:\n{transforms}")

    transformed = transforms(sample)
    print(f"\nOriginal shape: (3, {nt})")
    print(f"Transformed shape: {transformed.waveform.shape}")
    print(f"P indices: {sample.p_indices} -> {transformed.p_indices}")
    print(f"S indices: {sample.s_indices} -> {transformed.s_indices}")

    # Generate labels
    print("\n2. Label Generation Demo")
    print("-" * 40)

    labels = generate_phase_labels(transformed)
    for key, value in labels.items():
        print(f"  {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Waveform
    ax = axes[0]
    for i, name in enumerate(["E", "N", "Z"]):
        ax.plot(transformed.waveform[i] / 3 + i, label=name, alpha=0.7)
    ax.set_ylabel("Waveform")
    ax.legend(loc="upper right")

    # Phase labels
    ax = axes[1]
    ax.plot(labels["phase_pick"][1], label="P", color="blue")
    ax.plot(labels["phase_pick"][2], label="S", color="red")
    ax.fill_between(range(len(labels["phase_mask"])), labels["phase_mask"] * 0.5, alpha=0.2, label="mask")
    ax.set_ylabel("Phase Labels")
    ax.legend(loc="upper right")

    # Polarity
    ax = axes[2]
    ax.plot(labels["polarity"], label="Polarity", color="green")
    ax.fill_between(range(len(labels["polarity_mask"])), labels["polarity_mask"] * 0.5, alpha=0.2)
    ax.set_ylabel("Polarity")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Event
    ax = axes[3]
    ax.plot(labels["event_center"], label="Event Center", color="purple")
    ax.fill_between(range(len(labels["event_time_mask"])), labels["event_time_mask"] * 0.5, alpha=0.2, label="Time Mask")
    ax.fill_between(range(len(labels["event_center_mask"])), labels["event_center_mask"] * 0.3, alpha=0.1, label="Center Mask")
    ax.set_ylabel("Event")
    ax.set_xlabel("Sample")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("ceed_demo.png", dpi=150, bbox_inches="tight")
    print("\nSaved demo plot to ceed_demo.png")

    # Try loading real data if available
    print("\n3. Real Data Loading (if GCS credentials available)")
    print("-" * 40)

    try:
        dataset = CEEDDataset(
            region="SC",
            years=[2025],
            days=[9],
            transforms=default_train_transforms(crop_length=4096),
            min_snr=3.0,
        )
        print(f"Loaded {len(dataset)} samples")

        # Get a sample
        sample = dataset[0]
        print("\nSample tensors:")
        for key, value in sample.items():
            print(f"  {key}: {value.shape}")

    except Exception as e:
        print(f"Could not load real data: {e}")
        print("(This is expected if GCS credentials are not configured)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
