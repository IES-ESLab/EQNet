# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "numpy",
#     "torch",
#     "matplotlib",
#     "gcsfs",
# ]
# ///
"""
Generate verification figures for CEED dataset.

Saves figures to sources/figures/ folder:
1. transform_demo.png - Shows effect of each transform
2. label_generation.png - Shows label generation from phase indices
3. stacking_demo.png - Shows event stacking augmentation
4. real_data_samples.png - Shows real data samples from GCS
5. training_batch.png - Shows a training batch with augmentation
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from ceed import (
    Sample,
    LabelConfig,
    Normalize,
    RandomCrop,
    CenterCrop,
    RandomShift,
    FlipPolarity,
    DropChannel,
    AddGaussianNoise,
    StackEvents,
    StackNoise,
    Compose,
    generate_phase_labels,
    default_train_transforms,
    default_eval_transforms,
    CEEDDataset,
    SampleBuffer,
    LabelConfig,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def create_synthetic_sample(seed: int = 42, p_idx: int = 3000, s_idx: int = 4500) -> Sample:
    """Create a synthetic sample with realistic waveform pattern."""
    np.random.seed(seed)
    nt = 8192
    t = np.arange(nt)

    waveform = np.random.randn(3, nt).astype(np.float32) * 0.1  # Background noise

    # Add P arrival (impulsive)
    for ch in range(3):
        p_signal = np.exp(-((t - p_idx) ** 2) / 5000) * np.sin(2 * np.pi * 5 * (t - p_idx) / 100)
        p_signal *= np.exp(-(t - p_idx) / 200) * (t >= p_idx)
        waveform[ch] += p_signal * (1.0 if ch == 2 else 0.5)

    # Add S arrival (larger, lower frequency)
    for ch in range(3):
        s_signal = np.exp(-((t - s_idx) ** 2) / 8000) * np.sin(2 * np.pi * 2 * (t - s_idx) / 100)
        s_signal *= np.exp(-(t - s_idx) / 400) * (t >= s_idx)
        waveform[ch] += s_signal * (1.5 if ch < 2 else 0.8)

    return Sample(
        waveform=waveform.astype(np.float32),
        p_indices=[p_idx],
        s_indices=[s_idx],
        polarity_up=[p_idx],
        event_center=[(p_idx + s_idx) / 2],
        event_time=[p_idx - 200],
        snr=10.0,
        amp_signal=1.5,
        amp_noise=0.1,
        trace_id="synthetic/XX.STA01",
        sensor="HH",
    )


def plot_sample(ax, sample: Sample, title: str = "", show_picks: bool = True):
    """Plot a sample's waveform."""
    colors = ["#1f77b4", "#2ca02c", "#d62728"]  # E, N, Z
    labels = ["E", "N", "Z"]

    for i in range(3):
        offset = i * 2
        normalized = sample.waveform[i] / (np.abs(sample.waveform[i]).max() + 1e-10)
        ax.plot(normalized + offset, color=colors[i], linewidth=0.5, alpha=0.8)
        ax.text(-50, offset, labels[i], fontsize=9, va="center", ha="right")

    if show_picks:
        for p in sample.p_indices:
            if 0 <= p < sample.nt:
                ax.axvline(p, color="blue", linestyle="--", linewidth=1, alpha=0.7, label="P" if p == sample.p_indices[0] else "")
        for s in sample.s_indices:
            if 0 <= s < sample.nt:
                ax.axvline(s, color="red", linestyle="--", linewidth=1, alpha=0.7, label="S" if s == sample.s_indices[0] else "")

    ax.set_xlim(0, sample.nt)
    ax.set_ylim(-1, 6)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def figure_transform_demo():
    """Figure 1: Demonstrate effect of each transform."""
    print("Generating transform_demo.png...")

    sample = create_synthetic_sample()

    transforms = [
        ("Original", None),
        ("Normalize", Normalize()),
        ("RandomCrop(4096)", RandomCrop(4096)),
        ("RandomShift(500)", RandomShift(500)),
        ("FlipPolarity", FlipPolarity(p=1.0)),
        ("DropChannel", DropChannel(p=1.0)),
        ("AddGaussianNoise", AddGaussianNoise(snr_db_range=(5, 5))),
    ]

    fig, axes = plt.subplots(len(transforms), 1, figsize=(14, 2 * len(transforms)))

    for ax, (name, transform) in zip(axes, transforms):
        if transform is None:
            result = sample.copy()
        else:
            result = transform(sample.copy())
        plot_sample(ax, result, title=name)

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "transform_demo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved transform_demo.png")


def figure_label_generation():
    """Figure 2: Show label generation from phase indices."""
    print("Generating label_generation.png...")

    sample = create_synthetic_sample()
    # Use CenterCrop to ensure both P (2000) and S (3500) picks are visible
    # CenterCrop from position 1000-5096 would include both picks
    sample = CenterCrop(4096)(sample)

    labels = generate_phase_labels(sample)

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    # Waveform
    ax = axes[0]
    plot_sample(ax, sample, title="Waveform with P (blue) and S (red) picks")
    ax.legend(loc="upper right")

    # Phase pick labels
    ax = axes[1]
    ax.fill_between(range(sample.nt), labels["phase_pick"][0], alpha=0.3, label="Noise", color="gray")
    ax.plot(labels["phase_pick"][1], label="P label", color="blue", linewidth=1.5)
    ax.plot(labels["phase_pick"][2], label="S label", color="red", linewidth=1.5)
    ax.set_ylabel("Phase Labels")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")
    ax.set_title("Phase Pick Labels (Gaussian)")

    # Phase mask
    ax = axes[2]
    ax.fill_between(range(sample.nt), labels["phase_mask"], alpha=0.5, color="orange")
    ax.set_ylabel("Phase Mask")
    ax.set_ylim(0, 1.1)
    ax.set_title("Phase Mask (regions to compute loss)")

    # Polarity
    ax = axes[3]
    ax.plot(labels["polarity"], color="green", linewidth=1.5)
    ax.fill_between(range(sample.nt), labels["polarity_mask"], alpha=0.3, color="green")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Polarity")
    ax.set_ylim(0, 1.1)
    ax.set_title("Polarity Label (0=Down, 0.5=Unknown, 1=Up)")

    # Event center
    ax = axes[4]
    ax.plot(labels["event_center"], color="purple", linewidth=1.5)
    ax.fill_between(range(sample.nt), labels["event_mask"], alpha=0.3, color="purple")
    ax.set_ylabel("Event Center")
    ax.set_xlabel("Sample")
    ax.set_title("Event Center Label")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_generation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved label_generation.png")


def figure_stacking_demo():
    """Figure 3: Demonstrate event stacking augmentation."""
    print("Generating stacking_demo.png...")

    # Create two samples with different pick locations for stacking demo
    sample1 = create_synthetic_sample(seed=42, p_idx=2000, s_idx=3500)
    sample2 = create_synthetic_sample(seed=123, p_idx=5000, s_idx=6500)

    # Create buffer with sample2
    buffer = SampleBuffer(10)
    buffer.add(sample2)

    # Stack events
    stacker = StackEvents(max_events=1, max_shift=500)
    stacker.set_sample_fn(buffer.get_random)

    stacked = stacker(sample1.copy())

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    plot_sample(ax, sample1, title="Sample 1 (original)")

    ax = axes[1]
    plot_sample(ax, sample2, title="Sample 2 (to be stacked)")

    ax = axes[2]
    plot_sample(ax, stacked, title=f"Stacked Result (P: {stacked.p_indices}, S: {stacked.s_indices})")

    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "stacking_demo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved stacking_demo.png")


def figure_real_data():
    """Figure 4: Show real data samples from GCS."""
    print("Generating real_data_samples.png...")

    try:
        dataset = CEEDDataset(
            region="SC",
            years=[2025],
            days=[9],
            transforms=default_eval_transforms(crop_length=4096),
            min_snr=5.0,
        )

        n_samples = min(6, len(dataset))
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 2 * n_samples))

        for i in range(n_samples):
            batch = dataset[i]

            # Waveform
            ax = axes[i, 0]
            waveform = batch["data"].numpy()[:, 0, :]  # (3, nt)
            for ch in range(3):
                normalized = waveform[ch] / (np.abs(waveform[ch]).max() + 1e-10)
                ax.plot(normalized + ch * 2, linewidth=0.5)
            ax.set_title(f"Sample {i+1}: Waveform")
            ax.set_yticks([])

            # Phase labels
            ax = axes[i, 1]
            phase_pick = batch["phase_pick"].numpy()[:, 0, :]  # (3, nt)
            ax.plot(phase_pick[1], label="P", color="blue", linewidth=1)
            ax.plot(phase_pick[2], label="S", color="red", linewidth=1)
            ax.set_title(f"Sample {i+1}: Phase Labels")
            ax.set_ylim(0, 1.1)
            if i == 0:
                ax.legend(loc="upper right")

        axes[-1, 0].set_xlabel("Sample")
        axes[-1, 1].set_xlabel("Sample")

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "real_data_samples.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved real_data_samples.png")

    except Exception as e:
        print(f"  Could not load real data: {e}")
        print("  Skipping real_data_samples.png")


def figure_training_batch():
    """Figure 5: Show training batch with stacked events."""
    print("Generating training_batch.png...")

    try:
        dataset = CEEDDataset(
            region="SC",
            years=[2025],
            days=[9],
            transforms=default_train_transforms(crop_length=4096, enable_stacking=True),
            min_snr=3.0,
            buffer_size=100,
        )

        # Find samples that have been stacked (multiple P or S picks)
        n_samples = 4
        fig, axes = plt.subplots(n_samples, 3, figsize=(16, 3 * n_samples))

        stacked_count = 0
        for i in range(min(100, len(dataset))):
            if stacked_count >= n_samples:
                break

            # Get raw sample and transform it
            raw_sample = dataset.samples[i].copy()
            transformed = dataset.transforms(raw_sample)

            # Only use samples with stacking (multiple events)
            if len(transformed.p_indices) < 2 or len(transformed.s_indices) < 2:
                continue

            labels = generate_phase_labels(transformed)

            # Waveform with pick markers
            ax = axes[stacked_count, 0]
            for ch in range(3):
                normalized = transformed.waveform[ch] / (np.abs(transformed.waveform[ch]).max() + 1e-10)
                ax.plot(normalized + ch * 2, linewidth=0.5)
            # Mark picks
            for p in transformed.p_indices:
                ax.axvline(p, color="blue", linestyle="--", alpha=0.7, linewidth=1)
            for s in transformed.s_indices:
                ax.axvline(s, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.set_title(f"Stacked Sample: P={transformed.p_indices}")
            ax.set_yticks([])
            ax.set_xlim(0, transformed.nt)

            # Phase labels
            ax = axes[stacked_count, 1]
            ax.fill_between(range(transformed.nt), labels["phase_pick"][0], alpha=0.2, color="gray")
            ax.plot(labels["phase_pick"][1], label="P", color="blue", linewidth=1.5)
            ax.plot(labels["phase_pick"][2], label="S", color="red", linewidth=1.5)
            ax.set_title(f"S={transformed.s_indices}")
            ax.set_ylim(0, 1.1)
            if stacked_count == 0:
                ax.legend(loc="upper right")

            # Polarity and mask
            ax = axes[stacked_count, 2]
            ax.plot(labels["polarity"], color="green", linewidth=1.5, label="Polarity")
            ax.fill_between(range(transformed.nt), labels["polarity_mask"] * 0.5, alpha=0.3, color="green", label="Mask")
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_title("Polarity")
            ax.set_ylim(0, 1.1)
            if stacked_count == 0:
                ax.legend(loc="upper right")

            stacked_count += 1

        for ax in axes[-1]:
            ax.set_xlabel("Sample")

        plt.suptitle("Training Batch with Stacked Events (Multiple P/S per sample)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "training_batch.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved training_batch.png")

    except Exception as e:
        import traceback
        print(f"  Could not generate training batch: {e}")
        traceback.print_exc()
        print("  Skipping training_batch.png")


def figure_augmentation_comparison():
    """Figure 6: Compare original vs augmented samples side by side."""
    print("Generating augmentation_comparison.png...")

    sample = create_synthetic_sample()

    # Create before/after with different transforms
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))

    # Row 1: Normalize
    ax = axes[0, 0]
    plot_sample(ax, sample, title="Before Normalize")
    ax = axes[0, 1]
    normalized = Normalize()(sample.copy())
    plot_sample(ax, normalized, title="After Normalize")

    # Row 2: Crop
    ax = axes[1, 0]
    plot_sample(ax, sample, title="Before RandomCrop")
    ax = axes[1, 1]
    cropped = RandomCrop(4096)(sample.copy())
    plot_sample(ax, cropped, title="After RandomCrop(4096)")

    # Row 3: FlipPolarity
    ax = axes[2, 0]
    normalized = Normalize()(sample.copy())
    plot_sample(ax, normalized, title="Before FlipPolarity")
    ax = axes[2, 1]
    flipped = FlipPolarity(p=1.0)(normalized.copy())
    plot_sample(ax, flipped, title="After FlipPolarity")

    # Row 4: Full training transform
    ax = axes[3, 0]
    plot_sample(ax, sample, title="Original")
    ax = axes[3, 1]
    transforms = default_train_transforms(crop_length=4096, enable_stacking=False)
    augmented = transforms(sample.copy())
    plot_sample(ax, augmented, title="After Full Training Transforms")

    for ax in axes[-1]:
        ax.set_xlabel("Sample")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "augmentation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved augmentation_comparison.png")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating CEED Verification Figures")
    print(f"Output directory: {FIGURES_DIR}")
    print("=" * 60)

    figure_transform_demo()
    figure_label_generation()
    figure_stacking_demo()
    figure_augmentation_comparison()
    figure_real_data()
    figure_training_batch()

    print("\n" + "=" * 60)
    print("All figures generated!")
    print(f"Check {FIGURES_DIR}/ for output files")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".png"):
            path = os.path.join(FIGURES_DIR, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {f}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
