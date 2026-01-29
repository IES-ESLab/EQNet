"""
PhaseNet-DAS: Deep learning for DAS (Distributed Acoustic Sensing) phase picking.

Inherits from PhaseNet with DAS-specific configuration:
- Input: (batch, 1, nx, nt) - single channel DAS data
- Output: phase probabilities (P/S/Noise) and optionally event detection
"""
from .phasenet import PhaseNet


# DAS-specific UNet configuration overrides
DAS_UNET_CONFIG = dict(
    dim=8,  # reduced from 32 for faster training
    channels=1,  # DAS is single channel
    phase_channels=3,  # P, S, Noise
    # DAS: downsample both space and time with stride 4
    space_stride=4,
    time_stride=4,
    space_kernel=7,
    time_kernel=7,
)


class PhaseNetDAS(PhaseNet):
    """PhaseNet for DAS data.

    Inherits from PhaseNet with DAS-specific defaults:
    - Single channel input (channels=1)
    - Stride 4 downsampling in both dimensions
    - 7x7 kernels for spatial-temporal convolutions
    """

    def __init__(
        self,
        backbone: str = "unet",
        log_scale: bool = True,
        add_stft: bool = False,
        add_polarity: bool = False,
        add_event: bool = False,
        **kwargs,
    ) -> None:
        # Merge DAS defaults with user kwargs (user kwargs take precedence)
        das_kwargs = {**DAS_UNET_CONFIG, **kwargs}

        super().__init__(
            backbone=backbone,
            log_scale=log_scale,
            add_stft=add_stft,
            add_polarity=add_polarity,
            add_event=add_event,
            **das_kwargs,
        )


def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_event: bool = False,
    **kwargs,
) -> PhaseNetDAS:
    """Build a PhaseNet-DAS model."""
    return PhaseNetDAS(
        backbone=backbone,
        log_scale=log_scale,
        add_event=add_event,
        **kwargs,
    )
