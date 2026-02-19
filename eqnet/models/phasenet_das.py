"""
PhaseNet-DAS: Deep learning for DAS (Distributed Acoustic Sensing) phase picking.

Inherits from PhaseNet with DAS-specific configuration:
- Input: (batch, 1, nx, nt) - single channel DAS data
- Output: phase probabilities (P/S/Noise) and optionally event detection
"""
from .phasenet import PhaseNet


# DAS config: single channel, 2D spatial-temporal
# Includes pre-configured values for optional features when enabled
DAS_UNET_CONFIG = dict(
    dim=8,
    channels=1,
    phase_channels=3,
    time_kernel=7,
    time_stride=4,
    space_kernel=7,
    space_stride=4,
    init_conv_space_kernel=7,
    final_conv_space_kernel=3,
    # pre-configured for init_cross_embed=True / cross_embed_downsample=True
    init_cross_embed_space_kernel_sizes=(3, 7, 15),
    cross_embed_downsample_space_kernel_sizes=(4, 8),
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
