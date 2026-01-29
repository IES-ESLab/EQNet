"""PhaseNet with Time-Frequency (STFT) features."""
from .phasenet import PhaseNet


def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_stft: bool = True,
    add_polarity: bool = False,
    add_event: bool = False,
    **kwargs,
) -> PhaseNet:
    """Build PhaseNet with STFT features enabled."""
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_polarity=add_polarity,
        add_event=add_event,
        **kwargs,
    )
