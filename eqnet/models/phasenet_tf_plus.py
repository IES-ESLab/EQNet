"""PhaseNet TF Plus: Phase picking with STFT, polarity and event detection."""
from .phasenet import PhaseNet


def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_stft: bool = True,
    add_polarity: bool = True,
    add_event: bool = True,
    event_center_loss_weight: float = 1.0,
    event_time_loss_weight: float = 1.0,
    polarity_loss_weight: float = 1.0,
    **kwargs,
) -> PhaseNet:
    """Build PhaseNet TF Plus with STFT, polarity and event detection enabled."""
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_polarity=add_polarity,
        add_event=add_event,
        event_center_loss_weight=event_center_loss_weight,
        event_time_loss_weight=event_time_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
        **kwargs,
    )
