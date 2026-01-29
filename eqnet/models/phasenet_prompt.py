"""PhaseNet Prompt: Phase picking with prompt."""
from .phasenet import PhaseNet


def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_stft: bool = False,
    add_polarity: bool = True,
    add_event: bool = True,
    add_prompt: bool = True,
    event_center_loss_weight: float = 1.0,
    event_time_loss_weight: float = 1.0,
    polarity_loss_weight: float = 1.0,
    prompt_loss_weight: float = 1.0,
    **kwargs,
) -> PhaseNet:
    """Build PhaseNet with prompt."""
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_polarity=add_polarity,
        add_event=add_event,
        add_prompt=add_prompt,
        event_center_loss_weight=event_center_loss_weight,
        event_time_loss_weight=event_time_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
        prompt_loss_weight=prompt_loss_weight,
        **kwargs,
    )
