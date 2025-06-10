from .phasenet import PhaseNet


def build_model(
    backbone="unet",
    log_scale=True,
    add_stft=False,
    add_polarity=True,
    add_event=True,
    add_prompt=True,
    event_center_loss_weight=1.0,
    event_time_loss_weight=1.0,
    polarity_loss_weight=1.0,
    *args,
    **kwargs,
) -> PhaseNet:
    return PhaseNet(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_event=add_event,
        add_polarity=add_polarity,
        add_prompt=add_prompt,
        event_center_loss_weight=event_center_loss_weight,
        event_time_loss_weight=event_time_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
    )
