"""
PhaseNet-DAS Plus: DAS phase picking with event detection.

Extends PhaseNet-DAS with event detection capabilities:
- Phase picking (P/S/Noise)
- Event center detection
- Event time regression

Note: Polarity is not supported for DAS since it uses single-channel strain rate data.
"""
from .phasenet_das import PhaseNetDAS


def build_model(
    backbone: str = "unet",
    log_scale: bool = True,
    add_stft: bool = False,
    add_event: bool = True,
    event_center_loss_weight: float = 1.0,
    event_time_loss_weight: float = 1.0,
    **kwargs,
) -> PhaseNetDAS:
    """Build PhaseNet-DAS Plus with event detection enabled.

    Args:
        backbone: Backbone type ("unet")
        log_scale: Apply log transform to input
        add_stft: Add STFT features
        add_event: Enable event detection (default True for Plus variant)
        event_center_loss_weight: Weight for event center loss
        event_time_loss_weight: Weight for event time loss
        **kwargs: Additional arguments passed to PhaseNetDAS

    Returns:
        PhaseNetDAS model instance with event detection
    """
    return PhaseNetDAS(
        backbone=backbone,
        log_scale=log_scale,
        add_stft=add_stft,
        add_polarity=False,  # DAS doesn't support polarity
        add_event=add_event,
        event_center_loss_weight=event_center_loss_weight,
        event_time_loss_weight=event_time_loss_weight,
        **kwargs,
    )
