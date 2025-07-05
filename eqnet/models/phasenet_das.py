from .phasenet import PhaseNet, UNetHead, EventHead
from .unet import UNet
from .x_unet import XUnet


class PhaseNetDAS(PhaseNet):
    def __init__(
        self,
        backbone="xunet",
        log_scale=False,
        add_stft=False,
        add_polarity=False,
        add_event=False,
        add_prompt=False,
        event_center_loss_weight=1.0,
        event_time_loss_weight=1.0,
        polarity_loss_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone=backbone,
            log_scale=log_scale,
            add_stft=add_stft,
            add_polarity=add_polarity,
            add_event=add_event,
            add_prompt=add_prompt,
            event_center_loss_weight=event_center_loss_weight,
            event_time_loss_weight=event_time_loss_weight,
            polarity_loss_weight=polarity_loss_weight,
            **kwargs,
        )
        self.backbone_name = backbone
        self.add_stft = add_stft
        self.add_event = add_event
        self.add_polarity = add_polarity
        self.add_prompt = add_prompt
        self.event_center_loss_weight = event_center_loss_weight
        self.event_time_loss_weight = event_time_loss_weight
        self.polarity_loss_weight = polarity_loss_weight

        if backbone == "unet":
            self.backbone = UNet(
                channels=1,
                dim=16,
                out_dim=32,
                kernel_size=(7, 7),
                log_scale=log_scale,
                add_stft=add_stft,
                add_polarity=add_polarity,
                add_event=add_event,
                add_prompt=add_prompt,
            )
        elif backbone == "xunet":
            self.backbone = XUnet(
                channels=1,
                dim=16,
                out_dim=32,
                kernel_size=(1, 7, 7),
                log_scale=log_scale,
                add_stft=add_stft,
                add_polarity=add_polarity,
                add_event=add_event,
                add_prompt=add_prompt,
            )
        else:
            raise ValueError(f"Invalid backbone: {backbone}")

        if backbone == "unet":
            self.phase_picker = UNetHead(32, 3, feature_name="phase")
            if self.add_polarity:
                self.polarity_picker = UNetHead(32, 1, feature_name="polarity")
            if self.add_event:
                self.event_detector = UNetHead(32, 1, feature_name="event")
                self.event_timer = EventHead(32, 1, feature_name="event")

        elif backbone == "xunet":
            self.phase_picker = UNetHead(32, 3, feature_name="phase")
            if self.add_polarity:
                self.polarity_picker = UNetHead(32, 1, feature_name="polarity")
            if self.add_event:
                self.event_detector = UNetHead(32, 1, feature_name="event")
                self.event_timer = EventHead(32, 1, feature_name="event")
        else:
            raise ValueError("backbone only supports unet or xunet")


def build_model(backbone="unet", log_scale=True, *args, **kwargs) -> PhaseNetDAS:
    return PhaseNetDAS(backbone, log_scale, *args, **kwargs)
