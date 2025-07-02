from .phasenet_das import PhaseNetDAS


def build_model(backbone="unet", log_scale=True, add_event=True, *args, **kwargs) -> PhaseNetDAS:
    return PhaseNetDAS(backbone, log_scale=log_scale, add_event=add_event, *args, **kwargs)
