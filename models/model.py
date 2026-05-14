import torch.nn as nn
import segmentation_models_pytorch as smp


_ARCH_MAP = {
    "UnetPlusPlus":  smp.UnetPlusPlus,
    "Unet":          smp.Unet,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN":           smp.FPN,
    "PSPNet":        smp.PSPNet,
    "MAnet":         smp.MAnet,
}


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    arch_cls = _ARCH_MAP.get(m["architecture"])
    if arch_cls is None:
        raise ValueError(
            f"Unknown architecture '{m['architecture']}'. "
            f"Choose from: {list(_ARCH_MAP.keys())}"
        )

    model = arch_cls(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"],
        in_channels=m["in_channels"],
        classes=m["classes"],
        activation=m.get("activation"),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model  : {m['architecture']} + {m['encoder']}  "
        f"(in={m['in_channels']}ch, out={m['classes']}ch)\n"
        f"Params : {n_params:,}"
    )
    return model
