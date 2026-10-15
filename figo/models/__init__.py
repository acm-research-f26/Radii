"""Model registry for the custom (non-nnU-Net) Stage 1 baselines.

nnU-Net v2 (3d_fullres and Residual Encoder) is trained with the nnU-Net
framework itself; see ``figo.nnunet`` and ``notebooks/02_nnunet.ipynb``.
"""

from .aksunet import AKSUNet
from .davnet import DAVNet
from .mednext import MedNeXt
from .unet3d import UNet3D

MODELS = {
    "unet3d": UNet3D,
    "mednext": MedNeXt,
    "davnet": DAVNet,
    "aksunet": AKSUNet,
}

# deep supervision default per model (the plain U-Net baseline has none)
DEEP_SUPERVISION = {"unet3d": False, "mednext": True, "davnet": True, "aksunet": True}


def build_model(name: str, in_channels: int = 1, n_classes: int = 5, deep_supervision: bool | None = None):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    ds = DEEP_SUPERVISION[name] if deep_supervision is None else deep_supervision
    return MODELS[name](in_channels=in_channels, n_classes=n_classes, deep_supervision=ds)
