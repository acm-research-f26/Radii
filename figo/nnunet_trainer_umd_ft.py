"""Fine-tuning control trainer for the UMD nnU-Net experiment."""

import os

import torch

try:
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_UMD import nnUNetTrainer_UMD
except ImportError:
    from figo.nnunet_trainer_umd import nnUNetTrainer_UMD


def _unwrap_network(network: torch.nn.Module) -> torch.nn.Module:
    if hasattr(network, "module"):
        network = network.module
    if hasattr(network, "_orig_mod"):
        network = network._orig_mod
    return network


def _load_baseline_weights(network: torch.nn.Module, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    _unwrap_network(network).load_state_dict(checkpoint["network_weights"], strict=True)


class nnUNetTrainer_UMD_FT(nnUNetTrainer_UMD):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.num_epochs = int(os.environ["UMD_NNUNET_EPOCHS"])

    def initialize(self):
        was_initialized = self.was_initialized
        super().initialize()
        if was_initialized:
            return

        path = os.environ.get("UMD_BASELINE_CKPT")
        if not path:
            raise RuntimeError("UMD_BASELINE_CKPT must be set to the baseline checkpoint path")

        _load_baseline_weights(self.network, path, self.device)
        print(f"Loaded full baseline weights (heads included) from {path}")
