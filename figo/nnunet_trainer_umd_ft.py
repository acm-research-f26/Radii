"""Fine-tuning control trainer for the UMD nnU-Net experiment."""

import os

import torch

try:
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_UMD import nnUNetTrainer_UMD
except ImportError:
    from figo.nnunet_trainer_umd import nnUNetTrainer_UMD


class nnUNetTrainer_UMD_FT(nnUNetTrainer_UMD):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = int(os.environ["UMD_NNUNET_EPOCHS"])
