"""nnU-Net v2 trainer for Colab benchmarking. ``figo.nnunet install-trainer``
copies this file into the installed nnunetv2 package so ``-tr nnUNetTrainer_UMD``
can find it.

Differences from the stock nnUNetTrainer:
  * epochs capped (env ``UMD_NNUNET_EPOCHS``, default 250; keep it fixed for a run,
    since the poly LR schedule depends on it)
  * ``checkpoint_latest.pth`` every 5 epochs instead of 50, so ``--c`` resumes
    with little loss after a Colab disconnect
"""

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_UMD(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("UMD_NNUNET_EPOCHS", 250))
        self.save_every = 5
