"""Boundary-weighted loss experiment for the UMD nnU-Net trainer."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

try:
    from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_UMD_FT import nnUNetTrainer_UMD_FT
except ImportError:
    from figo.nnunet_trainer_umd_ft import nnUNetTrainer_UMD_FT

BOUNDARY_WEIGHT = 3.0
BOUNDARY_LOSS_ENABLED = True
BOUNDARY_RADIUS_MM = 2.0


def _pool_kernel(spacing, radius_mm=BOUNDARY_RADIUS_MM):
    if not spacing or any(float(s) <= 0 for s in spacing):
        raise ValueError("plans spacing must contain positive values")
    return tuple(
        1 if float(s) >= radius_mm else 2 * math.floor(radius_mm / float(s)) + 1
        for s in spacing
    )


def _spacing_at_scale(spacing, fullres_shape, target_shape):
    if len(spacing) != len(fullres_shape) or len(spacing) != len(target_shape):
        raise ValueError("spacing and target shapes must have matching dimensions")
    if any(size <= 0 for size in fullres_shape) or any(size <= 0 for size in target_shape):
        raise ValueError("full-resolution and target shapes must be positive")
    return tuple(
        float(voxel_spacing) * full_size / target_size
        for voxel_spacing, full_size, target_size in zip(spacing, fullres_shape, target_shape)
    )


def build_boundary_weight_map(target, cavity_label, spacing, weight=BOUNDARY_WEIGHT):
    ndim = len(spacing)
    if target.ndim == ndim + 2:
        if target.shape[1] != 1:
            raise ValueError("boundary weighting requires a single-channel label map")
        labels = target[:, 0]
    elif target.ndim == ndim + 1:
        labels = target
    else:
        raise ValueError(f"target rank {target.ndim} does not match {ndim}D spacing")

    kernel = _pool_kernel(spacing)
    cavity = labels.eq(cavity_label).unsqueeze(1).to(torch.float32)
    pool = getattr(F, f"max_pool{ndim}d")
    padding = tuple(k // 2 for k in kernel)
    pad = tuple(value for k in reversed(kernel) for value in (k // 2, k // 2))
    dilated = pool(cavity, kernel, stride=1, padding=padding) > 0
    inverse = F.pad(1 - cavity, pad, value=1)
    eroded = 1 - pool(inverse, kernel, stride=1) > 0
    boundary = torch.logical_xor(dilated, eroded)
    weighted = cavity.bool() | boundary
    return torch.where(weighted, torch.full_like(cavity, float(weight)), cavity.new_ones(()))


class BoundaryWeightedLoss(nn.Module):
    def __init__(self, base_loss, cavity_label, spacing, fullres_shape, weight=BOUNDARY_WEIGHT,
                 enabled=BOUNDARY_LOSS_ENABLED):
        super().__init__()
        if not isinstance(base_loss, DC_and_CE_loss):
            raise TypeError("boundary weighting requires nnU-Net's DC_and_CE_loss")
        self.base_loss = base_loss
        self.cavity_label = cavity_label
        self.spacing = tuple(float(s) for s in spacing)
        self.fullres_shape = tuple(int(s) for s in fullres_shape)
        self.weight = float(weight)
        self.enabled = enabled

    def weight_map_for_target(self, target):
        target_shape = target.shape[-len(self.spacing):]
        scale_spacing = _spacing_at_scale(self.spacing, self.fullres_shape, target_shape)
        return build_boundary_weight_map(target, self.cavity_label, scale_spacing, self.weight)

    def forward(self, net_output, target):
        if not self.enabled or self.weight == 1:
            return self.base_loss(net_output, target)

        weights = self.weight_map_for_target(target)
        target_map = target[:, 0] if target.ndim == net_output.ndim else target
        valid = None
        target_dice = target
        dice_weights = weights
        if self.base_loss.ignore_label is not None:
            valid = target != self.base_loss.ignore_label
            target_dice = torch.where(valid, target, 0)
            dice_weights = weights * valid

        dice_loss = self.base_loss.dc(net_output, target_dice, loss_mask=dice_weights) \
            if self.base_loss.weight_dice != 0 else 0

        if self.base_loss.weight_ce != 0:
            ce = self.base_loss.ce
            voxel_loss = F.cross_entropy(
                net_output,
                target_map.long(),
                weight=ce.weight,
                ignore_index=ce.ignore_index,
                reduction="none",
                label_smoothing=ce.label_smoothing,
            )
            ce_weights = weights[:, 0]
            if valid is not None:
                ce_weights = ce_weights * valid[:, 0]
            ce_loss = (voxel_loss * ce_weights).sum() / ce_weights.sum().clamp_min(1e-8)
        else:
            ce_loss = 0

        return self.base_loss.weight_ce * ce_loss + self.base_loss.weight_dice * dice_loss


class nnUNetTrainer_UMD_Boundary(nnUNetTrainer_UMD_FT):
    def _build_loss(self):
        loss = super()._build_loss()
        if isinstance(loss, DeepSupervisionWrapper):
            base_loss = loss.loss
        else:
            base_loss = loss
        if not isinstance(base_loss, DC_and_CE_loss):
            raise TypeError("boundary trainer requires a class-label DC_and_CE_loss")

        labels = self.dataset_json["labels"]
        cavity_labels = [
            label_id for name, label_id in labels.items()
            if name.casefold() == "uterine_cavity"
        ]
        if len(cavity_labels) != 1 or not isinstance(cavity_labels[0], int):
            raise ValueError("dataset.json must define one integer uterine_cavity label")
        weighted_loss = BoundaryWeightedLoss(
            base_loss,
            cavity_labels[0],
            self.configuration_manager.spacing,
            self.configuration_manager.patch_size,
        )
        if isinstance(loss, DeepSupervisionWrapper):
            loss.loss = weighted_loss
            return loss
        return weighted_loss
