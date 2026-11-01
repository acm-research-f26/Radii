import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from figo.nnunet_trainer_umd_boundary import (
    BOUNDARY_WEIGHT,
    BoundaryWeightedLoss,
    build_boundary_weight_map,
    _spacing_at_scale,
)


def make_base_loss():
    return DC_and_CE_loss(
        {"batch_dice": False, "smooth": 1e-5, "do_bg": False, "ddp": False},
        {},
        weight_ce=1,
        weight_dice=1,
        dice_class=MemoryEfficientSoftDiceLoss,
    )


def test_boundary_weight_map_marks_cavity_and_band_without_multiplying_overlap():
    target = torch.zeros((1, 1, 5, 9, 9), dtype=torch.long)
    target[0, 0, 2, 3:6, 3:6] = 17

    weights = build_boundary_weight_map(target, cavity_label=17, spacing=(2.0, 1.0, 1.0))

    assert weights.shape == target.shape
    assert torch.all(weights[0, 0, 2, 4, 4] == BOUNDARY_WEIGHT)
    assert torch.all(weights[0, 0, 2, 2, 4] == BOUNDARY_WEIGHT)
    assert torch.all(weights[0, 0, 2, 4, 2] == BOUNDARY_WEIGHT)
    assert torch.all(weights[0, 0, 1, 4, 4] == 1)
    assert torch.all(weights[0, 0, 2, 0, 0] == 1)
    assert set(weights.unique().tolist()) == {1, BOUNDARY_WEIGHT}


def test_weight_one_matches_control_loss_exactly():
    target = torch.randint(0, 4, (2, 1, 7, 9, 11))
    logits = torch.randn(2, 4, 7, 9, 11)
    weights = build_boundary_weight_map(target, cavity_label=2, spacing=(2.5, 1.0, 0.8), weight=1)
    base = make_base_loss()
    control = base(logits, target)
    boundary = BoundaryWeightedLoss(
        base, cavity_label=2, spacing=(2.5, 1.0, 0.8), fullres_shape=(7, 9, 11), weight=1
    )

    assert torch.all(weights == 1)
    assert torch.equal(boundary(logits, target), control)


def test_each_deep_supervision_target_shape_gets_its_own_weight_map():
    full_target = torch.zeros((1, 1, 16, 24, 24), dtype=torch.long)
    full_target[:, :, 4:12, 6:18, 6:18] = 1
    targets = [
        full_target,
        torch.nn.functional.interpolate(full_target.float(), size=(8, 12, 12), mode="nearest").long(),
        torch.nn.functional.interpolate(full_target.float(), size=(4, 6, 6), mode="nearest").long(),
    ]

    base = make_base_loss()
    spacing = (0.5, 0.5, 0.5)
    fullres_shape = full_target.shape[2:]
    loss = BoundaryWeightedLoss(base, cavity_label=1, spacing=spacing, fullres_shape=fullres_shape)
    scale_spacings = [
        _spacing_at_scale(spacing, fullres_shape, target.shape[2:])
        for target in targets
    ]
    maps = [loss.weight_map_for_target(target) for target in targets]
    outputs = [torch.randn((1, 2, *target.shape[2:])) for target in targets]
    deep_supervision_loss = DeepSupervisionWrapper(loss, [1.0, 0.5, 0.25])
    actual = deep_supervision_loss(outputs, targets)
    expected = sum(
        scale_weight * loss(output, target)
        for scale_weight, output, target in zip([1.0, 0.5, 0.25], outputs, targets)
    )

    assert [weight_map.shape for weight_map in maps] == [target.shape for target in targets]
    assert all(weight_map.device == target.device for weight_map, target in zip(maps, targets))
    assert all(torch.all(weight_map.max() == BOUNDARY_WEIGHT) for weight_map in maps)
    assert all(set(weight_map.unique().tolist()) == {1, BOUNDARY_WEIGHT} for weight_map in maps)
    assert torch.allclose(actual, expected)

    band_widths_mm = []
    for target, weight_map, scale_spacing in zip(targets, maps, scale_spacings):
        labels = target[0, 0]
        weights = weight_map[0, 0]
        cavity_coordinates = torch.where(labels[:, labels.shape[1] // 2, labels.shape[2] // 2] == 1)[0]
        weighted_coordinates = torch.where(weights[:, labels.shape[1] // 2, labels.shape[2] // 2] == BOUNDARY_WEIGHT)[0]
        band_widths_mm.append(
            (cavity_coordinates[0] - weighted_coordinates[0]).item() * scale_spacing[0]
        )

    for i in range(len(band_widths_mm) - 1):
        assert abs(band_widths_mm[i] - band_widths_mm[i + 1]) <= max(
            scale_spacings[i][0], scale_spacings[i + 1][0]
        )


class RejectCpuTransfers(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if device is None and len(args) > 1 and isinstance(args[1], (str, torch.device)):
            device = args[1]
        if device is not None and torch.device(device).type == "cpu":
            raise AssertionError(f"unexpected CPU tensor transfer in {func}")
        return func(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU loss check")
def test_boundary_loss_uses_gpu_tensors_without_moving_loss_to_cpu():
    target = torch.zeros((1, 1, 5, 8, 8), dtype=torch.long, device="cuda")
    target[:, :, 2, 3:5, 3:5] = 2
    logits = torch.randn(1, 3, 5, 8, 8, device="cuda", requires_grad=True)
    loss = BoundaryWeightedLoss(
        make_base_loss(),
        cavity_label=2,
        spacing=(2.5, 1.0, 0.8),
        fullres_shape=(5, 8, 8),
    ).cuda()

    with RejectCpuTransfers():
        result = loss(logits, target)
        result.backward()

    assert result.device.type == "cuda"
    assert logits.grad is not None and logits.grad.device.type == "cuda"
