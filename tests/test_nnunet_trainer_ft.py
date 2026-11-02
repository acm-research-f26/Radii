import pytest
import torch
from torch import nn

from figo.nnunet_trainer_umd_ft import _load_baseline_weights, _unwrap_network


def make_network():
    return nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))


def test_baseline_checkpoint_loads_all_weights_strictly(tmp_path):
    source = make_network()
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"network_weights": source.state_dict()}, checkpoint_path)
    target = make_network()

    _load_baseline_weights(target, str(checkpoint_path), torch.device("cpu"))

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    del checkpoint["network_weights"]["0.weight"]
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError):
        _load_baseline_weights(target, str(checkpoint_path), torch.device("cpu"))


class CompiledModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda module: module,
        nn.DataParallel,
        CompiledModule,
        lambda module: nn.DataParallel(CompiledModule(module)),
    ],
    ids=["plain", "data-parallel", "compiled", "nested"],
)
def test_unwrap_network_handles_supported_wrappers(wrap):
    network = make_network()

    assert _unwrap_network(wrap(network)) is network
