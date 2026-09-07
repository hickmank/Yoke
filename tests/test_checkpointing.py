"""Tests for torch checkpoint functions with dynamic model."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor

# Your checkpoint functions here or imported
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.checkpointing import save_model_and_optimizer_hdf5
from yoke.utils.checkpointing import load_model_and_optimizer_hdf5


class DummyNet(nn.Module):
    """Simple test model."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        """Initialization."""
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward map."""
        return self.linear(x)


@pytest.fixture
def dummy_model_args() -> dict:
    """Dummy model parameters dict."""
    return {"input_dim": 4, "output_dim": 2}


@pytest.fixture
def available_models() -> dict[str, type[nn.Module]]:
    """Dummy available models."""
    return {"DummyNet": DummyNet}


@pytest.fixture
def model_and_optimizer(dummy_model_args: dict) -> tuple[nn.Module, optim.Optimizer]:
    """Model and optimizer initialization function."""
    model = DummyNet(**dummy_model_args)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def test_checkpoint_save_and_load(
    model_and_optimizer: tuple[nn.Module, optim.Optimizer],
    dummy_model_args: dict,
    available_models: dict[str, type[nn.Module]],
) -> None:
    """Test saving and reloading, non-DDP."""
    model, optimizer = model_and_optimizer

    # Modify model weights to test restoration
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    # Take optimizer step to populate state dict
    dummy_input = torch.randn(1, 4)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pth")

        save_model_and_optimizer(
            model=model,
            optimizer=optimizer,
            epoch=5,
            filepath=ckpt_path,
            model_class=DummyNet,
            model_args=dummy_model_args,
        )

        loaded_model, loaded_optimizer, loaded_epoch = load_model_and_optimizer(
            filepath=ckpt_path,
            optimizer_class=optim.AdamW,
            optimizer_kwargs={"lr": 0.01},
            available_models=available_models,
            device="cpu",
        )

        # Check epoch was restored
        assert loaded_epoch == 5

        # Compare model parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters not restored correctly"

        # Compare optimizer states
        old_opt_state = optimizer.state_dict()
        new_opt_state = loaded_optimizer.state_dict()

        for k in old_opt_state["state"].keys():
            for subkey in old_opt_state["state"][k]:
                v1 = old_opt_state["state"][k][subkey]
                v2 = new_opt_state["state"][k][subkey]
                if isinstance(v1, torch.Tensor):
                    assert torch.allclose(v1, v2)
                else:
                    assert v1 == v2


class ScalarNet(nn.Module):
    """Model with a scalar (0-dim) parameter and buffer plus a normal layer."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        """Initialization."""
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # 0-dim parameter and buffer to exercise the HDF5 attribute code paths.
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("running_scalar", torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        """Forward map."""
        return self.linear(x) * self.scale


def test_hdf5_scalar_param_and_buffer_roundtrip() -> None:
    """Scalar (0-dim) params/buffers must survive an HDF5 save/load round-trip."""
    model = ScalarNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Set distinctive trained scalar values.
    with torch.no_grad():
        model.scale.fill_(3.5)
        model.running_scalar.fill_(2.25)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "scalar_ckpt.h5")
        save_model_and_optimizer_hdf5(
            model=model, optimizer=optimizer, epoch=3, filepath=ckpt_path
        )

        # Fresh model with default scalar values.
        fresh_model = ScalarNet()
        fresh_optimizer = optim.SGD(fresh_model.parameters(), lr=0.1)
        assert not torch.isclose(fresh_model.scale.detach(), torch.tensor(3.5))

        epoch = load_model_and_optimizer_hdf5(
            model=fresh_model, optimizer=fresh_optimizer, filepath=ckpt_path
        )

    assert epoch == 3
    assert torch.allclose(fresh_model.scale.detach(), torch.tensor(3.5)), (
        "Scalar parameter was not restored from HDF5 checkpoint!"
    )
    assert torch.allclose(fresh_model.running_scalar, torch.tensor(2.25)), (
        "Scalar buffer was not restored from HDF5 checkpoint!"
    )


def test_hdf5_optimizer_momentum_roundtrip() -> None:
    """Per-parameter optimizer state (SGD momentum) must survive HDF5 round-trip."""
    model = DummyNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Take a couple of steps to populate momentum buffers.
    for _ in range(2):
        optimizer.zero_grad()
        out = model(torch.randn(3, 4))
        loss = out.sum()
        loss.backward()
        optimizer.step()

    trained_state = optimizer.state_dict()
    # Sanity check: momentum buffers should exist and be non-trivial.
    momentum_buffers = [
        s["momentum_buffer"]
        for s in trained_state["state"].values()
        if "momentum_buffer" in s
    ]
    assert momentum_buffers, "Expected SGD momentum buffers to be populated!"

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "momentum_ckpt.h5")
        save_model_and_optimizer_hdf5(
            model=model, optimizer=optimizer, epoch=7, filepath=ckpt_path
        )

        fresh_model = DummyNet()
        # Copy weights so parameter identity/order matches for state restoration.
        fresh_optimizer = optim.SGD(fresh_model.parameters(), lr=0.1, momentum=0.9)
        epoch = load_model_and_optimizer_hdf5(
            model=fresh_model, optimizer=fresh_optimizer, filepath=ckpt_path
        )

    assert epoch == 7
    loaded_state = fresh_optimizer.state_dict()
    for key in trained_state["state"]:
        orig = trained_state["state"][key]
        loaded = loaded_state["state"][key]
        if "momentum_buffer" in orig:
            assert "momentum_buffer" in loaded, (
                "Optimizer momentum buffer missing after HDF5 load!"
            )
            assert torch.allclose(orig["momentum_buffer"], loaded["momentum_buffer"]), (
                "Optimizer momentum buffer not restored from HDF5 checkpoint!"
            )
