"""Tests for the ch_DDP_loderunner harness-local evaluation program."""

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_evaluator() -> ModuleType:
    """Load the harness-local evaluator without requiring applications as a package."""
    path = (
        Path(__file__).parents[2]
        / "applications/harnesses/ch_DDP_loderunner/eval_LodeRunner.py"
    )
    spec = importlib.util.spec_from_file_location("ch_loderunner_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_uses_checkpoint_fields_and_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator derives dataset fields and records directly from the checkpoint."""
    evaluator = _load_evaluator()
    captured: dict[str, object] = {}

    class FakeModel:
        """Checkpoint model with a saved field ordering."""

        default_vars = ["density_case", "Wvelocity"]

    def fake_load(*args: object, **kwargs: object) -> tuple[FakeModel, object, int]:
        """Return a model and the saved checkpoint epoch."""
        return FakeModel(), object(), 17

    class FakeDataset:
        """Capture the dataset configuration without loading LSC data."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Store construction arguments."""
            captured["dataset_args"] = args
            captured["dataset_kwargs"] = kwargs

    def fake_loader(*args: object, **kwargs: object) -> object:
        """Capture non-distributed dataloader settings."""
        captured["loader_kwargs"] = kwargs
        return object()

    def fake_epoch(**kwargs: object) -> None:
        """Capture a single evaluation epoch invocation."""
        captured["epoch_kwargs"] = kwargs

    monkeypatch.setattr(evaluator, "load_model_and_optimizer", fake_load)
    monkeypatch.setattr(evaluator, "LSC_rho2rho_temporal_DataSet", FakeDataset)
    monkeypatch.setattr(evaluator, "DataLoader", fake_loader)
    monkeypatch.setattr(evaluator, "eval_loderunner_epoch", fake_epoch)
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "testing.csv"
    evaluator.main(
        argparse.Namespace(
            checkpoint="final.pth",
            FILELIST_DIR="/filelists",
            LSC_NPZ_DIR="/data/",
            test_filelist="test.txt",
            batch_size=4,
            num_workers=2,
            test_batches=8,
            max_timeIDX_offset=3,
            test_rcrd_filename=str(output),
        )
    )

    dataset_kwargs = captured["dataset_kwargs"]
    assert isinstance(dataset_kwargs, dict)
    assert dataset_kwargs["file_prefix_list"] == "/filelists/test.txt"
    assert dataset_kwargs["max_timeIDX_offset"] == 3
    assert dataset_kwargs["deterministic"] is True
    assert dataset_kwargs["hydro_fields"].tolist() == FakeModel.default_vars
    epoch_kwargs = captured["epoch_kwargs"]
    assert isinstance(epoch_kwargs, dict)
    assert epoch_kwargs["epochIDX"] == 17
    assert epoch_kwargs["channel_map"] == [0, 1]
    assert output.with_suffix(".csv.metadata.json").exists()


def test_evaluator_rejects_model_without_saved_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The evaluator refuses checkpoints whose model has no field metadata."""
    evaluator = _load_evaluator()
    monkeypatch.setattr(
        evaluator,
        "load_model_and_optimizer",
        lambda *args, **kwargs: (object(), object(), 1),
    )
    with pytest.raises(ValueError, match="default_vars"):
        evaluator.main(
            argparse.Namespace(
                checkpoint="final.pth",
                FILELIST_DIR="/filelists",
                LSC_NPZ_DIR="/data/",
                test_filelist="test.txt",
                batch_size=4,
                num_workers=0,
                test_batches=1,
                max_timeIDX_offset=1,
                test_rcrd_filename="testing.csv",
            )
        )
