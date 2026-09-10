"""Tests for the ``yoke-plot-loss`` CLI entry point."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from yoke.cli import plot_loss


def _make_study(base: Path, studyIDX: int, n_trn: int, n_val: int) -> None:
    """Create a study dir with one training and one validation record CSV.

    Args:
        base (Path): Base directory to create the study under.
        studyIDX (int): Study index.
        n_trn (int): Number of training batch rows.
        n_val (int): Number of validation batch rows.
    """
    study_dir = base / f"study_{studyIDX:03d}"
    study_dir.mkdir(parents=True)
    trn = "\n".join(f"10, {b}, 0.5" for b in range(n_trn)) + "\n"
    val = "\n".join(f"10, {b}, 0.4" for b in range(n_val)) + "\n"
    (study_dir / f"training_study{studyIDX:03d}_epoch0010.csv").write_text(trn)
    (study_dir / f"validation_study{studyIDX:03d}_epoch0010.csv").write_text(val)


def test_main_savefig_creates_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` with ``--savefig`` writes the study PNG without a display."""
    _make_study(tmp_path, 1, n_trn=40, n_val=20)
    savedir = tmp_path / "images"
    monkeypatch.setattr(
        "sys.argv",
        [
            "yoke-plot-loss",
            "--basedir",
            str(tmp_path),
            "--IDX",
            "1",
            "-Nt",
            "10",
            "-Nv",
            "5",
            "--savefig",
            "--savedir",
            str(savedir),
        ],
    )

    plot_loss.main()

    assert (savedir / "study001_TandV_curve.png").exists()
