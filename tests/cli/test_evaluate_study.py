"""Tests for the ``yoke-evaluate-study`` CLI entry point."""

from pathlib import Path

import pytest

from yoke.cli import evaluate_study


def _write_eval_harness(harness_dir: Path) -> None:
    """Populate a minimal SLURM harness with evaluation artifacts.

    Args:
        harness_dir (Path): Directory to populate with harness config files.
    """
    (harness_dir / "hyperparameters.csv").write_text(
        "studyIDX,init_learnrate\n1,0.001\n2,0.002\n"
    )
    (harness_dir / "cp_files.txt").write_text("train.py\neval.py\n")
    (harness_dir / "train.py").write_text("print('train')\n")
    (harness_dir / "eval.py").write_text("print('eval')\n")
    (harness_dir / "training_input.tmpl").write_text("--studyIDX=<studyIDX>\n")
    (harness_dir / "training_slurm.tmpl").write_text(
        "python train.py @<INPUTFILE> <epochIDX>\n"
    )
    (harness_dir / "evaluation_input.tmpl").write_text(
        "--checkpoint=<CHECKPOINT>\n"
        "--output=testing_<studyIDX>_<STEM>_<init_learnrate>.csv\n"
    )
    (harness_dir / "evaluation_slurm.tmpl").write_text("python eval.py @<INPUTFILE>\n")


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    """Invoke the CLI main with a patched argv."""
    monkeypatch.setattr("sys.argv", argv)
    evaluate_study.main()


def test_evaluate_study_renders_and_submits_dryrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI renders checkpoint-specific files and prints the submit command."""
    _write_eval_harness(tmp_path)
    monkeypatch.chdir(tmp_path)

    # A pre-existing study directory with rendered eval templates + a checkpoint.
    study_dir = tmp_path / "runs" / "study_001"
    study_dir.mkdir(parents=True)
    (study_dir / "evaluation_input.tmpl").write_text(
        "--checkpoint=<CHECKPOINT>\n--output=testing_<studyIDX>_<STEM>_0.001.csv\n"
    )
    (study_dir / "evaluation_slurm.tmpl").write_text("python eval.py @<INPUTFILE>\n")
    checkpoint = study_dir / "study001_modelState_epoch0100.pth"
    checkpoint.write_text("w\n")

    _run_cli(
        monkeypatch,
        [
            "yoke-evaluate-study",
            "--studyIDX",
            "1",
            "--checkpoint",
            str(checkpoint),
            "--rundir",
            "./runs",
            "--dryrun",
        ],
    )

    out = capsys.readouterr().out
    stem = "study001_modelState_epoch0100"
    assert (study_dir / f"study001_evaluation_{stem}.input").exists()
    assert (study_dir / f"study001_evaluation_{stem}.slurm").exists()
    assert "[DRY RUN]" in out
    assert "sbatch" in out


def test_evaluate_study_renders_on_demand_for_old_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A study lacking eval templates gets them re-rendered from the CSV row."""
    _write_eval_harness(tmp_path)
    monkeypatch.chdir(tmp_path)

    study_dir = tmp_path / "runs" / "study_002"
    study_dir.mkdir(parents=True)
    checkpoint = study_dir / "study002_modelState_epoch0050.pth"
    checkpoint.write_text("w\n")
    assert not (study_dir / "evaluation_input.tmpl").exists()

    _run_cli(
        monkeypatch,
        [
            "yoke-evaluate-study",
            "--studyIDX",
            "2",
            "--checkpoint",
            str(checkpoint),
            "--rundir",
            "./runs",
            "--dryrun",
        ],
    )

    stem = "study002_modelState_epoch0050"
    # Templates rendered on demand, evaluator copied, CSV keys resolved.
    assert (study_dir / "evaluation_input.tmpl").exists()
    assert (study_dir / "eval.py").exists()
    input_data = (study_dir / f"study002_evaluation_{stem}.input").read_text()
    assert f"testing_002_{stem}_0.002.csv" in input_data
    assert str(checkpoint.resolve()) in input_data


def test_evaluate_study_ema_checkpoint_distinct_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An EMA checkpoint yields artifacts distinct from the ordinary one."""
    _write_eval_harness(tmp_path)
    monkeypatch.chdir(tmp_path)
    study_dir = tmp_path / "runs" / "study_001"
    study_dir.mkdir(parents=True)
    ema = study_dir / "study001_modelState_epoch0100_ema.pth"
    ema.write_text("w\n")

    _run_cli(
        monkeypatch,
        [
            "yoke-evaluate-study",
            "--studyIDX",
            "1",
            "--checkpoint",
            str(ema),
            "--rundir",
            "./runs",
            "--dryrun",
        ],
    )

    stem = "study001_modelState_epoch0100_ema"
    assert (study_dir / f"study001_evaluation_{stem}.input").exists()
    assert not (
        study_dir / "study001_evaluation_study001_modelState_epoch0100.input"
    ).exists()
