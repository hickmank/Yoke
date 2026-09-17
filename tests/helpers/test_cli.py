"""Test cli module."""

import argparse

import pytest

from yoke.helpers import cli


def _option_strings(parser: argparse.ArgumentParser) -> list[str]:
    """Collect all option strings registered on a parser.

    Args:
        parser (argparse.ArgumentParser): Parser to inspect.

    Returns:
        list[str]: Flattened list of option strings (e.g. ``--studyIDX``).
    """
    return [opt for action in parser._actions for opt in action.option_strings]


def test_add_default_args() -> None:
    """Ensure default argparser runs without crashing."""
    # Test default use case.
    cli.add_default_args()

    # Test use case of adding to existing parser.
    cli.add_default_args(argparse.ArgumentParser())


def test_add_default_args_excludes_studyIDX() -> None:
    """The launcher CLI must not expose --studyIDX (sourced from the CSV)."""
    parser = cli.add_default_args()
    assert "--studyIDX" not in _option_strings(parser)


def test_add_training_args_provides_studyIDX() -> None:
    """Train scripts obtain --studyIDX via add_training_args, not the launcher."""
    parser = cli.add_training_args(argparse.ArgumentParser())
    assert "--studyIDX" in _option_strings(parser)


def test_studyIDX_parses_from_at_file(tmp_path: object) -> None:
    """--studyIDX resolves from a rendered training_input @-file as an int.

    This mirrors how a harness feeds ``--studyIDX <studyIDX>`` to the train
    script via ``fromfile_prefix_chars="@"`` after template substitution.
    """
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser = cli.add_default_args(parser=parser)
    parser = cli.add_training_args(parser=parser)

    input_file = tmp_path / "study.input"
    input_file.write_text("--studyIDX\n3\n")

    args = parser.parse_args([f"@{input_file}"])
    assert args.studyIDX == 3
    assert isinstance(args.studyIDX, int)


def test_add_filepath_args() -> None:
    """Ensure filepath argparser runs without crashing."""
    cli.add_filepath_args(argparse.ArgumentParser())


def test_add_computing_args() -> None:
    """Ensure computing argparser runs without crashing."""
    cli.add_computing_args(argparse.ArgumentParser())


def test_multigpu_flag_deprecated_and_ignored() -> None:
    """--multigpu is deprecated: it warns and resolves to False (no-op)."""
    parser = cli.add_computing_args(argparse.ArgumentParser())

    # Default (flag absent) is False without a warning.
    args = parser.parse_args([])
    assert args.multigpu is False

    # Providing the flag warns and remains ignored (False).
    with pytest.warns(DeprecationWarning):
        args = parser.parse_args(["--multigpu"])
    assert args.multigpu is False


def test_add_model_args() -> None:
    """Ensure model argparser runs without crashing."""
    cli.add_model_args(argparse.ArgumentParser())


def test_add_training_args() -> None:
    """Ensure training argparser runs without crashing."""
    cli.add_training_args(argparse.ArgumentParser())


def test_add_step_lr_scheduler_args() -> None:
    """Ensure step lr scheduler argparser runs without crashing."""
    cli.add_step_lr_scheduler_args(argparse.ArgumentParser())


def test_add_cosine_lr_scheduler_args() -> None:
    """Ensure cosine lr scheduler argparser runs without crashing."""
    cli.add_cosine_lr_scheduler_args(argparse.ArgumentParser())


def test_add_scheduled_sampling_args() -> None:
    """Ensure scheduled sampling argparser runs without crashing."""
    cli.add_scheduled_sampling_args(argparse.ArgumentParser())
