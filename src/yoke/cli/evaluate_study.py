r"""Yoke CLI: Manually evaluate a checkpoint from a previously-run study.

This CLI tool renders and submits an evaluation job for a single checkpoint of
an existing study, run from a harness directory. It complements the automatic
post-training evaluation performed by
:class:`yoke.harnesses.trainer.HarnessTrainer`: where the trainer evaluates the
final checkpoint of a *finished* study exactly once, this tool lets a user
evaluate *any* checkpoint (an earlier epoch, an EMA companion, or a study that
predates the evaluation feature) on demand and repeatedly.

Usage:
    yoke-evaluate-study --studyIDX 5 \\
        --checkpoint runs/study_005/study005_modelState_epoch0100.pth

Expected Files in Harness Directory:
    - evaluation_input.tmpl
    - evaluation_slurm.tmpl (or evaluation_shell.tmpl)
    - eval_<harness>.py (listed in cp_files.txt)
    - <study_parameters>.csv

The checkpoint path is authoritative for the evaluated epoch; generated artifact
names derive from the checkpoint stem, so an ordinary checkpoint and its EMA
companion (``..._ema.pth``) never collide.
"""

import argparse
from pathlib import Path

from yoke.harnesses.base import HarnessStudy


def add_evaluate_study_args(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Add ``yoke-evaluate-study`` arguments to a parser.

    Args:
        parser (argparse.ArgumentParser | None): An optional existing parser.

    Returns:
        argparse.ArgumentParser: The parser with evaluation arguments added.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="yoke-evaluate-study",
            description="Evaluate a checkpoint from an existing Yoke study.",
        )
    parser.add_argument(
        "--studyIDX",
        type=int,
        required=True,
        help="Study index whose runs/study_### directory holds the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the checkpoint to evaluate. May be an ordinary checkpoint or "
            "an EMA companion (..._ema.pth). May live inside or outside the study "
            "directory."
        ),
    )
    parser.add_argument(
        "--rundir",
        type=str,
        default="./runs",
        help="Directory containing the study_### directories (default: ./runs).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="./hyperparameters.csv",
        help=(
            "Hyperparameter CSV used to re-render evaluation templates for studies "
            "that do not already contain them (default: ./hyperparameters.csv)."
        ),
    )
    parser.add_argument(
        "--cpFile",
        type=str,
        default="./cp_files.txt",
        help=(
            "Text file listing local files to copy into the study directory when "
            "rendering evaluation artifacts on demand (default: ./cp_files.txt)."
        ),
    )
    parser.add_argument(
        "--submissionType",
        choices=["slurm", "shell"],
        default="slurm",
        help="Which job-submission wrapper to use (default: slurm).",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help=(
            "Render evaluation files without submitting. The submit command is "
            "printed instead of run."
        ),
    )
    return parser


def _load_study_row(csv_path: str, studyIDX: int) -> dict | None:
    """Return the CSV row for ``studyIDX`` if the CSV exists, else ``None``.

    Args:
        csv_path (str): Path to the hyperparameter CSV.
        studyIDX (int): Study index to select.

    Returns:
        dict | None: The matching study substitution dict, or ``None`` when the
        CSV is absent or has no matching row.
    """
    if not Path(csv_path).exists():
        return None
    # Build a throwaway HarnessStudy purely to reuse its CSV parsing.
    loader = HarnessStudy.__new__(HarnessStudy)
    rows = HarnessStudy.load_hyperparameters(loader, csv_path)
    for row in rows:
        if int(row["studyIDX"]) == studyIDX:
            return row
    return None


def main() -> None:
    """Entry point for the ``yoke-evaluate-study`` console script.

    Parses arguments, constructs a :class:`HarnessStudy` from the harness
    configuration in the current directory, resolves the study directory and its
    CSV row (used only to re-render evaluation templates when they are missing),
    and renders/submits the evaluation for the requested checkpoint.
    """
    args = add_evaluate_study_args().parse_args()

    harness = HarnessStudy(
        rundir=args.rundir,
        template_dir=".",
        cp_file=args.cpFile,
        submission_type=args.submissionType,
        dryrun=args.dryrun,
    )
    study_dir = Path(args.rundir) / f"study_{args.studyIDX:03d}"
    study_row = _load_study_row(args.csv, args.studyIDX)

    submit_filename = harness.run_evaluation(
        str(study_dir),
        args.checkpoint,
        study=study_row,
    )
    if submit_filename is None:
        print("Evaluation is not configured for this harness; nothing submitted.")
    else:
        print(f"Prepared evaluation submission: {study_dir / submit_filename}")


if __name__ == "__main__":
    main()
