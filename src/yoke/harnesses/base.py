"""Base class for a Yoke harness."""

import os
import shutil
import pandas as pd
from pathlib import Path
from yoke.helpers import strings


class HarnessStudy:
    """HarnessStudy class.

    Defines class containing attributes for a Yoke Harness object. Methods of this
    class are then used to submit entries of a Yoke study.

    Templating contract:
        A harness is defined by a single input template (``training_input.tmpl``)
        and a single submission template (``training_slurm.tmpl`` or
        ``training_shell.tmpl``). Both are rendered by literal ``<KEY>``
        substitution (see :func:`yoke.helpers.strings.replace_keys`). The
        following keys are reserved and injected by this class rather than the
        hyperparameter CSV:

        - ``<studyIDX>`` — the study index; zero-padded to three digits.
        - ``<epochIDX>`` — the epoch index; always zero-padded to four digits. Set
          to ``0001`` for the first-launch submission by
          :meth:`generate_initial_inputs`, left as a placeholder in the
          continuation templates written by :meth:`generate_tmpl_inputs`, and set
          to the next epoch (``last_epoch + 1``) by :meth:`continuation_setup`.
        - ``<INPUTFILE>`` — the input file the submission script should read; set
          to the first-launch input by :meth:`generate_initial_inputs` and to the
          per-epoch restart input by :meth:`continuation_setup`.
        - ``<CONTINUATION>`` — present only in the continuation form; used with
          ``# <<optional:CONTINUATION>>`` blocks to include ``--continuation``.
        - ``<CHECKPOINT>`` — the checkpoint path to resume from; left as a
          placeholder in the continuation templates and substituted per epoch by
          :meth:`continuation_setup`.

        Lines bounded by ``# <<optional:KEY>>`` and ``# <<end>>`` are included in
        the rendered output only when ``KEY`` is present in the substitution dict.

    Args:
        rundir (str or Path): Output directory for the study run
        template_dir (str or Path): Directory containing .tmpl files
        cp_file (str or Path): File listing training files to copy per study
        submission_type (str): Job-submission system to prepare files for. One of
            ``"slurm"`` or ``"shell"``.
        dryrun (bool): Flag to turn off job submission.

    """

    #: Dispatch table mapping a submission system to its template filename, the
    #: output-script extension, and the command used to submit the script.
    SUBMISSION_SYSTEMS: dict[str, dict[str, str]] = {
        "slurm": {
            "template": "training_slurm.tmpl",
            "ext": "slurm",
            "submit": "sbatch",
        },
        "shell": {
            "template": "training_shell.tmpl",
            "ext": "sh",
            "submit": "source",
        },
    }
    EVALUATION_SYSTEMS: dict[str, dict[str, str]] = {
        "slurm": {
            "template": "evaluation_slurm.tmpl",
            "ext": "slurm",
            "submit": "sbatch",
        },
        "shell": {
            "template": "evaluation_shell.tmpl",
            "ext": "sh",
            "submit": "source",
        },
    }

    def __init__(
        self,
        rundir: str = "./runs",
        template_dir: str = ".",
        cp_file: str = "cp_files.txt",
        submission_type: str = "slurm",
        dryrun: bool = False,
    ) -> None:
        """Initialization for HarnessStudy."""
        self.rundir = Path(rundir)
        self.template_dir = Path(template_dir)
        self.cp_file = Path(cp_file)
        self.DRYRUN = dryrun

        submission_type = submission_type.lower()
        if submission_type not in self.SUBMISSION_SYSTEMS:
            valid = ", ".join(sorted(self.SUBMISSION_SYSTEMS))
            raise ValueError(
                f"Unknown submission type {submission_type!r}. "
                f"Supported types are: {valid}."
            )
        self.submission_type = submission_type
        self.submission_config = self.SUBMISSION_SYSTEMS[submission_type]

        # Template and base files
        self.input_template = self.template_dir / "training_input.tmpl"
        self.submission_template = self.template_dir / self.submission_config["template"]
        self.evaluation_input_template = self.template_dir / "evaluation_input.tmpl"
        self.evaluation_config = self.EVALUATION_SYSTEMS[submission_type]
        self.evaluation_submission_template = (
            self.template_dir / self.evaluation_config["template"]
        )
        evaluation_files = (
            self.evaluation_input_template,
            self.evaluation_submission_template,
        )
        if any(path.exists() for path in evaluation_files) and not all(
            path.exists() for path in evaluation_files
        ):
            raise ValueError(
                "Evaluation requires both evaluation_input.tmpl and "
                f"{self.evaluation_config['template']}."
            )
        self.evaluation_enabled = all(path.exists() for path in evaluation_files)

        self.rundir.mkdir(parents=True, exist_ok=True)

    def load_hyperparameters(self, csv_path: str) -> list[dict]:
        """Read hyperparameters from a CSV into a list of dicts.

        Args:
            csv_path (str): Path to the hyperparameter CSV. The first column is
                treated as the study index (``studyIDX``).

        Returns:
            list[dict]: One dictionary per study row, with the study index stored
            under the ``studyIDX`` key.
        """
        df = pd.read_csv(
            csv_path,
            sep=",",
            header=0,
            index_col=0,
            comment="#",
            engine="python",
        )

        study_list = []
        for idx in df.index.values:
            # Selecting a row as a Series coerces all numeric columns to a common
            # dtype. Keep the one-row DataFrame so integer template fields remain
            # integers when a study also contains floating-point hyperparameters.
            study = df.loc[[idx]].to_dict(orient="records")[0]
            study["studyIDX"] = int(idx)
            study_list.append(study)

        return study_list

    def render_template(self, template_path: Path, substitutions: dict) -> str:
        """Render a template file, honoring optional conditional blocks.

        Lines bounded by ``# <<optional:KEY>>`` and ``# <<end>>`` are included in
        the rendered output only when ``KEY`` is present in ``substitutions``.
        All remaining ``<KEY>`` tokens are substituted via
        :func:`yoke.helpers.strings.replace_keys`.

        Args:
            template_path (Path): Path to the template file to render.
            substitutions (dict): Mapping of keys to values for substitution.

        Returns:
            str: The rendered template contents.
        """
        with open(template_path) as f:
            lines = f.readlines()

        rendered = []
        skip_block = False
        for line in lines:
            if line.strip().startswith("# <<optional:"):
                key = line.strip().split(":")[1].rstrip(">>")
                skip_block = key not in substitutions
                continue
            if line.strip() == "# <<end>>":
                skip_block = False
                continue
            if skip_block:
                continue
            rendered.append(strings.replace_keys(substitutions, line))

        return "".join(rendered)

    def copy_files(self, study_dir: Path) -> None:
        """Copy the files listed in ``cp_file`` into the study directory.

        Args:
            study_dir (Path): Destination directory for the copied files.
        """
        with open(self.cp_file) as f:
            for line in f:
                file_path = line.strip()
                if file_path:
                    shutil.copy(file_path, study_dir)
                    print(f"[COPY] {file_path} -> {study_dir}")

    def generate_initial_inputs(self, study_dir: Path, study: dict) -> Path:
        """Generate the input and submission scripts for the first submission.

        Args:
            study_dir (Path): Directory in which to write the first-launch files.
            study (dict): Study substitution dictionary.

        Returns:
            Path: Path to the generated first-launch submission script.
        """
        sid = study["studyIDX"]
        # First-launch epoch is 1, zero-padded to four digits.
        study["epochIDX"] = f"{1:04d}"
        study["INPUTFILE"] = f"study{sid:03d}_START.input"

        # Ensure that the continuation and checkpoint arguments do not appear in the
        # intialization inputs.
        study.pop("CONTINUATION", None)

        # Render input and submission templates
        input_rendered = self.render_template(self.input_template, study)
        submit_rendered = self._render_submission_template(study)

        # Modify START files with substitutions
        ext = self.submission_config["ext"]
        input_path = study_dir / f"study{sid:03d}_START.input"
        submit_path = study_dir / f"study{sid:03d}_START.{ext}"

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(submit_path, "w") as f:
            f.write(submit_rendered)

        return submit_path

    def generate_tmpl_inputs(self, study_dir: Path, study: dict) -> None:
        """Generate the input and submission templates for job continuation.

        Args:
            study_dir (Path): Directory in which to write the continuation
                templates.
            study (dict): Study substitution dictionary.
        """
        study = dict(study)
        # For templates epochIDX and INPUTFILE should be left as variables.
        study.pop("epochIDX", None)
        study.pop("INPUTFILE", None)
        study["CONTINUATION"] = True

        # Render input and submission templates
        input_rendered = self.render_template(self.input_template, study)
        submit_rendered = self._render_submission_template(study)

        # Modify template files with substitutions
        input_path = study_dir / "training_input.tmpl"
        submit_path = study_dir / self.submission_config["template"]

        with open(input_path, "w") as f:
            f.write(input_rendered)
        with open(submit_path, "w") as f:
            f.write(submit_rendered)

    def generate_evaluation_templates(self, study_dir: Path, study: dict) -> None:
        """Render configured evaluation templates with late-bound values preserved.

        Args:
            study_dir (Path): Directory in which to write the evaluation templates.
            study (dict): Study substitution dictionary.
        """
        if not self.evaluation_enabled:
            return

        substitutions = dict(study)
        for key in ("CHECKPOINT", "INPUTFILE", "STEM", "epochIDX"):
            substitutions.pop(key, None)
        input_rendered = self.render_template(
            self.evaluation_input_template, substitutions
        )
        submit_rendered = self.render_template(
            self.evaluation_submission_template, substitutions
        )
        (study_dir / "evaluation_input.tmpl").write_text(input_rendered)
        (study_dir / self.evaluation_config["template"]).write_text(submit_rendered)

    def _render_submission_template(self, study: dict) -> str:
        """Return the rendered submission script for the selected system.

        The submission template file supplied by the harness is a complete
        submission script; its ``<KEY>`` tokens are substituted from ``study``.

        Args:
            study (dict): Study substitution dictionary.

        Returns:
            str: The rendered submission script.
        """
        with open(self.submission_template) as f:
            tmpl = f.read()

        return strings.replace_keys(study, tmpl)

    def submit_job(self, study_dir: Path, submit_path: Path) -> None:
        """Submit a job using the configured submission system.

        Args:
            study_dir (Path): Directory containing the submission script.
            submit_path (Path): Path to the submission script to submit.
        """
        submit_cmd = self.submission_config["submit"]
        submit_str = f"cd {study_dir}; {submit_cmd} {submit_path.name}; cd .."

        if self.DRYRUN:
            # Just print what would be executed
            print(f"[DRY RUN] Would execute: {submit_str}.")
        else:
            # Submit Job
            os.system(submit_str)

    def run_study(self, study: dict) -> None:
        """Run a single study: generate inputs, copy files, and submit job.

        Args:
            study (dict): Study substitution dictionary for a single CSV row.
        """
        # Make Study Directory
        study_dir = self.rundir / "study_{:03d}".format(study["studyIDX"])
        study_dir.mkdir(parents=True, exist_ok=True)

        self.copy_files(study_dir)
        self.generate_tmpl_inputs(study_dir, study)
        self.generate_evaluation_templates(study_dir, study)
        submit_path = self.generate_initial_inputs(study_dir, study)
        self.submit_job(study_dir, submit_path)

    @staticmethod
    def continuation_setup(
        checkpointpath: str,
        studyIDX: int,
        last_epoch: int,
        submission_type: str = "slurm",
    ) -> str:
        """Prepare restart submission files for continued training.

        This runs on the compute node from within a train script, using the
        continuation templates written by :meth:`generate_tmpl_inputs` (i.e.
        ``./training_input.tmpl`` and the submission template for the selected
        system). It substitutes ``<CHECKPOINT>`` into the input file and
        ``<INPUTFILE>`` / ``<epochIDX>`` into the submission script, writes the
        per-epoch restart files, and returns the new submission-script filename.

        Args:
            checkpointpath (str): Path to the model checkpoint to resume from.
            studyIDX (int): Study index, used in the generated file names.
            last_epoch (int): Number of epochs completed at this checkpoint.
            submission_type (str): Job-submission system, either ``"slurm"`` or
                ``"shell"``. Defaults to ``"slurm"``.

        Returns:
            str: Filename of the submission script for continued training.

        Raises:
            ValueError: If ``submission_type`` is not a supported system.
        """
        submission_type = submission_type.lower()
        if submission_type not in HarnessStudy.SUBMISSION_SYSTEMS:
            valid = ", ".join(sorted(HarnessStudy.SUBMISSION_SYSTEMS))
            raise ValueError(
                f"Unknown submission type {submission_type!r}. "
                f"Supported types are: {valid}."
            )
        config = HarnessStudy.SUBMISSION_SYSTEMS[submission_type]
        ext = config["ext"]

        # Input template is independent of the submission system.
        training_input_tmpl = "./training_input.tmpl"
        with open(training_input_tmpl) as f:
            training_input_data = f.read()

        new_training_input_data = training_input_data.replace(
            "<CHECKPOINT>", checkpointpath
        )

        input_str = "study{0:03d}_restart_training_epoch{1:04d}.input"
        new_training_input_filepath = input_str.format(studyIDX, last_epoch + 1)

        with open(os.path.join("./", new_training_input_filepath), "w") as f:
            f.write(new_training_input_data)

        # Render the submission script for the selected system.
        submission_tmpl = os.path.join("./", config["template"])
        with open(submission_tmpl) as f:
            submission_data = f.read()

        submit_str = "study{0:03d}_restart_training_epoch{1:04d}." + ext
        new_submission_filepath = submit_str.format(studyIDX, last_epoch + 1)

        submission_data = submission_data.replace(
            "<INPUTFILE>", new_training_input_filepath
        )
        submission_data = submission_data.replace("<epochIDX>", f"{last_epoch + 1:04d}")

        with open(os.path.join("./", new_submission_filepath), "w") as f:
            f.write(submission_data)

        return new_submission_filepath

    @staticmethod
    def evaluation_setup(
        checkpointpath: str,
        studyIDX: int,
        submission_type: str = "slurm",
    ) -> str:
        """Prepare checkpoint-specific evaluation input and submission files.

        This runs from a generated study directory after
        :meth:`generate_evaluation_templates` (or :meth:`run_evaluation`) has
        prepared its templates. The checkpoint path is authoritative: the epoch
        that appears in evaluation records comes from the checkpoint metadata
        itself, while the generated artifact names are derived from the
        checkpoint *stem*. Using the stem keeps the ordinary and EMA companion
        checkpoints (``..._epoch0100.pth`` vs ``..._epoch0100_ema.pth``) from
        colliding on output filenames.

        Args:
            checkpointpath (str): Path to the checkpoint being evaluated.
            studyIDX (int): Study index used in generated filenames.
            submission_type (str): Job-submission system, ``"slurm"`` or ``"shell"``.

        Returns:
            str: Filename of the generated evaluation submission script.

        Raises:
            ValueError: If ``submission_type`` is unsupported.
        """
        submission_type = submission_type.lower()
        if submission_type not in HarnessStudy.EVALUATION_SYSTEMS:
            valid = ", ".join(sorted(HarnessStudy.EVALUATION_SYSTEMS))
            raise ValueError(
                f"Unknown submission type {submission_type!r}. "
                f"Supported types are: {valid}."
            )
        config = HarnessStudy.EVALUATION_SYSTEMS[submission_type]
        # Derive artifact names from the checkpoint stem so that distinct
        # checkpoints (including EMA companions) never share output files.
        stem = Path(checkpointpath).stem
        input_filename = f"study{studyIDX:03d}_evaluation_{stem}.input"
        submit_filename = f"study{studyIDX:03d}_evaluation_{stem}.{config['ext']}"
        substitutions = {
            "CHECKPOINT": checkpointpath,
            "INPUTFILE": input_filename,
            "studyIDX": studyIDX,
            "STEM": stem,
        }
        input_data = strings.replace_keys(
            substitutions, Path("evaluation_input.tmpl").read_text()
        )
        submission_data = strings.replace_keys(
            substitutions, Path(config["template"]).read_text()
        )
        Path(input_filename).write_text(input_data)
        Path(submit_filename).write_text(submission_data)
        return submit_filename

    def run_evaluation(
        self,
        study_dir: str,
        checkpointpath: str,
        study: dict | None = None,
    ) -> str | None:
        """Render and submit an evaluation job for a single checkpoint.

        This is the shared render-and-submit path used by both automatic
        (:class:`~yoke.harnesses.trainer.HarnessTrainer`) and manual
        (``yoke-evaluate-study``) evaluation. It:

        1. Ensures the study directory holds rendered evaluation templates
           (``evaluation_input.tmpl`` and the submission template). If they are
           missing -- e.g. a study produced before this feature existed -- they
           are rendered on demand from ``self.template_dir`` using ``study``,
           reproducing the same CSV-derived keys training used.
        2. Ensures the evaluator program and any other ``cp_file`` entries are
           present in the study directory, copying them if necessary.
        3. Renders the checkpoint-specific input/submission files via
           :meth:`evaluation_setup` (from within ``study_dir``).
        4. Submits the rendered script with the evaluation submit command,
           honoring ``--dryrun`` exactly like :meth:`submit_job`.

        The ``studyIDX`` used for artifact naming is read from ``study`` when
        provided, otherwise parsed from the ``study_###`` directory name. The
        checkpoint path is authoritative for the evaluated epoch; artifact names
        derive from the checkpoint stem.

        Args:
            study_dir (str): Path to the ``runs/study_###`` directory.
            checkpointpath (str): Path to the checkpoint to evaluate. May be an
                ordinary checkpoint or an EMA companion (``..._ema.pth``). It is
                resolved to an absolute path so it remains valid after changing
                into ``study_dir``.
            study (dict | None): Substitution dictionary (a CSV row) used only to
                render missing evaluation templates and to supply ``studyIDX``.
                May be ``None`` when the study directory already contains rendered
                evaluation templates.

        Returns:
            str | None: The submission-script filename, or ``None`` if evaluation
            is not configured for this harness (no templates in the study
            directory and none defined by the harness).

        Raises:
            ValueError: If evaluation templates are missing from the study
                directory but the harness *does* define them, yet no ``study``
                row was supplied to render them; or if ``studyIDX`` cannot be
                determined.
        """
        study_dir = Path(study_dir)
        study_dir.mkdir(parents=True, exist_ok=True)

        # Determine studyIDX from the provided row or the directory name.
        if study is not None and "studyIDX" in study:
            studyIDX = int(study["studyIDX"])
        else:
            try:
                studyIDX = int(study_dir.name.split("_")[-1])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Could not determine studyIDX from directory {study_dir.name!r}; "
                    "pass a study dict containing 'studyIDX'."
                ) from exc

        eval_input = study_dir / "evaluation_input.tmpl"
        eval_submit = study_dir / self.evaluation_config["template"]

        # Render evaluation templates on demand for studies that predate the
        # feature (or whose templates were not generated at launch).
        if not (eval_input.exists() and eval_submit.exists()):
            if not self.evaluation_enabled:
                # Neither the study directory nor the harness defines evaluation
                # templates: this harness simply has no evaluation configured.
                print(
                    "Evaluation is not configured for this harness "
                    f"({self.template_dir} defines no evaluation templates and none "
                    f"are present in {study_dir}); nothing to do."
                )
                return None
            if study is None:
                raise ValueError(
                    "Evaluation templates are missing from the study directory and "
                    "no study row was provided to render them. Supply the CSV row "
                    "for this study so the templates can be reproduced."
                )
            self.generate_evaluation_templates(study_dir, study)
            self.copy_files(study_dir)

        # Resolve the checkpoint to an absolute path before changing directory.
        checkpoint_abs = str(Path(checkpointpath).resolve())

        config = self.EVALUATION_SYSTEMS[self.submission_type]
        cwd = os.getcwd()
        try:
            os.chdir(study_dir)
            submit_filename = self.evaluation_setup(
                checkpoint_abs, studyIDX, self.submission_type
            )
        finally:
            os.chdir(cwd)

        # Submit from within the study directory in a subshell so relative
        # paths in the submission script (its @-input file, --output/--error)
        # resolve there and the caller's working directory is left unchanged.
        submit_str = f"( cd {study_dir}; {config['submit']} {submit_filename} )"
        if self.DRYRUN:
            print(f"[DRY RUN] Would execute: {submit_str}.")
        else:
            os.system(submit_str)
        return submit_filename
