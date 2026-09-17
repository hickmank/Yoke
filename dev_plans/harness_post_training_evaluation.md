# Post-Training Harness Evaluation Plan

## Goal

Add an opt-in, repeatable test-set evaluation workflow for the common
non-Lightning DDP harnesses. A completed training study should submit a separate
evaluation job for its final checkpoint. The same harness artifacts must also
support explicitly evaluating any checkpoint produced by that study.

Evaluation is intentionally a separate job rather than a test phase in the
training allocation. Training requires DDP/NCCL resources and periodically
resubmits itself; evaluation does not train, checkpoint, or need multiple GPUs.
Separate jobs allow each harness to choose appropriate evaluation resources and
permit evaluation to be rerun independently after a failed job or a changed
analysis configuration.

This work excludes the intentionally bespoke `moving_mnist` and
`mnist_surrogate` demos and the Lightning harness.

## Existing Contract

### Initial study creation

`yoke-start-study` is a thin CLI. It parses only `--csv`, `--rundir`,
`--cpFile`, `--submissionType`, and `--dryrun`, creates one `HarnessStudy`, and
calls `run_study` for each CSV row. The first CSV column becomes `studyIDX`;
it is not a CLI flag.

For every CSV row, `HarnessStudy.run_study` currently:

1. Creates `runs/study_###/`.
2. Copies the paths listed in `cp_files.txt` into that directory.
3. Renders `training_input.tmpl` and the selected training submission template
   into first-launch files.
4. Writes rendered continuation templates into the same directory.
5. Submits the first training job unless `--dryrun` was supplied.

The template renderer applies literal `<KEY>` substitution from the CSV and
reserved values. It preserves unknown tokens. Conditional blocks are controlled
by `# <<optional:KEY>>` and `# <<end>>`.

### Training and continuation

Modern common DDP harnesses are thin wrappers around `HarnessTrainer`. They
provide a model builder, train/validation dataset builder, epoch function, and
optional scheduler/loss/hooks. `HarnessTrainer` owns DDP setup, dataloaders,
checkpointing, and continuation submission.

At the end of each job cycle, `HarnessTrainer.finalize` writes
`study###_modelState_epoch####.pth`. On rank 0, if the just-completed epoch is
not the total training epoch, it calls `HarnessStudy.continuation_setup`, which
renders a restart input and submission script inside the study directory and
submits it. Therefore a static dependent job rendered at initial launch cannot
identify the final continuation job; final evaluation must be triggered by the
last training process.

`ch_DDP_loderunner` is an older bespoke DDP script that duplicates this
lifecycle. It should become the reference migration target for this feature and
be converted to the existing `HarnessTrainer` pattern as part of the harness
work, rather than gaining another bespoke finalization path.

## User-Facing Harness Convention

An evaluation-enabled harness will contain these optional files alongside its
normal training files:

```text
evaluation_input.tmpl
evaluation_slurm.tmpl       # required for --submissionType slurm
evaluation_shell.tmpl       # required for --submissionType shell
eval_<harness>.py
```

The evaluation program must appear in that harness's `cp_files.txt` so it is
available from the isolated study directory. The normal training source remains
there as well. If common configuration is factored into a small module, that
module must also be copied.

The feature is enabled by passing `evaluate_after_training=True` to
`HarnessTrainer`. It is disabled by default, preserving every harness without
the new files and avoiding surprise test jobs.

The evaluation input template will include all data/test configuration and a
late-bound checkpoint field:

```text
--checkpoint
<CHECKPOINT>
--test_rcrd_filename
./testing_study<studyIDX>_epoch<epochIDX>.csv
```

It can also use ordinary CSV keys such as `<MAX_TIME_OFFSET>`, `<BATCH_SIZE>`,
`<INPUTFILE>`, `<studyIDX>`, and `<epochIDX>`. Evaluation-specific GPU count,
wall time, output/error names, and environment activation remain in the
harness's submission template, where training resources already live.

The generated files for an automatic or manual evaluation of epoch 100 would
be named:

```text
study005_evaluation_epoch0100.input
study005_evaluation_epoch0100.slurm
testing_study005_epoch0100.csv
```

Names must be checkpoint/epoch-specific. Evaluation must never append a new
run to an ambiguous generic `testing_evaluation.csv`.

## Shared Implementation

### 1. Extend `HarnessStudy` for optional evaluation artifacts

Modify `src/yoke/harnesses/base.py`.

1. Add an evaluation submission-system mapping parallel to
   `SUBMISSION_SYSTEMS`, using `evaluation_slurm.tmpl`/`.slurm` and
   `evaluation_shell.tmpl`/`.sh`.
2. During construction, determine whether evaluation is configured for the
   chosen submission type.
3. Treat no evaluation templates as a valid, disabled configuration.
4. Raise a clear error during `yoke-start-study` setup if exactly one required
   evaluation template is present, so a malformed enabled harness fails before
   jobs are submitted.
5. Add `generate_evaluation_templates(study_dir, study)` and invoke it from
   `run_study` after copying files. It renders CSV/study substitutions into
   `evaluation_input.tmpl` and the selected evaluation submission template,
   but deliberately leaves `<CHECKPOINT>`, `<INPUTFILE>`, and `<epochIDX>` for
   late binding. Store the rendered templates in the study directory under the
   conventional evaluation template names.
6. Add `HarnessStudy.evaluation_setup(checkpointpath, studyIDX, epochIDX,
   submission_type) -> str`. It runs from the study directory on rank 0, reads
   the prepared evaluation templates, replaces all late-bound tokens, writes
   the checkpoint-specific input/submission files, and returns the submission
   filename.
7. Reuse the existing submission command mapping for the final submission:
   `sbatch` for SLURM and `source` for shell. Do not introduce a new CLI or a
   separate study-directory layout.

`evaluation_setup` receives `epochIDX` rather than deriving an epoch from a
filename. The exact checkpoint path returned by training is authoritative,
while the epoch is needed only for stable, human-readable artifact names.

### 2. Add an opt-in final evaluation to `HarnessTrainer`

Modify `src/yoke/harnesses/trainer.py`.

1. Add a keyword-only `evaluate_after_training: bool = False` constructor
   argument and document it.
2. In `finalize`, keep the existing save sequence unchanged: invoke
   `on_before_save`, save the primary `.pth`, then determine `finished`.
3. Only after a successful primary checkpoint save, only on rank 0, and only
   when `finished` is true and the option is enabled:
   - Call `HarnessStudy.evaluation_setup` with `self.new_chkpt_path`,
     `self.args.studyIDX`, `self.epochIDX`, and `args.submissionType` (default
     `slurm`, matching continuation behavior).
   - Submit the returned script with the configured submission command.
4. When unfinished, submit only continuation and never evaluation.
5. Keep evaluation submission before teardown. It does not require a job
   dependency because the checkpoint write has returned and rank 0 has the
   final path; the new job is independent of the training allocation.
6. Do not place this behavior in `TrainerHooks.on_before_save`: that hook runs
   for every checkpoint cycle and before the primary checkpoint exists.

The boolean keeps the shared contract minimal. It avoids a callback whose
implementations would repeat template rendering and submission logic in every
harness. Model- and data-specific evaluation remains in the copied evaluator
script.

### 3. Provide a manual checkpoint path

No new `yoke-start-study` mode is needed. The generated evaluator remains
directly runnable, for example:

```bash
python eval_LodeRunner.py @study005_evaluation_epoch0100.input
```

For a checkpoint not automatically submitted, users call
`HarnessStudy.evaluation_setup` from the study directory (or use a small,
documented harness-local invocation wrapper) with that checkpoint and epoch,
then submit the returned script. The implementation should expose a narrow
public helper rather than replicate rendering logic in shell instructions.

Automatic final evaluation is exactly-once per successful final training job.
Manual invocation remains intentionally possible for earlier epochs and
re-evaluation. The generated filenames make overwrite behavior explicit. A
future enhancement may add a dedicated `yoke-evaluate-study` CLI only after
multiple users demonstrate that calling the helper is insufficient.

## `ch_DDP_loderunner` Reference Implementation

### Training script migration

Refactor `applications/harnesses/ch_DDP_loderunner/train_LodeRunner_ddp.py` to
match the established `se_DDP_loderunner` structure.

1. Keep its current parser and study-specific options.
2. Move the 20-field `channel_list` to a module-level `CHANNEL_LIST` constant.
3. Extract `make_model_args`, `build_optimizer`, and `build_dataset` with the
   current model and LSC temporal dataset settings.
4. Use `build_from_checkpoint` and `HarnessTrainer` instead of manually
   managing DDP, checkpoints, and `HarnessStudy.continuation_setup`.
5. Pass `epoch_kwargs={"channel_map": list(range(len(CHANNEL_LIST))),
   "dataset": "pli"}` and `evaluate_after_training=True`.
6. Preserve the current training CSV names, scheduler behavior, and CSV-driven
   resource substitutions.

This removes duplicated orchestration and ensures the new evaluation trigger
works identically to the other modern DDP harnesses.

### Evaluation script

Replace the current `eval/eval_LodeRunner.py` prototype with a copied,
harness-local `eval_LodeRunner.py` that has a deliberately narrow evaluation
parser. It must not call `add_training_args`, which currently creates a
duplicate `--pretrained_model` argument and carries irrelevant training state.

Its required arguments are:

```text
--checkpoint
--FILELIST_DIR
--LSC_NPZ_DIR
--test_filelist
--batch_size
--num_workers
--test_batches
--max_timeIDX_offset
--test_rcrd_filename
```

Implementation steps:

1. Use `load_model_and_optimizer` with an explicit registry containing
   `LodeRunner`; discard the restored optimizer after load.
2. Obtain the field ordering from the restored model's saved `default_vars`,
   not an independently maintained `number_channels` argument.
3. Construct `LSC_rho2rho_temporal_DataSet` for the test file list with the
   model's full field list and the rendered `max_timeIDX_offset` used by this
   study.
4. Use a non-distributed, non-shuffled `DataLoader`, `nn.MSELoss(reduction="none")`,
   and `channel_map=list(range(len(model.default_vars)))`.
5. Evaluate exactly one configured test pass. Use the checkpoint epoch as the
   record epoch; do not reset it to zero or loop `cycle_epochs` times.
6. Update the shared LodeRunner evaluation epoch/datastep code to run forward
   passes under `torch.inference_mode()` (or surround the test loop with it).
7. Write per-sample records to the checkpoint-specific CSV. Add a small
   sidecar JSON or a header/metadata companion containing checkpoint path,
   saved checkpoint epoch, test filelist, field list, time-offset policy,
   batch limit, and evaluation command/configuration.

The current temporal dataset chooses time pairs randomly. Before treating a
test result as comparable, inspect its sampling API and implement one explicit
evaluation policy: either deterministic index-to-time-pair selection in the
dataset or a documented seeded sampling mode with worker seeding. The evaluator
must record that seed/policy in its metadata. Do not describe repeated random
draws as multiple evaluation epochs.

### Templates and copied files

Move the active evaluation workflow out of the `eval/` subdirectory into the
standard optional harness files:

```text
applications/harnesses/ch_DDP_loderunner/
  eval_LodeRunner.py
  evaluation_input.tmpl
  evaluation_slurm.tmpl
```

Update `cp_files.txt` to copy `train_LodeRunner_ddp.py`, `eval_LodeRunner.py`,
`eval_START.input`, `eval_START.slurm`, and `avg_eval_csv.py` only after the new
workflow includes equivalent summary output or a clear result reader. Keep
evaluation scheduler settings independent from DDP training: one GPU, no DDP
environment setup, and its own output/error files.

The evaluation template must render test data options from the same CSV keys
or fixed harness values as training. In particular it must use
`<MAX_TIME_OFFSET>` and the 20-channel saved model configuration; it must not
hard-code the prototype's incompatible 8-channel default or time offset of 2.

## Adoption by Other Common Harnesses

After the reference flow is verified, add evaluation only where a meaningful
test dataset and inference metric already exist or can be specified.

1. LodeRunner and LodeRunner-ViT harnesses can share the LSC temporal evaluator
   pattern, with each evaluator's registry including its model class and with
   EMA runs explicitly selecting either the ordinary final checkpoint or the
   EMA companion checkpoint.
2. Cylex and two-frame variants need evaluator-specific dataset and datastep
   selection; they should reuse submission/rendering but not force the
   single-frame LodeRunner evaluator.
3. Array-output/surrogate and policy harnesses require their own thin evaluator
   because batches, model calls, and metrics differ. Their training scripts
   only add `evaluate_after_training=True` once their three evaluation files
   and deterministic result contract exist.
4. Diffusion harnesses require a diffusion-aware evaluation contract (noise
   schedule, sampling procedure, and metric) before opting in. Do not enable
   them merely because they can load a checkpoint.

This order gives all common DDP harnesses one shared lifecycle API without
pretending their scientific test metrics are interchangeable.

## Tests

### `tests/harnesses/test_base.py`

1. Verify initial study creation with evaluation templates writes the rendered
   templates into each study directory while preserving late-bound tokens.
2. Verify a harness with no evaluation templates continues to work unchanged.
3. Verify incomplete evaluation template pairs fail early with an actionable
   error.
4. Parameterize `evaluation_setup` for SLURM and shell. Assert exact input and
   submission filenames, checkpoint substitution, input-file substitution, and
   zero-padded study/epoch substitution.
5. Verify CSV substitutions, including test output filenames, occur before
   late-bound rendering.

### `tests/cli/test_start_study.py`

1. Extend the dry-run fixture with optional evaluation templates and copied
   evaluator source.
2. Assert each study directory contains both training continuation templates
   and evaluation templates, while dry-run still submits no jobs.
3. Keep existing tests proving that the CLI interface and standard-only
   harnesses remain unchanged.

### `tests/harnesses/test_trainer.py`

1. Extend the patched fixture to record `evaluation_setup` and submission
   calls.
2. Verify a finished `evaluate_after_training=True` trainer saves first, calls
   evaluation setup exactly once on rank 0 with the exact checkpoint path and
   final epoch, then submits the returned script.
3. Verify an unfinished cycle submits continuation only and never evaluation.
4. Verify the default flag and `evaluate_after_training=False` submit no
   evaluation.
5. Verify no duplicate evaluation occurs for nonzero ranks.

### Evaluation unit tests

Add focused tests for the new LodeRunner evaluator without real LSC data:

1. Mock checkpoint loading and assert model-derived field ordering is passed to
   the test dataset.
2. Assert the dataset receives the rendered test filelist and time offset.
3. Assert the checkpoint epoch is propagated to records and only one test pass
   runs.
4. Assert the evaluator rejects missing checkpoint/model metadata clearly.
5. Add a datastep/epoch test proving inference mode is used and no gradients
   are retained.
6. Add a deterministic sampling test once the dataset evaluation policy is
   selected.

## Verification

Use the Yoke-installed Python interpreter supplied by the user before running
tests or CLI commands. From the repository root, run:

```bash
pytest -Werror tests/harnesses/test_base.py tests/harnesses/test_trainer.py \
    tests/cli/test_start_study.py
ruff check src/yoke applications/harnesses/ch_DDP_loderunner tests
ruff format --check src/yoke applications/harnesses/ch_DDP_loderunner tests
```

Then run a `yoke-start-study --dryrun` invocation from
`applications/harnesses/ch_DDP_loderunner` with a small representative CSV and
inspect one `runs/study_###/` directory. Confirm it contains copied train/eval
scripts, training continuation templates, evaluation templates, and correctly
rendered start files. Finally, in a non-production scheduler allocation, call
`HarnessStudy.evaluation_setup` for a known checkpoint and inspect the rendered
evaluation input and submission script before submitting it.

## Non-Goals

- Do not add training, validation, or model saving to evaluation jobs.
- Do not change `yoke-start-study` flags or make `studyIDX` a CLI argument.
- Do not run test evaluation after every continuation checkpoint by default.
- Do not make a single model-agnostic evaluator that hides model/dataset/metric
  differences.
- Do not alter the Lightning or MNIST demo workflows in this feature.
