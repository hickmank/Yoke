# Migrate Remaining DDP Harnesses to HarnessTrainer

## Goal

Migrate every remaining non-MNIST, non-Lightning legacy DDP training harness
to `yoke.harnesses.trainer.HarnessTrainer`. Preserve each study's model,
dataset, optimizer, scheduler, records, checkpoint metadata, templates, and
continuation behavior while removing duplicated DDP orchestration.

In this plan, “new HarnessStudy format” means the established combined pattern:

1. `yoke-start-study` creates isolated study directories through
   `HarnessStudy` from a CSV and the training templates.
2. A thin copied training script uses `HarnessTrainer` for DDP setup, data
   loaders, epochs, checkpointing, continuation rendering, submission, and
   teardown.

`HarnessStudy` itself already supports all in-scope training harnesses. The
remaining migration work is in the training scripts that still manually
duplicate the lifecycle now owned by `HarnessTrainer`.

## Inventory

### Legacy DDP scripts to migrate

| Harness | Training script | Model | Dataset | Epoch function |
| --- | --- | --- | --- | --- |
| `ch_DDP_loderunner` | `train_LodeRunner_ddp.py` | `LodeRunner` | `LSC_rho2rho_temporal_DataSet` | `train_DDP_loderunner_epoch` |
| `ch_lsc_reward` | `train_lsc_reward.py` | `hybrid2vectorCNN` | `LSC_hfield_reward_DataSet` | `train_lsc_reward_epoch` |
| `ch_lsc_inverse` | `train_lsc_inverse.py` | `Image2VectorCNN` | `LSC_hfield2cntr_DataSet` | `train_DDP_array_epoch` |
| `ch_DDP_diffLDR` | `train_DDP_diffLDR.py` | `DiffusionLodeRunner` | `DiffusionLSC_temporal_DataSet` | `train_DDP_diffusion_loderunner_epoch` |

Each manually performs the same operations: distributed setup/teardown, model
restore/build, optimizer restore/build, `DDP` wrapping, distributed train and
validation loader construction, per-epoch sampler/timing, checkpoint creation,
and rank-zero continuation submission. These responsibilities are already
covered by `HarnessTrainer`.

### Already migrated

The following scripts already correctly use `HarnessTrainer` and require no
trainer migration:

- `se_DDP_loderunner/train_LodeRunner_ddp.py`
- `vt_DDP_loderunner/train_LodeRunner_ddp.py`
- `se_DDP_loderunner_cylex/train_LodeRunner_ddp_cylex.py`
- `se_DDP_loderunner_finetune_cylex/train_LodeRunner_ddp_cylex.py`
- `ch_ldrViT/train_ldrViT_ddp.py`
- `ch_ldrViT/train_ldrViT_2frame.py`
- `se_ldrViT/train_ldrViT_ddp.py`
- `vt_DDP_ldrViT/train_ldrViT_ddp.py`
- `ch_lsc_policy/train_lsc_policy.py`
- `lsc_action/train_lsc_action.py`

`lsc_action` needs a separate submission-template correction: its Python
script is already a DDP `HarnessTrainer` harness, but its SLURM template still
launches a single Python process without the DDP environment expected by
`setup_distributed`.

The deliberately bespoke MNIST demos and `ch_lightning_loderunner` are out of
scope.

## Invariants

Every migration must retain the existing `yoke-start-study` contract:

1. Launch from the harness directory with a hyperparameter CSV, `cp_files.txt`,
   and one selected `training_{slurm,shell}.tmpl`.
2. The first CSV column remains the source of `studyIDX` and names
   `runs/study_###/`.
3. The first job consumes `study###_START.input`; continuation jobs consume the
   rendered restart input created from `training_input.tmpl`.
4. Only rank 0 submits continuation work.
5. A continuation retains the optimizer state loaded from the checkpoint.
6. Each cycle writes `study###_modelState_epoch####.pth` using the concrete
   class and construction arguments of the built model.
7. Existing record naming and substitution keys keep working.
8. No training script keeps its own DDP setup, `DDP(...)`, distributed loader,
   save, continuation, `os.system`, barrier, or teardown code after migration.

Before changing any legacy harness, compare its rendered start and continuation
arguments with the new version. The migration must not accidentally turn a
template-specific setting into a parser default or silently change a CSV
parameter's meaning.

## Shared Migration Pattern

For each legacy script:

1. Retain the parser and its existing defaults/options.
2. Move study constants to uppercase module-level names where appropriate.
3. Extract `make_model_args(args) -> dict` for model reconstruction metadata.
4. Extract a `build_model(args, device)` function, or use
   `build_from_checkpoint(model_class, make_model_args, optimizer_kwargs=...)`
   where fresh and continuation construction are conventional.
5. Extract `build_optimizer(model, args)` only when the fresh optimizer differs
   from the shared default or needs special parameter groups.
6. Extract `build_dataset(args) -> (train_dataset, val_dataset)`.
7. Extract `build_scheduler(optimizer, args, last_epoch)` if used.
8. Instantiate `HarnessTrainer` under the `if __name__ == "__main__"` guard,
   passing the builders, current epoch function, and static `epoch_kwargs`.
9. Delete the manual orchestration code and no-longer-used imports.

Use `se_DDP_loderunner` as the baseline single-frame LodeRunner example,
`lsc_action` for array-output datasets, `ch_lsc_policy` for a bespoke model
builder/optimizer, and `ch_ldrViT` for `TrainerHooks`.

Do not extend `HarnessTrainer` solely for one legacy script if a small harness
local epoch wrapper can prepare dynamic per-run arguments. In particular,
device-specific tensors for diffusion can be created in a wrapper that receives
the trainer-provided `device` argument and forwards to the existing epoch
function.

## Migration Order

### 1. `ch_DDP_loderunner`

This is the lowest-risk conversion and should be performed first. It is nearly
the same as already-migrated `se_DDP_loderunner` and is the intended reference
for the post-training evaluation feature.

1. Replace manual imports for `os`, `time`, `torch.distributed`, `DDP`,
   distributed loading, saving, continuation, and parallel setup with
   `HarnessTrainer`, `build_adamw`, and `build_from_checkpoint`.
2. Promote its exact 20-field material/velocity list to `CHANNEL_LIST`.
3. Add `make_model_args` preserving the existing image/patch/window architecture
   and `embed_dim`/`block_structure` substitutions.
4. Add `build_dataset` preserving `half_image=True`, `max_file_checks=10`,
   `args.max_timeIDX_offset`, and the 20-field NumPy array for both train and
   validation datasets.
5. Add a fresh optimizer builder matching its current AdamW parameters: learning
   rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0.01`.
6. Add the existing constant-with-warmup scheduler with `warmup_steps=0` and
   constant LR `1e-4`.
7. Use `build_from_checkpoint(LodeRunner, make_model_args, ...)` with the
   legacy continuation optimizer arguments. Confirm whether its current
   continuation LR of `1e-6` versus fresh LR of `1e-4` is intentional; preserve
   it until the study owner explicitly changes it.
8. Pass the LodeRunner epoch function with the 20-channel identity map and
   `dataset="pli"`.
9. Enable the new evaluation option only after the post-training evaluation
   feature and its harness artifacts are implemented and tested.

### 2. `ch_lsc_reward`

This is a direct custom-builder migration with no expected trainer extension.

1. Preserve the full-image reward dataset configuration:
   `LSC_hfield_reward_DataSet`, design data, `density_throw`, and
   `half_image=False`.
2. Extract the exact `hybrid2vectorCNN` model-argument dictionary and ensure it
   is saved unchanged in checkpoints.
3. Use `build_from_checkpoint` if its fresh and continuation branches are
   standard; otherwise retain a focused `build_model` that returns the restored
   optimizer on continuation.
4. Preserve AdamW parameters, including its learning rate and weight decay, and
   retain the unscaled cosine-with-warmup scheduler driven by `anchor_lr`.
5. Pass `train_lsc_reward_epoch` directly with no new generic abstractions.
6. Verify the result uses the same record files, train/validation batch limits,
   and `TRAIN_PER_VAL` cadence as the legacy script.

### 3. `ch_lsc_inverse`

This is a direct array-output migration, using `lsc_action` as the closest
reference.

1. Preserve the `LSC_hfield2cntr_DataSet` arguments, including full images,
   `density_throw`, and the time-inclusive inverse target configuration.
2. Preserve the fixed `Image2VectorCNN` architecture and 29-dimensional output
   in `make_model_args`.
3. Use `train_DDP_array_epoch` and the current unscaled cosine schedule.
4. Preserve the current fresh optimizer (LR `1e-3`, zero weight decay) and
   explicitly document/review the differing continuation loader arguments
   (LR `1e-2`, weight decay `0.01`). A migration should retain the existing
   behavior unless it is confirmed as a defect in a separately scoped change.
5. Use a bespoke `build_model` if required to make that fresh/continuation
   optimizer distinction obvious rather than obscuring it in a generic helper.

### 4. `ch_DDP_diffLDR`

Migrate last because the model and epoch function have diffusion-specific
runtime inputs, but keep the solution harness-local.

1. Promote the active density/velocity channel list to `CHANNEL_LIST` without
   changing the study's selected fields.
2. Extract `make_model_args` preserving the `(10, 10)` patch size, architecture,
   and model class `DiffusionLodeRunner`.
3. Extract `build_dataset`; construct a `VPCosineNoiseSchedule` and pass it,
   `CHANNEL_LIST` as both input and output variables, `half_image=True`, and
   the configured maximum time offset into both temporal diffusion datasets.
4. Preserve the fresh/continuation AdamW arguments and cosine-with-warmup
   scheduler behavior.
5. Add a small local `run_epoch(**kwargs)` wrapper. It receives the trainer's
   `device`, constructs `in_vars` and `out_vars` from `CHANNEL_LIST` on that
   device, and calls `train_DDP_diffusion_loderunner_epoch`. This avoids adding
   a dynamic-kwargs feature to `HarnessTrainer` for one model family.
6. Confirm the shared trainer passes the DDP-wrapped model and that its generic
   checkpoint save path correctly unwraps it. Preserve the legacy model class
   and model args in the checkpoint.
7. Repair the study launch input before validation: its SLURM template uses
   `<train_script>`, but `study_template.csv` does not provide that column.
   Either add a `train_script` CSV field with `train_DDP_diffLDR.py` or replace
   the token with that fixed filename. Prefer the fixed filename unless this
   harness intentionally selects scripts per row.

## `lsc_action` Submission Remediation

Modify only `applications/harnesses/lsc_action/training_slurm.tmpl`; its Python
script is already migrated.

1. Request task/GPU resources consistently with `args.Ngpus` and `args.Knodes`.
2. Export `MASTER_ADDR` from the SLURM nodelist and a valid `MASTER_PORT` before
   launching Python.
3. Invoke the copied training script through `srun`, not bare `python`, so every
   task receives the SLURM rank/local-rank/world-size environment needed by
   `setup_distributed`.
4. Preserve the scheduler account/allocation, job naming, wall time, output,
   error, and environment activation conventions already used by this harness.
5. Dry-run the harness through `yoke-start-study` and inspect both its start and
   continuation submission files for fully substituted resource/script tokens.

## Template and Copy-File Checks

For all four legacy harnesses and the remediated `lsc_action` template:

1. Keep `training_input.tmpl`'s continuation block:
   `--continuation` and `--checkpoint <CHECKPOINT>` must occur only inside the
   `CONTINUATION` optional block.
2. Ensure each SLURM template references the copied script by a token that is
   supplied by its CSV/template, or by the exact copied filename.
3. Ensure `cp_files.txt` lists every local Python source the rendered job
   imports from the run directory. It must at minimum include the training
   script.
4. Do not add package source paths to `cp_files.txt`; reusable logic remains in
   installed `src/yoke`.
5. Test `slurm` and `shell` at the generic `HarnessStudy` layer; individual
   Venado-specific harness templates need only their configured submission type.

## Tests

Unit tests are limited to reusable code under `src/yoke`. Do not add unit tests
for scripts, templates, or other files under `applications/`.

The `HarnessTrainer` core is already covered by CPU tests that monkeypatch DDP,
collectives, loaders, checkpoint writes, and submissions. If a migration exposes
a missing reusable capability or a regression in the generic trainer, study
renderer, CLI, checkpointing, or data-loading code, add the corresponding test
under `tests/` mirroring its `src/yoke` module. Otherwise, no unit-test change is
needed for a harness-only refactor.

Keep and run the existing `tests/harnesses/test_trainer.py`,
`tests/harnesses/test_base.py`, and `tests/cli/test_start_study.py` to prove the
shared lifecycle behavior remains unchanged.

## Verification

Ask for the path to the installed Yoke Python interpreter before executing
tests or the CLI. Then, from the repository root, run:

```bash
pytest -Werror tests/harnesses tests/cli
ruff check src/yoke applications/harnesses tests
ruff format --check src/yoke applications/harnesses tests
```

For each migrated harness, run `yoke-start-study --dryrun` from its harness
directory with a small CSV. Inspect a generated `runs/study_###/` directory to
confirm all of the following:

1. The copied script is present.
2. Start input/submission files contain the selected CSV substitutions.
3. Continuation templates contain `<CHECKPOINT>`, `<INPUTFILE>`, and
   `<epochIDX>` where they must be late-bound.
4. Calling `HarnessStudy.continuation_setup` creates the expected restart input
   and submission files.
5. The SLURM script has the correct DDP task layout and command.

Finally, use a short real allocation per migrated harness to validate one fresh
cycle and one continuation cycle. Confirm the continuation checkpoint restores
without an optimizer/model-class mismatch and that only rank 0 submits the next
job.

## Non-Goals

- Do not migrate the Lightning or MNIST demo harnesses.
- Do not change scientific model architecture, channel selection, data splits,
  batch limits, optimizer hyperparameters, or scheduler recipes as part of this
  structural migration.
- Do not fold post-training evaluation into these migrations except for the
  separately planned `ch_DDP_loderunner` reference implementation.
- Do not add a new harness subclass or registry; the generic `HarnessStudy` and
  `HarnessTrainer` remain the shared implementation.
