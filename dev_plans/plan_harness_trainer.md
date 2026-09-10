# Dev Plan: Streamline Harnesses with a Unifying `HarnessTrainer`

---

## 0. Implementation status (living log)

**Completed: Phase 0 + Phase 1.** Remaining: Phase 2, Phase 3, and the later
Dec-2026 hard-removal PR.

### Phase 0 — DONE (deprecations + helpers + de-dup)

- **HDF5 checkpoint deprecation** (`src/yoke/utils/checkpointing.py`):
  `save_model_and_optimizer_hdf5` and `load_model_and_optimizer_hdf5` both emit
  `DeprecationWarning`s and carry docstring notes stating the **Dec-2026** hard
  removal. `load_*` is retained as a read-only shim; `save_*` is
  write-deprecated. The `h5py` dataset *cache* code was left untouched.
- **DataParallel / `--multigpu` deprecation**: `LodeRunner_DataParallel`
  (`src/yoke/utils/parallel.py`) warns on construction (Dec-2026). `--multigpu`
  is now a warn-and-ignore no-op implemented as a custom argparse action
  `_DeprecatedMultiGPUAction` in `src/yoke/helpers/cli.py` (kept parseable so
  existing `@`-input files don't break; always resolves to `False`).
- **Builder helpers** added in **`src/yoke/utils/builders.py`** (NEW file, not
  the pre-existing `parameters.py`): `build_adamw`, `move_optimizer_state_to_device`,
  `compute_last_epoch`, `default_mse_loss`, `checkpoint_name`, and the
  `build_from_checkpoint` factory (added during Phase 1). 100% test coverage in
  `tests/utils/test_builders.py`.
- **`setup_distributed`/`cleanup_distributed` de-duplication**: replaced the
  inlined copies with imports from `yoke.utils.parallel` in the **7 scripts not
  migrated in Phase 1** (`ch_lsc_policy`, `ch_lsc_inverse`, `ch_lsc_reward`,
  `ch_DDP_loderunner`, `ch_ldrViT/train_ldrViT_2frame.py`,
  `ch_ldrViT/train_ldrViT_ddp.py`, `se_DDP_loderunner_finetune_cylex`). The 5
  Phase-1 scripts dropped their inlined copies as part of the full rewrite.
- **Tests** updated to wrap deprecated calls in `pytest.warns` so `-Werror`
  stays clean (`tests/test_checkpointing.py`, `tests/test_training_utils.py`,
  `tests/test_parallel_utils.py`, `tests/helpers/test_cli.py`).

### Phase 1 — DONE (`HarnessTrainer` + canonical migrations)

- **`src/yoke/harnesses/trainer.py`** (NEW): `HarnessTrainer` + `TrainerHooks`.
  Always DDP, always `.pth`; no `parallel` switch and no `CheckpointIO`
  abstraction (per §1a). 100% line coverage via `tests/harnesses/test_trainer.py`
  (DDP orchestration tested on CPU by monkeypatching
  `setup_distributed`/`cleanup_distributed`, `DDP`, `make_distributed_dataloader`,
  `dist.barrier`, `torch.cuda.synchronize`, `save_model_and_optimizer`,
  `HarnessStudy.continuation_setup`, and `os.system`).
- **`build_from_checkpoint`** factory covers the common fresh-vs-continue
  `model_builder` pattern.
- **5 canonical scripts migrated** to thin wrappers:
  `se_DDP_loderunner`, `vt_DDP_loderunner`, `se_ldrViT`, `vt_DDP_ldrViT`,
  `se_DDP_loderunner_cylex`.
- **Latent bugs fixed**: `vt_DDP_ldrViT` now saves `model_class=LodeRunnerViT`
  (previously referenced the undefined `LodeRunner`); the cylex kwarg-spelling
  bug (`max_time_idx_offset` → `max_timeIDX_offset`) was fixed in
  `se_DDP_loderunner_finetune_cylex` (a Phase-2 script — fixed opportunistically
  since it is a clear latent bug; deliverable #8).
- **Docs**: `docs/source/harness_trainer.rst` (added to the index toctree) and
  an `AGENTS.md` §5a authoring section.

### Verification (Phase 0 + 1)

- `pytest -Werror`: all tests pass (593). Coverage 94% overall;
  `trainer.py`/`builders.py` at 100%.
- `ruff check` on the CI-linted set (`src`, `tests`, `applications/{evaluation,
  filelists,normalization,viewers}`): the only remaining errors are **17
  pre-existing** ones in untouched files (`noise_schedulers.py`,
  `datastep/loderunner.py`, `epoch/loderunner.py`); **zero** introduced by this
  work. `applications/harnesses/` is intentionally NOT part of the CI lint set,
  but all migrated scripts nonetheless pass `ruff check` + `ruff format --check`.

### Decisions / revelations during implementation

1. **`model_builder` returns a 5-tuple, not the 3-tuple sketched in §4.1.** The
   sketch showed `(model, model_args, starting_epoch)`. The real contract is
   **`(model, model_args, model_class, starting_epoch, optimizer)`**:
   - `model_class` is returned explicitly so the checkpoint's saved
     `model_class` matches the built model (this is what kills the
     `vt_DDP_ldrViT` bug structurally rather than by hand).
   - `optimizer` is returned because on **continuation** the optimizer state
     must be restored *together with* the model (that's how
     `load_model_and_optimizer` works). The builder returns the restored
     optimizer on continuation and `None` on a fresh run (the trainer then
     builds one via `optimizer_builder`). This avoids double-constructing the
     optimizer and keeps the restored state intact.
2. **`build_from_checkpoint` takes `optimizer_kwargs` for the *reload* path
   only.** Fresh-run optimizer construction still goes through the harness's
   `optimizer_builder` (default `build_adamw`). Several scripts pass a bespoke
   `optimizer_builder` because their fresh LR differs from the reload LR (e.g.
   `se_DDP_loderunner` uses fresh `lr=1e-4` but the original reload used
   `lr=1e-6`; cylex/reward use a fixed `lr=1e-6` with the scheduler driving the
   effective LR). Behavior was preserved exactly.
3. **`setup_distributed` now passes `device_id=device` to
   `init_process_group`.** Two scripts (`ch_lsc_policy`, `ch_lsc_inverse`)
   already did this (modern eager device binding); the other 9 did not. To
   de-duplicate without changing behavior for those two, the canonical
   `yoke.utils.parallel.setup_distributed` was updated to include `device_id`.
   Harmless/beneficial for the rest.
4. **The cylex "two scripts disagree" bug was in the *finetune* script.**
   `se_DDP_loderunner_cylex` already used the correct `max_timeIDX_offset`; it
   was `se_DDP_loderunner_finetune_cylex` that used the wrong
   `max_time_idx_offset`. Fixed there.
5. **`--multigpu` deprecation via a custom argparse action** (rather than
   warning at consumption time) so the warning fires once at parse time and the
   value is forced to `False` regardless of the `@`-file contents.
6. **`epoch_kwargs` carries the per-epoch-fn variance** (`channel_map`,
   `dataset` tag). The trainer assembles the fixed kwargs bundle and
   `dict.update`s `epoch_kwargs` on top. EMA-aware epoch functions return an
   `int` global-step; the trainer captures an `int` return into
   `self.global_step` (unused by the Phase-1 scripts, but wired for Phase 2).
7. **`finalize()` resubmission is submission-type aware.** It reads
   `args.submissionType` (default `"slurm"`) and uses
   `HarnessStudy.SUBMISSION_SYSTEMS[...]["submit"]` for the submit command
   instead of hard-coding `sbatch`, and passes `submission_type` through to
   `continuation_setup`. A `resubmit=False` flag disables resubmission entirely
   (for future demo use).
8. **Trainer methods stay small/overridable** (`setup_distributed`, `setup`,
   `train`, `finalize`, `teardown`, `run`) as the §4.1 escape hatch.

### Where to start next session (Phase 2)

Phase 2 covers the deviating scripts via `TrainerHooks`:
- **`ch_ldrViT` (both `train_ldrViT_ddp.py` and `train_ldrViT_2frame.py`)** —
  EMA + gradient clipping. The `train_DDP_loderunner_epoch` signature already
  accepts `ema_model`, `global_step`, `ema_update_after_step`, and `grad_clip`
  kwargs, so these thread through `epoch_kwargs` plus `on_after_ddp_wrap`
  (build the EMA `AveragedModel`), `on_after_step`/`on_before_save` (EMA update
  + companion save), and the `global_step` return handling already wired in the
  trainer. Grad-clip is passed to the epoch fn via `grad_clip` in
  `epoch_kwargs` (the epoch fn does the clipping); confirm whether an
  `on_before_optimizer_step` hook is actually needed or if `grad_clip` kwarg
  suffices — likely the latter, so the hook may go unused for these two.
- **`ch_lsc_policy`** — block-progressive (un)freezing (`on_epoch_start`) and
  per-block param-group LRs (custom `optimizer_builder`). Uses
  `train_lsc_policy_epoch`. Inspect its inlined freeze helpers.
- **`se_DDP_loderunner_finetune_cylex`** — backbone freeze schedule +
  pretrained-weight init. This needs a **bespoke `model_builder`** (load
  pretrained, optionally strip/replace, freeze backbone for N epochs) plus an
  `on_epoch_start` unfreeze hook. Note the kwarg-spelling bug here is already
  fixed. Read its `_set_requires_grad`/`_freeze_*` helpers first.

Practical starting point: read
`applications/harnesses/ch_ldrViT/train_ldrViT_ddp.py` and
`src/yoke/utils/ema.py`, then migrate `ch_ldrViT` first (it exercises the EMA +
grad-clip hook surface end-to-end and validates the `global_step` plumbing).
Add hook-ordering tests (grad-clip before step, EMA after step) — the
`TrainerHooks` order test already exists as a template.

**Phase 3** (later): migrate `lsc_action` to `.pth` + DDP (drop HDF5,
`--multigpu`, and `torch.jit.script`/`torch.compile`); keep `moving_mnist` and
`mnist_surrogate` bespoke. **Dec-2026 PR** (later): hard-remove the HDF5
save+load functions, `LodeRunner_DataParallel`, and `--multigpu`.

---

## 1. Motivation

Every harness under `applications/harnesses/` ships a `train_*.py` script. Across
the 13 non-Lightning scripts these are, in the large, **the same program copied
and lightly edited**. A survey of the scripts (line counts 184–676) shows that
~7 of them are near-identical modulo four or five injected values, and the rest
are the same skeleton plus one localized feature (EMA, freezing, RL specifics).

This creates real problems:
- **Copy drift.** Bug fixes and improvements do not propagate. Two concrete
  latent bugs already exist from copying: `vt_DDP_ldrViT` saves a
  `LodeRunnerViT` checkpoint with `model_class=LodeRunner`, and the two cylex
  scripts disagree on the dataset kwarg spelling (`max_timeIDX_offset` vs
  `max_time_idx_offset`).
- **Inconsistent behavior.** `save_model_and_optimizer` is rank-guarded in some
  scripts and unconditional in others; `setup_distributed` is inlined in most
  scripts but already lives in `yoke.utils.parallel` and is imported by exactly
  one (`ch_DDP_diffLDR`).
- **High authoring cost.** Creating a new study means copying ~450 lines and
  finding the ~10 lines that matter.

Note: reusable per-epoch/per-batch logic **already** lives in
`yoke.utils.training.epoch/` and `.../datastep/`. The boilerplate that remains
un-factored is the **orchestration layer** — the code *between* argument parsing
and the epoch call. That is what this plan targets.

## 1a. Scope-narrowing decisions (deprecations)

Two axes of variation exist only to support a single outlier harness
(`lsc_action`). Removing them collapses that outlier into the common path and
makes `HarnessTrainer` markedly simpler (no `checkpoint_io` abstraction, no
`parallel` mode switch). Both are approved simplifications:

### 1a.1 Deprecate HDF5 checkpoint save/load

- **Deprecate** `save_model_and_optimizer_hdf5` / `load_model_and_optimizer_hdf5`
  in `src/yoke/utils/checkpointing.py`. The `.pth` path
  (`save_model_and_optimizer` / `load_model_and_optimizer`) becomes the single
  checkpoint format for all new runs.
- **Only one training harness** uses HDF5 checkpoints: `lsc_action`
  (`train_lsc_action.py`, lines 26–27, 179, 291–294). It will be migrated to
  `.pth`.
- **Important — do NOT touch dataset caching.** `h5py` is also used for dataset
  *caching* in `src/yoke/datasets/lsc_dataset.py` and
  `src/yoke/datasets/load_npz_dataset.py`. That is unrelated to checkpointing and
  **stays**. This deprecation is strictly about the two checkpoint functions.
- **Downstream readers of existing `.hdf5` checkpoints** exist in evaluation
  scripts: `applications/evaluation/parameters2image.py`,
  `tk_parameters2image_slider.py`, `image_prediction_comparison.py`,
  `lsc_loderunner_anime.py`, `lsc_loderunner_create_gif.py`. These load
  *previously produced* artifacts (and several are largely legacy / non-functional
  code anyway).
- **Deprecation mechanics (resolved, Q6): remove the write path now, keep the
  reader until a dated removal.**
  - Deprecate `save_model_and_optimizer_hdf5` immediately with a
    `DeprecationWarning` and a docstring note pointing to `save_model_and_optimizer`;
    it is targeted for **hard removal in December 2026**.
  - **Retain `load_model_and_optimizer_hdf5` as a read-only compatibility shim**
    (also warning, also dated Dec 2026) so existing `.hdf5` checkpoints and the
    evaluation scripts that read them keep working in the meantime.
  - All warnings and docstrings must state the **Dec-2026** hard-removal date
    explicitly.
  - Move/adjust the affected tests under an explicit "deprecated" marker (or wrap
    in `pytest.warns`) so `-Werror` still passes. The small Yoke user base means
    little risk of orphaned artifacts. Hard removal (both functions) is a later,
    separate PR, keeping this deprecation independent of the trainer work.

### 1a.2 Deprecate non-DDP / vanilla DataParallel training

- **Deprecate** vanilla `nn.DataParallel` training and the `--multigpu` code
  path. DDP becomes the single parallelism model for all harnesses.
- **Only one training harness** uses `nn.DataParallel`: `lsc_action`
  (`train_lsc_action.py`, lines 187–201, gated by `--multigpu`). It will be
  migrated to the standard DDP flow (`setup_distributed` +
  `DistributedDataParallel`, `make_distributed_dataloader`).
- The `LodeRunner_DataParallel` class in `src/yoke/utils/parallel.py` is not used
  by any harness (only by its own test, `tests/test_parallel_utils.py`). Deprecate
  it now with a `DeprecationWarning` and a **Dec-2026** hard-removal note.
- The `--multigpu` argument in `yoke.helpers.cli.add_computing_args` becomes a
  no-op/deprecated flag (keep it parseable to avoid breaking existing
  `@`-input files, but warn and ignore); also dated for **Dec-2026** removal.
- Hard removal of `LodeRunner_DataParallel` + `--multigpu` happens in the same
  later cleanup PR as the HDF5 functions (Dec-2026 target).
- Net effect: `HarnessTrainer` needs **no** `parallel` switch — it is always DDP.
  The `mnist_surrogate` and `moving_mnist` demo scripts are single-process today
  but do not use `DataParallel`; they are kept bespoke (Q3) and are not affected
  by this deprecation.

## 2. The canonical training script (what is actually duplicated)

The DDP scripts follow this exact sequence (see `se_DDP_loderunner`,
`ch_lsc_inverse`, `ch_lsc_reward`, `se_ldrViT`, `vt_*`, `se_DDP_loderunner_cylex`):

1. Build argparse from `cli.add_*_args`, `parse_args`.
2. `setup_distributed()` → `(rank, world_size, local_rank, device)`.
3. Unpack args into locals.
4. Build `model_args` dict; register `available_models`.
5. **Continuation branch:** `if CONTINUATION: load_model_and_optimizer(...)`
   `else:` construct model, `.to(device)`, build `AdamW`, move optimizer state
   to device.
6. `loss_fn = nn.MSELoss(reduction="none")`.
7. Wrap in `DDP(model, device_ids=[local_rank], ...)`.
8. Compute `last_epoch` and build the LR scheduler.
9. Build train/val datasets and `make_distributed_dataloader(...)`.
10. `starting_epoch += 1; ending_epoch = min(...)`.
11. **Timed epoch loop:** `set_epoch`, `dist.barrier()` +
    `torch.cuda.synchronize()`, call the epoch function, sync again, print time.
12. `save_model_and_optimizer(...)` with `study{IDX:03d}_modelState_epoch{E:04d}.pth`.
13. rank-0: if not finished, `HarnessStudy.continuation_setup(...)` + `sbatch`.
14. `cleanup_distributed()`.

Steps 2, 7, 8, 10, 11, 12, 13, 14 are **pure boilerplate** — byte-for-byte
identical except formatting. Steps 4, 5, 6, 9 are boilerplate *shape* with a few
injected values. The genuinely study-specific content is small.

## 3. What actually varies (the trainer's parameters)

From the script survey, the real "knobs" are:

**Injected components (per study):**
- **Model**: class + `model_args` dict (or, for `lsc_action`, kwargs).
- **Dataset**: class(es) + kwargs; whether a test set is built; optional wrapper
  (e.g. `ButterfliedDataset`).
- **Epoch function**: one of `train_DDP_loderunner_epoch`,
  `train_DDP_diffusion_loderunner_epoch`, `train_lsc_policy_epoch`,
  `train_lsc_reward_epoch`, `train_DDP_array_epoch`, `train_array_csv_epoch`,
  `train_simple_loderunner_epoch`, plus epoch-fn-specific kwargs (`dataset=`
  tag, `channel_map`, `in_vars`/`out_vars`, `blocks`, `ema_model`/`global_step`).
- **LR scheduler**: `CosineWithWarmupScheduler` / `ConstantWithWarmupScheduler` /
  none, plus its args.
- **Optimizer**: effectively always `AdamW(betas=(0.9,0.999), eps=1e-8)`; only
  `lr` and `weight_decay` vary. (Keep configurable, default `AdamW`.)
- **Loss**: effectively always `nn.MSELoss(reduction="none")`. (Configurable,
  same default.)

**Behavioral toggles:**
- Parallelism mode: DDP / single-process (optional `DataParallel`) / single-GPU.
- Continuation format: `.pth` (`load/save_model_and_optimizer`) vs HDF5
  (`*_hdf5`); whether resubmission happens at all.
- EMA (on/off + warmup/decay params + companion checkpoint saving).
- Gradient clipping (`grad_clip`).
- Parameter freezing / fine-tuning (block-progressive unfreeze; backbone freeze
  for N epochs; pretrained-weights init).
- LR scaling by global batch size; `warmup_lr` override.
- `torch.compile` / `torch.jit.script` (only `lsc_action`).

## 4. Proposed design

### 4.1 A `HarnessTrainer` in `src/yoke/harnesses/`

Add `src/yoke/harnesses/trainer.py` containing a `HarnessTrainer` class that owns
the orchestration (steps 2, 7–14 above) and delegates the variable parts to
injected callables/objects. Sketch:

```python
class HarnessTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        model_builder: Callable[[argparse.Namespace, torch.device],
                                tuple[nn.Module, dict, int]],
        dataset_builder: Callable[..., tuple[Dataset, Dataset]],
        epoch_fn: Callable[..., int | None],
        optimizer_builder: Callable[[nn.Module, argparse.Namespace], Optimizer]
            = build_adamw,
        scheduler_builder: Callable[...] | None = None,
        loss_builder: Callable[[], nn.Module] = default_mse_loss,
        hooks: TrainerHooks | None = None,   # EMA, grad_clip, dynamic (un)freeze
    ) -> None: ...

    def setup(self) -> None:        # setup_distributed, model_builder(args, device),
                                    # loss, DDP wrap, scheduler, dataloaders
    def train(self) -> None:        # timed epoch loop -> epoch_fn
    def finalize(self) -> None:     # save .pth checkpoint, continuation_setup+submit
    def run(self) -> None:          # setup(); train(); finalize(); teardown()
```

Because HDF5 checkpointing and non-DDP training are deprecated (Section 1a),
there is **no** `parallel` mode switch and **no** `checkpoint_io` abstraction:
the trainer is always DDP and always writes `.pth`.

**Extensibility model (resolves Q1): hybrid + builder-owned surgery.** There are
two distinct kinds of customization and they are handled in two different places:

1. **Static model construction & surgery — the `model_builder`.** This is the
   author's own callable and may do arbitrary work: instantiate the model, load
   a pretrained checkpoint, strip encoder/decoder layers and splice in
   replacements, and freeze/unfreeze subnetworks. Crucially, **the builder also
   owns the continuation-reload branch**: it receives `args` (including
   `continuation`/`checkpoint`) and `device`, and returns a ready-to-train
   `(model, model_args, starting_epoch)`. This keeps *all* architecture knowledge
   in one place, so a surgically-modified model reloads its own modified
   architecture correctly on continuation — the trainer never assumes a standard
   architecture. (Helper builders like `build_from_checkpoint(...)` will be
   provided to cover the common fresh-vs-continue pattern so simple harnesses
   stay short.) Returning `model_args` alongside the model also lets the trainer
   save with a matching `model_class`, killing the `vt_DDP_ldrViT` mismatch bug.

2. **Dynamic, in-loop behavior — `TrainerHooks`.** A small object of optional
   callables with no-op defaults, invoked at defined points:
   `on_after_ddp_wrap`, `on_epoch_start` (e.g. progressive unfreeze scheduling),
   `on_before_optimizer_step` (grad clip), `on_after_step` (EMA update),
   `on_before_save` (EMA companion save). Hooks **compose**, so a study needing
   EMA *and* grad-clip *and* scheduled unfreezing combines them without the
   multiple-inheritance mess that subclassing would create. This covers
   `ch_ldrViT` (EMA + grad-clip), `ch_lsc_policy` (block (un)freezing), and
   `se_DDP_loderunner_finetune_cylex` (backbone freeze schedule).

3. **Escape hatch.** `HarnessTrainer` methods (`setup`/`train`/`finalize`) stay
   small and overridable for the rare case a study needs something no predefined
   hook anticipates — preferable to reverting to a copy-pasted script.

This directly supports the "load pretrained → strip encoder/decoder → replace →
freeze/unfreeze → fine-tune" workflow: the surgery lives in `model_builder`; any
epoch-scheduled unfreezing lives in an `on_epoch_start` hook.

- **`epoch_fn`** keeps the existing signatures; the trainer adapts by passing a
  well-defined kwargs bundle. Epoch functions already exist and are tested — the
  trainer only calls them.
- **Checkpointing** is always `.pth` via `save_model_and_optimizer` /
  `load_model_and_optimizer` (HDF5 deprecated, Section 1a.1). The trainer still
  owns the resubmission decision (some scripts, e.g. `moving_mnist`, never
  resubmit) via a simple flag rather than an IO abstraction.
- **Parallelism** is always DDP (vanilla DataParallel deprecated, Section 1a.2).
  DDP setup uses the existing
  `yoke.utils.parallel.setup_distributed`/`cleanup_distributed` (already the
  intended home; `ch_DDP_diffLDR` already imports them). No `parallel` switch.


### 4.2 Thin harness scripts

After the refactor, a canonical harness `train_*.py` becomes ~40–80 lines:
build the parser, define `model_builder`/`dataset_builder`, pick `epoch_fn` and
`scheduler_builder`, then:

```python
if __name__ == "__main__":
    args = parser.parse_args()
    HarnessTrainer(
        args,
        model_builder=build_loderunner,
        dataset_builder=build_lsc_temporal,
        epoch_fn=train_DDP_loderunner_epoch,
        scheduler_builder=cosine_from_args,
    ).run()
```

This thin-script pattern **is** the harness authoring interface. A central
trainer registry + generic `yoke-train` CLI was considered and **dropped**
(Q4): a fixed registry fights the arbitrary-`model_builder` flexibility from Q1
(bespoke pretrained-surgery builders don't fit a catalog) and adds a coupling
point. Revisit only if fully code-free harnesses become a concrete need.

## 5. Scope and phasing

This is a large surface. Propose an incremental, test-guarded rollout so nothing
breaks:

**Phase 0 — Deprecations + extract obvious helpers (low risk).**
- Deprecate HDF5 checkpoint functions and vanilla DataParallel / `--multigpu`
  (Section 1a): add `DeprecationWarning`s, docstring notes, and mark their tests
  deprecated so `-Werror` stays clean. (Hard removal is a later, separate PR.)
- Replace every inlined `setup_distributed`/`cleanup_distributed` with the
  `yoke.utils.parallel` versions. (12 scripts; pure deletion + import.)
- Add small builder helpers to `yoke.utils`: `build_adamw(args)`,
  `move_optimizer_state_to_device`, `compute_last_epoch(starting_epoch, ...)`,
  `default_mse_loss()`, `checkpoint_name(studyIDX, epochIDX)`. These are used by
  both the current scripts and the future trainer.

**Phase 1 — Introduce `HarnessTrainer` for the canonical DDP path.**
- Implement the class covering the 7 near-identical scripts (LodeRunner /
  LodeRunnerViT / cylex / reward / diffusion) — always DDP, always `.pth`
  checkpointing, with resubmission.
- Migrate `se_DDP_loderunner`, `vt_DDP_loderunner`, `se_ldrViT`,
  `vt_DDP_ldrViT`, `se_DDP_loderunner_cylex` first. Fix the `vt_DDP_ldrViT`
  `model_class` bug and the cylex kwarg-spelling inconsistency during migration.

**Phase 2 — Cover the deviating scripts via hooks.**
- EMA + grad clip: `ch_ldrViT` (both scripts).
- Block freezing: `ch_lsc_policy`.
- Backbone freeze / pretrained init: `se_DDP_loderunner_finetune_cylex`.

**Phase 3 — Fold in the former outlier + demo scripts.**
- `lsc_action`: **migrate to the common path** — convert HDF5 → `.pth`
  checkpointing and vanilla DataParallel → DDP (per Section 1a), drop
  `--multigpu`, and **drop the `torch.jit.script`/`torch.compile` usage**
  (removed outright — it never offered meaningful benefit). With the
  deprecations done and compilation removed, this harness fits `HarnessTrainer`
  directly.
- `moving_mnist`, `mnist_surrogate`: **kept bespoke** (resolved, Q3). These are
  single-process demo/tutorial scripts with their own loops, no resubmission, and
  (for `moving_mnist`) inline plotting. They do not use DataParallel, so no
  deprecation forces a change. Migrating them would reintroduce a single-process
  path contradicting the "always DDP" simplification, so they stay as minimal
  reference examples. They should be labeled as such in the harness docs.

**Lightning harness (`ch_lightning_loderunner`) is explicitly out of scope** —
Lightning already owns the loop; wrapping it in `HarnessTrainer` adds nothing.


## 6. Testing strategy

- **New unit tests** under `tests/harnesses/test_trainer.py`:
  - `HarnessTrainer.setup` builds model/optimizer for both fresh and continuation
    paths (monkeypatch `load_model_and_optimizer`).
  - The timed epoch loop calls `epoch_fn` the expected number of times with the
    expected kwargs (inject a fake `epoch_fn` recording calls).
  - `finalize` calls `save_model_and_optimizer` with matching
    `model_class`/`model_args`, and only resubmits when not finished (monkeypatch
    `HarnessStudy.continuation_setup` and the submit call).
  - Hooks fire in the correct order (grad-clip before step, EMA after step).
  - The DDP orchestration is exercised on CPU via the `gloo` backend with a
    single-process group (or by monkeypatching `setup_distributed`), so the
    timed epoch loop, save, and resubmission logic are testable without GPUs.
- **Migration guard:** for each migrated harness, keep behavior identical.
  Where feasible, add a tiny CPU/single-process smoke test that runs one or two
  fake batches end-to-end through the trainer.
- Must remain `pytest -Werror` clean; `ruff check`/`format` clean; google
  docstrings + full type annotations (per `AGENTS.md`).

## 7. Risks and mitigations

- **Behavioral drift during migration.** Mitigate by migrating one harness at a
  time, diffing rendered record/checkpoint filenames, and keeping the epoch
  functions untouched.
- **Over-abstraction.** The hook surface must stay minimal; if a script needs
  something the hooks can't express cleanly, leave it bespoke (Phase 3 escape
  hatch). The goal is removing *copy-paste*, not forcing every script into one
  mold.
- **DDP code is hard to unit test.** Cover the orchestration logic with a
  CPU `gloo` single-process group (or monkeypatched `setup_distributed`) and
  fakes; do not require GPUs in CI.
- **Large diff.** Phasing keeps each PR reviewable; the Phase 0 deprecations and
  parallel-helper de-duplication are independently valuable and low-risk.
- **Deprecation blast radius.** Deprecating HDF5 checkpoints affects evaluation
  scripts that *read* old `.hdf5` files (Section 1a.1). Mitigated by removing only
  the *write* path now and retaining the HDF5 *reader* as a read-only shim until
  the Dec-2026 hard-removal, so existing artifacts stay loadable. Do not conflate
  this with the separate `h5py` dataset-cache code, which is untouched.

## 8. Deliverables checklist

1. **[DONE]** Phase 0 deprecations: `DeprecationWarning`s + docstring notes (all
   stating the **Dec-2026** hard-removal date) on `save_model_and_optimizer_hdf5`,
   `load_model_and_optimizer_hdf5` (retained as a read-only shim),
   `LodeRunner_DataParallel`, and the `--multigpu` flag; tests updated to stay
   `-Werror` clean.
2. **[DONE]** `src/yoke/harnesses/trainer.py` — `HarnessTrainer` (+ `TrainerHooks`).
   No `CheckpointIO` abstraction and no `parallel` switch (always `.pth`, always
   DDP).
3. **[DONE]** Builder helpers in `src/yoke/utils/builders.py` (`build_adamw`,
   `move_optimizer_state_to_device`, `compute_last_epoch`, `default_mse_loss`,
   `checkpoint_name`, `build_from_checkpoint`).
4. **[DONE]** Phase 0 de-duplication of `setup_distributed`/`cleanup_distributed`.
5. Migrated harness scripts: **[DONE] Phase 1** (`se_DDP_loderunner`,
   `vt_DDP_loderunner`, `se_ldrViT`, `vt_DDP_ldrViT`, `se_DDP_loderunner_cylex`);
   **[TODO] Phase 2** hook-based ones; **[TODO] Phase 3** `lsc_action`.
6. **[DONE for Phase 1]** Tests under `tests/harnesses/` (`test_trainer.py`) and
   `tests/utils/test_builders.py`. Per-harness smoke tests still optional/TODO.
7. **[DONE]** Docs: `docs/source/harness_trainer.rst` (in the index toctree) and
   an `AGENTS.md` §5a authoring section.
8. **[DONE]** Fix the two latent bugs surfaced by the survey (`vt_DDP_ldrViT`
   model_class — fixed structurally via the trainer's model_class handling;
   cylex dataset kwarg spelling — fixed in `se_DDP_loderunner_finetune_cylex`).
9. **[TODO — later, separate PR, Dec-2026 target]** Hard-remove the HDF5
   checkpoint save **and** load functions, `LodeRunner_DataParallel`, and
   `--multigpu`.

## 9. Open questions

1. **[RESOLVED] Hooks vs. subclassing → hybrid + builder-owned surgery.**
   Static model construction/surgery (load pretrained, strip/replace layers,
   freeze) lives in `model_builder`, which also owns the continuation-reload
   branch. Dynamic in-loop behavior (EMA, grad-clip, scheduled unfreezing) uses
   composable `TrainerHooks`. Trainer methods stay overridable as a rare escape
   hatch. See Section 4.1.
2. **[RESOLVED] Optimizer/loss configurability → expose both builders now.**
   `HarnessTrainer` takes `optimizer_builder=build_adamw` and
   `loss_builder=default_mse_loss` with sensible defaults. Most harnesses use the
   defaults and pass nothing; `ch_lsc_policy`'s per-block param-group LRs slot in
   as a custom `optimizer_builder` with no trainer change. This keeps the builder
   injection symmetric with `model_builder`/`dataset_builder`.
3. **[RESOLVED] Demo scripts → keep bespoke as references.**
   `moving_mnist` and `mnist_surrogate` stay small standalone scripts serving as
   minimal onboarding examples. They are single-process and do not use
   DataParallel, so no deprecation forces a change. Migrating them would
   reintroduce the single-process branch removed in Q1a.2 (and require a plotting
   hook for `moving_mnist`), so they remain intentionally-bespoke demos.
4. **[RESOLVED] `yoke-train` CLI + registry → dropped.** The thin harness
   script calling `HarnessTrainer(...).run()` is the authoring interface. A
   central registry fights the arbitrary-builder flexibility from Q1 and adds a
   coupling/maintenance point; revisit only if code-free harnesses become a
   concrete ask. (Section 4.2 updated; former Section 4.3 removed.)
5. **[RESOLVED] Config surface → Python builders + CSV for scalars.** Harnesses
   inject Python callables (`model_builder`/`dataset_builder`/`epoch_fn`/
   `optimizer_builder`/`scheduler_builder`/`loss_builder`); scalar
   hyperparameters continue to flow through the hyperparameter CSV and `@`-input
   file. A declarative config would conflict with the Q1 arbitrary-surgery
   requirement, so builders remain the interface.
6. **[RESOLVED] Deprecation timeline → remove write path now, keep HDF5 reader
   until a dated hard-removal.** New runs always write `.pth`. The HDF5 *save*
   function, `LodeRunner_DataParallel`, and `--multigpu` are deprecated now and
   hard-removed by a target of **December 2026**. The HDF5 *load* function is
   retained as a small read-only compatibility shim so existing `.hdf5`
   checkpoints (and the — largely legacy — evaluation scripts that read them)
   keep working until the same Dec-2026 removal. Deprecation warnings and
   docstrings must state the Dec-2026 hard-removal date explicitly. The Yoke
   user base is small, so little is at risk of orphaning.
