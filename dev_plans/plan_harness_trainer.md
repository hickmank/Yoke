# Dev Plan: Streamline Harnesses with a Unifying `HarnessTrainer`

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
  checkpoint format.
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
  *previously produced* artifacts. Handle via a deprecation window rather than
  immediate removal (see below), so old checkpoints remain loadable while new
  runs write `.pth`.
- **Deprecation mechanics:** keep the two functions for now but emit a
  `DeprecationWarning` (and a note in their docstrings pointing to the `.pth`
  functions). Move their tests under an explicit "deprecated" marker so
  `-Werror` still passes. Schedule hard removal for a later, separate PR once no
  live `.hdf5` checkpoints need reading. This keeps the deprecation independent
  of the trainer work.

### 1a.2 Deprecate non-DDP / vanilla DataParallel training

- **Deprecate** vanilla `nn.DataParallel` training and the `--multigpu` code
  path. DDP becomes the single parallelism model for all harnesses.
- **Only one training harness** uses `nn.DataParallel`: `lsc_action`
  (`train_lsc_action.py`, lines 187–201, gated by `--multigpu`). It will be
  migrated to the standard DDP flow (`setup_distributed` +
  `DistributedDataParallel`, `make_distributed_dataloader`).
- The `LodeRunner_DataParallel` class in `src/yoke/utils/parallel.py` is not used
  by any harness (only by its own test, `tests/test_parallel_utils.py`). Mark it
  deprecated alongside this change; remove in the same later cleanup PR as the
  HDF5 functions.
- The `--multigpu` argument in `yoke.helpers.cli.add_computing_args` becomes a
  no-op/deprecated flag (keep it parseable to avoid breaking existing
  `@`-input files, but warn and ignore).
- Net effect: `HarnessTrainer` needs **no** `parallel` switch — it is always DDP.
  The `mnist_surrogate` and `moving_mnist` demo scripts are single-process today
  but do not use `DataParallel`; they are handled separately in Phase 3 and are
  not affected by this deprecation.

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
        model_builder: Callable[..., tuple[nn.Module, dict]],
        dataset_builder: Callable[..., tuple[Dataset, Dataset]],
        epoch_fn: Callable[..., int | None],
        optimizer_builder: Callable[[nn.Module, argparse.Namespace], Optimizer]
            = build_adamw,
        scheduler_builder: Callable[...] | None = None,
        loss_builder: Callable[[], nn.Module] = default_mse_loss,
        hooks: TrainerHooks | None = None,   # EMA, freezing, grad_clip, etc.
    ) -> None: ...

    def setup(self) -> None:        # setup_distributed, model+opt (w/ continuation),
                                    # loss, DDP wrap, scheduler, dataloaders
    def train(self) -> None:        # timed epoch loop -> epoch_fn
    def finalize(self) -> None:     # save .pth checkpoint, continuation_setup+submit
    def run(self) -> None:          # setup(); train(); finalize(); teardown()
```

Because HDF5 checkpointing and non-DDP training are deprecated (Section 1a),
there is **no** `parallel` mode switch and **no** `checkpoint_io` abstraction:
the trainer is always DDP and always writes `.pth`.

- **`model_builder`** returns `(model, model_args)` so `HarnessTrainer` can drive
  the continuation-vs-fresh branch (calling `load_model_and_optimizer` itself)
  and later pass `model_args` to `save_model_and_optimizer`. This centralizes the
  `model_class`/`model_args` pairing and kills the `vt_DDP_ldrViT` mismatch bug.
- **`epoch_fn`** keeps the existing signatures; the trainer adapts by passing a
  well-defined kwargs bundle. Epoch functions already exist and are tested — the
  trainer only calls them.
- **Behavioral toggles** are handled by a small `TrainerHooks` object (or a set
  of optional callables) with no-op defaults: `on_after_ddp_wrap`,
  `on_before_optimizer_step` (grad clip), `on_after_step` (EMA),
  `on_epoch_start` (freeze scheduling), `on_before_save` (EMA companion save).
  This lets the EMA (`ch_ldrViT`), freezing (`ch_lsc_policy`), and fine-tuning
  (`se_DDP_loderunner_finetune_cylex`) scripts opt in without forking the loop.
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

### 4.3 Optional CLI convenience (later)

Once `HarnessTrainer` exists, a small registry could map a `--trainer` name to
prebuilt builders, enabling a `yoke-train` console script analogous to
`yoke-start-study`. **Out of scope for the first pass** — the class is the win;
a generic CLI is a follow-up.

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
- `moving_mnist`, `mnist_surrogate`: demo/tutorial scripts (single-process, own
  loops, no resubmission). Decide whether to adopt `HarnessTrainer` (with a
  "no resubmission" flag) or intentionally keep them bespoke as minimal
  references. These do not use DataParallel, so the deprecation does not force a
  change; decide during Phase 3.

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
  scripts that *read* old `.hdf5` files (Section 1a.1). Use a soft deprecation
  window (warn, don't remove) so existing artifacts stay loadable until a later
  removal PR. Do not conflate this with the separate `h5py` dataset-cache code,
  which is untouched.

## 8. Deliverables checklist

1. Phase 0 deprecations: `DeprecationWarning`s + docstring notes on the HDF5
   checkpoint functions, `LodeRunner_DataParallel`, and the `--multigpu` flag;
   tests updated to stay `-Werror` clean.
2. `src/yoke/harnesses/trainer.py` — `HarnessTrainer` (+ `TrainerHooks`).
   No `CheckpointIO` abstraction and no `parallel` switch (always `.pth`, always
   DDP).
3. Builder helpers in `src/yoke/utils/` (optimizer/scheduler/checkpoint-name).
4. Phase 0 de-duplication of `setup_distributed`/`cleanup_distributed`.
5. Migrated canonical harness scripts (Phase 1), then hook-based ones (Phase 2),
   then `lsc_action` converted to `.pth` + DDP (Phase 3).
6. Tests under `tests/harnesses/` (and per-harness smoke tests where feasible).
7. Docs: a new `docs/source/harness_trainer.rst` and a section in
   `AGENTS.md`/`harnesses.rst` on authoring a harness with `HarnessTrainer`.
8. Fix the two latent bugs surfaced by the survey (`vt_DDP_ldrViT` model_class;
   cylex dataset kwarg spelling).
9. (Later, separate PR) Hard-remove the deprecated HDF5 checkpoint functions,
   `LodeRunner_DataParallel`, and `--multigpu` once no live `.hdf5` checkpoints
   need reading.

## 9. Open questions

1. **Hooks object vs. subclassing.** Prefer composable `TrainerHooks`
   callables (recommended) or allow subclassing `HarnessTrainer` for the
   EMA/freezing/finetune variants?
2. **Optimizer/loss configurability.** They are effectively constant today.
   Expose them as builders now (future-proof) or hardcode the `AdamW` + masked
   MSE defaults and add configurability only when a study needs it?
3. **Demo scripts.** Should `moving_mnist`/`mnist_surrogate` adopt
   `HarnessTrainer` (with a no-resubmission flag) or stay bespoke as minimal
   references?
4. **Eventual `yoke-train` CLI + registry** (Section 4.3): pursue after the class
   lands, or not at all?
5. **Config surface.** Keep injecting Python callables/builders (maximally
   flexible), or move toward a declarative config (dataclass/dict) describing
   model/dataset/epoch/scheduler? The former fits the current `@`-file + CSV
   harness flow better; confirm.
6. **Deprecation timeline.** How long a window before hard-removing the HDF5
   checkpoint functions and DataParallel — next release, or gated on migrating
   known old checkpoints?
