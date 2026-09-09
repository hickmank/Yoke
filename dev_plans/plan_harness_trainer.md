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
        parallel: str = "ddp",          # "ddp" | "dp" | "single"
        checkpoint_io: CheckpointIO = PthCheckpointIO(),  # or Hdf5CheckpointIO
        hooks: TrainerHooks | None = None,   # EMA, freezing, grad_clip, etc.
    ) -> None: ...

    def setup(self) -> None:        # parallel init, model+opt (w/ continuation),
                                    # loss, DDP wrap, scheduler, dataloaders
    def train(self) -> None:        # timed epoch loop -> epoch_fn
    def finalize(self) -> None:     # save checkpoint, continuation_setup+submit
    def run(self) -> None:          # setup(); train(); finalize(); teardown()
```

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
- **`checkpoint_io`** abstracts `.pth` vs HDF5 (`lsc_action`) and the
  resubmission decision (some scripts, e.g. `moving_mnist`, never resubmit).
- **`parallel`** selects DDP / DataParallel / single. DDP setup uses the existing
  `yoke.utils.parallel.setup_distributed`/`cleanup_distributed` (already the
  intended home; `ch_DDP_diffLDR` already imports them).

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

**Phase 0 — Extract obvious helpers (low risk).**
- Replace every inlined `setup_distributed`/`cleanup_distributed` with the
  `yoke.utils.parallel` versions. (12 scripts; pure deletion + import.)
- Add small builder helpers to `yoke.utils`: `build_adamw(args)`,
  `move_optimizer_state_to_device`, `compute_last_epoch(starting_epoch, ...)`,
  `default_mse_loss()`, `checkpoint_name(studyIDX, epochIDX)`. These are used by
  both the current scripts and the future trainer.

**Phase 1 — Introduce `HarnessTrainer` for the canonical DDP path.**
- Implement the class covering the 7 near-identical scripts (LodeRunner /
  LodeRunnerViT / cylex / reward / diffusion) with `parallel="ddp"`, `.pth`
  checkpointing, and resubmission.
- Migrate `se_DDP_loderunner`, `vt_DDP_loderunner`, `se_ldrViT`,
  `vt_DDP_ldrViT`, `se_DDP_loderunner_cylex` first. Fix the `vt_DDP_ldrViT`
  `model_class` bug and the cylex kwarg-spelling inconsistency during migration.

**Phase 2 — Cover the deviating scripts via hooks.**
- EMA + grad clip: `ch_ldrViT` (both scripts).
- Block freezing: `ch_lsc_policy`.
- Backbone freeze / pretrained init: `se_DDP_loderunner_finetune_cylex`.

**Phase 3 — Non-DDP / special cases (evaluate case-by-case).**
- `lsc_action` (jit/compile, HDF5, DataParallel), `moving_mnist` (no resubmit,
  plotting), `mnist_surrogate` (own loop). These may adopt `parallel="single"`
  and `checkpoint_io=Hdf5CheckpointIO`, or be intentionally left as bespoke
  scripts if forcing them into the trainer reduces clarity. Decide during
  Phase 3.

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
  - `parallel="single"` path runs on CPU without a process group (CI-friendly).
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
  single-process/CPU path and fakes; do not require GPUs in CI.
- **Large diff.** Phasing keeps each PR reviewable; Phase 0 alone (parallel
  helper de-duplication) is independently valuable and low-risk.

## 8. Deliverables checklist

1. `src/yoke/harnesses/trainer.py` — `HarnessTrainer` (+ `TrainerHooks`,
   `CheckpointIO` interface with `.pth` and HDF5 implementations).
2. Builder helpers in `src/yoke/utils/` (optimizer/scheduler/checkpoint-name).
3. Phase 0 de-duplication of `setup_distributed`/`cleanup_distributed`.
4. Migrated canonical harness scripts (Phase 1), then hook-based ones (Phase 2).
5. Tests under `tests/harnesses/` (and per-harness smoke tests where feasible).
6. Docs: a new `docs/source/harness_trainer.rst` and a section in
   `AGENTS.md`/`harnesses.rst` on authoring a harness with `HarnessTrainer`.
7. Fix the two latent bugs surfaced by the survey (`vt_DDP_ldrViT` model_class;
   cylex dataset kwarg spelling).

## 9. Open questions

1. **Hooks object vs. subclassing.** Prefer composable `TrainerHooks`
   callables (recommended) or allow subclassing `HarnessTrainer` for the
   EMA/freezing/finetune variants?
2. **Optimizer/loss configurability.** They are effectively constant today.
   Expose them as builders now (future-proof) or hardcode the `AdamW` + masked
   MSE defaults and add configurability only when a study needs it?
3. **Phase 3 inclusion.** Should `lsc_action`/`moving_mnist`/`mnist_surrogate`
   be migrated, or intentionally kept bespoke as reference/demo scripts?
4. **Eventual `yoke-train` CLI + registry** (Section 4.3): pursue after the class
   lands, or not at all?
5. **Config surface.** Keep injecting Python callables/builders (maximally
   flexible), or move toward a declarative config (dataclass/dict) describing
   model/dataset/epoch/scheduler? The former fits the current `@`-file + CSV
   harness flow better; confirm.
