"""Generic orchestration layer for Yoke DDP training harnesses.

Historically every harness under ``applications/harnesses/`` shipped a
``train_*.py`` script that was, in the large, the same program copied and
lightly edited: parse args, set up DDP, build a model (fresh or from a
checkpoint), wrap it in :class:`~torch.nn.parallel.DistributedDataParallel`,
build a scheduler and dataloaders, run a timed epoch loop, checkpoint, and
resubmit. Only a handful of values genuinely varied between scripts.

:class:`HarnessTrainer` owns that fixed *orchestration* while delegating the
variable parts to injected callables:

- ``model_builder(args, device) -> (model, model_args, model_class, start_epoch)``
  constructs the model for **both** the fresh and continuation paths (it owns
  the continuation-reload branch), returning the model class so the checkpoint
  is saved with a matching ``model_class``.
- ``dataset_builder(args) -> (train_dataset, val_dataset)`` builds the datasets.
- ``optimizer_builder(model, args) -> Optimizer`` (defaults to
  :func:`yoke.utils.builders.build_adamw`).
- ``scheduler_builder(optimizer, args, last_epoch) -> scheduler | None``
  (optional).
- ``loss_builder() -> nn.Module`` (defaults to
  :func:`yoke.utils.builders.default_mse_loss`).
- ``epoch_fn`` is one of the existing per-epoch functions in
  ``yoke.utils.training.epoch``; per-study extra kwargs (e.g. ``channel_map``,
  ``dataset``) are supplied via ``epoch_kwargs``.

Dynamic, in-loop customization (EMA, gradient clipping, progressive unfreeze)
is expressed through a small :class:`TrainerHooks` object of optional,
composable callables.

Because HDF5 checkpointing and vanilla ``DataParallel`` are deprecated, the
trainer is **always** DDP and **always** writes ``.pth`` checkpoints.
"""

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import Dataset

from yoke.harnesses.base import HarnessStudy
from yoke.utils.builders import (
    build_adamw,
    checkpoint_name,
    compute_last_epoch,
    default_mse_loss,
)
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.utils.dataload import make_distributed_dataloader
from yoke.utils.parallel import cleanup_distributed, setup_distributed

# Type aliases for the injected builder callables.
#
# ``model_builder`` returns a 5-tuple:
#   (model, model_args, model_class, starting_epoch, optimizer)
# where ``optimizer`` is ``None`` on a fresh run (the trainer then builds one
# via ``optimizer_builder``) and a restored optimizer on continuation (so its
# state is preserved). The model is returned already moved to ``device`` and
# *before* DDP wrapping.
ModelBuilder = Callable[
    [argparse.Namespace, torch.device],
    tuple[nn.Module, dict, type, int, Optimizer | None],
]
DatasetBuilder = Callable[[argparse.Namespace], tuple[Dataset, Dataset]]
OptimizerBuilder = Callable[[nn.Module, argparse.Namespace], Optimizer]
SchedulerBuilder = Callable[[Optimizer, argparse.Namespace, int], object | None]
LossBuilder = Callable[[], nn.Module]


@dataclass
class TrainerHooks:
    """Composable, optional callables invoked at defined points in the loop.

    Every hook defaults to ``None`` (a no-op). Studies needing EMA, gradient
    clipping, and/or progressive unfreezing supply the relevant hooks; the
    trainer invokes whichever are present. Hooks receive the live
    :class:`HarnessTrainer` instance so they can read/modify trainer state
    (model, optimizer, args, ...).

    Attributes:
        on_after_ddp_wrap: Called once, immediately after the model is wrapped
            in DDP (``on_after_ddp_wrap(trainer)``). Useful for building an EMA
            shadow from the wrapped model.
        on_epoch_start: Called at the start of each epoch
            (``on_epoch_start(trainer, epochIDX)``). Useful for scheduling a
            progressive unfreeze.
        on_before_optimizer_step: Called before each optimizer step
            (``on_before_optimizer_step(trainer)``). Useful for gradient
            clipping. NOTE: the optimizer step itself lives inside the epoch
            function; this hook is provided for epoch functions/harnesses that
            opt into calling it. Kept for API completeness and forward
            compatibility.
        on_after_step: Called after each optimizer step
            (``on_after_step(trainer)``). Useful for EMA updates.
        on_before_save: Called just before the checkpoint is written
            (``on_before_save(trainer, epochIDX)``). Useful for saving an EMA
            companion checkpoint or contributing ``extra_state``. May return a
            ``dict`` that is merged into the checkpoint's ``extra_state``.
    """

    on_after_ddp_wrap: Callable[["HarnessTrainer"], None] | None = None
    on_epoch_start: Callable[["HarnessTrainer", int], None] | None = None
    on_before_optimizer_step: Callable[["HarnessTrainer"], None] | None = None
    on_after_step: Callable[["HarnessTrainer"], None] | None = None
    on_before_save: Callable[["HarnessTrainer", int], dict | None] | None = None


class HarnessTrainer:
    """Own the boilerplate DDP training orchestration for a harness.

    The trainer executes the canonical sequence shared by the non-Lightning
    DDP harnesses (DDP setup, model/optimizer/scheduler/dataloader construction,
    a timed epoch loop, ``.pth`` checkpointing, and job resubmission) while
    delegating the study-specific parts to injected builders and hooks.

    Args:
        args (argparse.Namespace): Parsed command-line arguments. Must provide
            the standard training attributes (``studyIDX``, ``batch_size``,
            ``total_epochs``, ``cycle_epochs``, ``train_batches``,
            ``val_batches``, ``TRAIN_PER_VAL``, ``num_workers``,
            ``trn_rcrd_filename``, ``val_rcrd_filename``) as declared by
            :func:`yoke.helpers.cli.add_training_args`.
        model_builder (ModelBuilder): Callable returning
            ``(model, model_args, model_class, starting_epoch, optimizer)``. It
            owns both the fresh-construction and continuation-reload paths and
            returns the model *before* DDP wrapping (already moved to
            ``device``), the ``model_args`` dict, the model *class* (for a
            matching checkpoint ``model_class``), the epoch to resume from
            (``0`` for fresh), and either a restored ``optimizer`` (continuation)
            or ``None`` (fresh, in which case the trainer builds one via
            ``optimizer_builder``).
        dataset_builder (DatasetBuilder): Callable returning
            ``(train_dataset, val_dataset)``.
        epoch_fn (Callable): One of the ``yoke.utils.training.epoch`` functions.
            Called once per epoch. Its return value, if an ``int``, updates the
            trainer's ``global_step`` (used by EMA-aware epoch functions).
        optimizer_builder (OptimizerBuilder): Callable returning the optimizer.
            Defaults to :func:`yoke.utils.builders.build_adamw`.
        scheduler_builder (SchedulerBuilder | None): Callable returning an LR
            scheduler given ``(optimizer, args, last_epoch)``, or ``None`` for
            no scheduler. Defaults to ``None``.
        loss_builder (LossBuilder): Callable returning the loss module. Defaults
            to :func:`yoke.utils.builders.default_mse_loss`.
        epoch_kwargs (dict | None): Extra keyword arguments forwarded verbatim to
            ``epoch_fn`` (e.g. ``{"channel_map": [...]}`` or
            ``{"dataset": "cylex"}``). Defaults to ``None`` (empty).
        hooks (TrainerHooks | None): Optional in-loop hooks. Defaults to a
            no-op :class:`TrainerHooks`.
        resubmit (bool): Whether to resubmit a continuation job when training is
            not finished. Defaults to ``True``. Set ``False`` for demos that
            never resubmit.
        steps_per_epoch (int | None): Number of scheduler steps per epoch used
            to compute ``last_epoch`` on continuation. Defaults to
            ``args.train_batches``.
        time_epochs (bool): Whether to time and print each epoch. Defaults to
            ``True``.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        model_builder: ModelBuilder,
        dataset_builder: DatasetBuilder,
        epoch_fn: Callable[..., Any],
        optimizer_builder: OptimizerBuilder = build_adamw,
        scheduler_builder: SchedulerBuilder | None = None,
        loss_builder: LossBuilder = default_mse_loss,
        epoch_kwargs: dict | None = None,
        hooks: TrainerHooks | None = None,
        resubmit: bool = True,
        steps_per_epoch: int | None = None,
        time_epochs: bool = True,
    ) -> None:
        """Initialize the trainer with parsed args and injected components."""
        self.args = args
        self.model_builder = model_builder
        self.dataset_builder = dataset_builder
        self.epoch_fn = epoch_fn
        self.optimizer_builder = optimizer_builder
        self.scheduler_builder = scheduler_builder
        self.loss_builder = loss_builder
        self.epoch_kwargs = dict(epoch_kwargs) if epoch_kwargs else {}
        self.hooks = hooks if hooks is not None else TrainerHooks()
        self.resubmit = resubmit
        self.steps_per_epoch = (
            steps_per_epoch if steps_per_epoch is not None else args.train_batches
        )
        self.time_epochs = time_epochs

        # Distributed context, populated by :meth:`setup_distributed`.
        self.rank: int = 0
        self.world_size: int = 1
        self.local_rank: int = 0
        self.device: torch.device | None = None

        # Training objects, populated by :meth:`setup`.
        self.model: nn.Module | None = None
        self.model_args: dict = {}
        self.model_class: type | None = None
        self.optimizer: Optimizer | None = None
        self.loss_fn: nn.Module | None = None
        self.scheduler: object | None = None
        self.train_dataloader: object | None = None
        self.val_dataloader: object | None = None

        self.starting_epoch: int = 0
        self.last_epoch: int = -1
        self.ending_epoch: int = 0
        self.epochIDX: int = 0
        self.global_step: int = 0
        self.new_chkpt_path: str | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_distributed(self) -> None:
        """Initialize the DDP process group and record the rank/device."""
        (
            self.rank,
            self.world_size,
            self.local_rank,
            self.device,
        ) = setup_distributed()

    def setup(self) -> None:
        """Build model, optimizer, loss, scheduler, and dataloaders.

        Delegates model construction (fresh or continuation) to
        ``model_builder``, wraps the model in DDP, fires the
        ``on_after_ddp_wrap`` hook, builds the loss and optimizer, computes the
        scheduler's ``last_epoch``, builds the scheduler, and constructs the
        distributed dataloaders.
        """
        if self.device is None:
            raise RuntimeError("setup_distributed() must be called before setup().")

        # --- Model (owns fresh vs. continuation) ---
        (
            model,
            self.model_args,
            self.model_class,
            self.starting_epoch,
            optimizer,
        ) = self.model_builder(self.args, self.device)

        # --- Optimizer ---
        # On continuation the builder returns a restored optimizer (with state);
        # on a fresh run it returns None and the trainer builds one.
        if optimizer is None:
            self.optimizer = self.optimizer_builder(model, self.args)
        else:
            self.optimizer = optimizer

        # --- Loss ---
        self.loss_fn = self.loss_builder()

        # --- DDP wrap ---
        self.model = DDP(
            model, device_ids=[self.local_rank], output_device=self.local_rank
        )
        if self.hooks.on_after_ddp_wrap is not None:
            self.hooks.on_after_ddp_wrap(self)

        # --- LR scheduler ---
        self.last_epoch = compute_last_epoch(self.starting_epoch, self.steps_per_epoch)
        if self.scheduler_builder is not None:
            self.scheduler = self.scheduler_builder(
                self.optimizer, self.args, self.last_epoch
            )

        # --- Dataloaders ---
        train_dataset, val_dataset = self.dataset_builder(self.args)
        # NOTE: For DDP the batch_size is the per-GPU batch_size.
        self.train_dataloader = make_distributed_dataloader(
            train_dataset,
            self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.val_dataloader = make_distributed_dataloader(
            val_dataset,
            self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            rank=self.rank,
            world_size=self.world_size,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _epoch_call_kwargs(self, epochIDX: int) -> dict:
        """Assemble the keyword arguments for a single ``epoch_fn`` call.

        Args:
            epochIDX (int): Current epoch index.

        Returns:
            dict: The keyword arguments to pass to ``epoch_fn``.
        """
        kwargs = {
            "training_data": self.train_dataloader,
            "validation_data": self.val_dataloader,
            "num_train_batches": self.args.train_batches,
            "num_val_batches": self.args.val_batches,
            "model": self.model,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "LRsched": self.scheduler,
            "epochIDX": epochIDX,
            "train_per_val": self.args.TRAIN_PER_VAL,
            "train_rcrd_filename": self.args.trn_rcrd_filename,
            "val_rcrd_filename": self.args.val_rcrd_filename,
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        # Study-specific extra kwargs (e.g. channel_map, dataset tag, EMA).
        kwargs.update(self.epoch_kwargs)
        return kwargs

    def train(self) -> None:
        """Run the timed epoch loop, calling ``epoch_fn`` once per epoch."""
        print("Training Model . . .")
        self.starting_epoch += 1
        self.ending_epoch = min(
            self.starting_epoch + self.args.cycle_epochs,
            self.args.total_epochs + 1,
        )

        for epochIDX in range(self.starting_epoch, self.ending_epoch):
            self.epochIDX = epochIDX

            # DistributedSampler needs the epoch set for correct shuffling.
            self.train_dataloader.sampler.set_epoch(epochIDX)

            if self.hooks.on_epoch_start is not None:
                self.hooks.on_epoch_start(self, epochIDX)

            if self.time_epochs:
                dist.barrier()
                torch.cuda.synchronize(self.device)
                start_time = time.time()

            result = self.epoch_fn(**self._epoch_call_kwargs(epochIDX))
            # EMA-aware epoch functions return the updated global-step counter.
            if isinstance(result, int):
                self.global_step = result

            if self.time_epochs:
                torch.cuda.synchronize(self.device)
                dist.barrier()
                end_time = time.time()
                epoch_time = (end_time - start_time) / 60
                if self.rank == 0:
                    print(f"Completed epoch {epochIDX}...", flush=True)
                    print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self) -> None:
        """Checkpoint the model and, if unfinished, resubmit a continuation job.

        Writes ``study{IDX:03d}_modelState_epoch{E:04d}.pth`` via
        :func:`yoke.utils.checkpointing.save_model_and_optimizer` using the
        model class returned by ``model_builder`` (so the saved ``model_class``
        matches the built model). On rank 0, if the total epoch budget is not
        yet exhausted and ``resubmit`` is set, prepares and submits the next
        continuation job.
        """
        chkpt_name = checkpoint_name(self.args.studyIDX, self.epochIDX)
        self.new_chkpt_path = os.path.join("./", chkpt_name)

        extra_state: dict | None = None
        if self.hooks.on_before_save is not None:
            extra_state = self.hooks.on_before_save(self, self.epochIDX)

        save_model_and_optimizer(
            self.model,
            self.optimizer,
            self.epochIDX,
            self.new_chkpt_path,
            model_class=self.model_class,
            model_args=self.model_args,
            extra_state=extra_state,
        )

        if self.rank == 0 and self.resubmit:
            finished = self.epochIDX + 1 > self.args.total_epochs
            if not finished:
                submission_type = getattr(self.args, "submissionType", "slurm")
                new_submit_file = HarnessStudy.continuation_setup(
                    self.new_chkpt_path,
                    self.args.studyIDX,
                    last_epoch=self.epochIDX,
                    submission_type=submission_type,
                )
                config = HarnessStudy.SUBMISSION_SYSTEMS[submission_type.lower()]
                os.system(f"{config['submit']} {new_submit_file}")

    # ------------------------------------------------------------------
    # Orchestration entry points
    # ------------------------------------------------------------------
    def teardown(self) -> None:
        """Tear down the DDP process group."""
        cleanup_distributed()

    def run(self) -> None:
        """Execute the full lifecycle: distributed setup, train, finalize.

        This is the single entry point a thin harness script calls. It sets up
        the process group, builds everything, runs the epoch loop, checkpoints
        (and possibly resubmits), and always tears down the process group.
        """
        self.setup_distributed()
        try:
            self.setup()
            self.train()
            self.finalize()
        finally:
            self.teardown()
