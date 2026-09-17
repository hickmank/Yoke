"""Small, reusable builder helpers shared by harness scripts and HarnessTrainer.

These helpers factor out the boilerplate that is copy-pasted across the
non-Lightning training scripts under ``applications/harnesses/`` (optimizer
construction, moving optimizer state onto a device, computing the scheduler's
``last_epoch`` on continuation, the default loss, and checkpoint filenames).
They are intentionally tiny so both the current scripts and the future
:class:`yoke.harnesses.trainer.HarnessTrainer` can use them without behavioral
change.
"""

import argparse
from collections.abc import Callable

import torch
import torch.nn as nn

from yoke.utils.checkpointing import load_model_and_optimizer

# A ``model_builder`` for HarnessTrainer returns
# (model, model_args, model_class, starting_epoch, optimizer|None).
ModelBuilderReturn = tuple[nn.Module, dict, type, int, "torch.optim.Optimizer | None"]


def build_adamw(
    model: nn.Module,
    args: argparse.Namespace,
    *,
    lr: float | None = None,
    weight_decay: float | None = None,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Construct the canonical ``AdamW`` optimizer used by Yoke harnesses.

    Every non-Lightning harness builds ``AdamW`` with the same ``betas`` and
    ``eps``; only the learning rate and weight decay vary. This helper mirrors
    that pattern and pulls ``lr``/``weight_decay`` from ``args`` when not passed
    explicitly.

    Args:
        model (nn.Module): Model whose parameters the optimizer will update.
        args (argparse.Namespace): Parsed arguments. Used as a fallback source
            for ``lr`` (``args.init_learnrate``) and ``weight_decay``
            (``args.weight_decay``) when the corresponding keyword arguments are
            ``None``.
        lr (float, optional): Explicit learning rate. When ``None``, falls back
            to ``getattr(args, "init_learnrate", 1e-4)``.
        weight_decay (float, optional): Explicit weight decay. When ``None``,
            falls back to ``getattr(args, "weight_decay", 0.0)``.
        betas (tuple[float, float]): Adam beta coefficients. Defaults to
            ``(0.9, 0.999)``.
        eps (float): Adam epsilon. Defaults to ``1e-8``.

    Returns:
        torch.optim.AdamW: The constructed optimizer.
    """
    if lr is None:
        lr = getattr(args, "init_learnrate", 1e-4)
    if weight_decay is None:
        weight_decay = getattr(args, "weight_decay", 0.0)

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device | str
) -> None:
    """Move all tensor entries of an optimizer's state onto ``device`` in place.

    Freshly constructed optimizers have empty state, but this is still called by
    scripts so that reloaded state (and any lazily created momentum buffers) live
    on the training device.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose state to move.
        device (torch.device | str): Target device.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def compute_last_epoch(starting_epoch: int, steps_per_epoch: int) -> int:
    """Compute the LR scheduler's ``last_epoch`` from the resumed epoch.

    Yoke schedulers are stepped once per training batch, so on continuation the
    scheduler must be advanced to ``steps_per_epoch * (starting_epoch - 1)``. A
    fresh run (``starting_epoch == 0``) uses ``-1`` (PyTorch's "no steps taken"
    sentinel).

    Args:
        starting_epoch (int): Epoch index the run is resuming from (``0`` for a
            fresh run, i.e. before the first epoch increment).
        steps_per_epoch (int): Number of scheduler steps taken per epoch
            (typically ``train_batches``).

    Returns:
        int: The ``last_epoch`` value to pass to the scheduler constructor.
    """
    if starting_epoch == 0:
        return -1
    return steps_per_epoch * (starting_epoch - 1)


def default_mse_loss() -> nn.Module:
    """Return the canonical per-sample MSE loss used by Yoke harnesses.

    ``reduction="none"`` is used so per-sample losses can be recorded by the
    epoch functions.

    Returns:
        nn.Module: An ``nn.MSELoss(reduction="none")`` instance.
    """
    return nn.MSELoss(reduction="none")


def checkpoint_name(studyIDX: int, epochIDX: int) -> str:
    """Return the canonical checkpoint filename for a study/epoch.

    Args:
        studyIDX (int): Study index; zero-padded to three digits.
        epochIDX (int): Epoch index; zero-padded to four digits.

    Returns:
        str: Filename of the form ``study{IDX:03d}_modelState_epoch{E:04d}.pth``.
    """
    return f"study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth"


def build_from_checkpoint(
    model_class: type,
    model_args_fn: Callable[[argparse.Namespace], dict],
    *,
    optimizer_class: type = torch.optim.AdamW,
    optimizer_kwargs: dict | None = None,
) -> Callable[[argparse.Namespace, torch.device], ModelBuilderReturn]:
    """Build a ``HarnessTrainer`` ``model_builder`` for the fresh-vs-continue pattern.

    This helper covers the common case where a study builds a model fresh from
    ``model_args`` on a first launch and reloads the *same* architecture from a
    ``.pth`` checkpoint on continuation. For studies that perform architectural
    surgery (strip/replace layers, load a partial backbone), write a bespoke
    ``model_builder`` instead.

    The returned callable matches the
    :data:`yoke.harnesses.trainer.ModelBuilder` contract, returning
    ``(model, model_args, model_class, starting_epoch, optimizer)`` where
    ``optimizer`` is ``None`` on a fresh run (so the trainer builds it) and the
    restored optimizer on continuation.

    Args:
        model_class (type): The model class to instantiate/save.
        model_args_fn (Callable[[argparse.Namespace], dict]): Callable mapping
            parsed args to the ``model_args`` dict passed to ``model_class``.
        optimizer_class (type): Optimizer class used when reloading on
            continuation. Defaults to :class:`torch.optim.AdamW`.
        optimizer_kwargs (dict | None): Keyword arguments for the reload-time
            optimizer. When ``None``, defaults to the canonical AdamW settings
            with ``lr`` taken from ``args.init_learnrate`` (fallback ``1e-4``).

    Returns:
        Callable: A ``model_builder`` suitable for :class:`HarnessTrainer`.
    """

    def _model_builder(
        args: argparse.Namespace, device: torch.device
    ) -> tuple[nn.Module, dict, type, int, torch.optim.Optimizer | None]:
        """Construct the model fresh or reload it from a checkpoint."""
        model_args = model_args_fn(args)
        available_models = {model_class.__name__: model_class}

        if getattr(args, "continuation", False):
            kwargs = optimizer_kwargs
            if kwargs is None:
                kwargs = {
                    "lr": getattr(args, "init_learnrate", 1e-4),
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": getattr(args, "weight_decay", 0.01),
                }
            model, optimizer, starting_epoch = load_model_and_optimizer(
                args.checkpoint,
                optimizer_class=optimizer_class,
                optimizer_kwargs=kwargs,
                available_models=available_models,
                device=device,
            )
            print("Model state loaded for continuation.")
            return model, model_args, model_class, starting_epoch, optimizer

        # Fresh construction.
        model = model_class(**model_args)
        model.to(device)
        return model, model_args, model_class, 0, None

    return _model_builder
