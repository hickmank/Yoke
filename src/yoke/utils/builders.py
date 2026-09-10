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

import torch
import torch.nn as nn


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
