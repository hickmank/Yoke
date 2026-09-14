"""DDP training harness for the Gaussian policy CNN on lsc240420.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer`. This study
trains :class:`~yoke.models.policyCNNmodules.gaussian_policyCNN` on the layered
shaped-charge design problem, with two study-specific behaviors expressed
through the trainer's injection points:

- A **bespoke ``model_builder``** owns the freeze schedule: a fresh run freezes
  every parameter and then unfreezes the eight named sub-blocks (everything
  except the covariance head ``cov_mlp``); a continuation reload freezes
  ``cov_mlp`` only. It also returns the restored optimizer on continuation.
- A **custom ``optimizer_builder``** builds AdamW with per-block parameter groups
  whose base learning rates are scaled per sub-block (the cosine scheduler then
  scales all groups together).

The ``blocks`` list is forwarded to ``train_lsc_policy_epoch`` via
``epoch_kwargs`` for per-block gradient-norm monitoring.
"""

import argparse
import os

import torch

from yoke.datasets.lsc_dataset import LSC_hfield_policy_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.policyCNNmodules import gaussian_policyCNN
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.training.epoch.lsc_policy import train_lsc_policy_epoch

#############################################
# Inputs
#############################################
descr_str = "Uses DDP to train Gaussian policy architecture."
parser = argparse.ArgumentParser(
    prog="Gaussian Policy Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

#############################################
# Study-specific constants
#############################################
# Model arguments for dynamic reconstruction on continuation.
MODEL_ARGS = {
    "img_size": (1, 1120, 800),
    "input_vector_size": 28,
    "output_dim": 28,
    "min_variance": 1e-6,
    "features": 12,
    "depth": 15,
    "kernel": 3,
    "img_embed_dim": 32,
    "vector_embed_dim": 32,
    "size_reduce_threshold": (16, 16),
    "vector_feature_list": (16, 64, 64, 16),
    "output_feature_list": (16, 64, 64, 16),
}

# Base learning rate for the per-block parameter groups.
BASE_LR = 1e-2

# Sub-blocks that are unfrozen on a fresh run (everything except ``cov_mlp``),
# each paired with a name-matcher used for gradient-norm monitoring in the epoch
# function. The per-block LR multiplier scales ``BASE_LR`` in the optimizer.
BLOCK_LR_MULTIPLIERS = {
    "mean_mlp": 1.0,
    "vector_mlp": 2.0,
    "lin_embed_h1": 10.0,
    "lin_embed_h2": 10.0,
    "reduceH1": 5.0,
    "interpH1": 25.0,
    "reduceH2": 5.0,
    "interpH2": 25.0,
}

# (label, matcher) pairs forwarded to the epoch function for gradient logging.
BLOCKS = [
    ("mean head", lambda n: n.startswith("mean_mlp")),
    ("vector MLP", lambda n: n.startswith("vector_mlp")),
    ("h1 embed", lambda n: n.startswith("lin_embed_h1")),
    ("h2 embed", lambda n: n.startswith("lin_embed_h2")),
    ("CNN-H1 reduce", lambda n: n.startswith("reduceH1")),
    ("CNN-H1 interp", lambda n: n.startswith("interpH1")),
    ("CNN-H2 reduce", lambda n: n.startswith("reduceH2")),
    ("CNN-H2 interp", lambda n: n.startswith("interpH2")),
]


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, dict, type, int, torch.optim.Optimizer | None]:
    """Build the policy model with its freeze schedule (fresh or continuation).

    On a fresh run this freezes all parameters, then unfreezes the eight
    trainable sub-blocks (everything except the covariance head ``cov_mlp``) and
    returns ``None`` for the optimizer so the trainer builds the per-block
    optimizer via :func:`build_optimizer`. On continuation it reloads the model
    and optimizer from ``args.checkpoint`` and freezes ``cov_mlp``.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to place the model (and reloaded optimizer
            state) on.

    Returns:
        tuple: ``(model, model_args, model_class, starting_epoch, optimizer)``.
    """
    available_models = {"gaussian_policyCNN": gaussian_policyCNN}

    if getattr(args, "continuation", False):
        model, optimizer, starting_epoch = load_model_and_optimizer(
            args.checkpoint,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": BASE_LR,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.01,
            },
            available_models=available_models,
            device=device,
        )
        # Freeze the covariance head for continuation.
        for param in model.cov_mlp.parameters():
            param.requires_grad = False
        print("Model state loaded for continuation.")
        return model, MODEL_ARGS, gaussian_policyCNN, starting_epoch, optimizer

    # Fresh construction.
    model = gaussian_policyCNN(**MODEL_ARGS)
    model.to(device)

    # Freeze everything, then unfreeze the trainable sub-blocks.
    for param in model.parameters():
        param.requires_grad = False
    for _, matcher in BLOCKS:
        for name, param in model.named_parameters():
            if matcher(name):
                param.requires_grad = True

    return model, MODEL_ARGS, gaussian_policyCNN, 0, None


def build_optimizer(
    model: torch.nn.Module, args: argparse.Namespace
) -> torch.optim.AdamW:
    """Build AdamW with per-block parameter groups (fresh run only).

    Each unfrozen sub-block gets its own parameter group with a learning rate of
    ``BASE_LR`` scaled by the block's multiplier. Weight decay is disabled to
    match the study's mean-head-only fine-tuning recipe.

    Args:
        model (torch.nn.Module): Freshly constructed model (pre-DDP-wrap).
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer with per-block parameter groups.
    """
    param_groups = [
        {
            "params": getattr(model, block).parameters(),
            "lr": BASE_LR * multiplier,
        }
        for block, multiplier in BLOCK_LR_MULTIPLIERS.items()
    ]
    return torch.optim.AdamW(
        params=param_groups,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
    )


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train/validation policy datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = LSC_hfield_policy_DataSet(
        args.LSC_NPZ_DIR,
        filelist=train_filelist,
        design_file=design_file,
        half_image=False,
        field_list=["density_throw"],
    )
    val_dataset = LSC_hfield_policy_DataSet(
        args.LSC_NPZ_DIR,
        filelist=validation_filelist,
        design_file=design_file,
        half_image=False,
        field_list=["density_throw"],
    )
    return train_dataset, val_dataset


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the cosine-with-warmup LR scheduler.

    Args:
        optimizer (object): Optimizer the scheduler wraps.
        args (argparse.Namespace): Parsed command-line arguments.
        last_epoch (int): Scheduler ``last_epoch`` for continuation.

    Returns:
        CosineWithWarmupScheduler: The learning-rate scheduler.
    """
    return CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        warmup_steps=args.warmup_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    HarnessTrainer(
        args,
        model_builder=build_model,
        dataset_builder=build_dataset,
        epoch_fn=train_lsc_policy_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        epoch_kwargs={"blocks": BLOCKS},
    ).run()
