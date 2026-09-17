"""DDP training harness for DiffusionLodeRunner on temporal LSC data.

This thin wrapper delegates DDP setup, loaders, checkpointing, and continuation
handling to :class:`yoke.harnesses.trainer.HarnessTrainer`.
"""

import argparse

import numpy as np
import torch

from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.diffusion_bomberman import DiffusionLodeRunner
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.diff_loderunner import (
    train_DDP_diffusion_loderunner_epoch,
)


#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train DiffusionLodeRunner architecture on temporal prediction "
    "of the lsc240420 per-material density fields using score-clearbased diffusion."
)
parser = argparse.ArgumentParser(
    prog="DDP DiffusionLodeRunner Training",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Diffusion-specific parameters
parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=10,
    help="Maximum time offset for input/output image pairs.",
)

# Change some default filepaths
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

#############################################
# Study-specific constants
#############################################
CHANNEL_LIST = [
    "density_case",
    "density_cushion",
    "density_maincharge",
    "density_outside_air",
    "density_striker",
    "density_throw",
    "Uvelocity",
    "Wvelocity",
]


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the DiffusionLodeRunner construction arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for :class:`DiffusionLodeRunner`.
    """
    return {
        "default_vars": CHANNEL_LIST,
        "image_size": (1120, 400),
        "patch_size": (10, 10),
        "embed_dim": args.embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": tuple(args.block_structure),
        "window_sizes": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build this harness's AdamW optimizer.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The configured optimizer.
    """
    return build_adamw(model, args, lr=1e-4, weight_decay=0.01)


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the cosine-with-warmup learning-rate scheduler.

    Args:
        optimizer (object): Optimizer the scheduler wraps.
        args (argparse.Namespace): Parsed command-line arguments.
        last_epoch (int): Scheduler state for a fresh or continued run.

    Returns:
        CosineWithWarmupScheduler: The configured scheduler.
    """
    return CosineWithWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build train and validation temporal diffusion datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: The train and validation datasets.
    """
    noise_schedule = VPCosineNoiseSchedule()
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(CHANNEL_LIST),
        out_vars=np.array(CHANNEL_LIST),
        noise_schedule=noise_schedule,
    )
    val_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(CHANNEL_LIST),
        out_vars=np.array(CHANNEL_LIST),
        noise_schedule=noise_schedule,
    )
    return train_dataset, val_dataset


def run_epoch(**kwargs: object) -> None:
    """Run one diffusion epoch with channel indices on the trainer device.

    Args:
        **kwargs: Keyword arguments supplied by :class:`HarnessTrainer`.
    """
    device = kwargs["device"]
    in_vars = torch.tensor(list(range(len(CHANNEL_LIST))), device=device)
    out_vars = torch.tensor(list(range(len(CHANNEL_LIST))), device=device)
    train_DDP_diffusion_loderunner_epoch(**kwargs, in_vars=in_vars, out_vars=out_vars)


if __name__ == "__main__":
    args = parser.parse_args()

    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(
            DiffusionLodeRunner,
            make_model_args,
            optimizer_kwargs={
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=run_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
    ).run()
