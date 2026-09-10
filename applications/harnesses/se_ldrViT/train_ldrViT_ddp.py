"""DDP training harness for the LodeRunner-ViT architecture on lsc240420.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer`. Uses a
LodeRunner-ViT model, an ``anchor_lr`` AdamW optimizer, and a cosine-with-warmup
LR scheduler.
"""

import argparse

import numpy as np

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunnerViT
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch

#############################################
# Inputs
#############################################
descr_str = "Uses DDP to train LodeRunner-ViT architecture on lsc240420."
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner-ViT Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=1,
    help="Maximum time offset for input/output image pairs.",
)

# ViT backbone parameters
parser.add_argument(
    "--vit_embed_dim",
    type=int,
    default=512,
    help="Embedding dimension for the ViT backbone.",
)
parser.add_argument(
    "--vit_num_layers",
    type=int,
    default=12,
    help="Number of ViT layers in backbone.",
)
parser.add_argument(
    "--vit_num_heads",
    type=int,
    default=8,
    help="Number of ViT attention heads in backbone.",
)

# Change some default filepaths.
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
    "energy_case",
    "pressure_case",
    "density_cushion",
    "energy_cushion",
    "pressure_cushion",
    "density_maincharge",
    "energy_maincharge",
    "pressure_maincharge",
    "density_outside_air",
    "energy_outside_air",
    "pressure_outside_air",
    "density_striker",
    "energy_striker",
    "pressure_striker",
    "density_throw",
    "energy_throw",
    "pressure_throw",
    "Uvelocity",
    "Wvelocity",
]


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the LodeRunner-ViT ``model_args`` dict from parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for constructing :class:`LodeRunnerViT`.
    """
    return {
        "default_vars": CHANNEL_LIST,
        "image_size": (1120, 400),
        "patch_size": (10, 5),
        "embed_dim": args.vit_embed_dim,
        "num_heads": 8,
        "num_attention_heads": args.vit_num_heads,
        "attention_head_dim": int(args.vit_embed_dim / args.vit_num_heads),
        "num_layers": args.vit_num_layers,
        "mlp_ratio": 4.0,
        "concat_mlp": True,
        "verbose": False,
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the AdamW optimizer using ``anchor_lr`` as the fresh-run LR.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer.
    """
    return build_adamw(model, args, lr=args.anchor_lr, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train/validation LSC temporal datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        hydro_fields=np.array(CHANNEL_LIST),
        half_image=True,
    )
    val_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        hydro_fields=np.array(CHANNEL_LIST),
        half_image=True,
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
        model_builder=build_from_checkpoint(
            LodeRunnerViT,
            make_model_args,
            optimizer_kwargs={
                "lr": args.anchor_lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        epoch_kwargs={"channel_map": list(range(len(CHANNEL_LIST)))},
    ).run()
