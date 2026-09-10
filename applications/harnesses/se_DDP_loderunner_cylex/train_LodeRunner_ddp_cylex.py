"""DDP training harness for LodeRunner on the cx241203 (cylex) dataset.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer`. Trains
LodeRunner on the cylex temporal dataset using a small fixed optimizer LR
(``1e-6``) and a cosine-with-warmup scheduler whose peak LR is scaled by the
global batch size. The epoch function is told it is operating on the ``cylex``
dataset.
"""

import argparse

import numpy as np

from yoke.datasets.load_npz_dataset import TemporalDataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch

#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train LodeRunner architecture on single-timstep input and output "
    "of the cx241203 (cylex) per-material fields."
)
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(
    train_filelist="cx241203_prefixes_train_80pct.txt",
    validation_filelist="cx241203_prefixes_val_10pct.txt",
    test_filelist="cx241203_prefixes_test_10pct.txt",
)

#############################################
# Study-specific constants
#############################################
# 4 kinematic + 39 thermodynamic variable fields.
DEFAULT_VARS = [
    "Rcoord",
    "Zcoord",
    "Uvelocity",
    "Wvelocity",
    "density_Air",
    "energy_Air",
    "pressure_Air",
    "density_Al",
    "energy_Al",
    "pressure_Al",
    "density_Be",
    "energy_Be",
    "pressure_Be",
    "density_booster",
    "energy_booster",
    "pressure_booster",
    "density_Cu",
    "energy_Cu",
    "pressure_Cu",
    "density_U.DU",
    "energy_U.DU",
    "pressure_U.DU",
    "density_maincharge",
    "energy_maincharge",
    "pressure_maincharge",
    "density_N",
    "energy_N",
    "pressure_N",
    "density_Sn",
    "energy_Sn",
    "pressure_Sn",
    "density_Steel.alloySS304L",
    "energy_Steel.alloySS304L",
    "pressure_Steel.alloySS304L",
    "density_Polymer.Sylgard",
    "energy_Polymer.Sylgard",
    "pressure_Polymer.Sylgard",
    "density_Ta",
    "energy_Ta",
    "pressure_Ta",
    "density_Void",
    "energy_Void",
    "pressure_Void",
    "density_Water",
    "energy_Water",
    "pressure_Water",
]

# Optimizer learning rate (fixed; the scheduler drives the effective LR).
OPTIMIZER_LR = 1e-6

# Reference global batch size used to normalize the LR scaling (1 node, 4 GPUs,
# 10 samples/GPU).
REFERENCE_BATCHSIZE = 40.0


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the LodeRunner ``model_args`` dict for the cylex study.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for constructing :class:`LodeRunner`.
    """
    return {
        "default_vars": DEFAULT_VARS,
        "image_size": (1120, 400),
        "patch_size": (10, 5),
        "embed_dim": args.embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": tuple(args.block_structure),
        "window_sizes": [(8, 8), (8, 8), (4, 4), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the AdamW optimizer at the fixed cylex learning rate.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer.
    """
    return build_adamw(model, args, lr=OPTIMIZER_LR, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train/validation cylex temporal datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = TemporalDataSet(
        args.NPZ_DIR,
        args.CSV_FILEPATH,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=2,
        max_file_checks=10,
        half_image=True,
    )
    val_dataset = TemporalDataSet(
        args.NPZ_DIR,
        args.CSV_FILEPATH,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=2,
        max_file_checks=10,
        half_image=True,
    )
    return train_dataset, val_dataset


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the cosine-with-warmup scheduler with batch-size-scaled peak LR.

    Args:
        optimizer (object): Optimizer the scheduler wraps.
        args (argparse.Namespace): Parsed command-line arguments.
        last_epoch (int): Scheduler ``last_epoch`` for continuation.

    Returns:
        CosineWithWarmupScheduler: The learning-rate scheduler.
    """
    # Scale the anchor LR by the global batch size.
    lr_scale = np.sqrt(float(args.Ngpus) * float(args.Knodes) * float(args.batch_size))
    ddp_anchor_lr = args.anchor_lr * lr_scale / REFERENCE_BATCHSIZE

    return CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=ddp_anchor_lr,
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
            LodeRunner,
            make_model_args,
            optimizer_kwargs={
                "lr": OPTIMIZER_LR,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        epoch_kwargs={"dataset": "cylex"},
    ).run()
