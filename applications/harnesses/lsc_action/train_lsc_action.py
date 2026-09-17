"""DDP training harness for the tCNN surrogate on lsc240420.

Trains :class:`~yoke.models.surrogateCNNmodules.tCNNsurrogate` -- a transpose-CNN
that maps layered-shaped-charge (LSC) B-spline contour geometry parameters to a
density-field image. This is a modernized rebuild: it follows the current Yoke
harness conventions (always DDP, ``.pth`` checkpointing via
:class:`yoke.harnesses.trainer.HarnessTrainer`, a cosine-with-warmup LR
scheduler) rather than the legacy single-process / HDF5 / ``torch.compile`` path
the study originally used.

The reflected (full) density field is produced directly by
:class:`~yoke.datasets.lsc_dataset.LSC_cntr2hfield_DataSet` with
``half_image=False`` (a ``(1, 1120, 800)`` target), so no dataset wrapper is
needed. The array-output epoch function
(:func:`yoke.utils.training.epoch.array_output.train_DDP_array_epoch`) handles
the per-batch train/eval steps.
"""

import argparse
import os

from yoke.datasets.lsc_dataset import LSC_cntr2hfield_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.surrogateCNNmodules import tCNNsurrogate
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.array_output import train_DDP_array_epoch

#############################################
# Inputs
#############################################
descr_str = (
    "Trains a Transpose-CNN to reconstruct the density field of an LSC "
    "simulation from B-spline contour parameters, using DDP."
)
parser = argparse.ArgumentParser(
    prog="LSC Surrogate Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(design_file="design_lsc240420_MASTER.csv")

#############################################
# Study-specific constants
#############################################
# Number of B-spline contour-node scalar inputs (no sim-time channel).
INPUT_SIZE = 28

# Fixed optimizer learning rate; the cosine scheduler drives the effective LR.
OPTIMIZER_LR = 1e-6


def make_model_args(args: argparse.Namespace) -> dict:
    """Build the ``tCNNsurrogate`` ``model_args`` dict from parsed arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Keyword arguments for constructing :class:`tCNNsurrogate`.
    """
    return {
        "input_size": INPUT_SIZE,
        "linear_features": (7, 5, args.linearFeatures),
        "kernel": (3, 3),
        "nfeature_list": list(args.featureList),
        "output_image_size": (1120, 800),
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the AdamW optimizer at the fixed learning rate.

    Args:
        model (object): Model whose parameters are optimized.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        torch.optim.AdamW: The optimizer.
    """
    return build_adamw(model, args, lr=OPTIMIZER_LR, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train/validation LSC contour-to-field datasets.

    The datasets return the full (reflected) density field directly via
    ``half_image=False``, so no mirror/butterfly wrapper is required.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        tuple: ``(train_dataset, val_dataset)``.
    """
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    train_dataset = LSC_cntr2hfield_DataSet(
        args.LSC_NPZ_DIR,
        filelist=train_filelist,
        design_file=design_file,
        half_image=False,
        field_list=["density_throw"],
    )
    val_dataset = LSC_cntr2hfield_DataSet(
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
        model_builder=build_from_checkpoint(
            tCNNsurrogate,
            make_model_args,
            optimizer_kwargs={
                "lr": OPTIMIZER_LR,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_array_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
    ).run()
