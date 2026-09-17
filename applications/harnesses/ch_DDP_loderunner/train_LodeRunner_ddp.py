"""DDP training harness for LodeRunner on the lsc240420 PLI dataset."""

import argparse

import numpy as np

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import ConstantWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch

descr_str = (
    "Uses DDP to train LodeRunner architecture on single-timstep input and output "
    "of the lsc240420 per-material density fields."
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
parser.add_argument(
    "--noise_scale",
    type=float,
    default=0.0,
    help="Relative magnitude for Gaussian noise injection (e.g. 5e-5).",
)
parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=1,
    help="Maximum time offset for input/output image pairs.",
)
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

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
    """Build LodeRunner construction arguments from parsed arguments."""
    return {
        "default_vars": CHANNEL_LIST,
        "image_size": (1120, 400),
        "patch_size": (5, 5),
        "embed_dim": args.embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": tuple(args.block_structure),
        "window_sizes": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
        "noise_scale": 0.0,
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the fresh-run AdamW optimizer."""
    return build_adamw(model, args, lr=1e-4, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build the train and validation temporal LSC datasets."""
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    dataset_args = {
        "max_timeIDX_offset": args.max_timeIDX_offset,
        "max_file_checks": 10,
        "hydro_fields": np.array(CHANNEL_LIST),
        "half_image": True,
    }
    return (
        LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR, file_prefix_list=train_filelist, **dataset_args
        ),
        LSC_rho2rho_temporal_DataSet(
            args.LSC_NPZ_DIR, file_prefix_list=validation_filelist, **dataset_args
        ),
    )


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> ConstantWithWarmupScheduler:
    """Build the constant-with-warmup learning-rate scheduler."""
    return ConstantWithWarmupScheduler(
        optimizer, warmup_steps=0, lr_constant=1e-4, last_epoch=last_epoch
    )


if __name__ == "__main__":
    args = parser.parse_args()
    HarnessTrainer(
        args,
        model_builder=build_from_checkpoint(
            LodeRunner,
            make_model_args,
            optimizer_kwargs={
                "lr": 1e-6,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        epoch_kwargs={
            "channel_map": list(range(len(CHANNEL_LIST))),
            "dataset": "pli",
        },
        evaluate_after_training=True,
    ).run()
