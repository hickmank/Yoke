"""Train a Gaussian Policy network using DDP."""

import argparse
import os

import torch

from yoke.models.CNNmodules import Image2VectorCNN
from yoke.datasets.lsc_dataset import LSC_hfield2cntr_DataSet
from yoke.utils.training.epoch.array_output import train_DDP_array_epoch
from yoke.harnesses.trainer import HarnessTrainer
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.builders import build_adamw
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = "Uses DDP to train parameter-estimation CNN."
parser = argparse.ArgumentParser(
    prog="Gaussian Policy Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)


def make_model_args(args: argparse.Namespace) -> dict:
    """Build inverse-model construction arguments."""
    return {
        "img_size": (1, 1120, 800),
        "output_dim": 29,
        "size_threshold": (12, 12),
        "kernel": 3,
        "features": 16,
        "interp_depth": 12,
        "conv_onlyweights": True,
        "batchnorm_onlybias": True,
        "hidden_features": 32,
    }


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, dict, type, int, torch.optim.Optimizer | None]:
    """Build or restore the inverse model and its optimizer.

    The continuation loader intentionally retains its legacy optimizer arguments,
    which differ from the fresh optimizer configuration below.
    """
    model_args = make_model_args(args)
    if args.continuation:
        model, optimizer, starting_epoch = load_model_and_optimizer(
            args.checkpoint,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": 1e-2,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.01,
            },
            available_models={"Image2VectorCNN": Image2VectorCNN},
            device=device,
        )
        print("Model state loaded for continuation.")
        return model, model_args, Image2VectorCNN, starting_epoch, optimizer
    model = Image2VectorCNN(**model_args).to(device)
    return model, model_args, Image2VectorCNN, 0, None


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the fresh inverse-study AdamW optimizer."""
    return build_adamw(model, args, lr=1e-3, weight_decay=0.0)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build full-image train and validation inverse datasets."""
    dataset_args = {
        "design_file": os.path.abspath(args.LSC_DESIGN_DIR + args.design_file),
        "half_image": False,
        "include_time": True,
        "field_list": ["density_throw"],
    }
    return (
        LSC_hfield2cntr_DataSet(
            args.LSC_NPZ_DIR,
            filelist=args.FILELIST_DIR + args.train_filelist,
            **dataset_args,
        ),
        LSC_hfield2cntr_DataSet(
            args.LSC_NPZ_DIR,
            filelist=args.FILELIST_DIR + args.validation_filelist,
            **dataset_args,
        ),
    )


def build_scheduler(
    optimizer: object, args: argparse.Namespace, last_epoch: int
) -> CosineWithWarmupScheduler:
    """Build the unscaled cosine-with-warmup scheduler."""
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
        epoch_fn=train_DDP_array_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
    ).run()
