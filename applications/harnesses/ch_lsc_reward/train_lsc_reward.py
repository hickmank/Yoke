"""DDP training harness for the LSC reward network."""

import argparse
import os

import torch.nn as nn

from yoke.datasets.lsc_dataset import LSC_hfield_reward_DataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.hybridCNNmodules import hybrid2vectorCNN
from yoke.utils.builders import build_adamw, build_from_checkpoint
from yoke.utils.training.epoch.lsc_reward import train_lsc_reward_epoch

descr_str = (
    "Trains reward network architecture to calculate error between current and target "
    "density fields."
)
parser = argparse.ArgumentParser(
    prog="reward network training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)
parser.set_defaults(design_file="design_lsc240420_MASTER.csv")


def make_model_args(args: argparse.Namespace) -> dict:
    """Build reward-model construction arguments."""
    return {
        "img_size": (1, 1120, 800),
        "input_vector_size": 28,
        "output_dim": 1,
        "features": 12,
        "depth": 4,
        "kernel": 3,
        "img_embed_dim": 32,
        "vector_embed_dim": 32,
        "size_reduce_threshold": (16, 16),
        "vector_feature_list": (4, 4, 4, 4),
        "output_feature_list": (4, 4, 4, 4),
        "act_layer": nn.GELU,
        "norm_layer": nn.LayerNorm,
    }


def build_optimizer(model: object, args: argparse.Namespace) -> object:
    """Build the reward study's AdamW optimizer."""
    return build_adamw(model, args, lr=1e-6, weight_decay=0.01)


def build_dataset(args: argparse.Namespace) -> tuple[object, object]:
    """Build full-image train and validation reward datasets."""
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    dataset_args = {
        "design_file": design_file,
        "half_image": False,
        "field_list": ["density_throw"],
    }
    return (
        LSC_hfield_reward_DataSet(
            args.LSC_NPZ_DIR,
            filelist=args.FILELIST_DIR + args.train_filelist,
            **dataset_args,
        ),
        LSC_hfield_reward_DataSet(
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
        model_builder=build_from_checkpoint(
            hybrid2vectorCNN,
            make_model_args,
            optimizer_kwargs={
                "lr": 1e-6,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        ),
        dataset_builder=build_dataset,
        epoch_fn=train_lsc_reward_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
    ).run()
