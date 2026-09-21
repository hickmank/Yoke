"""Evaluate a saved LodeRunner checkpoint on an LSC test pass."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.training.epoch.loderunner import eval_loderunner_epoch


parser = argparse.ArgumentParser(
    prog="LodeRunner Evaluation",
    description="Evaluate one test pass for a saved checkpoint.",
    fromfile_prefix_chars="@",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--FILELIST_DIR", required=True)
parser.add_argument("--LSC_NPZ_DIR", required=True)
parser.add_argument("--test_filelist", required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--num_workers", type=int, required=True)
parser.add_argument("--test_batches", type=int, required=True)
parser.add_argument("--max_timeIDX_offset", type=int, required=True)
parser.add_argument("--test_rcrd_filename", required=True)


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint and write records for exactly one test pass.

    Args:
        args (argparse.Namespace): Parsed evaluation configuration.

    Raises:
        ValueError: If checkpoint model metadata lacks the saved field ordering.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, checkpoint_epoch = load_model_and_optimizer(
        args.checkpoint,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-6},
        available_models={"LodeRunner": LodeRunner},
        device=device,
    )
    if not hasattr(model, "default_vars") or not model.default_vars:
        raise ValueError(
            "Checkpoint model metadata must provide non-empty default_vars."
        )

    fields = list(model.default_vars)
    test_filelist = str(Path(args.FILELIST_DIR) / args.test_filelist)
    dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=test_filelist,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        hydro_fields=np.array(fields),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loderunner_epoch(
        testing_data=dataloader,
        num_test_batches=args.test_batches,
        model=model,
        channel_map=list(range(len(fields))),
        loss_fn=nn.MSELoss(reduction="none"),
        epochIDX=checkpoint_epoch,
        test_rcrd_filename=args.test_rcrd_filename,
        device=device,
        dataset="pli",
    )
    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "test_filelist": test_filelist,
        "fields": fields,
        "max_timeIDX_offset": args.max_timeIDX_offset,
        "test_batches": args.test_batches,
        "sampling_policy": (
            "random temporal pairs; test_batches bounds the number of samples"
        ),
        "command": sys.argv,
    }
    Path(f"{args.test_rcrd_filename}.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main(parser.parse_args())
