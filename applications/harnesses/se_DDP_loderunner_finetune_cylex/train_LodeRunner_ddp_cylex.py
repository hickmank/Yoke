"""DDP fine-tuning harness for LodeRunner on the cx241203 (cylex) dataset.

Thin wrapper around :class:`yoke.harnesses.trainer.HarnessTrainer`. This is the
fine-tuning sibling of ``se_DDP_loderunner_cylex``: it starts from an optional
pretrained checkpoint and optionally freezes the transformer *backbone* for the
first ``--freeze_backbone_epochs`` epochs, training only the variable-embedding
and unpatch head during that warmup phase before unfreezing everything.

The study-specific behavior is expressed through two trainer injection points:

- A **bespoke ``model_builder``** owns architecture knowledge: on a fresh run it
  constructs LodeRunner and (if ``--pretrained_model`` is given) shape-safely
  loads pretrained weights; on continuation it reloads model + optimizer. It
  also applies the backbone freeze for this job **before** DDP wrapping, based on
  the job's first (upcoming) epoch -- matching the original semantics and
  keeping the DDP reducer consistent with the trainable-parameter set.
- The frozen-phase learning rate may be overridden via ``--warmup_lr``; the
  ``scheduler_builder`` selects it when the job's first epoch falls inside the
  frozen phase (matching the original per-job behavior).

The freeze is applied at job granularity (before DDP wrap) rather than per-epoch:
toggling ``requires_grad`` after DDP construction would desynchronize DDP's
gradient reducer from the trainable-parameter set. In the study's usage the
freeze boundary aligns with job boundaries, so this is behavior-preserving.
"""

import argparse

import numpy as np
import torch

from yoke.datasets.load_npz_dataset import TemporalDataSet
from yoke.harnesses.trainer import HarnessTrainer
from yoke.helpers import cli
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.builders import build_adamw
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.training.epoch.loderunner import train_DDP_loderunner_epoch

#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to fine-tune LodeRunner architecture on the cx241203 (cylex) "
    "per-material fields, optionally freezing the backbone for warmup epochs."
)
parser = argparse.ArgumentParser(
    prog="DDP LodeRunner Fine-Tuning",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Fine-tuning / freezing options.
parser.add_argument(
    "--freeze_backbone_epochs",
    type=int,
    default=0,
    help="Freeze backbone for the first N epochs (train embeddings/head only).",
)
parser.add_argument(
    "--warmup_lr",
    type=float,
    default=None,
    help="Override anchor_lr during the frozen phase (scheduler anchor LR).",
)

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

# Backbone-warmup keeps only these modules trainable while the backbone is
# frozen. Missing attributes are silently skipped (guarded by ``hasattr``).
WARMUP_TRAINABLE_MODULES = ("parallel_embed", "var_embed_layer", "linear4unpatch")


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


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    """Set ``requires_grad`` on every parameter of ``module``.

    Args:
        module (torch.nn.Module): Module whose parameters to toggle.
        requires_grad (bool): Target ``requires_grad`` value.
    """
    for p in module.parameters():
        p.requires_grad = requires_grad


def apply_backbone_freeze(model: torch.nn.Module, freeze: bool) -> None:
    """Freeze or unfreeze the backbone of an (unwrapped) LodeRunner.

    When ``freeze`` is ``True`` all parameters are frozen and only the
    variable-embedding + unpatch-head modules in
    :data:`WARMUP_TRAINABLE_MODULES` are left trainable. When ``False`` every
    parameter is unfrozen.

    Args:
        model (torch.nn.Module): The unwrapped model (i.e. ``ddp.module``).
        freeze (bool): Whether to freeze the backbone.
    """
    if not freeze:
        _set_requires_grad(model, True)
        return

    _set_requires_grad(model, False)
    for attr in WARMUP_TRAINABLE_MODULES:
        if hasattr(model, attr):
            _set_requires_grad(getattr(model, attr), True)


def _load_pretrained_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Shape-safely load pretrained weights into ``model`` (weights-only).

    Accepts a raw ``state_dict`` or a checkpoint dict with a
    ``model_state_dict``/``state_dict`` entry, strips any ``module.`` prefix, and
    copies only tensors whose names and shapes match (mismatches are skipped).

    Args:
        model (torch.nn.Module): Freshly constructed model to initialize.
        ckpt_path (str): Path to the pretrained checkpoint.
    """
    print(f"Loading pretrained weights from {ckpt_path} (weights-only).")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt if isinstance(ckpt, dict) else {}

    # Strip a leading "module." (DDP) prefix if present.
    cleaned = {
        (k[len("module.") :] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    # Shape-safe load: keep only name+shape matches.
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in cleaned.items()
        if k in model_state
        and isinstance(v, torch.Tensor)
        and isinstance(model_state[k], torch.Tensor)
        and v.shape == model_state[k].shape
    }
    skipped = len(cleaned) - len(filtered)
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    print(
        f"Pretrained load complete: loaded {len(filtered)} tensors; skipped {skipped}."
    )
    if missing_keys:
        print(f"Missing keys after pretrained load (sample): {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys after pretrained load (sample): {unexpected_keys[:10]}")


def build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.nn.Module, dict, type, int, torch.optim.Optimizer | None]:
    """Build the LodeRunner model for a fresh or continuation run.

    Fresh runs optionally initialize from ``args.pretrained_model`` (shape-safe,
    weights-only) and return ``None`` for the optimizer so the trainer builds it.
    Continuation reloads model + optimizer from ``args.checkpoint``. The backbone
    freeze for this job is applied here, **before** DDP wrapping, based on the
    job's first (upcoming) epoch, so the DDP gradient reducer is built over
    exactly the trainable parameters.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        device (torch.device): Device to place the model on.

    Returns:
        tuple: ``(model, model_args, model_class, starting_epoch, optimizer)``.
    """
    model_args = make_model_args(args)
    available_models = {"LodeRunner": LodeRunner}

    if getattr(args, "continuation", False):
        model, optimizer, starting_epoch = load_model_and_optimizer(
            args.checkpoint,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": OPTIMIZER_LR,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.01,
            },
            available_models=available_models,
            device=device,
        )
        print("Model state loaded for continuation.")
    else:
        # Fresh construction, optionally initialized from pretrained weights.
        starting_epoch = 0
        optimizer = None
        model = LodeRunner(**model_args)
        if getattr(args, "pretrained_model", None) is not None:
            _load_pretrained_weights(model, args.pretrained_model)
        model.to(device)

    # Apply the backbone freeze for this job before DDP wrapping. The job's first
    # epoch is (starting_epoch + 1) because the trainer increments before the
    # loop begins.
    freeze_epochs = getattr(args, "freeze_backbone_epochs", 0) or 0
    if freeze_epochs > 0:
        upcoming_epoch = starting_epoch + 1
        freeze = upcoming_epoch <= freeze_epochs
        apply_backbone_freeze(model, freeze)
        state = "frozen" if freeze else "unfrozen"
        print(f"Backbone {state} for epoch {upcoming_epoch} (freeze<= {freeze_epochs}).")

    return model, model_args, LodeRunner, starting_epoch, optimizer


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

    If the job's first epoch falls inside the frozen phase and ``--warmup_lr`` is
    set, the frozen-phase anchor LR is used instead of ``--anchor_lr`` (matching
    the original per-job warmup behavior). The upcoming epoch is derived from
    ``last_epoch``: ``-1`` (a fresh run) implies the job starts at epoch 1;
    otherwise ``last_epoch == train_batches * (starting_epoch - 1)``.

    Args:
        optimizer (object): Optimizer the scheduler wraps.
        args (argparse.Namespace): Parsed command-line arguments.
        last_epoch (int): Scheduler ``last_epoch`` for continuation.

    Returns:
        CosineWithWarmupScheduler: The learning-rate scheduler.
    """
    # Derive the first epoch this job will train.
    if last_epoch < 0:
        upcoming_epoch = 1
    else:
        starting_epoch = last_epoch // args.train_batches + 1
        upcoming_epoch = starting_epoch + 1

    freeze_epochs = getattr(args, "freeze_backbone_epochs", 0) or 0
    effective_anchor_lr = args.anchor_lr
    if (
        freeze_epochs > 0
        and upcoming_epoch <= freeze_epochs
        and getattr(args, "warmup_lr", None) is not None
    ):
        effective_anchor_lr = args.warmup_lr

    # Scale the anchor LR by the global batch size.
    lr_scale = np.sqrt(float(args.Ngpus) * float(args.Knodes) * float(args.batch_size))
    ddp_anchor_lr = effective_anchor_lr * lr_scale / REFERENCE_BATCHSIZE

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
        model_builder=build_model,
        dataset_builder=build_dataset,
        epoch_fn=train_DDP_loderunner_epoch,
        optimizer_builder=build_optimizer,
        scheduler_builder=build_scheduler,
        epoch_kwargs={"dataset": "cylex"},
    ).run()
