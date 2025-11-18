"""Single-step training and evaluation functions for probabilistic LSC nets."""

import math

import torch
import torch.nn as nn
import torch.distributed as dist


def train_lsc_gaussian_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for LSC Gaussian inverse.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        x_true (torch.Tensor): Ground truth end vector
        pred_mean (torch.Tensor): Predicted end vector mean
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes

    """
    # Set model to train mode
    model.train()

    # Extract data
    Hfield, x_true = data
    Hfield = Hfield.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    pred_distribution = model(Hfield)
    pred_mean = pred_distribution.mean

    # Compute loss
    loss = loss_fn(pred_mean, x_true)
    per_sample_loss = loss.mean(dim=1)  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()

    # Step the optimizer
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return x_true, pred_mean, all_losses


def eval_lsc_gaussian_datastep(
    data: tuple,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible gaussian LSC inverse evaluation step.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    Returns:
        x_true (torch.Tensor): Ground truth end vector
        pred_mean (torch.Tensor): Predicted end vector mean
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    Hfield, x_true = data
    Hfield = Hfield.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred_distribution = model(Hfield)

    pred_mean = pred_distribution.mean

    # Compute loss
    loss = loss_fn(pred_mean, x_true)
    per_sample_loss = loss.mean(dim=1)  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return x_true, pred_mean, all_losses


def train_lsc_NLL_datastep(
    data: tuple,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible training step for NLL fine-tuning of the LSC Gaussian inverse.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        optimizer (torch.optim): optimizer for training set
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Number of total DDP processes

    Returns:
        all_NLLs (torch.Tensor): Concatenated per-sample NLL from all processes

    """
    # Set model to train mode
    model.train()

    # Extract data
    Hfield, x_true = data
    Hfield = Hfield.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    pred_dist = model(Hfield)

    # Per-sample negative log-likelihood (shape: [B])
    # log_prob expects x_true of shape [B, D] matching the distribution's event shape
    per_sample_nll = -pred_dist.log_prob(x_true)

    # Scalar loss for backprop
    loss = per_sample_nll.mean()

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Step the optimizer
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_nll) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_nll)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_NLLs = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_NLLs = None

    return all_NLLs


def eval_lsc_NLL_datastep(
    data: tuple,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A DDP-compatible evaluation step for NLL fine-tuning of the LSC Gaussian inverse.

    Args:
        data (tuple): tuple of model input and corresponding ground truth
        model (loaded pytorch model): model to train
        loss_fn (torch.nn Loss Function): loss function for training set
        device (torch.device): device index to select
        rank (int): Rank of device
        world_size (int): Total number of DDP processes

    Returns:
        all_NLLs (torch.Tensor): Concatenated per-sample NLLs from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data
    Hfield, x_true = data
    Hfield = Hfield.to(device, non_blocking=True)
    x_true = x_true.to(device, non_blocking=True)

    # Forward pass
    with torch.no_grad():
        pred_dist = model(Hfield)

    # Per-sample negative log-likelihood (shape: [B])
    # log_prob expects x_true of shape [B, D] matching the distribution's event shape
    per_sample_nll = -pred_dist.log_prob(x_true)

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_nll) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_nll)

    # Rank 0 concatenates and saves or returns all losses
    if rank == 0:
        all_NLLs = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_NLLs = None

    return all_NLLs
