"""Functions to train and evaluate an lsc240420 inverse network over a single epoch."""

import torch
import numpy as np
from contextlib import nullcontext

from yoke.utils.training.datastep.lsc_gaussian import (
    train_lsc_gaussian_datastep,
    train_lsc_NLL_datastep,
    eval_lsc_gaussian_datastep,
    eval_lsc_NLL_datastep,
)


def train_lsc_gaussian_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    num_train_batches: int,
    num_val_batches: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Epoch training of LSC Gaussian inverse network.

    Function to complete a training epoch on the LSC gaussian inverse network for the
    layered shaped charge design problem. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        num_train_batches (int): Number of batches in training epoch
        num_val_batches (int): Number of batches in validation epoch
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        LRsched (torch.optim.lr_scheduler._LRScheduler): Learning-rate scheduler called
                                                         every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        rank (int): rank of process
        world_size (int): number of total processes
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:
        # Iterate over training batches
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Perform a single training step
            x_true, pred_mean, train_losses = train_lsc_gaussian_datastep(
                traindata, model, optimizer, loss_fn, device, rank, world_size,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
                # Save training losses
                batch_records = np.column_stack(
                    [
                        np.full(len(train_losses), epochIDX),
                        np.full(len(train_losses), trainbatch_ID),
                        train_losses.cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Validation loop
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        model.eval()
        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:
            with torch.no_grad():
                for valbatch_ID, valdata in enumerate(validation_data):
                    # Stop when number of training batches is reached
                    if valbatch_ID >= num_val_batches:
                        break

                    x_true, pred_mean, val_losses = eval_lsc_gaussian_datastep(
                        valdata,
                        model,
                        loss_fn,
                        device,
                        rank,
                        world_size,
                    )

                    # Save validation record (rank 0 only)
                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(val_losses), epochIDX),
                                np.full(len(val_losses), valbatch_ID),
                                val_losses.cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def train_lsc_NLL_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    num_train_batches: int,
    num_val_batches: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Epoch negative-loglikelihood training of LSC Gaussian inverse network.

    Function to complete a training epoch on the LSC gaussian inverse network for the
    layered shaped charge design problem. Training and validation information
    is saved to successive CSV files.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        num_train_batches (int): Number of batches in training epoch
        num_val_batches (int): Number of batches in validation epoch
        model (torch.nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        LRsched (torch.optim.lr_scheduler._LRScheduler): Learning-rate scheduler called
                                                         every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        rank (int): rank of process
        world_size (int): number of total processes
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:
        # Iterate over training batches
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Perform a single training step
            train_NLLs = train_lsc_NLL_datastep(
                traindata, model, optimizer, loss_fn, device, rank, world_size,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
                # Save training losses
                batch_records = np.column_stack(
                    [
                        np.full(len(train_NLLs), epochIDX),
                        np.full(len(train_NLLs), trainbatch_ID),
                        train_NLLs.cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Validation loop
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        model.eval()
        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:
            with torch.no_grad():
                for valbatch_ID, valdata in enumerate(validation_data):
                    # Stop when number of training batches is reached
                    if valbatch_ID >= num_val_batches:
                        break

                    val_NLLs = eval_lsc_NLL_datastep(
                        valdata,
                        model,
                        loss_fn,
                        device,
                        rank,
                        world_size,
                    )

                    # Save validation record (rank 0 only)
                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(val_NLLs), epochIDX),
                                np.full(len(val_NLLs), valbatch_ID),
                                val_NLLs.cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")
