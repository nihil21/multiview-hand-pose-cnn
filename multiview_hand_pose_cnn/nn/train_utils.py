import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
from time import time
from tqdm.notebook import tqdm
from multiview_hand_pose_cnn.nn.multiview_hand_pose_cnn import MultiViewHandPoseCNN, MultiViewHandPoseCNNBranch
from typing import Callable, Iterable, Optional, Union


# Lambda computing the mean
mean: Callable[[Iterable[float]], float] = lambda l: sum(l) / len(l)


def log_cosh(pred: torch.Tensor, true: torch.Tensor):
    return torch.log(torch.cosh(true - pred)).mean()


# Train function util
def train(model: Union[MultiViewHandPoseCNN, MultiViewHandPoseCNNBranch],
          data: DataLoader,
          criterion: Callable[[torch.FloatTensor, torch.FloatTensor],
                              torch.FloatTensor],
          optimizer: Optimizer,
          device: torch.device,
          projection: Optional[int] = None,
          verbose: Optional[bool] = False) -> float:
    loss_data = []

    # Activate train mode
    model.train()

    if verbose:
        data = tqdm(data, leave=False)
    for (batch_proj, batch_heats) in data:
        optimizer.zero_grad()
        loss = 0
        if isinstance(model, MultiViewHandPoseCNN):
            # Extract ground truth heat map: (batch_size, 21, 3, 18, 18) -> 3 x (batch_size, 21, 18, 18)
            xy_true_heats = batch_heats[:, :, 0]
            yz_true_heats = batch_heats[:, :, 1]
            zx_true_heats = batch_heats[:, :, 2]
            # Move tensors to GPU
            batch_proj = batch_proj.to(device)
            xy_true_heats = xy_true_heats.to(device)
            yz_true_heats = yz_true_heats.to(device)
            zx_true_heats = zx_true_heats.to(device)

            # Make prediction
            xy_pred_heats, yz_pred_heats, zx_pred_heats = model(batch_proj)
            # Compute loss
            xy_loss = sum([criterion(xy_pred_heats[:, i], xy_true_heats[:, i]) for i in range(21)])
            yz_loss = sum([criterion(yz_pred_heats[:, i], yz_true_heats[:, i]) for i in range(21)])
            zx_loss = sum([criterion(zx_pred_heats[:, i], zx_true_heats[:, i]) for i in range(21)])
            loss = xy_loss + yz_loss + zx_loss

        elif isinstance(model, MultiViewHandPoseCNNBranch) and projection is not None:
            # Extract ground truth heat map: (batch_size, 21, 3, 18, 18) -> (batch_size, 21, 18, 18)
            true_heats = batch_heats[:, :, projection]
            # Move tensors to GPU
            batch_proj = batch_proj[:, projection].unsqueeze(1)
            batch_proj = batch_proj.to(device)
            true_heats = true_heats.to(device)

            # Make prediction
            pred_heats = model(batch_proj)
            # Compute loss
            loss = sum([criterion(pred_heats[:, i], true_heats[:, i]) for i in range(21)])

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update history
        loss_data.append(loss.item())

    return mean(loss_data)


# Evaluate function util
def evaluate(model: Union[MultiViewHandPoseCNN, MultiViewHandPoseCNNBranch],
             data: DataLoader,
             criterion: Callable[[torch.FloatTensor, torch.FloatTensor],
                                 torch.FloatTensor],
             device: torch.device,
             projection: Optional[int] = None,
             verbose: Optional[bool] = False) -> float:
    loss_data = []

    # Activate eval mode
    model.eval()

    with torch.no_grad():
        if verbose:
            data = tqdm(data, leave=False)
        for (batch_proj, batch_heats) in data:
            loss = 0
            if isinstance(model, MultiViewHandPoseCNN):
                # Extract ground truth heat map: (batch_size, 21, 3, 18, 18) -> 3 x (batch_size, 21, 18, 18)
                xy_true_heats = batch_heats[:, :, 0]
                yz_true_heats = batch_heats[:, :, 1]
                zx_true_heats = batch_heats[:, :, 2]
                # Move tensors to GPU
                batch_proj = batch_proj.to(device)
                xy_true_heats = xy_true_heats.to(device)
                yz_true_heats = yz_true_heats.to(device)
                zx_true_heats = zx_true_heats.to(device)

                # Make prediction
                xy_pred_heats, yz_pred_heats, zx_pred_heats = model(batch_proj)
                # Compute loss
                xy_loss = sum([criterion(xy_pred_heats[:, i], xy_true_heats[:, i]) for i in range(21)])
                yz_loss = sum([criterion(yz_pred_heats[:, i], yz_true_heats[:, i]) for i in range(21)])
                zx_loss = sum([criterion(zx_pred_heats[:, i], zx_true_heats[:, i]) for i in range(21)])
                loss = xy_loss + yz_loss + zx_loss

            elif isinstance(model, MultiViewHandPoseCNNBranch) and projection is not None:
                # Extract ground truth heat map: (batch_size, 21, 3, 18, 18) -> (batch_size, 21, 18, 18)
                true_heats = batch_heats[:, :, projection]
                # Move tensors to GPU
                batch_proj = batch_proj[:, projection].unsqueeze(1)
                batch_proj = batch_proj.to(device)
                true_heats = true_heats.to(device)

                # Make prediction
                pred_heats = model(batch_proj)
                # Compute loss
                loss = sum([criterion(pred_heats[:, i], true_heats[:, i]) for i in range(21)])

            # Update history
            loss_data.append(loss.item())

    return mean(loss_data)


# Training loop function util
def training_loop(model: Union[MultiViewHandPoseCNN, MultiViewHandPoseCNNBranch],
                  train_data: DataLoader,
                  optimizer: Optimizer,
                  criterion: Callable[[torch.FloatTensor, torch.FloatTensor],
                                      torch.FloatTensor],
                  epochs: int,
                  device: torch.device,
                  val_data: Optional[DataLoader] = None,
                  projection: Optional[int] = None,
                  lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                  early_stopping: Optional[bool] = False,
                  patience: Optional[int] = 5,
                  tolerance: Optional[float] = 1e-4,
                  checkpoint_path: Optional[str] = None,
                  verbose: Optional[bool] = True):
    history = {'loss': [],
               'val_loss': []}

    # Initialize variables for early stopping
    min_val_loss = np.inf
    no_improve_counter = 0

    for ep in range(epochs):
        if verbose:
            print('-' * 100)
            print(f'Epoch {ep + 1}/{epochs}')

        start = time()
        train_loss = train(model, train_data, criterion, optimizer, device, projection, verbose)
        end = time()

        history['loss'].append(train_loss)

        if verbose:
            print(f'\tLoss: {train_loss:.5f} [Time elapsed: {end - start:.2f} s]')

        # Do validation if required
        if val_data is not None:
            start = time()
            val_loss = evaluate(model, val_data, criterion, device, projection, verbose)
            end = time()

            history['val_loss'].append(val_loss)
            if verbose:
                print(f'\tValidation loss: {val_loss:.5f} [Time elapsed: {end - start:.2f} s]')

            if early_stopping and checkpoint_path:
                # If validation loss is lower than minimum, update minimum
                if val_loss < min_val_loss - tolerance:
                    min_val_loss = val_loss
                    no_improve_counter = 0
                    print(f'New minimum val loss: {val_loss:.5f}')

                    # Save model
                    torch.save(model.state_dict(), checkpoint_path)
                # otherwise increment counter
                else:
                    no_improve_counter += 1
                # If loss did not improve for 'patience' epochs, break
                if no_improve_counter == patience:
                    if verbose:
                        print(f'Early stopping: no improvement in validation loss for '
                              f'{patience} epochs from {min_val_loss:.5f}')
                    # Restore model to best
                    model.load_state_dict(torch.load(checkpoint_path))
                    model.eval()
                    break

        # If lr scheduling is used, invoke next step
        if lr_scheduler:
            lr_scheduler.step()

    return history
