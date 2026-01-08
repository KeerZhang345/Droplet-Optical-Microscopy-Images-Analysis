import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List


def masked_mse_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Masked mean squared error loss.

    :param
    ----------
    reconstructed : torch.Tensor
        Reconstructed images, shape (B, C, H, W)
    original : torch.Tensor
        Original images, shape (B, C, H, W)
    mask : torch.Tensor
        Binary mask, shape (B, 1, H, W) or (B, H, W)

    :return
    -------
    torch.Tensor
        Scalar loss value
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)

    loss = (((reconstructed - original) ** 2) * mask).mean()
    return loss


def train_ae_conv_fc(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    model_path: str,
    early_stopping_patience: int = 50
) -> Tuple[List[float], List[float], nn.Module]:
    """
    Train convolutional autoencoder with masked reconstruction loss.

    - masked MSE loss
    - validation loop
    - ReduceLROnPlateau scheduler
    - early stopping
    - best-model checkpointing

    :param
    ----------
    model : nn.Module
        AEConvFC model
    train_loader : DataLoader
        Yields (images, masks)
    val_loader : DataLoader
        Yields (images, masks)
    device : torch.device
        CUDA or CPU
    num_epochs : int
        Maximum number of epochs
    lr : float
        Initial learning rate
    model_path : str
        Path to save best model weights
    early_stopping_patience : int
        Number of epochs without improvement before stopping

    :return
    -------
    train_losses : list of float
    val_losses : list of float
    model : nn.Module
        Model loaded with best weights
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(num_epochs):
        # -------------------------
        # Training phase
        # -------------------------
        model.train()
        train_loss_epoch = 0.0

        for batch in train_loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            _, reconstructed = model(images)
            loss = masked_mse_loss(reconstructed, images, masks)

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= len(train_loader)
        train_losses.append(train_loss_epoch)

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

                _, reconstructed = model(images)
                loss = masked_mse_loss(reconstructed, images, masks)

                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

        scheduler.step(val_loss_epoch)

        # -------------------------
        # Logging
        # -------------------------
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss_epoch:.6f} | "
                f"Val Loss: {val_loss_epoch:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        # -------------------------
        # Early stopping + checkpoint
        # -------------------------
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # -------------------------
    # Load best model
    # -------------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return train_losses, val_losses, model
