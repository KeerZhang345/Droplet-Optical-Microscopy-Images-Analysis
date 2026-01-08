import torch
import torch.nn as nn


class AEConvFC(nn.Module):
    """
    Convolutional Autoencoder with Fully-Connected latent bottleneck.

    Architecture (fixed, as in manuscript figure):

    Input:        (B, 3, 32, 32)
    Encoder:
        Conv1     → (B, 16, 16, 16)
        Conv2     → (B, 32,  8,  8)
        Conv3     → (B, 64,  4,  4)
        Flatten   → (B, 1024)
        FC1       → (B, latent_dim)

    Decoder:
        FC2       → (B, 1024)
        Reshape   → (B, 64, 4, 4)
        DeConv3   → (B, 32,  8,  8)
        DeConv2   → (B, 16, 16, 16)
        DeConv1   → (B,  3, 32, 32)

    Notes
    -----
    - latent_dim is intentionally user-defined.
    - Masking is NOT handled here (loss-level responsibility).
    - This module is inference- and training-compatible.
    """

    def __init__(self, latent_dim: int, input_channels: int = 3):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(64 * 4 * 4, latent_dim)



        self.decoder_fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                16, input_channels, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input images into latent vectors.

        :param
        ----------
        x : torch.Tensor
            Shape (B, 3, 32, 32)

        :return
        -------
        latent : torch.Tensor
            Shape (B, latent_dim)
        """
        x = self.encoder_conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        latent = self.encoder_fc(x)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors back to image space.

        :param
        ----------
        latent : torch.Tensor
            Shape (B, latent_dim)

        :return
        -------
        recon : torch.Tensor
            Shape (B, 3, 32, 32)
        """
        x = self.decoder_fc(latent)
        x = x.reshape(x.size(0), 64, 4, 4)
        recon = self.decoder_conv(x)
        return recon

    def forward(self, x: torch.Tensor):
        """
        Full autoencoder forward pass.

        :return
        -------
        latent : torch.Tensor
            Shape (B, latent_dim)
        recon : torch.Tensor
            Shape (B, 3, 32, 32)
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return latent, recon
