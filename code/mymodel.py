"""Variational autoencoder (VAE).

TODO: complete the class `VAE`.

You are free to design the architecture (type and number of layers, their
sizes, the latent dimension), but keep the interface described in the
docstrings: the notebook and the other files rely on it.

Notation
    B      : number of images in a batch
    x      : images,                   shape (B, 1, 32, 32), values in [0, 1]
    z      : latent variables,         shape (B, latent_dim)
    mu     : mean of q(z|x),           shape (B, latent_dim)
    logvar : log-variance of q(z|x),   shape (B, latent_dim)
    x_hat  : decoded images,           shape (B, 1, 32, 32), values in [0, 1]
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        """Define the layers of the encoder and of the decoder.

        The notebook creates the model with `VAE()`, so every argument you
        add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.
        """
        super().__init__()
        # TODO: set self.latent_dim and define the layers
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=2, num_channels=8), nn.ReLU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=4, num_channels=16), nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=8, num_channels=32), nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=16, num_channels=64), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64*4*4, latent_dim)
        self.fc_logvar = nn.Linear(64*4*4, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64*4*4), nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=8, num_channels=32), nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=4, num_channels=16), nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ), nn.GroupNorm(num_groups=2, num_channels=8), nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )


    def encode(self, x):
        """Encoder: parameters of q(z|x) = N(z; mu, diag(exp(logvar))).

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (mu, logvar), each of shape (B, latent_dim).
        """
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)
        

    def reparameterize(self, mu, logvar):
        """Draw z ~ N(mu, diag(exp(logvar))) with the reparameterization
        trick, so that gradients can flow back to mu and logvar.

        Args:
            mu, logvar: shape (B, latent_dim).

        Returns:
            z, shape (B, latent_dim).
        """
        # TODO
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return z

    def decode(self, z):
        """Decoder: pixel-wise mean of p(x|z).

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            x_hat, shape (B, 1, 32, 32), values in [0, 1].
        """
        # TODO
        return torch.sigmoid(self.dec(z))

    def forward(self, x):
        """Encode x, draw z from q(z|x), and decode z.

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (x_hat, mu, logvar).
        """
        # TODO
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
