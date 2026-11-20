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

Two-stage VAE (Dai & Wipf, "Diagnosing and Enhancing VAE Models", ICLR 2019)
    stage 1   the convolutional VAE x <-> z below (encode1 / decode1)
    stage 2   models the distribution of z, so that decoding z ~ N(0, I) lands
              where the stage-1 decoder has seen data:
                mode 1  whitening: z = m + L u, with m and L L^T the mean and
                        covariance of the stage-1 latents (exactly invertible,
                        so the reconstruction is unchanged)
                mode 2  whitening, then a small MLP VAE w <-> u on the
                        whitened latents w, with a Gaussian decoder of learned
                        variance gamma (Stage2 below)
    The mode is a buffer (saved in the state_dict); 0 means stage 1 only.
    encode / decode go through the active stages, so the notebook and
    util/result.py see one VAE whose latent variable is u:
        encode(x) = stage-2 encoder( whiten( mu1(x) ) )
        decode(u) = decode1( unwhiten( stage-2 decoder(u) ) )
    The stages are trained one after the other by mytrain.train_one_epoch.
"""

import torch
import torch.nn as nn

MODE_STAGE1, MODE_WHITEN, MODE_VAE = 0, 1, 2


class Stage2(nn.Module):
    """VAE of the whitened stage-1 latents w <-> u, both of dimension d.
    MLP encoder and decoder; the decoder is a Gaussian N(w_hat, gamma I) with a
    learned variance gamma = exp(log_gamma)."""

    def __init__(self, d, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden, d)
        self.fc_logvar = nn.Linear(hidden, d)
        self.dec = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )
        self.log_gamma = nn.Parameter(torch.zeros(()))

    def encode(self, w):
        h = self.enc(w)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, u):
        return self.dec(u)


class VAE(nn.Module):
    def __init__(self, latent_dim=32, stage2_hidden=512):
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

        # ---- stage 2
        self.s2 = Stage2(latent_dim, stage2_hidden)
        self.register_buffer("stage2_mode", torch.tensor(MODE_STAGE1))
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_chol", torch.eye(latent_dim))      # lower triangular L
        self.register_buffer("z_chol_inv", torch.eye(latent_dim))  # L^{-1}

    # ------------------------------------------------------------ stage 1

    def encode1(self, x):
        """Stage-1 encoder: (mu, logvar) of q(z|x)."""
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode1(self, z):
        """Stage-1 decoder: pixel-wise mean of p(x|z), in [0, 1]."""
        return torch.sigmoid(self.dec(z))

    # ------------------------------------------------------------ whitening

    def set_whitening(self, mean, cov, mode):
        """Store the mean and the Cholesky factor of the covariance of the
        stage-1 latents, and switch to `mode` (MODE_WHITEN or MODE_VAE).
        The factorization runs on the CPU in float64 (some devices, such as
        DirectML, have no linear-algebra routines); the forward pass then only
        needs matrix products."""
        d = self.latent_dim
        cov = cov.detach().cpu().double() + 1e-6 * torch.eye(d, dtype=torch.float64)
        chol = torch.linalg.cholesky(cov)
        chol_inv = torch.linalg.solve_triangular(chol, torch.eye(d, dtype=torch.float64), upper=False)
        self.z_mean.copy_(mean.detach().cpu().float())
        self.z_chol.copy_(chol.float())
        self.z_chol_inv.copy_(chol_inv.float())
        self.stage2_mode.fill_(mode)

    def whiten(self, z):
        """w = L^{-1} (z - m), row by row."""
        return (z - self.z_mean) @ self.z_chol_inv.T

    def unwhiten(self, w):
        """z = m + L w, row by row."""
        return self.z_mean + w @ self.z_chol.T

    # ------------------------------------------------------------ interface

    def encode(self, x):
        """Encoder: parameters of q(z|x) = N(z; mu, diag(exp(logvar))).

        With stage 2 active, the latent variable is u (see the module
        docstring), and (mu, logvar) are those of q(u|x).

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (mu, logvar), each of shape (B, latent_dim).
        """
        mu, logvar = self.encode1(x)
        mode = int(self.stage2_mode)
        if mode == MODE_STAGE1:
            return mu, logvar
        w = self.whiten(mu)
        if mode == MODE_VAE:
            return self.s2.encode(w)
        # whitening only: the posterior covariance becomes L^-1 diag(var) L^-T,
        # whose diagonal is returned (only its mean w is used for evaluation)
        var_w = logvar.exp() @ (self.z_chol_inv ** 2).T
        return w, var_w.log()


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
        mode = int(self.stage2_mode)
        if mode == MODE_VAE:
            z = self.s2.decode(z)
        if mode != MODE_STAGE1:
            z = self.unwhiten(z)
        return self.decode1(z)

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
