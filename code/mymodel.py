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

Whitening of the latents
    The convolutional VAE x <-> z below (encode1 / decode1) is trained as
    usual. After the training, mytrain fits the mean m and the covariance
    L L^T of the latents of the training images, and the model switches to
    the whitened latent variable w:
        encode(x) = L^{-1} ( mu1(x) - m )
        decode(w) = decode1( m + L w )
    so that decoding w ~ N(0, I) lands where the decoder has seen data. The
    transform is exactly invertible, so the reconstruction is unchanged.
    The switch is a buffer (`whitened`, saved in the state_dict).

Skip connections (module constants, so that other scripts can switch them,
e.g. `mymodel.Z_SKIP = False` before creating the model)
    RESIDUAL  every resolution step of the encoder and of the decoder is a
              residual block, conv-GN-ReLU-conv-GN plus a shortcut (identity,
              or a 1x1 conv / 2x2 transposed conv when the shape changes),
              then ReLU. Off: one conv-GN-ReLU per step.
    Z_SKIP    generative skip connections (Dieng, Kim, Rush & Blei, "Avoiding
              Latent Variable Collapse with Generative Skip Models", AISTATS
              2019): z enters every layer of the decoder, h_l = f(W_l h_{l-1}
              + U_l z). For a convolution, U_l z is a Linear(z -> channels)
              added to every pixel, after the GroupNorm and before the ReLU;
              the output layer gets one too. The decoder cannot then lose z
              as it gets deeper, which keeps the latent variables in use.
    There is no connection from the encoder to the decoder (as in a U-Net):
    it would bypass z, and images could not be generated from z alone.
    With both off, the model is the plain convolutional VAE.
"""

import torch
import torch.nn as nn

RESIDUAL = True     # residual blocks inside the encoder and inside the decoder
Z_SKIP = True       # z enters every decoder layer (generative skip model)


def _norm(c):
    """GroupNorm with 4 channels per group."""
    return nn.GroupNorm(num_groups=c // 4, num_channels=c)


def _conv(c_in, c_out, resample):
    """3x3 convolution: resample "down" halves the size (stride 2), "up"
    doubles it (4x4 transposed convolution, stride 2), None keeps it."""
    if resample == "up":
        return nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False)
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=2 if resample == "down" else 1,
                     padding=1, bias=False)


class Block(nn.Module):
    """One resolution step, c_in -> c_out channels, with resample None,
    "down" or "up".

    residual False:  conv - GN (+ U z) - ReLU
    residual True:   conv - GN (+ U1 z) - ReLU - conv - GN (+ U2 z),
                     plus the shortcut, then ReLU
    z_dim > 0 adds the generative skip U z (decoder only)."""

    def __init__(self, c_in, c_out, resample=None, residual=True, z_dim=0):
        super().__init__()
        self.residual = residual
        self.conv1, self.norm1 = _conv(c_in, c_out, resample), _norm(c_out)
        self.z1 = nn.Linear(z_dim, c_out) if z_dim else None
        if residual:
            self.conv2, self.norm2 = _conv(c_out, c_out, None), _norm(c_out)
            self.z2 = nn.Linear(z_dim, c_out) if z_dim else None
            if c_in == c_out and resample is None:
                self.shortcut = nn.Identity()
            elif resample == "up":
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2, bias=False),
                    _norm(c_out))
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=1,
                              stride=2 if resample == "down" else 1, bias=False),
                    _norm(c_out))
        self.act = nn.ReLU()

    @staticmethod
    def _add_z(h, proj, z):
        return h if proj is None else h + proj(z)[:, :, None, None]

    def forward(self, x, z=None):
        h = self.act(self._add_z(self.norm1(self.conv1(x)), self.z1, z))
        if not self.residual:
            return h
        h = self._add_z(self.norm2(self.conv2(h)), self.z2, z)
        return self.act(h + self.shortcut(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=32, channels=(32, 64, 128, 256),
                 residual=None, z_skip=None):
        """Define the layers of the encoder and of the decoder.

        The notebook creates the model with `VAE()`, so every argument you
        add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.

        channels: widths (c1, c2, c3, c4) of the four encoder convolutions
        (32x32 -> 16 -> 8 -> 4 -> 4); the decoder mirrors them. Each GroupNorm
        has 4 channels per group, so every width must be a multiple of 4.

        residual, z_skip: the skip connections of the module docstring
        (None: the module constants RESIDUAL and Z_SKIP).
        """
        super().__init__()
        # TODO: set self.latent_dim and define the layers
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = channels

        self.residual = RESIDUAL if residual is None else residual
        self.z_skip = Z_SKIP if z_skip is None else z_skip
        res, zd = self.residual, (latent_dim if self.z_skip else 0)

        # encoder: 32x32 -> 16 -> 8 -> 4 -> 4
        self.enc = nn.ModuleList([
            Block(1, c1, "down", res),
            Block(c1, c2, "down", res),
            Block(c2, c3, "down", res),
            Block(c3, c4, None, res),
        ])
        self.fc_mu = nn.Linear(c4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(c4*4*4, latent_dim)

        # decoder: z -> 4x4 -> 8 -> 8 -> 16 -> 32, z also enters every block
        # and the output layer when z_skip
        self.dec_fc = nn.Sequential(nn.Linear(latent_dim, c4*4*4), nn.ReLU(),
                                    nn.Unflatten(1, (c4, 4, 4)))
        self.dec = nn.ModuleList([
            Block(c4, c3, "up", res, zd),
            Block(c3, c2, None, res, zd),
            Block(c2, c1, "up", res, zd),
        ])
        self.dec_out = nn.ConvTranspose2d(c1, 1, kernel_size=4, stride=2, padding=1)
        self.dec_out_z = nn.Linear(latent_dim, 1) if self.z_skip else None

        # ---- whitening of the latents (identity until set_whitening)
        self.register_buffer("whitened", torch.tensor(False))
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_chol", torch.eye(latent_dim))      # lower triangular L
        self.register_buffer("z_chol_inv", torch.eye(latent_dim))  # L^{-1}

    # ------------------------------------------------------------ VAE without whitening

    def encode1(self, x):
        """Encoder without whitening: (mu, logvar) of q(z|x)."""
        h = x
        for block in self.enc:
            h = block(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode1(self, z):
        """Decoder without whitening: pixel-wise mean of p(x|z), in [0, 1]."""
        h = self.dec_fc(z)
        for block in self.dec:
            h = block(h, z)
        logits = self.dec_out(h)
        if self.dec_out_z is not None:
            logits = logits + self.dec_out_z(z)[:, :, None, None]
        return torch.sigmoid(logits)

    # ------------------------------------------------------------ whitening

    def set_whitening(self, mean, cov):
        """Store the mean and the Cholesky factor of the covariance of the
        latents, and switch the model to the whitened latent variable.
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
        self.whitened.fill_(True)

    def whiten(self, z):
        """w = L^{-1} (z - m), row by row."""
        return (z - self.z_mean) @ self.z_chol_inv.T

    def unwhiten(self, w):
        """z = m + L w, row by row."""
        return self.z_mean + w @ self.z_chol.T

    # ------------------------------------------------------------ interface

    def encode(self, x):
        """Encoder: parameters of q(z|x) = N(z; mu, diag(exp(logvar))).

        When whitened, the latent variable is w (see the module docstring),
        and (mu, logvar) are those of q(w|x).

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (mu, logvar), each of shape (B, latent_dim).
        """
        mu, logvar = self.encode1(x)
        if not self.whitened:
            return mu, logvar
        # the posterior covariance becomes L^-1 diag(var) L^-T, whose diagonal
        # is returned (only its mean w is used for evaluation)
        var_w = logvar.exp() @ (self.z_chol_inv ** 2).T
        return self.whiten(mu), var_w.log()

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
        if self.whitened:
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
