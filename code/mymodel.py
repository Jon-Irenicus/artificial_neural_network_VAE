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
    The stages are trained one after the other by mytrain.train_one_epoch.

    As in the paper, stage 2 is only the prior of stage 1: the reconstruction
    uses stage 1 alone, and only the generation goes through stage 2.
        encode(x)            = whiten( mu1(x) ) = w, the stage-1 posterior
                               mean in whitened coordinates
        decode(w), w from encode (or reparameterize of it):
                               decode1( unwhiten(w) )            reconstruction
        decode(u), any other tensor, e.g. u ~ N(0, I):
                               decode1( unwhiten( stage-2 decoder(u) ) )
                                                                 generation
    The notebook and util/result.py call the same decode for both, so encode
    marks the tensors it returns (attribute STAGE1_TAG, kept by
    reparameterize) and decode reads the mark. A new tensor computed from
    them (e.g. mu.clone(), mu.to(device) to another device) loses the mark
    and is decoded as a sample of the prior.

Skip connections of stage 1 (module constants, so that other scripts can
switch them, e.g. `mymodel.Z_SKIP = False` before creating the model)
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
    With both off, stage 1 is the plain convolutional VAE.

ENC_DROPOUT: dropout on the flattened encoder features, just before fc_mu and
fc_logvar (0: none). Active only in training mode: model.eval() (the
evaluation, the whitening statistics, the stage-2 inputs) turns it off.
"""

import torch
import torch.nn as nn

MODE_STAGE1, MODE_WHITEN, MODE_VAE = 0, 1, 2

# attribute set on the tensors returned by encode: decode then reconstructs
# with stage 1 alone (see the module docstring)
STAGE1_TAG = "_stage1_latent"


def _tag(t):
    setattr(t, STAGE1_TAG, True)
    return t


def _is_tagged(t):
    return getattr(t, STAGE1_TAG, False)

RESIDUAL = True     # residual blocks inside the encoder and inside the decoder
Z_SKIP = True       # z enters every decoder layer (generative skip model)
ENC_DROPOUT = 0   # dropout just before the encoder FC layers (0: none)
# initial log gamma of the stage-2 decoder. Not 0: gamma = 1 with w_hat = 0 is
# the N(0, I) fit of the whitened latents, a fixed point where stage 2 ignores u
# (posterior collapse) and every generated image decodes the same z.
STAGE2_LOG_GAMMA_INIT = -3.0


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
        self.log_gamma = nn.Parameter(torch.tensor(STAGE2_LOG_GAMMA_INIT))

    def encode(self, w):
        h = self.enc(w)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, u):
        return self.dec(u)


class VAE(nn.Module):
    def __init__(self, latent_dim=32, stage2_hidden=512, channels=(32, 64, 128, 256),
                 residual=None, z_skip=None, enc_dropout=None):
        """Define the layers of the encoder and of the decoder.

        The notebook creates the model with `VAE()`, so every argument you
        add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.

        channels: widths (c1, c2, c3, c4) of the four encoder convolutions
        (32x32 -> 16 -> 8 -> 4 -> 4); the decoder mirrors them. Each GroupNorm
        has 4 channels per group, so every width must be a multiple of 4.

        residual, z_skip: the skip connections of the module docstring
        (None: the module constants RESIDUAL and Z_SKIP).

        enc_dropout: dropout probability before fc_mu / fc_logvar (None: the
        module constant ENC_DROPOUT). It has no parameters, so a saved model
        loads whatever its value.
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
        self.enc_dropout = ENC_DROPOUT if enc_dropout is None else enc_dropout
        self.enc_drop = nn.Dropout(self.enc_dropout)
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

        # ---- stage 2
        self.s2 = Stage2(latent_dim, stage2_hidden)
        self.register_buffer("stage2_mode", torch.tensor(MODE_STAGE1))
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_chol", torch.eye(latent_dim))      # lower triangular L
        self.register_buffer("z_chol_inv", torch.eye(latent_dim))  # L^{-1}

    # ------------------------------------------------------------ stage 1

    def encode1(self, x):
        """Stage-1 encoder: (mu, logvar) of q(z|x)."""
        h = x
        for block in self.enc:
            h = block(h)
        h = self.enc_drop(h.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode1(self, z):
        """Stage-1 decoder: pixel-wise mean of p(x|z), in [0, 1]."""
        h = self.dec_fc(z)
        for block in self.dec:
            h = block(h, z)
        logits = self.dec_out(h)
        if self.dec_out_z is not None:
            logits = logits + self.dec_out_z(z)[:, :, None, None]
        return torch.sigmoid(logits)

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

        With whitening or stage 2 active, (mu, logvar) are those of the
        stage-1 posterior in whitened coordinates (see the module docstring);
        the returned tensors are marked so that decode reconstructs them with
        stage 1 alone.

        Args:
            x: images, shape (B, 1, 32, 32).

        Returns:
            (mu, logvar), each of shape (B, latent_dim).
        """
        mu, logvar = self.encode1(x)
        if int(self.stage2_mode) == MODE_STAGE1:
            return mu, logvar
        # the posterior covariance becomes L^-1 diag(var) L^-T, whose diagonal
        # is returned (only its mean w is used for evaluation)
        var_w = logvar.exp() @ (self.z_chol_inv ** 2).T
        return _tag(self.whiten(mu)), _tag(var_w.log())


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
        # a sample of the posterior of encode stays a stage-1 latent
        return _tag(z) if _is_tagged(mu) else z

    def decode(self, z):
        """Decoder: pixel-wise mean of p(x|z).

        With stage 2 active, z from encode (marked) is decoded by stage 1
        alone, any other z (a sample of the prior N(0, I)) goes through the
        stage-2 decoder first (see the module docstring).

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            x_hat, shape (B, 1, 32, 32), values in [0, 1].
        """
        mode = int(self.stage2_mode)
        if mode == MODE_VAE and not _is_tagged(z):
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
