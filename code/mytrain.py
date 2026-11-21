"""Training of the VAE: loss function and one epoch of training.

TODO: complete `loss_function` and `train_one_epoch`.

Besides the loss and the epoch of training, this file holds the data
augmentation of the training images, applied batch by batch inside
`train_one_epoch` (the notebook gives a DataLoader without transform):

  AUGMENT   weak geometric and stroke transforms of the training images.
            The augmented image is both the input and the target.
  CORRUPT   denoising: the encoder sees a corrupted copy (Gaussian noise,
            salt-and-pepper pixels or a masked block), the loss compares the
            reconstruction with the clean (augmented) image.

All the transforms are pure torch, use the global torch RNG (reproducible
with torch.manual_seed) and run on the device of the batch. The settings are
module-level constants, so other scripts can switch them, e.g.
`mytrain.AUGMENT = False`.

Preview of the transforms:  python code/mytrain.py   (saves augment_preview.png)
"""

import math

import torch
import torch.nn.functional as F

from mymodel import MODE_STAGE1, MODE_VAE, MODE_WHITEN

# ------------------------------------------------------------------ settings

AUGMENT = True      # weak geometric / stroke augmentation of the training images
CORRUPT = False      # corrupt the encoder input only, reconstruct the clean image

# Probability that each transform is applied to an image (independently)
P_AFFINE = 0.5
P_PIECEWISE = 0.3
P_ELASTIC = 0.3
P_STROKE = 0.3
P_GAMMA = 0.2

# Magnitudes (each image draws uniformly up to these values)
ROTATE_DEG = 10.0       # rotation in [-10, 10] degrees
SCALE = 0.10            # zoom factor in [0.9, 1.1]
SHEAR = 0.15            # horizontal shear factor in [-0.15, 0.15]
TRANSLATE_PX = 2.0      # shift in [-2, 2] pixels along each axis
PIECEWISE_GRID = 4      # 4 x 4 cells: (4+1)^2 control points moved independently
PIECEWISE_PX = 1.0      # max shift of a control point, in pixels
ELASTIC_ALPHA = 10.0    # Simard et al. (2003) elastic distortion: scale of the
ELASTIC_SIGMA = 4.0     # smoothed random field (alpha) and its Gaussian width (sigma)
                        # alpha 10, sigma 4: about 0.6 pixel of displacement on average
STROKE_MAX = 0.5        # blend weight toward a 3x3 dilation (thicker) or erosion (thinner)
GAMMA_MAX = 1.3         # x ** gamma with gamma in [1/1.3, 1.3] (softer / sharper edges)

# Two-stage VAE (see code/mymodel.py). train_one_epoch trains stage 1 for
# STAGE1_EPOCHS epochs; at the end of the last of them it fits the whitening,
# then (STAGE2 = "vae") trains the stage-2 VAE in the following epochs, with
# stage 1 frozen. NUM_EPOCHS of code/myoptim.py adds the two.
STAGE2 = "vae"           # None (stage 1 only, no quotes), "whiten", or "vae" (whiten + stage-2 VAE)
STAGE1_EPOCHS = 188
STAGE2_EPOCHS = 130     # epochs of the stage-2 VAE (used only with STAGE2 = "vae")
STAGE2_AUGMENT = True   # stage-1 latents of augmented images (if AUGMENT) for stage 2
STAGE2_INPUT = "mean"   # latents for stage 2: "sample" z ~ q(z|x) or "mean" mu of q(z|x)
                        # "sample" collapses: the posterior noise (std ~0.6) makes the
                        # whitened latents almost N(0, I), so stage 2 has nothing to learn.
                        # "mean" also matches mymodel.encode, which whitens mu.
STAGE2_STAT_PASSES = 20 # passes over the training images to fit the whitening
STAGE2_REPEAT = 8       # each image is encoded this many times per stage-2 step
STAGE2_LOG_GAMMA_MIN = -12.0  # lower bound of log gamma of the stage-2 decoder

# EMA of the weights: an exponential moving average of the stage-1 weights is
# kept beside them, updated after every optimizer step with the decay
# min(EMA_DECAY, (1 + n) / (10 + n)) at step n (warm-up, so that the random
# initial weights are forgotten quickly). At the end of the last stage-1 epoch
# it replaces the weights, before the whitening is fitted, so the notebook
# saves, samples and evaluates the EMA weights. 0.995 averages about the last
# 200 steps (25 epochs of 8 batches). The learning rate is left as set by
# myoptim.get_optimizer.
EMA = True
EMA_DECAY = 0.995

# Denoising: probability that an image is corrupted, then one of three
# corruptions chosen uniformly
P_CORRUPT = 0.5
NOISE_STD_MAX = 0.3     # Gaussian noise, std in [0, 0.3]
PIXEL_MAX = 0.2         # salt-and-pepper: fraction of pixels set to 0 or 1, in [0, 0.2]
BLOCK_MIN_PX = 4        # masked block (set to 0): side in [4, 10] pixels,
BLOCK_MAX_PX = 10       # centered in the middle half of the image


# ------------------------------------------------------------------ augmentation

def _uniform(shape, low, high, device):
    return torch.rand(shape, device=device) * (high - low) + low


def _gaussian_blur(field, sigma):
    """Separable Gaussian blur of each channel of field (B, C, H, W)."""
    radius = min(int(math.ceil(3 * sigma)), min(field.shape[-2:]) - 1)
    t = torch.arange(-radius, radius + 1, dtype=field.dtype, device=field.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = field.size(1)
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    field = F.conv2d(F.pad(field, (radius, radius, 0, 0), mode="reflect"), kx, groups=c)
    field = F.conv2d(F.pad(field, (0, 0, radius, radius), mode="reflect"), ky, groups=c)
    return field


def _affine_theta(b, size, device):
    """Random rotation, zoom, shear and translation, as (B, 2, 3) matrices of
    affine_grid (normalized coordinates in [-1, 1])."""
    angle = _uniform(b, -1, 1, device) * math.radians(ROTATE_DEG)
    scale = _uniform(b, 1 - SCALE, 1 + SCALE, device)
    shear = _uniform(b, -SHEAR, SHEAR, device)
    shift = _uniform((b, 2), -TRANSLATE_PX, TRANSLATE_PX, device) * 2 / size
    cos, sin = angle.cos(), angle.sin()
    # rotation @ shear, times the zoom
    a = torch.stack([
        torch.stack([cos, cos * shear - sin], dim=1),
        torch.stack([sin, sin * shear + cos], dim=1),
    ], dim=1) * scale.view(b, 1, 1)
    return torch.cat([a, shift.unsqueeze(2)], dim=2)


def _piecewise_field(b, h, w, device):
    """Displacement field (B, 2, H, W), in normalized coordinates, of a coarse
    grid of control points moved at random and interpolated bilinearly in
    between (a piecewise warp, like a piecewise affine transform)."""
    g = PIECEWISE_GRID + 1
    ctrl = _uniform((b, 2, g, g), -PIECEWISE_PX, PIECEWISE_PX, device) * 2 / w
    return F.interpolate(ctrl, size=(h, w), mode="bilinear", align_corners=True)


def _elastic_field(b, h, w, device):
    """Displacement field (B, 2, H, W), in normalized coordinates, of the
    elastic distortion of Simard et al. (2003): a uniform random field in
    [-1, 1] smoothed by a Gaussian of width sigma, times alpha pixels."""
    field = _uniform((b, 2, h, w), -1, 1, device)
    return _gaussian_blur(field, ELASTIC_SIGMA) * ELASTIC_ALPHA * 2 / w


def augment(x):
    """Weak random transforms of a batch x (B, 1, H, W) in [0, 1], each
    applied to an image with its own probability:

      geometric   affine (rotation, zoom, shear, shift), piecewise warp and
                  elastic distortion, combined into one sampling grid so that
                  the image is interpolated only once (bicubic)
      stroke      thickness: blend toward a 3x3 dilation or erosion
      intensity   gamma, which makes the anti-aliased edges softer or sharper

    Returns a new batch of the same shape, in [0, 1].
    """
    b, _, h, w = x.shape
    dev = x.device

    use_affine = torch.rand(b, device=dev) < P_AFFINE
    use_piecewise = torch.rand(b, device=dev) < P_PIECEWISE
    use_elastic = torch.rand(b, device=dev) < P_ELASTIC
    use_warp = use_affine | use_piecewise | use_elastic

    if use_warp.any():
        identity = torch.eye(2, 3, device=dev).expand(b, 2, 3)
        theta = torch.where(use_affine.view(b, 1, 1), _affine_theta(b, w, dev), identity)
        grid = F.affine_grid(theta, (b, 1, h, w), align_corners=False)
        disp = torch.zeros(b, 2, h, w, device=dev)
        if use_piecewise.any():
            disp = disp + _piecewise_field(b, h, w, dev) * use_piecewise.view(b, 1, 1, 1)
        if use_elastic.any():
            disp = disp + _elastic_field(b, h, w, dev) * use_elastic.view(b, 1, 1, 1)
        grid = grid + disp.permute(0, 2, 3, 1)
        # bicubic blurs less than bilinear: the VAE would also learn the blur
        warped = F.grid_sample(x, grid, mode="bicubic", padding_mode="zeros",
                               align_corners=False)
        x = torch.where(use_warp.view(b, 1, 1, 1), warped, x)

    use_stroke = torch.rand(b, device=dev) < P_STROKE
    if use_stroke.any():
        t = _uniform(b, -STROKE_MAX, STROKE_MAX, dev).view(b, 1, 1, 1)
        thick = F.max_pool2d(x, 3, stride=1, padding=1)
        thin = -F.max_pool2d(-x, 3, stride=1, padding=1)
        stroked = torch.where(t > 0, x + t * (thick - x), x - t * (thin - x))
        x = torch.where(use_stroke.view(b, 1, 1, 1), stroked, x)

    use_gamma = torch.rand(b, device=dev) < P_GAMMA
    if use_gamma.any():
        log_g = math.log(GAMMA_MAX)
        gamma = torch.exp(_uniform(b, -log_g, log_g, dev)).view(b, 1, 1, 1)
        x = torch.where(use_gamma.view(b, 1, 1, 1), x.clamp_min(0) ** gamma, x)

    return x.clamp(0, 1)


def corrupt(x):
    """Corrupted copy of a batch x (B, 1, H, W) in [0, 1] for the encoder
    input. With probability P_CORRUPT an image gets one of: Gaussian noise,
    salt-and-pepper pixels, or a masked block; the others are unchanged (the
    evaluation encodes clean images, so the encoder keeps seeing them too)."""
    b, _, h, w = x.shape
    dev = x.device

    # Gaussian noise
    std = _uniform(b, 0, NOISE_STD_MAX, dev).view(b, 1, 1, 1)
    noisy = (x + std * torch.randn_like(x)).clamp(0, 1)

    # salt and pepper
    frac = _uniform(b, 0, PIXEL_MAX, dev).view(b, 1, 1, 1)
    hit = torch.rand_like(x) < frac
    salted = torch.where(hit, (torch.rand_like(x) < 0.5).to(x.dtype), x)

    # masked block, centered in the middle half of the image
    side = torch.randint(BLOCK_MIN_PX, BLOCK_MAX_PX + 1, (b, 2), device=dev)
    center = torch.stack([torch.randint(h // 4, 3 * h // 4, (b,), device=dev),
                          torch.randint(w // 4, 3 * w // 4, (b,), device=dev)], dim=1)
    top_left = center - side // 2
    rows = torch.arange(h, device=dev).view(1, h, 1)
    cols = torch.arange(w, device=dev).view(1, 1, w)
    inside = ((rows >= top_left[:, 0].view(b, 1, 1)) & (rows < (top_left[:, 0] + side[:, 0]).view(b, 1, 1))
              & (cols >= top_left[:, 1].view(b, 1, 1)) & (cols < (top_left[:, 1] + side[:, 1]).view(b, 1, 1)))
    masked = x.masked_fill(inside.unsqueeze(1), 0.0)

    kind = torch.randint(0, 3, (b,), device=dev).view(b, 1, 1, 1)
    out = torch.where(kind == 0, noisy, torch.where(kind == 1, salted, masked))
    use = (torch.rand(b, device=dev) < P_CORRUPT).view(b, 1, 1, 1)
    return torch.where(use, out, x)


def make_batch(x):
    """(encoder input, reconstruction target) for a batch of training images,
    with AUGMENT and CORRUPT as set above."""
    if AUGMENT:
        x = augment(x)
    x_in = corrupt(x) if CORRUPT else x
    return x_in, x


# ------------------------------------------------------------------ training

def loss_function(x, x_hat, mu, logvar):
    """VAE loss (negative ELBO) of a batch, averaged over its images.

        loss  = recon + kl

        recon : reconstruction loss, the binary cross-entropy between x_hat
                and x, summed over the pixels of an image
        kl    : KL divergence KL( q(z|x) || p(z) ) with the prior
                p(z) = N(0, I), summed over the latent dimensions

    Both terms are averaged over the B images of the batch.

    Args:
        x:      images,                  shape (B, 1, 32, 32), values in [0, 1].
        x_hat:  decoded images,          shape (B, 1, 32, 32), values in [0, 1].
        mu:     mean of q(z|x),          shape (B, latent_dim).
        logvar: log-variance of q(z|x),  shape (B, latent_dim).

    Returns:
        (loss, recon, kl): three scalar tensors.
    """
    # TODO
    B = x.size(0)
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / B
    kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()
    loss = recon + kl
    return loss, recon, kl


def train_one_epoch(model, optimizer, dataloader, device, kl_weight=1.0):
    """Train `model` for one epoch, i.e. one pass over `dataloader`.

    For every mini-batch: move the images to `device`, run the model,
    compute the loss with `loss_function`, back-propagate, and update the
    parameters with `optimizer`. Put the model in training mode first.

    Args:
        model:      the VAE, already on `device`.
        optimizer:  the optimizer from myoptim.get_optimizer.
        dataloader: yields batches of images, shape (B, 1, 32, 32), values
                    in [0, 1] (there are no labels).
        device:     torch.device to compute on.
        kl_weight:  weight of the KL term in the loss that is back-propagated
                    (KL warm-up of compare_configs.py; the returned loss is
                    always recon + kl).

    Returns:
        dict {"loss": ..., "recon": ..., "kl": ...} of Python floats: the
        average of each term per image over the whole epoch.
    """
    # TODO
    if STAGE2 not in (None, "whiten", "vae"):
        raise ValueError(f'mytrain.STAGE2 = {STAGE2!r}: use None (no quotes), "whiten" or "vae"')
    model.train()
    model.epochs_trained = getattr(model, "epochs_trained", 0) + 1
    stage1 = int(model.stage2_mode) != MODE_VAE
    acc_loss = acc_recon = acc_kl = 0

    for x in dataloader:
        b = x.size(0)
        if not stage1:
            loss, recon, kl = _stage2_step(model, optimizer, x, device)
        else:
            # augmentation on the CPU batch: cheap at 32 x 32, and the same on every device
            x_in, x = make_batch(x)
            x_in, x = x_in.to(device), x.to(device)
            mu, logvar = model.encode1(x_in)
            x_hat = model.decode1(model.reparameterize(mu, logvar))
            loss, recon, kl = loss_function(x, x_hat, mu, logvar)
            optimizer.zero_grad()
            (loss if kl_weight == 1.0 else recon + kl_weight * kl).backward()
            optimizer.step()
            ema_update(model)
        acc_loss += loss.item() * b
        acc_recon += recon.item() * b
        acc_kl += kl.item() * b

    # end of stage 1: EMA weights into the model, then switch to stage 2
    if model.epochs_trained == STAGE1_EPOCHS and stage1:
        ema_apply(model)
        if STAGE2 and int(model.stage2_mode) == MODE_STAGE1:
            start_stage2(model, dataloader, device, STAGE2)

    n = len(dataloader.dataset)
    losses = {"loss": acc_loss / n, "recon": acc_recon / n, "kl": acc_kl / n}
    return losses


# ------------------------------------------------------------------ EMA of the weights

def _stage1_params(model):
    return [(n, p) for n, p in model.named_parameters() if not n.startswith("s2.")]


@torch.no_grad()
def ema_update(model):
    """After an optimizer step of stage 1 (EMA): update the EMA of the
    stage-1 weights, kept in model.ema (not in the state_dict)."""
    if not EMA or getattr(model, "ema_applied", False):
        return
    if getattr(model, "ema", None) is None:
        model.ema, model.ema_steps = {n: p.detach().clone() for n, p in _stage1_params(model)}, 0
    model.ema_steps += 1
    beta = min(EMA_DECAY, (1 + model.ema_steps) / (10 + model.ema_steps))
    for n, p in _stage1_params(model):
        model.ema[n].mul_(beta).add_(p.detach(), alpha=1 - beta)


@torch.no_grad()
def _swap_ema(model):
    for n, p in _stage1_params(model):
        tmp = p.detach().clone()
        p.copy_(model.ema[n])
        model.ema[n].copy_(tmp)


def ema_apply(model):
    """Put the EMA weights into the model for good (end of stage 1)."""
    if getattr(model, "ema", None) is not None and not getattr(model, "ema_applied", False):
        _swap_ema(model)
    ema_discard(model)


def ema_discard(model):
    """Forget the EMA: the weights of the model are final (also after loading
    weights that already are EMA weights, as compare_configs.py does)."""
    model.ema, model.ema_applied = None, True


class ema_weights:
    """Context manager: the model has its EMA weights inside the block, its
    training weights again after it (to measure during the training). Does
    nothing without an EMA."""

    def __init__(self, model):
        self.model = model
        self.active = getattr(model, "ema", None) is not None

    def __enter__(self):
        if self.active:
            _swap_ema(self.model)
        return self.model

    def __exit__(self, *exc):
        if self.active:
            _swap_ema(self.model)
        return False


# ------------------------------------------------------------------ stage 2

def _stage2_images(x):
    """Training images whose stage-1 latents are used by stage 2."""
    return augment(x) if (AUGMENT and STAGE2_AUGMENT) else x


def _stage1_latents(model, x):
    """Stage-1 latents of a batch: z ~ q(z|x), or its mean (STAGE2_INPUT)."""
    mu, logvar = model.encode1(x)
    return model.reparameterize(mu, logvar) if STAGE2_INPUT == "sample" else mu


@torch.no_grad()
def start_stage2(model, dataloader, device, kind="vae"):
    """Fit the whitening to the stage-1 latents of the training images
    (STAGE2_STAT_PASSES passes, with fresh augmentation and z samples each
    time) and switch the model to stage 2: kind "whiten" or "vae".

    All its random numbers (shuffling, augmentation, z samples) come from the
    CPU generator, so a caller can keep its training unchanged by wrapping it
    in torch.random.fork_rng(devices=[])."""
    if kind not in ("whiten", "vae"):
        raise ValueError(f'start_stage2: kind = {kind!r}, expected "whiten" or "vae"')
    model.eval()
    zs = []
    for _ in range(STAGE2_STAT_PASSES):
        for x in dataloader:
            mu, logvar = model.encode1(_stage2_images(x).to(device))
            mu, logvar = mu.cpu(), logvar.cpu()
            if STAGE2_INPUT == "sample":
                mu = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            zs.append(mu)
    z = torch.cat(zs).double()
    mean = z.mean(0)
    cov = torch.cov(z.T)
    model.set_whitening(mean, cov, MODE_VAE if kind == "vae" else MODE_WHITEN)
    model.train()


def stage2_loss(w, w_hat, mu, logvar, log_gamma):
    """Negative ELBO of the stage-2 VAE, averaged over the batch, with the
    Gaussian decoder N(w_hat, gamma I), gamma = exp(log_gamma) learned
    (the constant d/2 log(2 pi) is left out)."""
    d = w.size(1)
    recon = (((w - w_hat) ** 2).sum(1) / (2 * log_gamma.exp())
             + 0.5 * d * log_gamma).mean()
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1).mean()
    return recon + kl, recon, kl


def _stage2_step(model, optimizer, x, device):
    """One update of the stage-2 VAE on a batch of training images, encoded
    STAGE2_REPEAT times by the frozen stage 1."""
    x = x.repeat(STAGE2_REPEAT, 1, 1, 1)
    model.eval()            # frozen stage 1 without dropout, as at evaluation
    with torch.no_grad():
        w = model.whiten(_stage1_latents(model, _stage2_images(x).to(device)))
    model.train()
    mu, logvar = model.s2.encode(w)
    w_hat = model.s2.decode(model.reparameterize(mu, logvar))
    log_gamma = model.s2.log_gamma.clamp(min=STAGE2_LOG_GAMMA_MIN)
    loss, recon, kl = stage2_loss(w, w_hat, mu, logvar, log_gamma)
    optimizer.zero_grad(set_to_none=True)   # stage-1 parameters keep grad None:
    loss.backward()                         # AdamW skips them (no weight decay)
    optimizer.step()
    return loss, recon, kl


# ------------------------------------------------------------------ preview

if __name__ == "__main__":
    # python code/mytrain.py : saves augment_preview.png in the project folder
    import sys
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "util"))
    from data import get_dataset

    torch.manual_seed(0)
    x = get_dataset("train").images[::50][:10]          # 10 images, 2 of each digit

    disp = _elastic_field(256, 32, 32, x.device) * 32 / 2
    print(f"elastic displacement: mean {disp.norm(dim=1).mean():.2f} px, "
          f"max {disp.norm(dim=1).max():.2f} px")

    rows = [("original", x)]
    for i in range(3):
        rows.append((f"augmented {i + 1}", augment(x)))
    aug = augment(x)
    saved = P_CORRUPT
    P_CORRUPT = 1.0                                       # corrupt every image for the preview
    for i in range(2):
        rows.append((f"corrupted input {i + 1}", corrupt(aug)))
    P_CORRUPT = saved
    rows.append(("its target", aug))

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.25 * len(rows)),
                             constrained_layout=True)
    for ax, (title, images) in zip(axes, rows):
        ax.imshow(images.squeeze(1).permute(1, 0, 2).reshape(32, -1),
                  cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, loc="left")
        ax.axis("off")
    path = root / "augment_preview.png"
    fig.savefig(path, dpi=130)
    print(f"saved {path}")
