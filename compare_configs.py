"""Compare VAE settings by FID and reconstruction error.

For every combination of latent dimension and KL warm-up, a VAE is trained
with the code in the `code` folder (mymodel, myoptim, mytrain) and measured.

When it is measured:
  periodic      every `--eval-every` epochs, to compare numbers of epochs.
  early stop    with `--early-stop`, the validation loss (recon + kl on the
                held-out training images) is computed after every epoch, and
                training stops when it has not improved by more than
                `--min-delta` for `--patience` epochs. The weights of the best
                epoch are restored and measured. Each setting thus gets its
                own best epoch. During a KL warm-up the validation loss is
                naturally high, so the best epoch is searched only after it.

KL warm-up (KL annealing): the weight of the KL term in the loss that is
back-propagated grows linearly from 0 at epoch 1 to 1 at epoch `warmup + 1`,
and stays 1 afterwards. `warmup 0` is the plain loss recon + kl. Both are
trained with `train_one_epoch` of code/mytrain.py (the warm-up through its
argument kl_weight, which the notebook does not use).

EMA of the weights (--ema, default EMA of code/mytrain.py): the model is
measured with its EMA weights (periodic measures, validation loss, best
epoch of the early stop); after the last epoch they replace the weights.

Stage-2 VAE (--stage2-epochs N, default 0 = off): after the stage-1 training
of a setting (its best epoch with --early-stop, else its last epoch), the
whitening is fitted and the stage-2 VAE of code/mymodel.py is trained on top
of the frozen stage 1 for up to N epochs, with train_one_epoch as in the
notebook (STAGE2_INPUT, STAGE2_REPEAT, ... of code/mytrain.py), and a fresh
optimizer from myoptim.get_optimizer. It is measured (rows with stage2
"vae", their epoch is the stage-2 epoch and stage1_epoch the epoch of the
stage 1 below it):
  periodic      every `--stage2-eval-every` stage-2 epochs (default 50).
  early stop    with --early-stop, the stage-2 validation loss (negative
                ELBO of the stage-2 VAE on the whitened latents of the
                held-out images) is computed after every stage-2 epoch;
                `--stage2-patience` and `--stage2-min-delta` play the roles
                of --patience and --min-delta. The best stage-2 weights are
                restored and measured.
As in util/result.py, the reconstruction uses stage 1 alone (code/mymodel.py:
decode reconstructs the tensors of encode without stage 2), so the
reconstruction error of the vae rows is that of the stage 1 below them; only
the FID depends on stage 2. The best epoch of the stage-2 validation loss is
not always the one of the lowest FID: look at the periodic measures too.

Data:
  default   no test image is used, as the README asks for the training.
            The model is trained on (500 - val_size) training images.
            Reconstruction error and early stopping: on the val_size
            held-out training images.
            FID: generated images against all 500 training images.
            The values differ from those of the notebook (fewer images give a
            larger FID), so compare them with each other, not with the
            notebook.

  --test    measured on the 5,000 test images like util/result.py (with
            --n-gen 5000 the FID is computed the same way). Trained on all 500
            training images, or on (500 - val_size) with --early-stop, which
            still stops on the held-out training images. Choosing the settings
            on the test images makes the final result optimistic: use it only
            if that is acceptable for the assignment.

Device (--device, default auto):
  auto      cuda (NVIDIA, or AMD with a ROCm build of PyTorch), then mps
            (Apple silicon), then DirectML (any DirectX 12 GPU on Windows,
            with the package torch-directml), then the CPU.
  cuda      NVIDIA GPU, or AMD GPU with a ROCm build of PyTorch ("rocm" is
            accepted as an alias: ROCm builds use the device name cuda).
  dml       DirectML (pip install torch-directml).
  mps, cpu
The Inception-v3 network of the FID runs on the device. The images are
enlarged to 299 x 299 before it on the CPU (the bilinear interpolation of
DirectML gives different values), and the statistics of the features and the
Frechet distance are computed on the CPU (DirectML has no eigenvalue routine)
with the float32 formulas of torcheval.

Whitening (--whiten, default both): the second stage of code/mymodel.py, which
decodes z = m + L u with u ~ N(0, I), m and L L^T the mean and covariance of
the latents of the training images (mytrain.start_stage2). At every
measurement the model is measured as trained ("none") and with the whitening
fitted to its training images ("whiten"), then switched back, so that its
training goes on unchanged (the random numbers of the fit are not taken from
those of the training). The reconstruction error is the same for both: the
whitening is exactly invertible. Whatever STAGE2 of code/mytrain.py says, the
model is never switched during the training.

Seeds (--seeds, default 0): every setting is trained once per seed, and the
values are averaged over the seeds (mean +- standard deviation), so that a
setting is not judged on one lucky or unlucky run. The seed sets the initial
weights, the order of the mini-batches and, when training images are held
out, which ones. All the settings use the same seeds, so they are compared on
the same held-out images. The seeds are the outer loop: if the run is
interrupted, the finished seeds have been run for every setting.

Examples:
    python compare_configs.py --early-stop
    python compare_configs.py --early-stop --seeds 0 1 2
    python compare_configs.py --early-stop --latent-dims 4 8 16 32 --patience 40
    python compare_configs.py --latent-dims 4 8 16 --warmups 0 20 50
    python compare_configs.py --epochs 300 --eval-every 25 --n-gen 1000
    python compare_configs.py --early-stop --device dml
    python compare_configs.py --early-stop --aug off --corrupt off   # baseline
    python compare_configs.py --early-stop --aug on  --corrupt off
    python compare_configs.py --early-stop --aug both --corrupt both     # 4 combinations
    python compare_configs.py --test --latent-dims 32 48 64 --warmups 0 --epochs 268 --n-gen 5000 --seeds 0 1 2
    python compare_configs.py --test --latent-dims 32 --warmups 0 --epochs 268 --residual off --z-skip off
    python compare_configs.py --early-stop --latent-dims 64 --warmups 0 --stage2-epochs 600
    python compare_configs.py --test --latent-dims 64 --warmups 0 --epochs 239 --stage2-epochs 300 --stage2-eval-every 25

Augmentation (AUGMENT) and denoising input corruption (CORRUPT) are those of
code/mytrain.py, in both training paths (with and without warm-up); --aug and
--corrupt override them for the run, and with "both" each setting is trained
twice, without and with (like --latent-dims and --warmups, they multiply the
number of settings). Compare runs with the same --seeds: they
use the same held-out images. The validation loss and the measures always use
clean, unaugmented images.

Output: a table of the means over the seeds in the terminal, and in the
folder `compare/`
    run_<time>.csv        one line per measurement (per seed)
    run_<time>_mean.csv   mean and standard deviation over the seeds
    run_<time>.png        FID against reconstruction error, both against
                          epoch (means, with the standard deviation as error
                          bars), and the validation loss with --early-stop;
                          with --stage2-epochs, a second row of the same
                          plots against the stage-2 epoch
"""

import argparse
import csv
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
for folder in ("code", "util"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

from torcheval.metrics.image.fid import FIDInceptionV3  # noqa: E402

import mytrain                                          # noqa: E402
from data import get_dataset                            # noqa: E402
import mymodel                                          # noqa: E402
from mymodel import MODE_STAGE1, MODE_VAE, VAE          # noqa: E402
from myoptim import BATCH_SIZE, get_optimizer           # noqa: E402
from mytrain import (ema_apply, ema_discard, ema_weights, loss_function,  # noqa: E402
                     start_stage2, stage2_loss, train_one_epoch)
from result import _to_rgb                              # noqa: E402

FID_BATCH = 50      # images given to Inception-v3 at a time
Z_SEED = 0          # seed of the latent variables of the generated images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latent-dims", type=int, nargs="+", default=[8, 16, 32],
                   help="latent dimensions to compare (default 8 16 32)")
    p.add_argument("--warmups", type=int, nargs="+", default=[0, 30],
                   help="KL warm-up lengths in epochs, 0 = off (default 0 30)")
    p.add_argument("--epochs", type=int, default=None,
                   help="epochs of training of each setting; with --early-stop the "
                        "maximum (default 200, or 1000 with --early-stop)")
    p.add_argument("--eval-every", type=int, default=None,
                   help="measure every this many epochs, 0 = never (default 50, "
                        "or 0 with --early-stop)")
    p.add_argument("--early-stop", action="store_true",
                   help="stop each setting at its own best epoch of the validation loss")
    p.add_argument("--patience", type=int, default=30,
                   help="early stop: epochs without improvement before stopping (default 30)")
    p.add_argument("--min-delta", type=float, default=0.1,
                   help="early stop: smallest decrease of the validation loss that "
                        "counts as an improvement (default 0.1)")
    p.add_argument("--n-gen", type=int, default=500,
                   help="generated images per FID (default 500)")
    p.add_argument("--val-size", type=int, default=100,
                   help="held-out training images (default 100)")
    p.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0],
                   help="seeds of the split and of the training; every setting is "
                        "trained once per seed and the results are averaged (default 0)")
    p.add_argument("--test", action="store_true",
                   help="measure on the test images")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "rocm", "dml", "mps", "cpu"],
                   help="device to compute on (default auto)")
    p.add_argument("--aug", choices=["on", "off", "both"], default=None,
                   help="augmentation of the training images; both: a model trained "
                        "without and one with (default: AUGMENT of code/mytrain.py)")
    p.add_argument("--corrupt", choices=["on", "off", "both"], default=None,
                   help="denoising input corruption; both: a model trained without and "
                        "one with (default: CORRUPT of code/mytrain.py)")
    p.add_argument("--residual", choices=["on", "off"], default=None,
                   help="residual blocks in the encoder and decoder (default: RESIDUAL "
                        "of code/mymodel.py)")
    p.add_argument("--z-skip", choices=["on", "off"], default=None,
                   help="z into every decoder layer, generative skip model (default: "
                        "Z_SKIP of code/mymodel.py)")
    p.add_argument("--enc-dropout", type=float, default=None, metavar="P",
                   help="dropout just before the encoder FC layers, 0 for none "
                        "(default: ENC_DROPOUT of code/mymodel.py)")
    p.add_argument("--ema", choices=["on", "off"], default=None,
                   help="EMA of the weights (default: EMA of code/mytrain.py)")
    p.add_argument("--whiten", choices=["off", "on", "both"], default="both",
                   help="measure each model without (off), with (on) or both without "
                        "and with (both) the whitening of its latents (default both)")
    p.add_argument("--stage2-epochs", type=int, default=0, metavar="N",
                   help="train the stage-2 VAE for up to N epochs after the stage 1 of "
                        "every setting, 0 = off (default 0)")
    p.add_argument("--stage2-eval-every", type=int, default=50, metavar="K",
                   help="measure the stage-2 VAE every K stage-2 epochs, 0 = never "
                        "(default 50)")
    p.add_argument("--stage2-patience", type=int, default=30,
                   help="early stop of stage 2: epochs without improvement (default 30)")
    p.add_argument("--stage2-min-delta", type=float, default=0.1,
                   help="early stop of stage 2: smallest decrease of the stage-2 "
                        "validation loss that counts (default 0.1)")
    args = p.parse_args()
    if args.stage2_epochs < 0 or args.stage2_eval_every < 0:
        p.error("--stage2-epochs and --stage2-eval-every must be >= 0")
    # the whitening and stage 2 are started here, never by train_one_epoch
    # (which would switch the model at STAGE1_EPOCHS)
    mytrain.STAGE2 = None
    args.stage2_kinds = {"off": ["none"], "on": ["whiten"], "both": ["none", "whiten"]}[args.whiten]
    # values of AUGMENT and CORRUPT to train with (set per setting in main)
    def values(option, default):
        return {"on": [True], "off": [False], "both": [False, True], None: [default]}[option]
    args.aug_values = values(args.aug, mytrain.AUGMENT)
    args.corrupt_values = values(args.corrupt, mytrain.CORRUPT)
    if args.residual is not None:
        mymodel.RESIDUAL = args.residual == "on"
    if args.z_skip is not None:
        mymodel.Z_SKIP = args.z_skip == "on"
    if args.enc_dropout is not None:
        if not 0 <= args.enc_dropout < 1:
            p.error("--enc-dropout must be in [0, 1)")
        mymodel.ENC_DROPOUT = args.enc_dropout
    if args.ema is not None:
        mytrain.EMA = args.ema == "on"
    args.seeds = list(dict.fromkeys(args.seeds))    # without repeats, in order
    if args.epochs is None:
        args.epochs = 1000 if args.early_stop else 200
    # EMA: its weights replace the weights after the last epoch (and are
    # swapped in for every measure before)
    mytrain.STAGE1_EPOCHS = args.epochs
    if args.eval_every is None:
        args.eval_every = 0 if args.early_stop else 50
    return args


# ---------------------------------------------------------------- device

def _cuda_device():
    name = torch.cuda.get_device_name(0)
    kind = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
    return torch.device("cuda"), f"cuda ({kind}, {name})"


def _dml_device():
    import torch_directml
    return torch_directml.device(), f"DirectML ({torch_directml.device_name(0)})"


def _has_dml():
    try:
        import torch_directml
        return torch_directml.is_available()
    except ImportError:
        return False


def get_device(name):
    """Return (device, description) for --device `name`."""
    if name == "auto":
        if torch.cuda.is_available():
            return _cuda_device()
        if torch.backends.mps.is_available():
            return torch.device("mps"), "mps (Apple silicon)"
        if _has_dml():
            return _dml_device()
        return torch.device("cpu"), "cpu"
    if name in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            sys.exit(f"--device {name}: torch.cuda.is_available() is False. This PyTorch "
                     f"({torch.__version__}) has no CUDA/ROCm support or sees no GPU.")
        return _cuda_device()
    if name == "dml":
        if not _has_dml():
            sys.exit("--device dml: the package torch-directml is not installed or sees "
                     "no GPU (pip install torch-directml).")
        return _dml_device()
    if name == "mps":
        if not torch.backends.mps.is_available():
            sys.exit("--device mps: not available")
        return torch.device("mps"), "mps (Apple silicon)"
    return torch.device("cpu"), "cpu"


# ---------------------------------------------------------------- measures

class FID:
    """FID against a fixed set of real images, computed like torcheval's
    FrechetInceptionDistance (used by util/result.py), in float32 like it, so
    that the values match those of the notebook. The features of the real
    images are computed once."""

    def __init__(self, real_images, device):
        self.device = device
        self.model = FIDInceptionV3().to(device).eval().requires_grad_(False)
        self.real_mean, self.real_cov = self._stats(self._features(real_images))

    @torch.no_grad()
    def _features(self, images):
        # FIDInceptionV3.forward enlarges the images to 299 x 299 (bilinear)
        # and runs the network. The enlargement is done here on the CPU, because
        # the bilinear interpolation of DirectML gives different values (on the
        # same model: FID 43.4 instead of 39.3, even the wrong order between
        # models); only the network runs on the device.
        feats = []
        for x in images.split(FID_BATCH):
            x = F.interpolate(_to_rgb(x.cpu()), size=(299, 299), mode="bilinear",
                              align_corners=False)
            feats.append(self.model.model(x.to(self.device)).cpu().float())
        return torch.cat(feats)

    @staticmethod
    def _stats(feats):
        # the formulas of torcheval: from the sum and the sum of products
        n = feats.shape[0]
        mean = (feats.sum(dim=0) / n).unsqueeze(0)
        cov = (feats.T @ feats - n * mean.T @ mean) / (n - 1)
        return mean.squeeze(0), cov

    @torch.no_grad()
    def __call__(self, model, n_gen):
        model.eval()
        z = torch.randn(n_gen, model.latent_dim,
                        generator=torch.Generator().manual_seed(Z_SEED))
        images = torch.cat([model.decode(zb.to(self.device)).cpu()
                            for zb in z.split(FID_BATCH)])
        mean, cov = self._stats(self._features(images))
        diff = (self.real_mean - mean).square().sum()
        trace = self.real_cov.trace() + cov.trace()
        sqrt_eig = torch.linalg.eigvals(self.real_cov @ cov).sqrt().real.sum()
        return (diff + trace - 2 * sqrt_eig).item()


@torch.no_grad()
def reconstruction_error(model, images, device):
    """Same as util/result.py: decode mu, squared error summed over the
    pixels, averaged over the images."""
    model.eval()
    total = 0.0
    for x in images.split(256):
        x = x.to(device)
        mu, _ = model.encode(x)
        total += (x - model.decode(mu)).pow(2).sum().item()
    return total / len(images)


@torch.no_grad()
def validation_loss(model, images, device):
    """Average loss recon + kl per image (z sampled as in training)."""
    model.eval()
    total = 0.0
    for x in images.split(256):
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, _, _ = loss_function(x, x_hat, mu, logvar)
        total += loss.item() * x.size(0)
    return total / len(images)


@torch.no_grad()
def stage2_validation_loss(model, images, device, repeats=8):
    """Average negative ELBO of the stage-2 VAE (stage2_loss of mytrain) on
    the whitened stage-1 latents of the images (clean, each encoded `repeats`
    times to smooth out the sampling noise)."""
    model.eval()
    total, n = 0.0, 0
    log_gamma = model.s2.log_gamma.clamp(min=mytrain.STAGE2_LOG_GAMMA_MIN)
    for x in images.split(256):
        x = x.to(device).repeat(repeats, 1, 1, 1)
        w = model.whiten(mytrain._stage1_latents(model, x))
        mu, logvar = model.s2.encode(w)
        w_hat = model.s2.decode(model.reparameterize(mu, logvar))
        loss, _, _ = stage2_loss(w, w_hat, mu, logvar, log_gamma)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


# ---------------------------------------------------------------- training

def train_epoch_warmup(model, optimizer, loader, device, beta):
    """train_one_epoch with the loss recon + beta * kl back-propagated (same
    augmentation, input corruption and EMA). Returns the
    averages of recon + kl, recon and kl, like train_one_epoch."""
    return train_one_epoch(model, optimizer, loader, device, kl_weight=beta)


def beta_at(epoch, warmup):
    return 1.0 if warmup <= 0 else min(1.0, (epoch - 1) / warmup)


def cpu_state(model, prefix=""):
    """Copy of the state_dict on the CPU (only the keys starting with prefix)."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()
            if k.startswith(prefix)}


def images_of(subset):
    return torch.stack([subset[i] for i in range(len(subset))])


# ---------------------------------------------------------------- output

def mean_std(values):
    """Mean and sample standard deviation (0 for a single value)."""
    values = list(values)
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def onoff(flag):
    return "on" if flag else "off"


def setting_of(r):
    """The setting a model was trained with: (latent_dim, warmup, aug, corrupt)."""
    return (r["latent_dim"], r["warmup"], r["aug"], r["corrupt"])


def setting_label(s, aggs):
    """'latent 32, warm-up 0', plus aug / corrupt when they vary in the run."""
    d, w, aug, corrupt = s
    return (f"latent {d}, warm-up {w}"
            + (f", aug {aug}" if len({a['aug'] for a in aggs}) > 1 else "")
            + (f", corrupt {corrupt}" if len({a['corrupt'] for a in aggs}) > 1 else ""))


def aggregate(rows):
    """Mean and standard deviation over the seeds of the measurements of one
    setting: the periodic ones of the same epoch together, and the best epochs
    (early stop) together."""
    groups = {}
    for r in rows:
        key = setting_of(r) + (r["kind"], r["stage2"],
                               r["epoch"] if r["kind"] == "periodic" else None)
        groups.setdefault(key, []).append(r)
    aggs = []
    for (latent_dim, warmup, aug, corrupt, kind, stage2, _), rs in groups.items():
        a = {"latent_dim": latent_dim, "warmup": warmup, "aug": aug, "corrupt": corrupt,
             "kind": kind, "stage2": stage2,
             "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs)}
        for k in ("epoch", "recon_error", "fid"):
            a[k], a[k + "_std"] = mean_std(r[k] for r in rs)
        aggs.append(a)
    return aggs


def fmt(a, key, digits=2):
    """'mean +- std' over the seeds, or the value alone for a single seed."""
    if a["n_seeds"] == 1:
        return f"{a[key]:.{digits}f}"
    return f"{a[key]:.{digits}f} ± {a[key + '_std']:.{digits}f}"


STAGE2_KINDS = ["none", "whiten", "vae"]


def _plot_measures(axes, aggs, all_aggs, configs, color, multi):
    """FID vs reconstruction error, FID vs epoch and reconstruction error vs
    epoch of the measurements `aggs`, on axes[0..2]."""
    kinds = sorted({a["stage2"] for a in aggs}, key=STAGE2_KINDS.index)
    for c, stage2 in [(c, s) for c in configs for s in kinds]:
        w = c[1]
        # whitening: same color, dotted line and hollow markers
        style = dict(color=color[c],
                     linestyle=("-" if w == 0 else "--") if stage2 != "whiten"
                     else (":" if w == 0 else "-."),
                     label=setting_label(c, all_aggs)
                           + ("" if len(kinds) == 1 else f", {stage2}"))
        if stage2 == "whiten":
            style["markerfacecolor"] = "none"
        mine = [a for a in aggs if setting_of(a) == c and a["stage2"] == stage2]
        has_periodic = any(a["kind"] == "periodic" for a in mine)
        for kind, marker in (("periodic", "o"), ("best", "*")):
            sel = sorted((a for a in mine if a["kind"] == kind), key=lambda a: a["epoch"])
            if not sel:
                continue
            ep = [a["epoch"] for a in sel]
            fid = [a["fid"] for a in sel]
            rec = [a["recon_error"] for a in sel]
            err = {k: ([a[k + "_std"] for a in sel] if multi else None)
                   for k in ("epoch", "fid", "recon_error")}
            if kind == "periodic":
                err["epoch"] = None             # the same epoch for every seed
            s = dict(style, marker=marker, markersize=4 if kind == "periodic" else 12,
                     capsize=2)
            if kind == "best":
                s.update(linestyle="none")
                if has_periodic:
                    s.pop("label")
            axes[0].errorbar(rec, fid, xerr=err["recon_error"], yerr=err["fid"], **s)
            axes[0].annotate(f"{ep[-1]:.0f}", (rec[-1], fid[-1]), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")
            axes[1].errorbar(ep, fid, xerr=err["epoch"], yerr=err["fid"], **s)
            axes[2].errorbar(ep, rec, xerr=err["epoch"], yerr=err["recon_error"], **s)


def _plot_curves(ax, curves_of, all_aggs, configs, color, multi, title):
    """Validation loss per epoch, one curve per seed, * at the best epoch."""
    for c in configs:
        style = dict(color=color[c], linestyle="-" if c[1] == 0 else "--",
                     label=setting_label(c, all_aggs))
        for i, (curve, best_epoch) in enumerate(curves_of.get(c, [])):
            s = style if i == 0 else {k: v for k, v in style.items() if k != "label"}
            ax.plot(range(1, len(curve) + 1), curve, linewidth=1.0,
                    alpha=0.6 if multi else 1.0, **s)
            if best_epoch:
                ax.plot(best_epoch, curve[best_epoch - 1], marker="*",
                        markersize=12, color=color[c])
    curves = [curve for runs in curves_of.values() for curve, _ in runs]
    tail = [v for c in curves for v in c[10:]] or [v for c in curves for v in c]
    lo, hi = min(tail), max(tail)
    pad = 0.05 * (hi - lo + 1e-8)
    ax.set(xlabel="epoch", ylabel="validation loss", ylim=(lo - pad, hi + pad),
           title=title + (", one curve per seed" if multi else ""))


def save_plot(aggs, val_curves, s2_curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # one color per setting (latent_dim, warmup, aug, corrupt)
    configs = sorted({setting_of(a) for a in aggs} | set(val_curves) | set(s2_curves))
    color = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(configs)}
    multi = any(a["n_seeds"] > 1 for a in aggs)
    s1_aggs = [a for a in aggs if a["stage2"] != "vae"]
    s2_aggs = [a for a in aggs if a["stage2"] == "vae"]
    rows = [r for r in (("stage 1", s1_aggs, val_curves), ("stage 2", s2_aggs, s2_curves))
            if r[1] or r[2]]
    n_axes = 4 if (val_curves or s2_curves) else 3
    fig, axes = plt.subplots(len(rows), n_axes, figsize=(5 * n_axes, 4.5 * len(rows)),
                             constrained_layout=True, squeeze=False)
    over = " (mean ± std over the seeds)" if multi else ""
    for ax_row, (name, row_aggs, curves) in zip(axes, rows):
        _plot_measures(ax_row, row_aggs, aggs, configs, color, multi)
        unit = "stage-2 epoch" if name == "stage 2" else "epoch"
        prefix = "stage 2: " if len(rows) > 1 and name == "stage 2" else ""
        ax_row[0].set(xlabel="reconstruction error", ylabel="FID",
                      title=prefix + "FID vs reconstruction error (lower left is better)")
        ax_row[1].set(xlabel=unit, ylabel="FID", title=prefix + "FID" + over)
        ax_row[2].set(xlabel=unit, ylabel="reconstruction error",
                      title=prefix + "reconstruction error" + over)
        if curves:
            _plot_curves(ax_row[3], curves, aggs, configs, color, multi,
                         prefix + ("stage-2 validation loss" if name == "stage 2"
                                   else "validation loss") + " (* best epoch)")
            ax_row[3].set(xlabel=unit)
        elif n_axes == 4:
            ax_row[3].axis("off")
        for ax in ax_row:
            ax.grid(alpha=0.3)
        if row_aggs:
            ax_row[1].legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    device, device_name = get_device(args.device)
    full = get_dataset("train")

    use_val = args.early_stop or not args.test
    n_train = len(full) - args.val_size if use_val else len(full)

    def split(seed):
        """(training images, held-out images or None) for this seed."""
        if not use_val:
            return full, None
        train_set, val_set = random_split(
            full, [n_train, args.val_size], generator=torch.Generator().manual_seed(seed))
        return train_set, images_of(val_set)

    if args.test:
        test_images = get_dataset("test").images
        real_images = test_images
        mode = (f"TEST mode: train on {n_train} training images, "
                f"measure on {len(test_images)} test images")
    else:
        real_images = full.images
        mode = (f"default mode: train on {n_train} training images, "
                f"recon error on {args.val_size} held-out, "
                f"FID against {len(real_images)} training images")
    if args.early_stop:
        mode += f"; early stop on {args.val_size} held-out training images"

    if args.eval_every > 0:
        eval_epochs = set(range(args.eval_every, args.epochs + 1, args.eval_every))
        if not args.early_stop:
            eval_epochs.add(args.epochs)
    else:
        eval_epochs = set() if args.early_stop else {args.epochs}

    n_configs = (len(args.latent_dims) * len(args.warmups)
                 * len(args.aug_values) * len(args.corrupt_values))
    print(mode)
    print(f"augmentation {' / '.join(map(onoff, args.aug_values))} | "
          f"input corruption {' / '.join(map(onoff, args.corrupt_values))} | "
          f"EMA {'on' if mytrain.EMA else 'off'} | "
          f"residual {'on' if mymodel.RESIDUAL else 'off'} | "
          f"z skip {'on' if mymodel.Z_SKIP else 'off'} | "
          f"encoder dropout {mymodel.ENC_DROPOUT:g}")
    print(f"device {device_name} | batch {BATCH_SIZE} | {n_configs} settings x "
          f"{len(args.seeds)} seeds {args.seeds} | "
          f"{'max ' if args.early_stop else ''}{args.epochs} epochs | "
          f"{args.n_gen} generated images per FID")
    if args.early_stop:
        print(f"early stop: patience {args.patience}, min_delta {args.min_delta}"
              + (f"; also measured every {args.eval_every} epochs" if eval_epochs else ""))
    elif eval_epochs:
        print(f"measured at epochs {sorted(eval_epochs)}")
    if args.stage2_epochs:
        print(f"stage-2 VAE: {'max ' if args.early_stop else ''}{args.stage2_epochs} epochs "
              f"(STAGE2_INPUT {mytrain.STAGE2_INPUT!r}, STAGE2_REPEAT {mytrain.STAGE2_REPEAT})"
              + (f" | measured every {args.stage2_eval_every} stage-2 epochs"
                 if args.stage2_eval_every else "")
              + (f" | early stop: patience {args.stage2_patience}, "
                 f"min_delta {args.stage2_min_delta}" if args.early_stop else ""))
    if args.test:
        print("note: settings chosen on the test images give an optimistic final result")

    t0 = time.time()
    print(f"computing the Inception features of {len(real_images)} real images ...", flush=True)
    fid = FID(real_images, device)
    print(f"  done in {time.time() - t0:.0f} s")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "compare"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"run_{stamp}.csv"
    fields = ["latent_dim", "warmup", "aug", "corrupt", "seed", "kind", "stage2", "epoch",
              "stage1_epoch", "stop_epoch", "val_loss", "train_loss", "train_recon",
              "train_kl", "recon_error", "fid"]
    rows, val_curves, s2_curves = [], {}, {}

    def measure(writer, f, latent_dim, warmup, seed, kind, epoch, tr, model, recon_images,
                loader, stop_epoch="", val_loss="", kinds=None, stage1_epoch=""):
        """Measure the model (its EMA weights, if any) for each of `kinds`
        (default args.stage2_kinds; ["vae"] for a model already in stage 2);
        returns the text 'recon error ... | FID ...' for the log."""
        with ema_weights(model):
            return _measure(writer, f, latent_dim, warmup, seed, kind, epoch, tr, model,
                            recon_images, loader, stop_epoch, val_loss,
                            kinds or args.stage2_kinds, stage1_epoch)

    def _measure(writer, f, latent_dim, warmup, seed, kind, epoch, tr, model, recon_images,
                 loader, stop_epoch, val_loss, kinds, stage1_epoch):
        texts = []
        for stage2 in kinds:
            if stage2 == "whiten":
                with torch.random.fork_rng(devices=[]):      # training unchanged
                    start_stage2(model, loader, device, "whiten")
            rec = reconstruction_error(model, recon_images, device)
            fd = fid(model, args.n_gen)
            if stage2 != "vae":                              # back to stage 1
                model.stage2_mode.fill_(MODE_STAGE1)
            model.train()
            row = {"latent_dim": latent_dim, "warmup": warmup, "aug": onoff(mytrain.AUGMENT),
                   "corrupt": onoff(mytrain.CORRUPT), "seed": seed, "kind": kind,
                   "stage2": stage2, "stage1_epoch": stage1_epoch,
                   "epoch": epoch, "stop_epoch": stop_epoch, "val_loss": val_loss,
                   "train_loss": tr["loss"], "train_recon": tr["recon"],
                   "train_kl": tr["kl"], "recon_error": rec, "fid": fd}
            rows.append(row)
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()})
            f.flush()
            name = "" if len(kinds) == 1 else f" ({stage2})"
            texts.append(f"recon error{name} {rec:6.2f} | FID{name} {fd:7.2f}")
        return " | ".join(texts)

    def run_stage2(writer, f, model, loader, val_images, recon_images, setting, seed,
                   stage1_epoch):
        """Fit the whitening, train the stage-2 VAE on top of the frozen stage 1
        (as train_one_epoch does in the notebook) and measure it."""
        latent_dim, warmup = setting[0], setting[1]
        ema_apply(model)                # final stage-1 weights (no-op if already)
        start_stage2(model, loader, device, "vae")
        optimizer = get_optimizer(model)    # stage-1 grads stay None: not updated
        print(f"  --- stage 2 on stage-1 epoch {stage1_epoch}", flush=True)
        t_s2 = time.time()
        curve = []
        best_loss, best_epoch, best_state, best_tr = float("inf"), 0, None, None
        wait, stop_epoch, epoch, tr = 0, None, 0, None
        measured_last = False
        try:
            for epoch in range(1, args.stage2_epochs + 1):
                tr = train_one_epoch(model, optimizer, loader, device)
                measured_last = False
                if args.early_stop:
                    vl = stage2_validation_loss(model, val_images, device)
                    model.train()
                    curve.append(vl)
                    if vl < best_loss - args.stage2_min_delta:
                        best_loss, best_epoch, wait = vl, epoch, 0
                        best_state, best_tr = cpu_state(model, "s2."), tr
                    else:
                        wait += 1
                    if epoch % 50 == 0:
                        print(f"  s2 epoch {epoch:4d} | train loss {tr['loss']:8.2f} | "
                              f"val loss {vl:8.2f} | best {best_loss:8.2f} @ {best_epoch} | "
                              f"wait {wait}/{args.stage2_patience} | "
                              f"log gamma {model.s2.log_gamma.item():6.2f} | "
                              f"{time.time() - t_s2:4.0f} s", flush=True)
                if args.stage2_eval_every and epoch % args.stage2_eval_every == 0:
                    text = measure(writer, f, latent_dim, warmup, seed, "periodic", epoch,
                                   tr, model, recon_images, loader, kinds=["vae"],
                                   stage1_epoch=stage1_epoch)
                    measured_last = True
                    print(f"  s2 epoch {epoch:4d} | train loss {tr['loss']:8.2f} "
                          f"(kl {tr['kl']:5.2f}) | log gamma "
                          f"{model.s2.log_gamma.item():6.2f} | {text} | "
                          f"{time.time() - t_s2:5.0f} s", flush=True)
                if args.early_stop and wait >= args.stage2_patience:
                    stop_epoch = epoch
                    break
        except KeyboardInterrupt:
            if not args.early_stop:
                raise
            print("  interrupted; measuring the best stage-2 epoch so far "
                  "(Ctrl+C again to stop everything)")
            stop_epoch = epoch

        if not args.early_stop:
            if epoch and not measured_last:     # the last stage-2 epoch, once
                text = measure(writer, f, latent_dim, warmup, seed, "periodic", epoch, tr,
                               model, recon_images, loader, kinds=["vae"],
                               stage1_epoch=stage1_epoch)
                print(f"  s2 epoch {epoch:4d} (last) | {text}", flush=True)
            return
        if stop_epoch is None:
            stop_epoch = epoch
            print(f"  reached the maximum of {args.stage2_epochs} stage-2 epochs without "
                  f"early stopping; try a larger --stage2-epochs")
        if best_state is None:
            if tr is None:
                print("  no stage-2 epoch was trained")
                return
            best_epoch, best_tr = epoch, tr
        else:
            model.load_state_dict(best_state, strict=False)
        s2_curves.setdefault(setting, []).append((curve, best_epoch))
        text = measure(writer, f, latent_dim, warmup, seed, "best", best_epoch, best_tr,
                       model, recon_images, loader, stop_epoch=stop_epoch,
                       val_loss=best_loss if best_state else "", kinds=["vae"],
                       stage1_epoch=stage1_epoch)
        print(f"  s2 stopped at epoch {stop_epoch} | BEST stage-2 epoch {best_epoch} | "
              f"val loss {best_loss:8.2f} | log gamma {model.s2.log_gamma.item():6.2f} | "
              f"{text} | {time.time() - t_s2:5.0f} s", flush=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        try:
            for seed in args.seeds:
                train_set, val_images = split(seed)
                recon_images = test_images if args.test else val_images
                for latent_dim in args.latent_dims:
                    for warmup, aug, corrupt in [(w, a, c) for w in args.warmups
                                                 for a in args.aug_values
                                                 for c in args.corrupt_values]:
                        # used by train_one_epoch (make_batch) and start_stage2
                        mytrain.AUGMENT, mytrain.CORRUPT = aug, corrupt
                        setting = (latent_dim, warmup, onoff(aug), onoff(corrupt))
                        print(f"\n=== seed {seed}: latent_dim {latent_dim}, "
                              f"KL warm-up {warmup} epochs, augmentation {onoff(aug)}, "
                              f"input corruption {onoff(corrupt)}")
                        torch.manual_seed(seed)
                        loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
                        model = VAE(latent_dim=latent_dim).to(device)
                        optimizer = get_optimizer(model)
                        t_cfg = time.time()

                        curve = []
                        best_loss, best_epoch, best_state, best_tr = \
                            float("inf"), 0, None, None
                        wait, stop_epoch = 0, None
                        epoch = 0
                        try:
                            for epoch in range(1, args.epochs + 1):
                                if warmup <= 0:
                                    tr = train_one_epoch(model, optimizer, loader, device)
                                else:
                                    tr = train_epoch_warmup(model, optimizer, loader, device,
                                                            beta_at(epoch, warmup))

                                if args.early_stop:
                                    with ema_weights(model):    # EMA weights, if any
                                        vl = validation_loss(model, val_images, device)
                                        curve.append(vl)
                                        if epoch > warmup:      # KL weight is 1 from here
                                            if vl < best_loss - args.min_delta:
                                                best_loss, best_epoch, wait = vl, epoch, 0
                                                best_state, best_tr = cpu_state(model), tr
                                            else:
                                                wait += 1
                                    if epoch % 50 == 0:
                                        print(f"  epoch {epoch:4d} | train loss "
                                              f"{tr['loss']:7.2f} | val loss {vl:7.2f} | "
                                              f"best {best_loss:7.2f} @ {best_epoch} | "
                                              f"wait {wait}/{args.patience} | "
                                              f"{time.time() - t_cfg:4.0f} s", flush=True)

                                if epoch in eval_epochs:
                                    text = measure(writer, f, latent_dim, warmup, seed,
                                                   "periodic", epoch, tr, model,
                                                   recon_images, loader)
                                    print(f"  epoch {epoch:4d} | train loss {tr['loss']:7.2f} "
                                          f"(kl {tr['kl']:5.2f}) | {text} | "
                                          f"{time.time() - t_cfg:5.0f} s", flush=True)

                                if args.early_stop and wait >= args.patience:
                                    stop_epoch = epoch
                                    break
                        except KeyboardInterrupt:
                            if not args.early_stop:
                                raise
                            print("  interrupted; measuring the best epoch so far "
                                  "(Ctrl+C again to stop everything)")
                            stop_epoch = epoch

                        if args.early_stop:
                            if stop_epoch is None:
                                stop_epoch = epoch
                                print(f"  reached the maximum of {args.epochs} epochs "
                                      f"without early stopping; try a larger --epochs")
                            if best_state is None:
                                print("  no epoch after the warm-up: measuring the last epoch")
                                best_epoch, best_tr = epoch, tr
                            else:
                                model.load_state_dict(best_state)
                                ema_discard(model)      # best_state holds the EMA weights
                            val_curves.setdefault(setting, []).append(
                                (curve, best_epoch))
                            text = measure(writer, f, latent_dim, warmup, seed, "best",
                                           best_epoch, best_tr, model, recon_images, loader,
                                           stop_epoch=stop_epoch,
                                           val_loss=best_loss if best_state else "")
                            print(f"  stopped at epoch {stop_epoch} | BEST epoch {best_epoch} "
                                  f"| val loss {best_loss:7.2f} | {text} | "
                                  f"{time.time() - t_cfg:5.0f} s", flush=True)

                        if args.stage2_epochs and epoch:
                            run_stage2(writer, f, model, loader, val_images, recon_images,
                                       setting, seed,
                                       best_epoch if args.early_stop else epoch)
        except KeyboardInterrupt:
            print("\ninterrupted; reporting the measurements so far")

    if not rows:
        print("no measurement")
        return

    aggs = aggregate(rows)
    multi = any(a["n_seeds"] > 1 for a in aggs)
    mean_path = out_dir / f"run_{stamp}_mean.csv"
    mean_fields = ["latent_dim", "warmup", "aug", "corrupt", "kind", "stage2", "n_seeds",
                   "seeds", "epoch", "epoch_std", "recon_error", "recon_error_std",
                   "fid", "fid_std"]
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fields)
        writer.writeheader()
        for a in aggs:
            writer.writerow({k: (round(a[k], 4) if isinstance(a[k], float) else a[k])
                             for k in mean_fields})

    def epoch_text(a):
        return f"{a['epoch']:.0f}" if a["kind"] == "periodic" else fmt(a, "epoch", 0)

    w_val = 17 if multi else 9
    print(f"\n{'mean over the seeds, ' if multi else ''}sorted by FID"
          + (" (epoch: the stage-2 epoch for stage 2 vae)"
             if any(a["stage2"] == "vae" for a in aggs) else "") + ":")
    print(f"{'latent':>6} {'warm-up':>7} {'aug':>3} {'corrupt':>7} {'kind':>8} {'stage 2':>7} "
          f"{'epoch':>{11 if multi else 5}} "
          f"{'seeds':>5} {'recon err':>{w_val}} {'FID':>{w_val}}")
    for a in sorted(aggs, key=lambda a: a["fid"]):
        print(f"{a['latent_dim']:6d} {a['warmup']:7d} {a['aug']:>3} {a['corrupt']:>7} "
              f"{a['kind']:>8} {a['stage2']:>7} "
              f"{epoch_text(a):>{11 if multi else 5}} {a['n_seeds']:5d} "
              f"{fmt(a, 'recon_error'):>{w_val}} {fmt(a, 'fid'):>{w_val}}")
    if len({a["n_seeds"] for a in aggs}) > 1:
        print("note: the settings do not all have the same number of seeds (interrupted run)")

    # Settings that no other setting beats on both mean values
    pareto = [a for a in aggs
              if not any(o["fid"] <= a["fid"] and o["recon_error"] <= a["recon_error"]
                         and (o["fid"] < a["fid"] or o["recon_error"] < a["recon_error"])
                         for o in aggs)]
    print(f"\nnot beaten on both {'mean ' if multi else ''}values by another measurement:")
    for a in sorted(pareto, key=lambda a: a["fid"]):
        unit = "stage-2 epoch" if a["stage2"] == "vae" else "epoch"
        print(f"  {setting_label(setting_of(a), aggs)}, {unit} {epoch_text(a)} "
              f"({a['kind']}, stage 2 {a['stage2']}): recon error {fmt(a, 'recon_error')}, "
              f"FID {fmt(a, 'fid')}")

    # one line per setting (the best epoch does not depend on the whitening)
    best_aggs = [a for a in aggs if a["kind"] == "best"
                 and a["stage2"] == args.stage2_kinds[0]]
    if best_aggs:
        print(f"\nSTAGE1_EPOCHS of code/mytrain.py "
              f"({'mean best epoch over the seeds; ' if multi else ''}"
              f"trained there on all {len(full)} images; "
              f"the second value keeps the number of optimizer steps):")
        for a in best_aggs:
            epochs = [r["epoch"] for r in rows if r["kind"] == "best"
                      and r["stage2"] == a["stage2"] and setting_of(r) == setting_of(a)]
            print(f"  {setting_label(setting_of(a), aggs)}: "
                  f"{round(a['epoch'])}  (or about {round(a['epoch'] * n_train / len(full))})"
                  + (f"  best epochs per seed {epochs}" if len(epochs) > 1 else ""))

    # stage 2: best epoch of the early stop, and the periodic measure of lowest FID
    s2_best = [a for a in aggs if a["kind"] == "best" and a["stage2"] == "vae"]
    if s2_best:
        print(f"\nSTAGE2_EPOCHS of code/mytrain.py "
              f"({'mean best stage-2 epoch over the seeds; ' if multi else ''}"
              f"by the stage-2 validation loss; the second value keeps the number of "
              f"optimizer steps on all {len(full)} images):")
        for a in s2_best:
            epochs = [r["epoch"] for r in rows if r["kind"] == "best"
                      and r["stage2"] == "vae" and setting_of(r) == setting_of(a)]
            print(f"  {setting_label(setting_of(a), aggs)}: "
                  f"{round(a['epoch'])}  (or about {round(a['epoch'] * n_train / len(full))})"
                  f" | recon error {fmt(a, 'recon_error')}, FID {fmt(a, 'fid')}"
                  + (f"  best epochs per seed {epochs}" if len(epochs) > 1 else ""))
    s2_periodic = [a for a in aggs if a["kind"] == "periodic" and a["stage2"] == "vae"]
    if s2_periodic:
        print(f"\nstage 2, periodic measure of lowest {'mean ' if multi else ''}FID "
              f"per setting:")
        for c in sorted({setting_of(a) for a in s2_periodic}):
            a = min((a for a in s2_periodic if setting_of(a) == c), key=lambda a: a["fid"])
            print(f"  {setting_label(c, aggs)}: stage-2 epoch {a['epoch']:.0f} | "
                  f"recon error {fmt(a, 'recon_error')}, FID {fmt(a, 'fid')}")

    png_path = out_dir / f"run_{stamp}.png"
    save_plot(aggs, val_curves, s2_curves, png_path)
    print(f"\nsaved {csv_path}")
    print(f"saved {mean_path}")
    print(f"saved {png_path}")
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
