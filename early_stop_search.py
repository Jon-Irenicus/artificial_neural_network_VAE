"""Find the best number of epochs with early stopping on a validation split.

The training images are split into a training part and a validation part.
The model is trained with the code in the `code` folder (mymodel, myoptim,
mytrain) and the validation loss is measured after every epoch. Training
stops when the validation loss has not improved by more than `min_delta`
for `patience` epochs. The best epoch is the value to put in NUM_EPOCHS of
code/myoptim.py.

The test images are never used, since the results are measured on them.

Run from anywhere:
    python early_stop_search.py
    python early_stop_search.py --patience 50 --min-delta 0.05 --max-epochs 2000
    python early_stop_search.py --val-size 100 --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent
for folder in ("code", "util"):
    if str(ROOT / folder) not in sys.path:
        sys.path.insert(0, str(ROOT / folder))

import mytrain                                    # noqa: E402
from data import get_dataset                      # noqa: E402
from mymodel import VAE                           # noqa: E402
from myoptim import BATCH_SIZE, get_optimizer     # noqa: E402
from mytrain import loss_function, train_one_epoch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patience", type=int, default=30,
                   help="epochs without improvement before stopping (default 30)")
    p.add_argument("--min-delta", type=float, default=0.1,
                   help="smallest decrease of the validation loss that counts "
                        "as an improvement (default 0.1)")
    p.add_argument("--max-epochs", type=int, default=1000,
                   help="upper bound on the number of epochs (default 1000)")
    p.add_argument("--val-size", type=int, default=100,
                   help="number of training images held out for validation (default 100)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed of the split and of the training (default 0)")
    p.add_argument("--print-every", type=int, default=10,
                   help="print a line every this many epochs (default 10)")
    p.add_argument("--no-plot", action="store_true",
                   help="do not save the plot of the loss curves")
    p.add_argument("--aug", choices=["on", "off"], default=None,
                   help="augmentation of the training images (default: AUGMENT of "
                        "code/mytrain.py)")
    p.add_argument("--corrupt", choices=["on", "off"], default=None,
                   help="denoising input corruption (default: CORRUPT of code/mytrain.py)")
    args = p.parse_args()
    if args.aug is not None:
        mytrain.AUGMENT = args.aug == "on"
    if args.corrupt is not None:
        mytrain.CORRUPT = args.corrupt == "on"
    return args


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """Average loss, recon and kl per image over `loader`."""
    model.eval()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    count = 0
    for x in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        loss, recon, kl = loss_function(x, x_hat, mu, logvar)
        n = x.size(0)
        totals["loss"] += loss.item() * n
        totals["recon"] += recon.item() * n
        totals["kl"] += kl.item() * n
        count += n
    return {k: v / count for k, v in totals.items()}


def save_plot(history, best_epoch, stop_epoch, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(epochs, history["train"], label="train", color="#9a9a9a", linewidth=1.5)
    ax.plot(epochs, history["val"], label="validation", color="#2a78d6", linewidth=2)
    ax.axvline(best_epoch, color="#d6542a", linestyle="--", linewidth=1.2,
               label=f"best epoch {best_epoch}")
    if stop_epoch is not None:
        ax.axvline(stop_epoch, color="#9a9a9a", linestyle=":", linewidth=1.2,
                   label=f"stopped at {stop_epoch}")
    # Skip the first epochs so that the interesting part is visible
    tail = history["val"][min(10, len(history["val"]) - 1):] + \
        history["train"][min(10, len(history["train"]) - 1):]
    lo, hi = min(tail), max(tail)
    ax.set_ylim(lo - 0.05 * (hi - lo + 1e-8), hi + 0.05 * (hi - lo + 1e-8))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss per image")
    ax.set_title(f"batch {BATCH_SIZE}, patience {args.patience}, "
                 f"min_delta {args.min_delta}, val {args.val_size}, seed {args.seed}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    full = get_dataset("train")
    train_size = len(full) - args.val_size
    train_set, val_set = random_split(
        full, [train_size, args.val_size],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    model = VAE().to(device)
    optimizer = get_optimizer(model)

    print(f"device {device} | train {train_size} / val {args.val_size} images | "
          f"batch {BATCH_SIZE}")
    print(f"patience {args.patience} | min_delta {args.min_delta} | "
          f"max epochs {args.max_epochs} | seed {args.seed}")
    print(f"augmentation {'on' if mytrain.AUGMENT else 'off'} | "
          f"input corruption {'on' if mytrain.CORRUPT else 'off'}")
    print(optimizer.__class__.__name__, {k: v for k, v in optimizer.defaults.items()
                                         if k in ("lr", "weight_decay", "momentum", "betas")})
    print("-" * 78)

    history = {"train": [], "val": []}
    best_loss, best_epoch, best_val = float("inf"), 0, None
    wait, stop_epoch = 0, None
    start = time.time()

    try:
        for epoch in range(1, args.max_epochs + 1):
            tr = train_one_epoch(model, optimizer, train_loader, device)
            va = evaluate(model, val_loader, device)
            history["train"].append(tr["loss"])
            history["val"].append(va["loss"])

            improved = va["loss"] < best_loss - args.min_delta
            if improved:
                best_loss, best_epoch, best_val, wait = va["loss"], epoch, va, 0
            else:
                wait += 1

            if epoch == 1 or epoch % args.print_every == 0 or improved and epoch <= 10:
                print(f"epoch {epoch:5d} | train {tr['loss']:7.2f} | "
                      f"val {va['loss']:7.2f} (recon {va['recon']:7.2f}, kl {va['kl']:6.2f}) | "
                      f"best {best_loss:7.2f} @ {best_epoch:<5d} | wait {wait:3d}/{args.patience}")

            if wait >= args.patience:
                stop_epoch = epoch
                print(f"early stop at epoch {epoch} "
                      f"(no improvement > {args.min_delta} for {args.patience} epochs)")
                break
        else:
            print(f"reached max epochs {args.max_epochs} without early stopping; "
                  f"try a larger --max-epochs")
    except KeyboardInterrupt:
        print("\ninterrupted; reporting the best epoch so far")

    elapsed = time.time() - start
    print("-" * 78)
    if best_val is None:
        print("no epoch finished")
        return
    print(f"best epoch      : {best_epoch}")
    print(f"best val loss   : {best_val['loss']:.2f} "
          f"(recon {best_val['recon']:.2f}, kl {best_val['kl']:.2f})")
    print(f"time            : {elapsed:.0f} s for {len(history['val'])} epochs")
    # The final training uses all images, so each epoch has more steps
    scaled = round(best_epoch * train_size / len(full))
    print(f"-> NUM_EPOCHS = {best_epoch} in code/myoptim.py "
          f"(or about {scaled} to keep the same number of optimizer steps "
          f"with all {len(full)} images)")

    if not args.no_plot:
        path = ROOT / "early_stop.png"
        save_plot(history, best_epoch, stop_epoch, args, path)
        print(f"plot            : {path}")


if __name__ == "__main__":
    main()
