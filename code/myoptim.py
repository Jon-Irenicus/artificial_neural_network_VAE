"""Optimization settings: batch size, number of epochs, optimizer.

TODO: choose BATCH_SIZE and NUM_EPOCHS, and complete `get_optimizer`.
"""

import torch

from mytrain import STAGE1_EPOCHS, STAGE2, STAGE2_EPOCHS

# Used by the notebook. The values below are only a starting point.
BATCH_SIZE = 32     # number of images in a mini-batch
# number of passes over the training data: stage 1, then the stage-2 VAE
# (the two-stage settings are in code/mytrain.py)
NUM_EPOCHS = STAGE1_EPOCHS + (STAGE2_EPOCHS if STAGE2 == "vae" else 0)


def get_optimizer(model):
    """Create the optimizer that updates the parameters of `model`.

    The choice of the optimizer and of its settings (learning rate, ...) is
    yours.

    Args:
        model: the VAE defined in mymodel.py.

    Returns:
        a torch.optim.Optimizer.
    """
    # TODO
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
