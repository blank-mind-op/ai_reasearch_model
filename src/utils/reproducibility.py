# src/utils/reproducibility.py

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.logging import get_logger

log = get_logger(__name__)

def set_seed(seed : int = 42) -> None:
    """
    Set all random seeds to the same value.

    Call this ONCE, right after setup_logging(), before you
    create any model, dataset, or dataloader.

    Why every library needs its own seed:
        Python's random, NumPy, and PyTorch each have their own
        independent random number generators. Setting only torch's
        seed leaves NumPy's generator unseeded — data augmentation
        that uses NumPy (e.g. albumentations) will still vary.

    Why PYTHONHASHSEED:
        Python randomises the hash order of dicts and sets by default.
        Setting this makes dict iteration order deterministic too.

    Why cudnn.deterministic:
        Some CUDA operations have non-deterministic implementations
        that are faster. Setting this forces the deterministic version.
        Small speed cost, full reproducibility.

    Why cudnn.benchmark = False:
        When True, CUDA benchmarks several kernel implementations and
        picks the fastest for your input size. This selection itself
        uses randomness and varies between runs. Disable it.
    """
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)

    log.debug("seed.set", seed = seed)