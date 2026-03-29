# src/data/datamodule.py

from __future__ import annotations

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split
import torch

from src.config.schema import DataConfig
from src.utils.logging import get_logger

log = get_logger(__name__)

# CIFAR-10 class names in label order
# label 0 = airplane, label 1 = automobile, etc.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

class DataModule:
    """
    Owns everything about data: downloading, transforms,
    splitting, and DataLoader creation.

    The Trainer never touches datasets directly.
    It calls datamodule.setup() then uses
    datamodule.train_loader and datamodule.val_loader.

    Why this separation?
        The Trainer should not know or care whether you're
        loading CIFAR-10 or a custom folder of medical images.
        Swapping datasets means replacing this file only.
        The Trainer, model, and config don't change.

    Usage:
        dm = DataModule(cfg.data)
        dm.setup()
        for x, y in dm.train_loader:
            ...
    """

    def __init__(self, cfg : DataConfig) -> None:
        self.cfg = cfg
        self._train_ds = None
        self._val_ds = None
    
    # ──────────────────────────────────────────────────────────
    # Transforms
    # ──────────────────────────────────────────────────────────
    def _train_transform(self) -> T.Compose:
        """
        Augmentation pipeline for training data only.

        Why augment training but not validation?
            Augmentation artificially creates variation so the model
            sees slightly different versions of each image each epoch.
            This prevents memorisation and improves generalisation.
            On validation you want consistent, deterministic results
            so metrics are comparable between runs — augmentation
            would add random noise to your val metrics.

        What each transform does:
            RandomCrop(32, padding=4)
                Pads the 32×32 image with 4px of zeros on each side
                making it 40×40, then randomly crops back to 32×32.
                Teaches the model that the subject can be anywhere
                in the frame — not always perfectly centred.

            RandomHorizontalFlip()
                50% chance of mirroring the image left-to-right.
                An airplane flying left is still an airplane.
                Does NOT use vertical flip — flipped cars and
                animals look unnatural and hurt accuracy.

            ColorJitter(brightness, contrast, saturation)
                Randomly varies colour properties.
                Teaches the model that a cat is a cat regardless
                of lighting conditions.

            ToTensor()
                Converts PIL Image (H, W, C) uint8 [0,255]
                to torch Tensor (C, H, W) float32 [0.0, 1.0]
                Always the second-to-last transform.

            Normalize(mean, std)
                Shifts and scales each channel so pixel values
                are roughly zero-mean, unit-variance.
                Without this, different channels have different
                scales and gradient descent is much slower.
                Always the very last transform.
        """
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
            ),
            T.ToTensor(),
            T.Normalize(self.cfg.mean, self.cfg.std),
        ])
    
    def _val_transform(self) -> T.Compose: # underscore is used to hint that the function is internal
        """
        Deterministic pipeline for validation data.

        No augmentation — only the transforms needed to make
        pixel values compatible with what the model expects.
        These two transforms must ALWAYS match what was done
        during training — same mean, same std.
        """

        return T.Compose([
            T.ToTensor(),
            T.Normalize(self.cfg.mean, self.cfg.std),
        ])
    
    # ──────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        Download data and create train/val splits.

        Called once by the Trainer at the start of fit().

        Why is setup() separate from __init__?
            __init__ should be instant — just store the config.
            setup() does the expensive work: downloading 170MB,
            reading file lists, splitting indices.
            This means you can create a DataModule object at the
            top of train.py without immediately triggering a
            download, and the Trainer controls when data is loaded.

        Why two separate dataset objects with different transforms?
            random_split() splits indices, not data.
            Both train_ds and val_ds_full point to the same
            underlying CIFAR-10 data on disk.
            But train_ds uses _train_transform() (with augmentation)
            and val_ds uses _val_transform() (without augmentation).
            This is the correct way to have different transforms
            for train and val when using random_split().
        """

        # Full dataset with training transforms (augmentation)
        full_train = torchvision.datasets.CIFAR10(
            root=self.cfg.data_dir,
            train=True,
            download=True,
            transform=self._train_transform(),
        )

        # Same data, but with val transforms (no augmentation)
        # We need this because random_split shares indices between
        # train and val — val samples must not be augmented
        # train = True because we want the original dataset and create the split ourselves
        # train = False will be for testing and whould not be used for validation
        full_val = torchvision.datasets.CIFAR10(
            root=self.cfg.data_dir,
            train=True,
            download=False, # Already downloaded before
            transform=self._val_transform(),
        )

        # Datasets in pytorch are lazy
        # transforms are applied at access time to stored
        # this way we can define different transforms for train and val on the same data
        # even if an index overlaps with train and val , when we read it using full_train[idx],
        # transforms are applied, but when full_val[idx], transforms are not applied

        # ── Reproducible split ────────────────────────────────
        # Generator with fixed seed ensures the same 45,000/5,000
        # split every run — the same images are always in train,
        # the same images are always in val.
        # Without this, val accuracy comparisons between runs
        # are meaningless (different images in val each time).

        n_val = int(len(full_train) * self.cfg.val_split)
        n_train = len(full_train) - n_val

        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(
            range(len(full_train)),
            [n_train, n_val],
            generator=generator,
        )

        # Subset wraps a dataset and exposes only a specific
        # list of indices — train_indices from full_train (augmented)
        # and val_indices from full_val (not augmented)
        self._train_ds = Subset(full_train, train_indices.indices)
        self._val_ds = Subset(full_val, val_indices.indices)

        log.info(
            "data.ready",
            train_samples=n_train,
            val_samples=n_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )

    # ──────────────────────────────────────────────────────────
    # DataLoaders
    # ──────────────────────────────────────────────────────────

    def _make_loader(
            self,
            dataset,
            shuffle : bool,
    ) -> DataLoader:
        """
        Create a DataLoader with the right performance settings.

        Args:
            dataset: the Dataset to wrap
            shuffle: True for train (random order each epoch),
                     False for val (consistent, deterministic order)

        Key parameters explained:

            num_workers:
                Spawns N separate processes that load and preprocess
                batches in parallel while the GPU trains on the
                current batch.
                Typical value: number of CPU cores / 2.
                Set to 0 for debugging — single process is easier
                to trace errors in.

            pin_memory:
                Allocates CPU tensors in page-locked (pinned) memory.
                Pinned memory transfers to GPU ~2× faster than
                normal memory.
                Always True when training on GPU.
                Has no effect on CPU-only training.

            persistent_workers:
                Keeps worker processes alive between epochs.
                Without this, workers are killed and restarted each
                epoch which takes 1-2 seconds per epoch.
                Set False when num_workers=0.

            prefetch_factor:
                Each worker preloads this many batches ahead.
                So while the GPU trains on batch N, workers are
                already loading batches N+1 and N+2.
                Set None when num_workers=0.
        """

        return DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=(
                2 if self.cfg.num_workers > 0 else None
            ),
            # drop_last=True drops the final incomplete batch
            # during training — prevents batch norm issues with
            # very small last batches (e.g. batch of 3)
            drop_last=True if shuffle else False,
        )
    
    @property
    def train_loader(self) -> DataLoader:
        """
        DataLoader for training — augmented, shuffled, drops last batch.
        Calling this property creates a new DataLoader each time,
        which is correct — each epoch should have fresh shuffling.
        """
        assert self._train_ds is not None , (
            "Call setup() before accessing train loader"
        )
        return self._make_loader(self._train_ds, shuffle=True)
    
    @property
    def val_loader(self) -> DataLoader:
        """
        DataLoader for validation — no augmentation, no shuffle,
        no dropped batches (we want metrics on every sample).
        """
        assert self._val_ds is not None , (
            "Call setup() before accessing the val_loader"
        )
        return self._make_loader(self._val_ds, shuffle=False)
    
    


