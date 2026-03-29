# src/data/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logging import get_logger

log = get_logger(__name__)

class ImageFolderDataset(Dataset):
    """
    A Dataset for image classification from a folder structure.

    Expected folder layout:
        root/
          class_a/
            img001.jpg
            img002.jpg
          class_b/
            img003.jpg
            ...

    Each subfolder is one class. The class label is the folder's
    alphabetical index — so class_a=0, class_b=1, etc.

    Why write this instead of using torchvision.ImageFolder?
        torchvision.ImageFolder does the same thing but is a black
        box. Writing it yourself means you understand exactly what
        __getitem__ does, you can add custom logic (e.g. caching,
        multi-label), and you're not dependent on torchvision's API.
        For CIFAR-10 specifically we use the built-in dataset, but
        this class is what you'll use for any custom image dataset.

    Why load lazily in __getitem__ instead of loading everything
    in __init__?
        __init__ is called once. If you load all images there, you
        wait minutes before training starts and need RAM for the
        entire dataset. __getitem__ is called once per sample per
        epoch — images are loaded on demand, only when needed.
    """

    def __init__(
            self,
            root : Path,
            transform : Callable | None = None,
    ) -> None:
        
        self.root = root
        self.transform = transform

        # Build list of (image_path, label) pairs
        # sorted() ensures consistent class ordering across runs
        self.classes = sorted([
            d.name for d in self.root.iterdir() if d.is_dir()
        ])

        if len(self.classes) == 0:
            raise ValueError(
                f"No class folder found in {self.root}."
                f"Expected structure : root/class_name/image.jpg"
            )
        
        # Map class name → integer label
        self.class_to_idx = {
            cls : idx for idx , cls in enumerate(self.classes)
        }

        # Collect all (path, label) pairs
        self.samples : list[tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = root / cls
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for img_path in sorted(cls_dir.glob(ext)):
                    self.samples.append(
                        (img_path, self.class_to_idx[cls])
                    )
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No images found in {self.root}. "
                f"Supported formats: jpg, jpeg, png, webp"
            )
        
        log.debug(
            root=str(self.root),
            num_classes=len(self.classes),
            num_samples=len(self.samples),
            classes=self.classes,
        )
    
    def __len__(self) ->int:
        """
        Total number of samples.
        DataLoader calls this to know how many batches to produce.
        """
        return len(self.samples)

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, int]:
        """
        Return one (image_tensor, label) pair.

        Called by the DataLoader for each sample in a batch.
        This is where the image actually gets loaded from disk —
        not in __init__.

        Args:
            idx: integer index into self.samples

        Returns:
            tuple of (transformed image tensor, integer label)
        """

        path, label = self.samples[idx]

        # Open image with PIL
        # .convert("RGB") handles:
        #   - greyscale images (L mode) → converts to 3-channel
        #   - images with alpha channel (RGBA) → drops alpha
        # Without this, a single greyscale image in your dataset
        # causes a cryptic shape error at batch collation time

        image = Image.open(path).convert("RGB")

        # Apply transforms (ToTensor, Normalize, augmentation)
        # If no transform given, return PIL Image as-is
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, label)


