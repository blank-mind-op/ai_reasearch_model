# src/models/base.py

from __future__ import annotations

from pathlib import Path
from typing import Any


import torch
from torch import nn

from src.utils.logging import get_logger


log = get_logger(__name__)

class BaseModel(nn.Module):
    """
    Parent class for every model in this project.

    Why inherit from this instead of nn.Module directly?
    Because every model needs the same utilities — parameter
    counting, logging info, saving, loading — and writing them
    once here means every model gets them for free.

    Your concrete model only needs to define __init__ and forward.
    Everything else comes from here.

    Usage:
        class ResNetClassifier(BaseModel):
            def __init__(self, cfg: ModelConfig) -> None:
                super().__init__()
                # define your layers here

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # define your forward pass here
                return x
    """

    # ──────────────────────────────────────────────────────────
    # Parameter counting
    # ──────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only : bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: if True, only count parameters that
                            will be updated during training
                            (i.e. requires_grad=True).
                            If False, count everything including
                            frozen layers.

        Why this matters:
            A ResNet-18 has ~11M parameters total.
            When you freeze the backbone and only train the head,
            you have ~5K trainable parameters.
            Knowing this tells you if your freezing actually worked.
        """

        # A if condition else B
        # To understand this bit on code start from the inner if
        # p.requires_grad if trainable_only else True :  resolve this first
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )
    
    def log_info(self) -> None:
        """
        Log model architecture info.
        Call this once right after creating the model in train.py.

        Prints something like:
            model.info  architecture=ResNetClassifier
                        trainable=11,181,642  total=11,181,642
        """
        log.info(
            "model.info",
            architecture=self.__class__.__name__,
            trainable_params=f"{self.num_parameters(trainable_only=True)}",
            total_params=f"{self.num_parameters(trainable_only=False)}",
        )
    
    # ──────────────────────────────────────────────────────────
    # Saving and loading
    # ──────────────────────────────────────────────────────────

    def save(self, path : Path) -> None:
        """
        Save model weights to disk.

        We save state_dict(), not the whole model object.

        Why state_dict() and not torch.save(model)?
            Saving the full model object uses Python pickle and
            ties the saved file to your exact class definition.
            If you rename a layer or refactor the class, the file
            becomes unloadable.
            state_dict() is just a dict of parameter names → tensors.
            It loads into any model with the same architecture,
            regardless of how the class is organised.
        """
        # Type hints are not enforced at runtime, this redunduncy is a safety mechanism
        # Very common in good code
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        log.info("model.saved", path=str(path))
    
    def load(self, path : Path, strict : bool = True) -> None:
        """
        Load weights from a state_dict file.

        Args:
            path:   path to the .pt file saved by save()
            strict: if True, the saved weights must exactly match
                    the current model's parameter names.
                    if False, mismatched keys are ignored —
                    useful when loading a pretrained model that
                    has a different head than yours.

        Why map_location="cpu"?
            If the model was saved on GPU 0 and you load it on a
            machine with GPU 2 (or no GPU), PyTorch would crash
            trying to restore it to the original device.
            Loading to CPU first then moving to the right device
            works on any hardware configuration.
        """
        state = torch.load(
            Path(path),
            map_location="cpu",     # load to CPU first, move later
            weights_only=True,      # security : only load tensors
        )
        self.load_state_dict(state, strict=strict)
        log.info("model.loaded", path=str(path), strict=strict)
    
    # ──────────────────────────────────────────────────────────
    # Forward pass — must be implemented by subclasses
    # ──────────────────────────────────────────────────────────

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Subclasses MUST override this.

        Raises NotImplementedError if called directly on BaseModel —
        which would mean you forgot to implement forward() in your
        concrete model.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward"
        )

    
