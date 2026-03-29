# src/models/resnet.py

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.config.schema import ModelConfig
from src.models.base import BaseModel
from src.utils.logging import get_logger

log = get_logger(__name__)

class ResNetClassifier(BaseModel):
    """
    Pretrained ResNet with a custom classification head.

    This model knows about:
        - Which backbone to load
        - How to replace the head for a custom number of classes
        - The forward pass

    This model knows NOTHING about:
        - Which device it runs on (the Trainer handles that)
        - Batch size (comes from the DataLoader)
        - Learning rate (belongs to the optimizer)
        - Augmentation (belongs to the DataModule)
        - Saving/loading (inherited from BaseModel)

    Why pretrained weights?
        ResNet was trained on 1.2 million ImageNet images.
        Its early layers already detect edges, corners, and textures.
        Its middle layers detect shapes and parts.
        We keep all of that and only replace the final layer
        that maps features → class predictions.
        This is called transfer learning — we transfer knowledge
        from ImageNet to our task.
        Result: 91%+ accuracy in 20 epochs instead of ~60% from scratch.

    Architecture after modification:
        ResNet backbone (pretrained, kept as-is)
            ↓ outputs a vector of size 512 (resnet18) or 2048 (resnet50)
        Dropout (regularisation — randomly zeros some values during training)
            ↓
        Linear(512 → num_classes)
            ↓ outputs raw logits, one per class
    """
    # calls nn.module __init__ because BaseModel has not __init__ implementation
    def __init__(self, cfg : ModelConfig) -> None:
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────
        # "IMAGENET1K_V1" = the standard ImageNet pretrained weights
        # None = random initialisation (no pretrained weights)

        weights = "IMAGENET1K_V1" if cfg.pretrained else None

        # python does not have block scopes for if , only functions and classes
        if cfg.backbone == "resnet18":
            backbone = tv_models.resnet18(weights=weights)
            # resnet18's final layer outputs a 512-dim vector
            in_features = 512
        elif cfg.backbone == "resnet50":
            backbone = tv_models.resnet50(weights=weights)
            # resnet50's final layer outputs a 2048-dim vector
            in_features = 2048
        else:
            # This should never happen because Pydantic validates
            # backbone against Literal["resnet18", "resnet50"]
            # before we ever reach this code
            raise ValueError(f"Unknown backbone : {cfg.backbone}")
        
        # ── Replace the classification head ───────────────────
        # The original head: Linear(in_features, 1000)
        # 1000 = number of ImageNet classes
        #
        # We replace it with our own head:
        # Dropout → Linear(in_features, num_classes)
        #
        # Dropout(p) randomly zeros p% of inputs during training.
        # This forces the network not to rely on any single feature,
        # which reduces overfitting.
        # During model.eval() dropout is automatically disabled.

        backbone.fc = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(in_features=in_features, out_features=cfg.num_classes),
        )

        self.backbone = backbone

        log.debug(
            "model.created",
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            in_features=in_features,
            dropout=cfg.dropout,
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x: image tensor of shape [batch_size, 3, H, W]
               Values should be normalised (mean~0, std~1)
               NOT raw pixels (0-255) or unnormalised (0.0-1.0)

        Returns:
            logits: tensor of shape [batch_size, num_classes]
                    Raw scores — NOT probabilities, NOT softmax output.

        Why return raw logits and not softmax probabilities?
            nn.CrossEntropyLoss expects raw logits and applies
            log_softmax internally. If you apply softmax here AND
            CrossEntropyLoss applies it again, you get wrong gradients
            and training quietly fails.
            Only apply softmax at inference time when you want
            actual probabilities.
        """
        return self.backbone(x)