from pathlib import Path
from src.utils.logging import setup_logging
from src.config.schema import load_config
from src.data.datamodule import DataModule

setup_logging()
cfg = load_config(Path("configs/base.yaml"))

# ── Check 1: config values ─────────────────────────────────────
print("── Config check ──────────────────────────────────────────")
print(f"pretrained:  {cfg.model.pretrained}")
print(f"precision:   {cfg.training.precision}")
print(f"device:      {cfg.training.device}")
print(f"num_workers: {cfg.data.num_workers}")
print(f"mean:        {cfg.data.mean}")
print(f"std:         {cfg.data.std}")

# ── Check 2: pixel values after transforms ─────────────────────
print("\n── Pixel value check ─────────────────────────────────────")
dm = DataModule(cfg.data)
dm.setup()
x, y = next(iter(dm.train_loader))

print(f"Batch shape: {x.shape}")
print(f"Pixel min:   {x.min():.3f}")    # expect ~-2.4
print(f"Pixel max:   {x.max():.3f}")    # expect ~+2.7
print(f"Pixel mean:  {x.mean():.3f}")   # expect ~0.0
print(f"Pixel std:   {x.std():.3f}")    # expect ~1.0

# ── Check 3: are pretrained weights actually loaded ────────────
print("\n── Pretrained weights check ──────────────────────────────")
import torch
from src.models.resnet import ResNetClassifier

# Load with pretrained=True
model_pretrained = ResNetClassifier(cfg.model)

# Load with pretrained=False (random weights)
from src.config.schema import ModelConfig
model_random = ResNetClassifier(
    ModelConfig(pretrained=False, num_classes=10)
)

# If pretrained weights loaded correctly, the first conv layer
# weights should be very different from random initialisation
w_pretrained = model_pretrained.backbone.layer1[0].conv1.weight.data
w_random     = model_random.backbone.layer1[0].conv1.weight.data

are_different = not torch.allclose(w_pretrained, w_random)
print(f"Pretrained vs random weights differ: {are_different}")
# Should be True — if False, pretrained weights did not load

# Check the weight range — pretrained conv weights are small and structured
print(f"Pretrained weight mean: {w_pretrained.mean():.4f}")
print(f"Pretrained weight std:  {w_pretrained.std():.4f}")
print(f"Random weight mean:     {w_random.mean():.4f}")
print(f"Random weight std:      {w_random.std():.4f}")

# ── Check 4: forward pass with one batch ──────────────────────
print("\n── Forward pass check ────────────────────────────────────")
model_pretrained.eval()
with torch.no_grad():
    logits = model_pretrained(x.to(cfg.training.device))

probs = torch.softmax(logits, dim=-1)
print(f"Output shape:       {logits.shape}")        # [128, 10]
print(f"Mean max prob:      {probs.max(dim=-1).values.mean():.3f}")
# Random model: ~0.15 (uniform across 10 classes)
# Pretrained model: >0.30 even before any training (has prior knowledge)
print(f"Predicted classes:  {logits.argmax(-1)[:10].tolist()}")