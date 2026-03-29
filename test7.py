import torch
import torch.nn as nn
from torch import GradScaler

from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.config.schema import ModelConfig, DataConfig
from src.models.resnet import ResNetClassifier
from src.data.datamodule import DataModule
from src.training.loops import train_epoch, eval_epoch

setup_logging(level="DEBUG")
set_seed(42)

# ── Decide device ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Build a small model ───────────────────────────────────────
model_cfg = ModelConfig(backbone="resnet18", num_classes=10, pretrained=False)
model = ResNetClassifier(model_cfg).to(device)

# ── Build data pipeline ───────────────────────────────────────
data_cfg = DataConfig(batch_size=64, num_workers=0)
dm = DataModule(data_cfg)
dm.setup()

# ── Build optimizer and loss ──────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler(enabled=False)   # fp32 for testing

# ── Decide AMP settings ───────────────────────────────────────
use_amp   = False
amp_dtype = torch.float32

# ── Run one training epoch ────────────────────────────────────
print("\n── Running train_epoch ──────────────────────────────────")
train_metrics, global_step = train_epoch(
    model=model,
    loader=dm.train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    scaler=scaler,
    use_amp=use_amp,
    amp_dtype=amp_dtype,
    grad_clip_norm=1.0,
    grad_accumulation_steps=1,
    log_every_n_steps=100,
    global_step=0,
)
print(f"Train metrics: {train_metrics}")
print(f"Global step:   {global_step}")

# ── Run one eval epoch ────────────────────────────────────────
print("\n── Running eval_epoch ───────────────────────────────────")
val_metrics = eval_epoch(
    model=model,
    loader=dm.val_loader,
    criterion=criterion,
    device=device,
    use_amp=use_amp,
    amp_dtype=amp_dtype,
)
print(f"Val metrics: {val_metrics}")

# ── Sanity checks ─────────────────────────────────────────────
print("\n── Sanity checks ────────────────────────────────────────")

# Loss should be around log(10) ≈ 2.3 for random weights
# on a 10-class problem (no better than chance)
import math
expected_loss = math.log(10)
print(f"Expected loss (random weights): ~{expected_loss:.2f}")
print(f"Actual train loss:               {train_metrics['loss']:.2f}")
print(f"Actual val loss:                 {val_metrics['loss']:.2f}")

# Accuracy should be around 0.10 (10% = random for 10 classes)
print(f"\nExpected accuracy (random): ~0.10")
print(f"Actual train accuracy:       {train_metrics['acc']:.2f}")
print(f"Actual val accuracy:         {val_metrics['acc']:.2f}")

# All keys present
assert "loss"     in train_metrics, "missing loss"
assert "acc"      in train_metrics, "missing acc"
assert "top5_acc" in train_metrics, "missing top5_acc"
print("\nAll metric keys present: OK")

# global_step should equal number of batches
# (45000 samples / 64 batch_size, drop_last=True)
expected_steps = len(dm.train_loader)
print(f"\nExpected global_step: {expected_steps}")
print(f"Actual global_step:   {global_step}")
