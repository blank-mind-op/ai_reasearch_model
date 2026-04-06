import torch
import torch.nn as nn
from pathlib import Path
from torch.cuda.amp import GradScaler

from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.config.schema import load_config
from src.data.datamodule import DataModule
from src.models.resnet import ResNetClassifier

setup_logging(level="INFO")
cfg = load_config(Path("configs/base.yaml"))
set_seed(42)

# ── Build pieces ───────────────────────────────────────────────
model = ResNetClassifier(cfg.model).to("cuda")
dm    = DataModule(cfg.data)
dm.setup()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
loader    = dm.train_loader

# ── Manual training loop — no abstractions ─────────────────────
print(f"Total batches in loader: {len(loader)}")

model.train()
optimizer.zero_grad(set_to_none=True)

total_correct = 0
total_samples = 0
total_loss    = 0.0

for step, (x, y) in enumerate(loader):
    x = x.to("cuda", non_blocking=True)
    y = y.to("cuda", non_blocking=True)

    logits = model(x)
    loss   = criterion(logits, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    total_correct += (logits.argmax(-1) == y).sum().item()
    total_samples += y.size(0)
    total_loss    += loss.item()

    # Print first 5 steps to confirm the loop is running
    if step < 5:
        acc_so_far = total_correct / total_samples
        print(f"  step={step+1:03d}  loss={loss.item():.4f}  "
                f"acc_so_far={acc_so_far:.4f}")

print(f"\nFinal step reached: {step + 1}")
print(f"Epoch loss:   {total_loss / len(loader):.4f}")
print(f"Epoch acc:    {total_correct / total_samples:.4f}")
print(f"Expected acc > 0.60 with pretrained weights")