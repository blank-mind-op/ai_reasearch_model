import torch
import torch.nn as nn
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.profiler import record_function

from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.config.schema import load_config
from src.data.datamodule import DataModule
from src.models.resnet import ResNetClassifier

setup_logging()
cfg = load_config(Path("configs/base.yaml"))
set_seed(42)

model     = ResNetClassifier(cfg.model).to("cuda")
dm        = DataModule(cfg.data)
dm.setup()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler(enabled=False)
loader    = dm.train_loader

grad_accumulation_steps = 1
log_every_n_steps       = 999999
global_step             = 0
use_amp                 = False
amp_dtype               = torch.float32
grad_clip_norm          = None
device                  = torch.device("cuda")

model.train()
optimizer.zero_grad(set_to_none=True)

# ── Manually replicate loops.py exactly ───────────────────────
from torch import autocast

print(f"Starting loop over {len(loader)} batches")

for step, (x, y) in enumerate(loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with record_function("forward"):
        with autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(x)
            loss   = criterion(logits, y) / grad_accumulation_steps

    scaler.scale(loss).backward()

    is_update_step = (
        (step + 1) % grad_accumulation_steps == 0
        or (step + 1) == len(loader)
    )

    print(f"step={step}  is_update_step={is_update_step}  "
          f"loss={loss.item():.4f}")

    if is_update_step:
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if step >= 4:   # only run 5 steps
        print(f"\nStopping early at step {step}")
        break

print(f"\nglobal_step: {global_step}")