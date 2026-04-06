import torch
import torch.nn as nn
from pathlib import Path
from torch.cuda.amp import GradScaler

from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.config.schema import load_config
from src.data.datamodule import DataModule
from src.models.resnet import ResNetClassifier
from src.training.loops import train_epoch

setup_logging(level="INFO")
cfg = load_config(Path("configs/base.yaml"))
set_seed(42)

model     = ResNetClassifier(cfg.model).to("cuda")
dm        = DataModule(cfg.data)
dm.setup()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler    = GradScaler(enabled=False)

# ── Count how many batches train_epoch actually processes ──────
original_len = len(dm.train_loader)
print(f"Loader has {original_len} batches")

metrics, global_step = train_epoch(
    model=model,
    loader=dm.train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=torch.device("cuda"),
    scaler=scaler,
    use_amp=False,
    amp_dtype=torch.float32,
    grad_clip_norm=None,
    grad_accumulation_steps=1,
    log_every_n_steps=999999,   # suppress step logs
    global_step=0,
)

print(f"global_step after epoch: {global_step}")
print(f"Expected global_step:    {original_len}")
print(f"Epoch acc:  {metrics['acc']:.4f}")
print(f"Epoch loss: {metrics['loss']:.4f}")
print(f"\nIf global_step={original_len} and acc>0.50 — loops.py is fine")
print(f"If global_step=1 and acc~0.10  — bug is in loops.py")