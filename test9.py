import torch
import torch.nn as nn
from pathlib import Path

from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed
from src.config.schema import load_config
from src.data.datamodule import DataModule
from src.models.resnet import ResNetClassifier
from src.training.trainer import Trainer

# ── Setup ──────────────────────────────────────────────────────
setup_logging(level="INFO")
cfg = load_config(Path(r"configs\base.yaml"))

# Override to 2 epochs so the test runs fast
cfg.training.max_epochs = 2
cfg.data.num_workers    = 0    # simpler for testing
cfg.training.profile    = False

set_seed(cfg.training.seed)

# ── Build components ───────────────────────────────────────────
model      = ResNetClassifier(cfg.model)
model.log_info()

datamodule = DataModule(cfg.data)

optimizer  = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.optimizer.lr,
    weight_decay=cfg.optimizer.weight_decay,
)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.training.max_epochs,
    eta_min=cfg.scheduler.min_lr,
)

# ── Run training ───────────────────────────────────────────────
trainer = Trainer(
    cfg=cfg,
    model=model,
    datamodule=datamodule,
    optimizer=optimizer,
    scheduler=scheduler,
)

print("\n── Starting 2-epoch training run ─────────────────────────")
final_metrics = trainer.fit()

print(f"\n── Final metrics ─────────────────────────────────────────")
print(f"  val loss: {final_metrics.get('loss', 'N/A')}")
print(f"  val acc:  {final_metrics.get('acc', 'N/A')}")
print(f"  val top5: {final_metrics.get('top5_acc', 'N/A')}")

# ── Verify checkpoints were created ───────────────────────────
from pathlib import Path
best = Path("checkpoints/best.pt")
assert best.exists(), "best.pt was not created"
print(f"\nbest.pt exists: OK")

import json
summary = json.loads(
    Path("checkpoints/best_summary.json").read_text()
)
print(f"best_summary.json: {summary}")

# ── Verify the model returns right output shape ────────────────
model.eval()
x = torch.randn(4, 3, 32, 32).to(cfg.training.device)
with torch.no_grad():
    out = model(x)
assert out.shape == (4, 10), f"Wrong output shape: {out.shape}"
print(f"\nPost-training output shape: {out.shape} — OK")

print("\n── All checks passed ─────────────────────────────────────")