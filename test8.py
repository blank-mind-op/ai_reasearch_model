import torch
import torch.nn as nn
from pathlib import Path

from src.utils.logging import setup_logging
from src.config.schema import ModelConfig, CheckpointConfig
from src.models.resnet import ResNetClassifier
from src.training.checkpoint import CheckpointManager

setup_logging(level="DEBUG")

# ── Setup ──────────────────────────────────────────────────────
model_cfg = ModelConfig(backbone="resnet18", num_classes=10, pretrained=False)
model     = ResNetClassifier(model_cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

ckpt_cfg  = CheckpointConfig(
    save_dir=Path("checkpoints/test"),
    save_top_k=3,
    monitor_metric="val_acc",
    monitor_mode="max",
)
manager = CheckpointManager(ckpt_cfg)

# ── Simulate 6 epochs with improving then declining accuracy ───
print("\n── Saving 6 checkpoints (top-k=3) ──────────────────────")
fake_accuracies = [0.72, 0.81, 0.87, 0.91, 0.89, 0.93]

for epoch, acc in enumerate(fake_accuracies):
    manager.save(
        epoch=epoch,
        metric=acc,
        model=model,
        optimizer=optimizer,
        extra={"val_metrics": {"acc": acc, "loss": 1.0 - acc}},
    )
    files = list(Path("checkpoints/test").glob("epoch_*.pt"))
    print(f"  Epoch {epoch} | acc={acc:.2f} | "
            f"checkpoints on disk: {len(files)}")

# ── Check that only top-3 remain ──────────────────────────────
print("\n── Checkpoints remaining on disk ────────────────────────")
remaining = sorted(Path("checkpoints/test").glob("epoch_*.pt"))
for f in remaining:
    print(f"  {f.name}")

assert len(remaining) == 3, f"Expected 3, got {len(remaining)}"
print(f"\nCorrectly kept top 3 — OK")

# ── Check best.pt has the highest accuracy ─────────────────────
print(f"\nBest metric tracked: {manager.best_metric:.2f}")  # 0.93
assert manager.best_metric == 0.93

# Read the human-readable summary
import json
summary = json.loads(
    Path("checkpoints/test/best_summary.json").read_text()
)
print(f"Best summary: {summary}")

# ── Test load_best restores weights correctly ──────────────────
print("\n── Testing load_best ────────────────────────────────────")

# Corrupt the model weights so we can verify loading restores them
original_weight = model.backbone.fc[1].weight.data.clone()
model.backbone.fc[1].weight.data.fill_(999.0)  # corrupt

state = manager.load_best(model, optimizer)

restored_weight = model.backbone.fc[1].weight.data
weights_match = torch.allclose(original_weight, restored_weight)
print(f"Weights correctly restored: {weights_match}")
print(f"Loaded from epoch: {state['epoch']}")   # should be 5
print(f"Best val_acc:      {state['metric']}")  # should be 0.93

# ── Test resume epoch logic ────────────────────────────────────
resume_from_epoch = state["epoch"] + 1
print(f"\nIf resuming, start from epoch: {resume_from_epoch}")  # 6

# ── Cleanup test checkpoints ──────────────────────────────────
import shutil
shutil.rmtree("checkpoints/test")
print("\nTest checkpoints cleaned up — OK")

import heapq

# Simulating max mode (accuracy) — we want to evict lowest accuracy
heap = []
accuracies = [0.72, 0.81, 0.87, 0.91, 0.89, 0.93]

for acc in accuracies:
    sort_key = acc          # max mode: use +metric
    heapq.heappush(heap, (sort_key, acc))
    if len(heap) > 3:
        removed = heapq.heappop(heap)
        print(f"Evicted accuracy: {removed[1]:.2f}")

print(f"\nKept: {sorted([x[1] for x in heap], reverse=True)}")
# Should keep: [0.93, 0.91, 0.89]

print()

# Simulating min mode (loss) — we want to evict highest loss
heap = []
losses = [1.45, 1.12, 0.87, 0.54, 0.61, 0.21]

for loss in losses:
    sort_key = -loss        # min mode: negate
    heapq.heappush(heap, (sort_key, loss))
    if len(heap) > 3:
        removed = heapq.heappop(heap)
        print(f"Evicted loss: {removed[1]:.2f}")

print(f"\nKept: {sorted([x[1] for x in heap])}")
# Should keep: [0.21, 0.54, 0.61]