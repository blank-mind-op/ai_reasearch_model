
from src.utils.logging import setup_logging
from src.config.schema import DataConfig
from src.data.datamodule import DataModule, CIFAR10_CLASSES

setup_logging()

# Create and setup the datamodule
cfg = DataConfig(batch_size=64, num_workers=0) # 0 workers for testing
dm = DataModule(cfg)
dm.setup() # downloads CIFAR-10 if not already present

# ── Check train loader ────────────────────────────────────────
train_loader = dm.train_loader
x, y = next(iter(train_loader))

print(f"Batch input shape:  {x.shape}")   # [64, 3, 32, 32]
print(f"Batch label shape:  {y.shape}")   # [64]
print(f"Pixel value range:  {x.min():.2f} to {x.max():.2f}")
# should be roughly -2.0 to 2.0 after normalisation
# NOT 0.0 to 1.0 — that would mean normalisation didn't apply

print(f"\nFirst 8 labels: {y[:8].tolist()}")
print(f"Their classes:  {[CIFAR10_CLASSES[i] for i in y[:8]]}")

# ── Check val loader ──────────────────────────────────────────
val_loader = dm.val_loader
x_val, y_val = next(iter(val_loader))
print(f"\nVal batch shape: {x_val.shape}")   # [64, 3, 32, 32]

# ── Verify no overlap between train and val ───────────────────
train_indices = set(dm._train_ds.indices)
val_indices = set(dm._val_ds.indices)

overlap = train_indices & val_indices

print(f"\nTrain/val overlap: {len(overlap)} samples")  # must be 0

# ── Verify sizes are right ────────────────────────────────────    
print(f"Train samples: {len(dm._train_ds)}")   # 45,000
print(f"Val samples:   {len(dm._val_ds)}")     # 5,000
print(f"Total:         {len(dm._train_ds) + len(dm._val_ds)}")  # 50,000

