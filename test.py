from pathlib import Path
from src.config.schema import load_config

cfg = load_config(Path("configs/base.yaml"))

# These should all print the right values
print(cfg.model.backbone)          # resnet18
print(cfg.training.max_epochs)     # 20
print(cfg.optimizer.lr)            # 0.001
print(cfg.data.batch_size)         # 128

# Test that validation catches errors
from src.config.schema import ModelConfig
try:
    bad = ModelConfig(num_classes=-5)
except Exception as e:
    print(f"Caught error: {e}")    # should print a validation error