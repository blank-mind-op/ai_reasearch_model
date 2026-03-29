import torch
from src.utils.logging import setup_logging
from src.config.schema import ModelConfig
from src.models.resnet import ResNetClassifier

setup_logging()

# Create the model
cfg = ModelConfig(backbone="resnet18", num_classes=10, pretrained=False)
model = ResNetClassifier(cfg)

# Log its info — should show ~11M params
model.log_info()

# Test the forward pass with a fake batch
# Shape: [4 images, 3 channels, 32x32 pixels]
x = torch.randn(4, 3, 32, 32)
model.eval()
with torch.no_grad():
    out = model(x)

print(f"\nInput shape:  {x.shape}")         # [4, 3, 32, 32]
print(f"Output shape: {out.shape}")         # [4, 10]
print(f"Output is logits (not softmax): "
      f"{not torch.allclose(out.sum(dim=-1), torch.ones(4))}")

# Test parameter counting
print(f"\nTotal params:     {model.num_parameters(trainable_only=False):,}")
print(f"Trainable params: {model.num_parameters(trainable_only=True):,}")

# Test save and load
from pathlib import Path
model.save(Path("checkpoints/test_model.pt"))
model.load(Path("checkpoints/test_model.pt"))
print("\nSave and load: OK")
