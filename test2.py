from src.utils.logging import setup_logging, get_logger
from src.utils.reproducibility import set_seed


# Set up logging first
setup_logging(level="DEBUG")

# Get a logger for this module
log = get_logger("test")

# These are the exact calls you'll use throughout the project
log.info("training.start", epoch=20, device="cuda")
log.info("epoch.done", epoch=5, loss=0.32, acc=0.91)
log.warning("checkpoint.not_found", path="checkpoints/best.pt")
log.debug("step", step=100, loss=0.41, lr="1.00e-03")

# Set seed and verify reproducibility
set_seed(42)

import torch
a = torch.randn(3)

set_seed(42)
b = torch.randn(3)

print(f"\nSame result both times: {torch.allclose(a, b)}")

