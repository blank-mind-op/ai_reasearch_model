from src.utils.logging import setup_logging
from src.utils.profiling import profiler
import torch

setup_logging()

# Simulate what the trainer will do
# enabled=False — should do nothing at all
with profiler(enabled=False) as prof:
    print(f"prof is None: {prof is None}")   # True
    for step in range(10):
        x = torch.randn(128, 3, 32, 32)
        # prof is None so this branch never runs
        if prof is not None:
            prof.step()

print("Disabled profiler: no output above — correct")
