# src/utils/profiling.py

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.profiler as torch_profiler

from src.utils.logging import get_logger

log = get_logger(__name__)

@contextmanager
def profiler(
    enabled : bool = True,
    output_dir : Path = Path("./profiler_logs"),
    profile_steps : int = 5,
    record_shapes : bool = True,
    profile_memory : bool = True,
) -> Iterator[torch_profiler.profile | None]:
    """
    Context manager that wraps torch.profiler.

    When enabled=False it does absolutely nothing — zero overhead.
    This means you can add profiling to your training loop and
    control it entirely from config, without touching the loop code.

    Usage in trainer.py:
        with profiler(enabled=cfg.profile, output_dir=...) as prof:
            for step, batch in enumerate(loader):
                train_step(batch)
                if prof is not None:
                    prof.step()   # tells profiler to advance schedule

    View results:
        tensorboard --logdir ./profiler_logs

    Args:
        enabled:        False = do nothing, True = run profiler
        output_dir:     where to write TensorBoard trace files
        profile_steps:  how many steps to actually record
                        (after warmup — see schedule below)
        record_shapes:  record input tensor shapes per operation
                        useful for spotting unexpected reshapes
        profile_memory: track GPU memory allocation per operation
                        useful for finding memory hogs
    """
    if not enabled:
        # yield None so the caller can do `if prof is not None: prof.step()`
        yield None
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "profiler.start",
        output_dir=output_dir,
        profile_steps=profile_steps,
    )

    # The schedule controls exactly which steps get recorded:
    #
    #   wait=1    — skip the very first step
    #               (JIT compilation happens here, distorts timings)
    #
    #   warmup=1  — run the profiler but discard results
    #               (cache warming — first real step is often slower)
    #
    #   active=profile_steps — actually record these steps
    #
    #   repeat=1  — do this cycle once then stop
    #
    # Total steps before profiler stops: wait + warmup + active = 1+1+5 = 7

    schedule = torch_profiler.schedule(
        wait=1,
        warmup=1,
        active=profile_steps,
        repeat=1,
    )

    # on_trace_ready is called when the schedule completes —
    # it writes the trace to output_dir in TensorBoard format

    trace_handler = torch_profiler.tensorboard_trace_handler(
        str(output_dir)
    )

    with torch_profiler.profile(
        activities=[
            # CPU operations — Python overhead, data loading
            torch_profiler.ProfilerActivity.CPU,
            # CUDA operations — actual GPU kernels
            torch_profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        # with_stack records the Python call stack per operation
        # makes the flame graph in TensorBoard much more useful
        with_stack=True,
    ) as prof:
        yield prof
    

    # Print a quick summary to terminal so you get instant feedback
    # without opening TensorBoard
    print("\n── Profiler summary (top 10 by CUDA time) ──────────────")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10,
        )
    )
    print(f"\nFull trace written to: {output_dir}")
    print("View with: tensorboard --logdir", output_dir)
    log.info("profiler.done", output_dir=str(output_dir))

# prof.step() :  a step corresponds to one iteration in the training loop
# Training step 0 → WAIT    → profiler OFF
# Training step 1 → WARMUP  → profiler ON but NOT saving
# Training step 2 → ACTIVE  → RECORD
# Training step 3 → ACTIVE  → RECORD
# Training step 4 → ACTIVE  → RECORD
# Training step 5 → ACTIVE  → RECORD
# Training step 6 → ACTIVE  → RECORD