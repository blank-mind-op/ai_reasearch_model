# src/training/loops.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch import GradScaler, autocast

# They simplified the API again:

# Before: torch.cuda.amp.autocast
# Then:   torch.amp.autocast("cuda")
# Now:    torch.autocast("cuda")

from torch.profiler import record_function
from torch.utils.data import DataLoader

from src.training.metrics import MetricTracker
from src.utils.logging import get_logger

log = get_logger(__name__)


def train_epoch(
        model : nn.Module,
        loader : DataLoader,
        optimizer : torch.optim.Optimizer,
        criterion : nn.Module,
        device : torch.device,
        scaler : GradScaler,
        use_amp : bool,
        amp_dtype : torch.dtype,
        grad_clip_norm : float | None,
        grad_accumulation_steps : int,
        log_every_n_steps : int,
        global_step : int,
) -> tuple[dict[str, float], int]:
    """
    Run one full pass over the training data.

    Args:
        model:                  the neural network
        loader:                 DataLoader for training data
        optimizer:              AdamW, SGD, etc.
        criterion:              loss function (CrossEntropyLoss)
        device:                 torch.device("cuda") etc.
        scaler:                 GradScaler for fp16 AMP
                                (does nothing when fp16 disabled)
        use_amp:                whether to use mixed precision
        amp_dtype:              torch.bfloat16 or torch.float16
        grad_clip_norm:         max gradient norm, None = disabled
        grad_accumulation_steps: accumulate gradients over N steps
                                before calling optimizer.step()
        log_every_n_steps:      how often to log step-level metrics
        global_step:            total steps taken so far across
                                all epochs — used for logging

    Returns:
        tuple of:
            metrics dict: {"loss": float, "acc": float, "top5_acc": float}
            updated global_step: int
    """
    model.train()
    # model.train() enables:
    #   - Dropout: randomly zeros neurons to prevent overfitting
    #   - BatchNorm: uses current batch statistics (not running stats)

    tracker = MetricTracker()

    # Zero gradients at the start of the epoch, not inside the loop
    # set_to_none=True is faster than zeroing — it frees the memory
    # instead of writing zeros, which saves a memory write operation
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(loader):
        # ── Move data to device ───────────────────────────────
        # non_blocking=True makes the transfer asynchronous —
        # the CPU doesn't wait for the GPU to confirm receipt.
        # This overlaps the data transfer with other CPU work,
        # making the pipeline slightly faster.

        
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
        
        # ── Forward pass ──────────────────────────────────────
        # autocast runs the forward pass in bf16/fp16 automatically.
        # PyTorch decides which operations to run in reduced precision
        # and which must stay in fp32 (e.g. loss computation).
        # When use_amp=False, autocast does nothing — same as fp32.

        
        with autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(x)
            # Divide loss by accumulation steps.
            # Why? Gradients accumulate additively across
            # N backward passes. Without dividing,
            # the accumulated gradient is N times larger
            # than a single large-batch gradient would be.
            # Dividing here makes it equivalent.

            loss = criterion(logits, y) / grad_accumulation_steps
        
        # ── Backward pass ─────────────────────────────────────
        # scaler.scale(loss) multiplies the loss by the scaler's
        # scale factor before backward(). This prevents fp16
        # gradients from underflowing to zero.
        # When fp16 is disabled, scaler is a no-op and this is
        # equivalent to loss.backward().
        
        scaler.scale(loss).backward()
        
        # ── Optimizer step ────────────────────────────────────
        # We only update weights every grad_accumulation_steps steps.
        # The condition also handles the last batch in the epoch
        # which may not complete a full accumulation cycle.

        is_update_step = (
            (step + 1) % grad_accumulation_steps == 0
            or (step + 1) == len(loader)
        )

        if is_update_step:
            
            # Gradient clipping
            # Before clipping we must unscale the gradients —
            # scaler multiplied them earlier and we need the
            # real values to compare against grad_clip_norm.
            # When fp16 is disabled this is a no-op.
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_norm,
                )

            # scaler.step() calls optimizer.step() internally
            # but first checks if any gradients are inf/nan
            # (which happens with fp16). If so, it skips the
            # step and adjusts the scale factor downward.
            # When fp16 disabled: equivalent to optimizer.step()

            scaler.step(optimizer)

            # Updates the scale factor for next iteration.
            # Grows it if no inf/nan were found (more precision),
            # shrinks it if they were (more stability).
            # When fp16 disabled: no-op.

            scaler.update()

            # clear gradient for next accumulation cycle

            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1 # counts the number of weights updates

            # Step-level logging — only on update steps so the
            # step number matches actual weight updates

            if global_step % log_every_n_steps == 0:
                log.debug(
                    "train.step",
                    step=global_step,
                    loss=round(loss.item() * grad_accumulation_steps, 4),
                    lr=f"{optimizer.param_groups[0]["lr"]:2e}",
                )
        
        # ── Accumulate metrics ────────────────────────────────
        # .detach() removes the tensor from the computation graph.
        # Without this, every loss value would keep a reference to
        # the full forward/backward graph, causing memory to grow
        # every step until you run out of GPU RAM.
        #
        # We multiply back by grad_accumulation_steps because we
        # divided earlier for gradient scaling — the tracker should
        # see the real loss value, not the scaled one.
        tracker.update(
            loss=loss.detach().item() * grad_accumulation_steps,
            logits=logits.detach(),
            labels=y,
        )

    return tracker.compute(), global_step


@torch.inference_mode()
def eval_epoch(
    model : nn.Module,
    loader : DataLoader,
    criterion : nn.Module,
    device : torch.device,
    use_amp : bool,
    amp_dtype : torch.dtype,
) -> dict[str, float]:
    """
    Run one full pass over the validation data.

    @torch.no_grad() decorator disables gradient tracking for
    the entire function. This means:
        - No computation graph is built during forward passes
        - ~30% less memory usage
        - ~20% faster than with gradients enabled
    We don't need gradients here because we're not updating weights.

    Args:
        model:     the neural network
        loader:    DataLoader for validation data
        criterion: loss function
        device:    torch.device
        use_amp:   whether to use mixed precision
        amp_dtype: torch.bfloat16 or torch.float16

    Returns:
        metrics dict: {"loss": float, "acc": float, "top5_acc": float}
    """

    model.eval()
    # model.eval() disables:
    #   - Dropout: all neurons active during evaluation
    #   - BatchNorm: uses running mean/std instead of batch stats
    # Forgetting model.eval() before validation is one of the most
    # common bugs — metrics will be noisier and slightly worse.

    tracker = MetricTracker()

    for x, y in loader:
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)

        with autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(x)
            loss = criterion(logits, y)

        # No .detach() needed here because @torch.no_grad()
        # already prevents any graph from being built

        tracker.update(
            loss=loss.item(),
            logits=logits,
            labels=y,
        )

    return tracker.compute()


