# src/training/trainer.py

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch import GradScaler

from src.config.schema import ExperimentConfig
from src.data.datamodule import DataModule
from src.training.checkpoint import CheckpointManager
from src.training.loops import train_epoch, eval_epoch
from src.utils.logging import get_logger
from src.utils.profiling import profiler

log = get_logger(__name__)

class Trainer:
    """
    Orchestrates the full training lifecycle.

    The Trainer owns:
        - device management (moves model to the right device)
        - AMP setup (GradScaler, dtype)
        - the epoch loop
        - calling train_epoch() and eval_epoch()
        - logging epoch-level summaries
        - checkpointing after each eval
        - scheduler stepping
        - resuming from a checkpoint

    The Trainer does NOT own:
        - model architecture (lives in src/models/)
        - data loading (lives in src/data/)
        - metric computation (lives in src/training/metrics.py)
        - per-step training logic (lives in src/training/loops.py)
        - hyperparameter values (lives in configs/)

    This separation means you can swap any component without
    touching the Trainer. New model? New dataset? New metrics?
    The Trainer doesn't change.

    Usage:
        trainer = Trainer(cfg, model, datamodule, optimizer, scheduler)
        trainer.fit()
    """
    def __init__(
        self,
        cfg : ExperimentConfig,
        model : nn.Module,
        datamodule : DataModule,
        optimizer : torch.optim.optimizer,
        scheduler : torch.optim.lr_scheduler.LRScheduler | None = None,
        loss_fn : nn.Module | None = None,
    ) -> None:
        self.cfg = cfg
        self.tcfg = cfg.training
        self.ocfg = cfg.optimizer
        self.device = torch.device(self.tcfg.device)
        
        # ── Model ─────────────────────────────────────────────
        # Move to device here, in the Trainer — never inside the model.
        # The model is device-agnostic by design.
        self.model = model.to(self.device)
        
        # torch.compile() fuses GPU operations into optimised kernels.
        # First call takes ~30s to compile. Every call after is faster.
        # Keep False during development, enable for long training runs.
        if self.tcfg.compile_model:
            log.info("model.compiling", note="first run takes about 30s")
            self.model = torch.compile(self.model)
        
        # ── Other components ──────────────────────────────────
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Default loss: CrossEntropyLoss with label smoothing.
        # label_smoothing=0.1 means instead of training toward
        # [0,0,1,0,0] (one-hot) we train toward [0.01,0.01,0.91,0.01,0.01].
        # Prevents the model from being overconfident.
        # Typical accuracy gain: +0.3 to +0.7%.
        
        # Note: the python or operator does short circuit evaluation, it returns the first truthy
        #       value, so if loss_fn is defined the expression si True without evaluating the rest,
        #       so loss_fn is returned immediatly without looking at the rest and
        #       nn.CrossEntropyLoss() is not even created
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(label_smoothing=0.1)
        
        
        self.ckpt = CheckpointManager(cfg.checkpoint)
        
        # ── AMP setup ─────────────────────────────────────────
        # bf16: modern GPUs (A100, 3090, 4090) — no scaler needed
        # fp16: older GPUs — GradScaler prevents gradient underflow
        # fp32: CPU or when numerical issues arise — no AMP at all
        self._use_amp = self.tcfg.precision in ("fp16", "bf16")
        self._amp_type = (
            torch.bfloat16
            if self.tcfg.precision == "bf16"
            else torch.float16
        )
        
        # GradScaler is only active for fp16.
        # When enabled=False it's a complete no-op —
        # scaler.scale(loss) just returns loss unchanged.
        self._scaler = GradScaler(
            enabled=(self.tcfg.precision == "fp16")
        )
        
        # ── Training state ────────────────────────────────────
        # global_step counts total optimizer steps across all epochs.
        # Used for step-level logging so you can plot loss vs step
        # rather than loss vs epoch (more granular).
        
        self._global_step : int  = 0
    
    # ═══════════════════════════════════════════════════════════
    # Public entry point
    # ═══════════════════════════════════════════════════════════
    
    def fit(self) -> dict[str, float]:
        """
        Run the full training loop from start (or checkpoint) to finish.

        Flow:
            1. Setup data (download, split, create loaders)
            2. Optionally resume from checkpoint
            3. For each epoch:
                a. train_epoch()
                b. eval_epoch() (every eval_every_n_epochs epochs)
                c. checkpoint if eval was run
                d. step scheduler
            4. Load best checkpoint
            5. Return final val metrics

        Returns:
            Final validation metrics from the best checkpoint epoch.
        """
        # ── Setup data ────────────────────────────────────────
        self.datamodule.setup()
        train_loader = self.datamodule.train_loader
        val_loader = self.datamodule.val_loader
        
        
        log.info(
            "training.start",
            experiment=self.cfg.name,
            epochs=self.tcfg.max_epochs,
            device=str(self.device),
            precision=self.tcfg.precision,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            amp=self._use_amp,
        )
        
        # ── Maybe resume from checkpoint ──────────────────────
        start_epoch = self._maybe_resume()
        
        # ── Epoch loop ────────────────────────────────────────
        # The profiler context wraps the entire loop.
        # When cfg.training.profile=False it's a transparent no-op.
        # When True it records the first few steps and writes traces.
        
        with profiler(
            enabled=self.tcfg.profile,
            output_dir=Path("profiler_logs"),
            profile_steps=self.tcfg.profile_steps,
        ) as prof:
            
            for epoch in range(start_epoch, self.tcfg.max_epochs):
                train_metrics, self._global_step = train_epoch(
                    model=self.model,
                    loader=train_loader,
                    optimizer=self.optimizer,
                    criterion=self.loss_fn,
                    device=self.device,
                    scaler=self._scaler,
                    use_amp=self._use_amp,
                    amp_dtype=self._amp_type,
                    grad_clip_norm=self.ocfg.grad_clip_norm,
                    grad_accumulation_steps=self.ocfg.grad_accumulation_steps,
                    log_every_n_steps=self.tcfg.log_every_n_steps,
                    global_step=self._global_step,
                )
                
                log.info(
                    "epoch.train",
                    epoch=epoch,
                    loss=round(train_metrics["loss"], 4),
                    acc=round(train_metrics["acc"], 4),
                    top5=round(train_metrics["top5_acc"], 4),
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    step=self._global_step,
                )
                
                # ── Eval ──────────────────────────────────────
                # We don't eval every epoch by default —
                # eval_every_n_epochs=1 means every epoch,
                # eval_every_n_epochs=2 means every other epoch, etc.
                # Skipping eval occasionally speeds up short-epoch runs.
                should_eval = (
                    (epoch % self.tcfg.eval_every_n_epochs == 0)
                    or (epoch == self.tcfg.max_epochs - 1)
                    # always eval on the last epoch
                )
                
                if should_eval:
                    val_metrics = eval_epoch(
                        model=self.model,
                        loader=val_loader,
                        criterion=self.loss_fn,
                        device=self.device,
                        use_amp=self._use_amp,
                        amp_dtype=self._amp_type,
                    )

                    log.info(
                        "epoch.val",
                        epoch=epoch,
                        loss=round(val_metrics["loss"], 4),
                        acc=round(val_metrics["acc"], 4),
                        top5=round(val_metrics["top5_acc"], 4),
                    )
                    
                    # ── Checkpoint ────────────────────────────
                    # Determine which metric to monitor based on config.
                    # "val_acc" → use acc, "val_loss" → use loss.
                    monitored_value = (
                        val_metrics["acc"]
                        if "acc" in self.cfg.checkpoint.monitor_metric
                        else val_metrics["loss"]
                    )

                    self.ckpt.save(
                        epoch=epoch,
                        metric=monitored_value,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        extra={
                            "train_metrics": train_metrics,
                            "val_metrics": val_metrics,
                            "global_step": self._global_step,
                        },
                    )
                    
                # ── Scheduler step ────────────────────────────
                # Step after eval so the logged lr matches the lr
                # used for that epoch's training.
                # Most schedulers step once per epoch.
                if self.scheduler is not None:
                    self.scheduler.step()

                # ── Profiler step ─────────────────────────────
                # Tells the profiler to advance its schedule.
                # When profiler is disabled, prof is None — no-op.
                if prof is not None:
                    prof.step()
            
        # ── Load best and return ──────────────────────────────
        log.info(
            "training.done",
            best_metric=round(self.ckpt.best_metric, 4),
            monitor=self.cfg.checkpoint.monitor_metric,
        )

        # Load the best weights back into the model so the caller
        # gets the best version, not the last version.
        final_state = self.ckpt.load_best(self.model)
        return final_state.get("val_metrics", {})
    
    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    def _maybe_resume(self) -> int:
        """
        If resume_from is set in config, load that checkpoint
        and return the epoch to start from.
        Otherwise return 0 (start from the beginning).

        Why epoch + 1?
            The checkpoint stores the epoch it was saved ON.
            If we saved at epoch 12, we want to resume FROM epoch 13.
            Without +1 we would redo epoch 12.
        """
        if self.cfg.checkpoint.resume_from is None:
            return 0

        log.info(
            "training.resuming",
            path=str(self.cfg.checkpoint.resume_from),
        )

        state = self.ckpt.load_from_path(
            path=self.cfg.checkpoint.resume_from,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        # Move optimizer state to the right device.
        # Optimizer states are loaded to CPU by map_location="cpu"
        # but they need to be on the same device as the model params.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)

        # Restore global step count so logging continues correctly
        self._global_step = state.get("global_step", 0)

        start_epoch = state["epoch"] + 1
        log.info(
            "training.resumed",
            start_epoch=start_epoch,
            best_metric=round(state["metric"], 4),
            global_step=self._global_step,
        )
        return start_epoch
        
        
        
        