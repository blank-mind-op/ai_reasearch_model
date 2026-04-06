# src/training/checkpoint.py

from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config.schema import CheckpointConfig
from src.utils.logging import get_logger

log = get_logger(__name__)

class CheckpointManager:
    """
    Manages saving and loading model checkpoints.

    Behaviour:
        - After each eval epoch, call save() with the current metric.
        - If this is the best metric seen so far → update best.pt
        - Keep only the top-k checkpoints on disk, delete the rest.
        - Load best.pt at any time with load_best().

    File structure on disk:
        checkpoints/
            best.pt                      ← always the best run
            epoch_0005_acc_0.8731.pt     ← one of the top-k
            epoch_0012_acc_0.9043.pt
            epoch_0019_acc_0.9187.pt

    What gets saved in each .pt file:
        {
            "epoch": 12,
            "metric": 0.9043,
            "metric_name": "val_acc",
            "model_state": {...},        ← model weights
            "optimizer_state": {...},    ← optimizer momentum etc.
            "scheduler_state": {...},    ← lr schedule position
            "config": {...},             ← full config as dict
        }

    Why save optimizer and scheduler state?
        model weights alone let you evaluate or run inference.
            But to RESUME training faithfully you also need:
            - optimizer state: AdamW stores momentum for every
                parameter. Without it, the first epoch after
                resuming behaves like a cold start.
            - scheduler state: tells the scheduler which epoch
                we're on so lr continues from where it left off,
                not from the beginning.

    Usage:
        ckpt = CheckpointManager(cfg.checkpoint)
        # after each eval:
        ckpt.save(epoch=5, metric=0.87, model=model,
                    optimizer=optimizer, scheduler=scheduler)
        # at the end or for inference:
        state = ckpt.load_best(model, optimizer, scheduler)
        resume_epoch = state["epoch"] + 1
    """
    
    def __init__(self, cfg : CheckpointConfig) -> None:
        self.cfg = cfg
        self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
        # Min-heap storing (sort_key, filepath) pairs.
        # We use a heap so finding and removing the worst
        # checkpoint is O(log k) instead of O(k).
        #
        # For monitor_mode="max" (e.g. accuracy — higher is better):
        #   sort_key = -metric (negate so the worst = largest heap value)
        # For monitor_mode="min" (e.g. loss — lower is better):
        #   sort_key = +metric (worst = largest value, which is correct)
        
        self._heap : list[tuple[float, str]] = []
        # Track the best metric seen so far
        # Initialised to worst possible value so any real metric beats it
        
        self.best_metric : float = (
            float("-inf") if cfg.monitor_mode == "max"
            else float("inf")
        )
        
        log.debug(
            "checkpoint.manager.ready",
            save_dir=str(cfg.save_dir),
            save_top_k=cfg.save_top_k,
            monitor_metric=cfg.monitor_metric,
            monotor_mode=cfg.monitor_mode,
        )
        
    # ──────────────────────────────────────────────────────────
    # Saving
    # ──────────────────────────────────────────────────────────
    
    def _is_better(self, metric : float) -> bool:
        """
        Returns True if this metric is better than the
        best seen so far, according to monitor_mode.
        """
        if self.cfg.monitor_mode == "max":
            return metric > self.best_metric
        if self.cfg.monitor_mode == "min":
            return metric < self.best_metric
    
    def _sort_key(self, metric: float) -> float:
        """
        Convert metric to a heap key where SMALLER = WORSE,
        so heappop() always removes the worst checkpoint.

        max mode (accuracy — higher is better):
            worst = lowest metric
            lowest metric is already the smallest number
            → use +metric directly, no change needed

        min mode (loss — lower is better):
            worst = highest metric
            highest metric is the largest number, not the smallest
            → negate so the highest loss becomes the most negative
            (smallest) heap key and gets popped first
        """
        return metric if self.cfg.monitor_mode == "max" else -metric
    
    def save(
        self,
        epoch : int,
        metric : float,
        model : nn.Module,
        optimizer : torch.optim.Optimizer,
        scheduler : torch.optim.lr_scheduler.LRScheduler | None = None,
        extra : dict[str, Any] | None = None,
    ) -> Path:
        """
        Save a checkpoint and manage the top-k pool.

        Args:
            epoch:      current epoch number (0-indexed)
            metric:     the monitored metric value (e.g. val_acc=0.91)
            model:      the model to save
            optimizer:  the optimizer to save
            scheduler:  the lr scheduler to save (optional)
            extra:      any additional data to include in the checkpoint
                        e.g. {"train_metrics": {...}, "val_metrics": {...}}

        Returns:
            Path to the saved checkpoint file
        """
        
        # ── Build filename ────────────────────────────────────
        # Include epoch and metric in the filename so you can
        # identify checkpoints at a glance without loading them
        
        metric_str = f"{metric:.4f}".replace(".", "_")
        filename = (
            f"epoch_{epoch:04d}_"
            f"{self.cfg.monitor_metric}_{metric_str}.pt"
        )
        
        path = self.cfg.save_dir / filename
        
        # ── Build state dict ──────────────────────────────────
        state : dict[str, Any] = {
            "epoch" : epoch,
            "metric" : metric,
            "metric_name" : self.cfg.monitor_metric,
            "model_state" : model.state_dict(),
            "optimizer_state" : optimizer.state_dict(),
            "scheduler_state" : (
                scheduler.state_dict() if scheduler is not None else None
                ),
            **(extra or {}),
        }
        
        # ── Save to disk ──────────────────────────────────────
        
        torch.save(state, path)
        log.debug(
            "checkpoint.saved",
            epoch=epoch,
            metric=round(metric, 4),
            path=str(path),
        )
        
        # ── Update best.pt if this is the best result ─────────
        if self._is_better(metric):
            self.best_metric = metric
            best_path = self.cfg.save_dir / "best.pt"
            torch.save(state, best_path)
            
            # Also save a human-readable JSON summary next to best.pt
            # so you can inspect the best result without loading the .pt
            
            summary = {
                "epoch" : epoch,
                "metric_name" : self.cfg.monitor_metric,
                "metric" : round(metric, 4),
                "checkpoint_file" : filename,
            }
            
            (self.cfg.save_dir / "best_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            
            log.info(
                "checkpoint.new_best",
                epoch=epoch,
                metric_name=self.cfg.monitor_metric,
                metric=round(metric, 4),
            )
            
        # ── Add to heap, evict worst if over top-k ────────────
        
        heapq.heappush(
            self._heap,
            (self._sort_key(metric), str(path)),
        )
        
        # When we have more than save_top_k checkpoints,
        # remove the one with the worst metric.
        # heappop removes the smallest heap key = worst checkpoint.
        
        while len(self._heap) > self.cfg.save_top_k:
            _, worst_path = heapq.heappop(self._heap)
            Path(worst_path).unlink(missing_ok=True)
            log.debug("checkpoint.evicted", path=worst_path)
        
        return path
    
    # ──────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────
    
    def load_best(
        self,
        model : nn.Module,
        optimizer : torch.optim.Optimizer | None = None,
        scheduler : torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict[str, Any]:
        """
        Load weights from best.pt into model.
        Optionally also restores optimizer and scheduler state
        (needed when resuming training, not needed for inference).

        Args:
            model:     model to load weights into
            optimizer: if provided, restore optimizer state
            scheduler: if provided, restore scheduler state

        Returns:
            The full state dict — use state["epoch"] + 1 to get
            the epoch to resume from.

        Why map_location="cpu"?
            Checkpoints remember which GPU they were saved on.
            If you saved on cuda:0 and load on a machine with
            cuda:1 or no GPU, PyTorch crashes without map_location.
            Loading to CPU first then moving to the right device
            works on any hardware configuration.

        Why weights_only=False?
            weights_only=True is safer (prevents arbitrary code
            execution from malicious .pt files) but it can't load
            optimizer state dicts which contain Python objects.
            We use False here because we trust our own checkpoints.
            For loading third-party checkpoints, prefer True.
        """
        
        best_path = self.cfg.save_dir / "best.pt"
        
        if not best_path.exists():
            raise FileNotFoundError(
                f"No best checkpoint found at {best_path}. "
                f"Has training been run yet?"
            )
            
        # Load to CPU first — safe regardless of original device
        state = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )
        
        # Load model weights
        model.load_state_dict(state["model_state"])
        
        # Restore optimizer state if provided
        # Only do this when RESUMING training — not for inference
        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        
        # Restore scheduler state if provided
        if scheduler is not None and "scheduler_state" in state:
            if state["scheduler_state"] is not None:
                scheduler.load_state_dict(state["scheduler_state"])
                
        log.info(
            "checkpoint.loader",
            epoch=state["epoch"],
            metric_name=state.get("metric_name"),
            metric=round(state["metric"], 4),
            path=str(best_path),
        )
        
        return state
    
    
    def load_from_path(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict[str, Any]:
        """
        Load a specific checkpoint by path.
        Same as load_best() but for any checkpoint file,
        not just best.pt.

        Useful when you want to load a specific epoch,
        not necessarily the best one.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(state["model_state"])

        if optimizer is not None and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])

        if scheduler is not None and "scheduler_state" in state:
            if state["scheduler_state"] is not None:
                scheduler.load_state_dict(state["scheduler_state"])

        log.info(
            "checkpoint.loaded_from_path",
            epoch=state["epoch"],
            metric=round(state["metric"], 4),
            path=str(path),
        )

        return state