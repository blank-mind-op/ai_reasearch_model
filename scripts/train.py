# scripts/train.py

# ─────────────────────────────────────────────────────────────
# The entrypoint. Wires components together, calls trainer.fit().
# Contains zero logic — everything lives in src/.
#
# Run:
#   cd cifar_project
#   python scripts/train.py
#   python scripts/train.py --config configs/base.yaml
#   python scripts/train.py --config configs/experiments/run_001.yaml
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure Python can find src/ regardless of where you run from
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.config.schema import load_config
from src.data.datamodule import DataModule
from src.models.resnet import ResNetClassifier
from src.training.trainer import Trainer
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import set_seed

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 training")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to YAML config file",
    )
    return p.parse_args()

def build_optimizer(
    cfg, model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Differential learning rates.

    The backbone already has good ImageNet features — it needs
    small updates to adapt to CIFAR-10.
    The head is randomly initialised — it needs large updates
    to learn from scratch.

    Using the same lr for both wastes the pretrained backbone
    or makes the head learn too slowly. Differential lr fixes both.

        backbone lr = cfg.optimizer.lr / 10   (fine adjustment)
        head lr     = cfg.optimizer.lr        (fast learning)
    """
    ocfg = cfg.optimizer

    backbone_params = [
        p for n, p in model.named_parameters()
        if "backbone.fc" not in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "backbone.fc" in n
    ]

    param_groups = [
        {"params": backbone_params, "lr": ocfg.lr / 10},
        {"params": head_params,     "lr": ocfg.lr},
    ]

    if ocfg.name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            weight_decay=ocfg.weight_decay,
        )
    elif ocfg.name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=ocfg.momentum,
            weight_decay=ocfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {ocfg.name}")
    

def build_scheduler(
    cfg, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scfg = cfg.scheduler
    tcfg = cfg.training

    if scfg.name == "none":
        return None

    if scfg.name == "cosine":
        if scfg.warmup_epochs > 0:
            # Warmup: lr ramps from 10% → 100% over warmup_epochs
            # Then cosine decay from 100% → min_lr over remaining epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=scfg.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=tcfg.max_epochs - scfg.warmup_epochs,
                eta_min=scfg.min_lr,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[scfg.warmup_epochs],
            )

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tcfg.max_epochs,
            eta_min=scfg.min_lr,
        )

    if scfg.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scfg.step_size,
            gamma=scfg.gamma,
        )

    raise ValueError(f"Unknown scheduler: {scfg.name}")

def main() -> None:
    args = parse_args()

    # ── 1. Load and validate config ───────────────────────────
    # Pydantic validates every field here.
    # A bad value raises ValidationError immediately —
    # before any model is built, data downloaded, or GPU allocated.
    cfg = load_config(args.config)

    # ── 2. Set up logging ─────────────────────────────────────
    # Must happen before any log.info() calls.
    # Creates logs/ directory if it doesn't exist.
    setup_logging(
        log_file=Path("logs") / f"{cfg.name}.log",
        level="INFO",
    )
    log = get_logger(__name__)
    log.info("config.loaded", name=cfg.name, path=str(args.config))

    # ── 3. Set all random seeds ───────────────────────────────
    # Must happen before model creation, data loading,
    # and any other operation that uses random numbers.
    set_seed(cfg.training.seed)

    # ── 4. Build model ────────────────────────────────────────
    model = ResNetClassifier(cfg.model)
    model.log_info()

    # ── 5. Build data pipeline ────────────────────────────────
    # setup() is called inside trainer.fit() — not here.
    # This means the DataModule is created instantly,
    # data is only downloaded when training actually starts.
    datamodule = DataModule(cfg.data)

    # ── 6. Build optimizer and scheduler ──────────────────────
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    log.info(
        "optimizer.ready",
        name=cfg.optimizer.name,
        backbone_lr=cfg.optimizer.lr / 10,
        head_lr=cfg.optimizer.lr,
        scheduler=cfg.scheduler.name,
        warmup_epochs=cfg.scheduler.warmup_epochs,
    )

    # ── 7. Build trainer ──────────────────────────────────────
    trainer = Trainer(
        cfg=cfg,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # ── 8. Train ──────────────────────────────────────────────
    final_metrics = trainer.fit()

    # ── 9. Report final result ────────────────────────────────
    log.info(
        "run.complete",
        val_acc=round(final_metrics.get("acc", 0), 4),
        val_loss=round(final_metrics.get("loss", 0), 4),
        val_top5=round(final_metrics.get("top5_acc", 0), 4),
        best_checkpoint="checkpoints/best.pt",
    )


# ─────────────────────────────────────────────────────────────
# This guard is mandatory on Windows.
#
# On Windows, Python uses "spawn" to create worker processes —
# each worker starts a fresh Python interpreter and re-imports
# this file. Without this guard, each worker tries to start
# training again, which spawns more workers, infinitely.
#
# This guard says: "only run main() if this file is the
# actual entry point — not if it's being imported by a worker."
#
# With this guard in place, num_workers=4 in the YAML works
# correctly on Windows.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()