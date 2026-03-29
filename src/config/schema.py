# src/config/schema.py

from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator

# ──────────────────────────────────────────────────────────────
# MODEL CONFIG
# Contains only the values the model's __init__ needs.
# If we  swap to a different architecture, we replace this class.
# The rest of the configs never change.
# ──────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):

    # Which backbone to load from torchvision
    # Literal means ONLY these exact strings are valid —
    # "resnet999" would raise a ValidationError immediately
    backbone : Literal["resnet18", "resnet50"] = "resnet18"

    # Number of output classes — your dataset determines this
    # gt=0 means "greater than 0" — catches num_classes=0 or negative
    num_classes : int = Field(10, gt=0)

    # Use ImageNet pretrained weights?
    # True = faster convergence, better results with less data
    # False = train from scratch, useful to understand the difference
    pretrained : bool = True

    # Dropout probability on the classifier head
    # ge=0.0 = greater than or equal to 0
    # lt=1.0 = less than 1 (dropout of 1.0 would zero everything)
    dropout : float = Field(0.3, ge=0, lt=1)

# ──────────────────────────────────────────────────────────────
# DATA CONFIG
# Everything about loading and preprocessing data.
# The model never sees this — only the DataModule uses it.
# ──────────────────────────────────────────────────────────────
class DataConfig(BaseModel):

    # Where to download/find the dataset on disk
    data_dir : Path = Path('./data')

    # How many images per batch
    # Larger = faster training, more GPU memory needed
    # Typical range: 32 to 512
    batch_size : int = Field(128, gt=0)

    # Parallel workers that load data while GPU trains
    # Rule of thumb: num_workers = number of CPU cores / 2
    # Set to 0 for debugging (single process, easier to trace errors)
    num_workers : int = Field(4, ge=0)

    # Stages data in CPU memory for faster CPU→GPU transfer
    # Always True when training on GPU
    pin_memory : bool = True

    # Fraction of training data used for validation
    # 0.1 = 10% val, 90% train
    val_split : float = Field(0.1, gt=0, lt=1)

    # Per-channel mean and std of CIFAR-10 (pre-computed)
    # Used to normalise pixel values to roughly zero-mean unit-variance
    # We compute these once for the dataset and hard-code them here
    mean :  list[float] = [0.4914, 0.4822, 0.4465]
    std :   list[float] = [0.2470, 0.2435, 0.2616]


# ──────────────────────────────────────────────────────────────
# OPTIMIZER CONFIG
# How the model's weights get updated each step.
# ──────────────────────────────────────────────────────────────
class OptimizerConfig(BaseModel):
    # Which optimizer to use
    # AdamW is the safe default for almost everything
    # SGD can be better for vision with careful tuning
    name : Literal["adamw", "sgd"] = "adamw"

    # Learning rate — the single most important hyperparameter
    # Too high: loss explodes. Too low: learns nothing.
    # Typical range: 1e-4 to 1e-2
    lr : float = Field(1e-3, gt=0)

    # L2 regularisation — penalises large weights to prevent overfitting
    weight_decay : float = Field(1e-4, gt=0)

    # Clip gradient norm to this value
    # Prevents "exploding gradients" — when one bad batch
    # sends weights flying in the wrong direction
    # None = disabled
    grad_clip_norm : float | None = Field(1.0, gt=0)

    # Accumulate gradients over N steps before updating weights
    # Simulates a batch N times larger without using more GPU memory
    # Most of the time we leave this at 1 (No accumulation)
    grad_accumulation_steps : int = Field(1, ge=1)


# ──────────────────────────────────────────────────────────────
# SCHEDULER CONFIG
# How the learning rate changes over training.
# A fixed lr rarely gives the best result — schedulers help.
# ──────────────────────────────────────────────────────────────
class SchedulerConfig(BaseModel):

    # cosine: smooth decay from lr → min_lr. Best default.
    # step:   multiply lr by gamma every step_size epochs. More aggressive.
    # none:   fixed lr throughout. Useful as a baseline.
    name : Literal["cosine", "step", "none"] = "cosine"

    # Cosine scheduler decays to this minimum
    min_lr : float = Field(1e-6, ge=0)

    # How many epochs to slowly ramp up lr from near-zero to full lr
    # Helps stability in the first few epochs when weights are random
    warmup_epochs : int = Field(2, ge=0)

    # Step scheduler: reduce lr by this factor every step_size epochs
    # Step size reduces learning rate every step_size epochs by gamma
    step_size : int = Field(10, gt=0)
    gamma : float = Field(0.1, gt=0, lt=1)


# ──────────────────────────────────────────────────────────────
# TRAINING CONFIG
# Controls the training loop itself.
# ──────────────────────────────────────────────────────────────
class TrainingConfig(BaseModel):

    # Total number of complete passes through the training data
    max_epochs : int = Field(20, gt=0)

    # Random seed — same seed = same result every run
    seed : int = 42

    # Device :"cuda" for NVIDIA GPU, "mps" for Apple Silicon, "cpu" for no GPU
    device : Literal["cuda", "mps", "cpu"] = "cuda"

    # Numerical precision for the forward pass
    # bf16: best default for modern GPUs (A100, 3090, 4090) — no scaler needed
    # fp16: older GPUs — requires GradScaler
    # fp32: CPU or when you hit numerical issues
    precision : Literal["bf16", "fp16", "fp32"] = "bf16"


    # torch.compile() fuses GPU kernels for ~10-15% extra speed
    # First run takes ~30s to compile. Set False while developing.
    compile_model : bool = False

    # Print a log line every N optimizer steps
    log_every_n_steps : int = Field(50, gt=0)

    # Run validation every N epochs
    # 1 = validate after every epoch (safe default)
    eval_every_n_epochs : int = Field(1, ge=1)

    # ── New fields ────────────────────────────────────────────
    # Set to True to run the profiler for the first few steps
    # Leave False during normal training — zero overhead when disabled
    profile: bool = False

    # How many steps to profile when profile=True
    # 5 is enough to get stable timings without slowing training much
    profile_steps: int = Field(5, gt=0)


# ──────────────────────────────────────────────────────────────
# CHECKPOINT CONFIG
# Controls how model weights are saved to disk.
# ──────────────────────────────────────────────────────────────
class CheckpointConfig(BaseModel):
    
    # Where to save checkpoint files
    save_dir : Path = Path("./checkpoints")

    # Keep only the k best checkpoints — delete the rest
    # Prevents filling your disk with hundreds of .pt files
    save_top_k : int = Field(3, ge=1)

    # Which metric to use when deciding "best
    monitor_metric : Literal["val_loss", "val_acc"] = "val_acc"

    # "max" for accuracy (higher is better)
    # "min" for loss (lower is better)
    monitor_mode: Literal["min", "max"] = "max"

    # Set to a checkpoint path to resume an interrupted training run
    # None = start from scratch
    resume_from: Path | None = None


# ──────────────────────────────────────────────────────────────
# EXPERIMENT CONFIG
# The top-level object that nests all the above.
# This is the single object you pass everywhere.
# ──────────────────────────────────────────────────────────────
class ExperimentConfig(BaseModel):
    # Human-readable name — used for log file names and experiment folders
    name : str = "cifar10_resnet18"

    # Nested configs — each section owns its own concern
    model : ModelConfig = ModelConfig()
    data : DataConfig = DataConfig()
    optimizer : OptimizerConfig = OptimizerConfig()
    scheduler : SchedulerConfig = SchedulerConfig()
    training : TrainingConfig = TrainingConfig()
    checkpoint : CheckpointConfig = CheckpointConfig()

    # Cross-field validation — checks that combinations make sense
    # This runs AFTER all individual fields are validated (mode = "after")
    @model_validator(mode="after")
    def check_precision_device(self) -> "ExperimentConfig": # quotes are optional if __future__, allows forward references
        if self.training.precision == "bf16" and self.training.device == "cpu":
            raise ValueError(
                "bf16 is not supported on cpu."
                "Use precision fp32 when device is cpu"
            )
        return self


# ──────────────────────────────────────────────────────────────
# LOADER FUNCTION
# The only function in this file.
# Reads a YAML file → validates it → returns a typed object.
# ──────────────────────────────────────────────────────────────
def load_config(path : Path) -> ExperimentConfig:
    """
    Load and validate a YAML config file.

    Raises:
        ValidationError: if any field has wrong type or fails a constraint
        FileNotFoundError: if the path doesn't exist
    """
    import yaml
    raw : dict = yaml.safe_load(path.read_text()) or {}
    return ExperimentConfig(**raw)