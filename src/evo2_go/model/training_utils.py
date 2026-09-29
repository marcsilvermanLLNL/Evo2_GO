"""Training utilities for Evo2 embedding to GO prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim import Optimizer


@dataclass(frozen=True)
class WarmupCosineLRConfig:
    """Configuration for the model handoff learning-rate schedule."""

    peak_lr: float = 3e-4
    warmup_steps: int = 2000
    max_steps: int = 200000
    final_lr_fraction: float = 0.1


@dataclass(frozen=True)
class AdamWConfig:
    """Configuration for the model handoff AdamW optimizer."""

    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass(frozen=True)
class GradientClippingConfig:
    """Configuration for global gradient norm clipping."""

    max_norm: float = 1.0
    norm_type: float = 2.0
    error_if_nonfinite: bool = True


@dataclass(frozen=True)
class FmaxResult:
    """CAFA-style gene-averaged Fmax summary."""

    fmax: float
    threshold: float
    precision: float
    recall: float
    thresholds: tuple[float, ...]
    precision_by_threshold: tuple[float, ...]
    recall_by_threshold: tuple[float, ...]
    f1_by_threshold: tuple[float, ...]


def _validate_adamw_config(config: AdamWConfig) -> None:
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if not 0.0 <= config.beta1 < 1.0:
        raise ValueError("beta1 must be in the interval [0, 1)")
    if not 0.0 <= config.beta2 < 1.0:
        raise ValueError("beta2 must be in the interval [0, 1)")
    if config.eps <= 0.0:
        raise ValueError("eps must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")


def _validate_gradient_clipping_config(config: GradientClippingConfig) -> None:
    if config.max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    if config.norm_type <= 0.0:
        raise ValueError("norm_type must be positive")


def build_adamw_optimizer(
    model: nn.Module,
    config: AdamWConfig | None = None,
) -> AdamW:
    """Build AdamW over every trainable model parameter."""

    config = config or AdamWConfig()
    _validate_adamw_config(config)

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("model has no trainable parameters")

    parameter_ids = [id(parameter) for _, parameter in named_parameters]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("model.named_parameters() returned duplicate parameters")

    optimizer = AdamW(
        [parameter for _, parameter in named_parameters],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    optimizer_parameter_ids = {
        id(parameter)
        for param_group in optimizer.param_groups
        for parameter in param_group["params"]
    }
    expected_parameter_ids = set(parameter_ids)
    if optimizer_parameter_ids != expected_parameter_ids:
        missing_names = [
            name
            for name, parameter in named_parameters
            if id(parameter) not in optimizer_parameter_ids
        ]
        raise RuntimeError(
            "AdamW optimizer parameter coverage mismatch; "
            f"missing trainable parameters: {missing_names}"
        )

    return optimizer


def clip_global_grad_norm(
    model: nn.Module,
    config: GradientClippingConfig | None = None,
) -> torch.Tensor:
    """Clip trainable model gradients by global norm and return the pre-clip norm.

    In AMP training, call this after ``GradScaler.unscale_(optimizer)`` and
    before ``optimizer.step()``.
    """

    config = config or GradientClippingConfig()
    _validate_gradient_clipping_config(config)

    parameters_with_grad = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not parameters_with_grad:
        return torch.tensor(0.0)

    return torch.nn.utils.clip_grad_norm_(
        parameters_with_grad,
        max_norm=config.max_norm,
        norm_type=config.norm_type,
        error_if_nonfinite=config.error_if_nonfinite,
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    step: int,
    best_fmax: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model/training state needed to resume or evaluate a run."""

    if step < 0:
        raise ValueError("step must be non-negative")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "step": step,
        "best_fmax": best_fmax,
        "metadata": metadata or {},
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, checkpoint_path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint and restore provided objects in place."""

    checkpoint_path = Path(path)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    else:
        # Older checkpoints may contain torch.__version__ as a TorchVersion
        # metadata object. Keep weights_only loading, but allow that safe type.
        from torch.torch_version import TorchVersion

        with safe_globals([TorchVersion]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


@torch.no_grad()
def cafa_gene_averaged_fmax(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    thresholds: torch.Tensor | None = None,
) -> FmaxResult:
    """Compute CAFA-style gene-averaged Fmax from probabilities and targets."""

    if probabilities.shape != targets.shape:
        raise ValueError(
            "Fmax shape mismatch: "
            f"probabilities shape {tuple(probabilities.shape)} != "
            f"targets shape {tuple(targets.shape)}"
        )
    if probabilities.ndim != 2:
        raise ValueError("Fmax expects 2D tensors of shape (n_genes, n_terms)")
    if probabilities.shape[0] == 0:
        raise ValueError("Fmax requires at least one gene")

    probabilities = probabilities.to(dtype=torch.float32)
    targets_bool = targets.to(dtype=torch.float32) > 0.5
    if thresholds is None:
        thresholds = torch.linspace(0.01, 0.99, 99, device=probabilities.device)
    else:
        thresholds = thresholds.to(device=probabilities.device, dtype=torch.float32)
    if thresholds.ndim != 1 or thresholds.numel() == 0:
        raise ValueError("thresholds must be a non-empty 1D tensor")

    best_f1 = -1.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    true_counts = targets_bool.sum(dim=1).to(dtype=torch.float32)
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tp = (predicted & targets_bool).sum(dim=1).to(dtype=torch.float32)
        fp = (predicted & ~targets_bool).sum(dim=1).to(dtype=torch.float32)
        fn = (~predicted & targets_bool).sum(dim=1).to(dtype=torch.float32)

        predicted_counts = tp + fp
        precision_per_gene = torch.where(
            predicted_counts > 0.0,
            tp / predicted_counts,
            torch.zeros_like(tp),
        )
        recall_per_gene = torch.where(
            true_counts > 0.0,
            tp / (tp + fn),
            torch.zeros_like(tp),
        )

        precision = float(precision_per_gene.mean().item())
        recall = float(recall_per_gene.mean().item())
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (
            precision + recall
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold.item())
            best_precision = precision
            best_recall = recall

    return FmaxResult(
        fmax=best_f1,
        threshold=best_threshold,
        precision=best_precision,
        recall=best_recall,
        thresholds=tuple(float(value.item()) for value in thresholds),
        precision_by_threshold=tuple(precision_values),
        recall_by_threshold=tuple(recall_values),
        f1_by_threshold=tuple(f1_values),
    )


def _validate_warmup_cosine_config(config: WarmupCosineLRConfig) -> None:
    if config.peak_lr <= 0.0:
        raise ValueError("peak_lr must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.max_steps < config.warmup_steps:
        raise ValueError("max_steps must be >= warmup_steps")
    if not 0.0 <= config.final_lr_fraction <= 1.0:
        raise ValueError("final_lr_fraction must be in the interval [0, 1]")


def warmup_cosine_learning_rate(
    step: int,
    config: WarmupCosineLRConfig | None = None,
) -> float:
    """Return the LR for a one-based optimizer step.

    ``step=1`` is the first optimizer update. During warmup, LR increases
    linearly to ``peak_lr`` at ``step=warmup_steps``. After warmup, LR follows
    cosine decay to ``final_lr_fraction * peak_lr`` at ``step=max_steps``.
    ``step=0`` returns 0.0 for initialization and checkpoint bookkeeping.
    """

    config = config or WarmupCosineLRConfig()
    _validate_warmup_cosine_config(config)

    if step < 0:
        raise ValueError("step must be non-negative")
    if step == 0:
        return 0.0 if config.warmup_steps > 0 else config.peak_lr

    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.peak_lr * (step / config.warmup_steps)

    if config.max_steps == config.warmup_steps:
        return config.peak_lr * config.final_lr_fraction

    decay_steps = config.max_steps - config.warmup_steps
    decay_step = min(max(step - config.warmup_steps, 0), decay_steps)
    decay_progress = decay_step / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    final_lr = config.peak_lr * config.final_lr_fraction
    return final_lr + (config.peak_lr - final_lr) * cosine


def set_optimizer_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    """Set the same learning rate on every optimizer parameter group."""

    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


class WarmupCosineLRScheduler:
    """Small manual-step LR helper for optimizer-driven training loops."""

    def __init__(
        self,
        optimizer: Optimizer,
        config: WarmupCosineLRConfig | None = None,
        *,
        last_step: int = 0,
    ) -> None:
        if last_step < 0:
            raise ValueError("last_step must be non-negative")

        self.optimizer = optimizer
        self.config = config or WarmupCosineLRConfig()
        _validate_warmup_cosine_config(self.config)
        self.last_step = last_step
        set_optimizer_learning_rate(
            self.optimizer,
            warmup_cosine_learning_rate(self.last_step, self.config),
        )

    def get_lr(self, step: int | None = None) -> float:
        """Return the LR for ``step`` or the scheduler's current step."""

        return warmup_cosine_learning_rate(
            self.last_step if step is None else step,
            self.config,
        )

    def step(self, step: int | None = None) -> float:
        """Advance or set the scheduler step, update optimizer groups, and return LR."""

        if step is None:
            step = self.last_step + 1
        if step < 0:
            raise ValueError("step must be non-negative")

        self.last_step = step
        learning_rate = self.get_lr()
        set_optimizer_learning_rate(self.optimizer, learning_rate)
        return learning_rate

    def state_dict(self) -> dict[str, int]:
        """Return checkpoint state for this lightweight scheduler."""

        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Restore scheduler step state and update the optimizer LR."""

        self.last_step = int(state_dict["last_step"])
        if self.last_step < 0:
            raise ValueError("last_step must be non-negative")
        set_optimizer_learning_rate(self.optimizer, self.get_lr())
