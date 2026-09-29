"""Model components for Evo2 embedding to GO term prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SequenceEncoderConfig:
    """Configuration for the Evo2 embedding projection branch."""

    input_dim: int = 4096
    hidden_dim: int = 2048
    embed_dim: int = 1024
    dropout: float = 0.1
    bias: bool = True
    normalize_eps: float = 1e-12


class SequenceEncoder(nn.Module):
    """Project Evo2 sequence embeddings into the shared normalized latent space."""

    def __init__(self, config: SequenceEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or SequenceEncoderConfig()

        # Validate the architectural dimensions early so configuration mistakes
        # fail before a training job starts allocating model state.
        if self.config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if not 0.0 <= self.config.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        # First projection block: normalize raw Evo2 embedding magnitudes, then
        # expand/reshape the 4096-d input into the hidden representation used by
        # the projection head.
        self.input_norm = nn.LayerNorm(self.config.input_dim)
        self.input_projection = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
            bias=self.config.bias,
        )
        self.activation = nn.GELU()
        self.input_dropout = nn.Dropout(p=self.config.dropout)

        # Second projection block: normalize the hidden activations before the
        # final linear map into the shared sequence/GO latent space.
        self.hidden_norm = nn.LayerNorm(self.config.hidden_dim)
        self.output_projection = nn.Linear(
            self.config.hidden_dim,
            self.config.embed_dim,
            bias=self.config.bias,
        )

        # Final dropout regularizes the latent representation before L2
        # normalization turns dot products with GO vectors into cosine scores.
        self.output_dropout = nn.Dropout(p=self.config.dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized sequence representations.

        Args:
            embeddings: Tensor of shape ``(batch_size, input_dim)``. Embedding
                storage may be FP8/FP16 upstream, but this branch consumes
                float32 values.
        """

        if embeddings.ndim != 2:
            raise ValueError(
                "SequenceEncoder expects a 2D tensor of shape "
                f"(batch_size, {self.config.input_dim}); got shape {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.config.input_dim:
            raise ValueError(
                "SequenceEncoder input feature dimension mismatch: "
                f"expected {self.config.input_dim}, got {embeddings.shape[-1]}"
            )

        # The storage dtype may be FP8/FP16 upstream, but the projection head
        # operates in float32 at its boundary for a stable model contract.
        x = embeddings.to(dtype=torch.float32)

        # Block 1: normalize input scale, project to hidden_dim, apply the
        # nonlinearity, and regularize the hidden representation.
        x = self.input_norm(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.input_dropout(x)

        # Block 2: normalize hidden activations and project into embed_dim,
        # the shared latent dimension used for sequence/GO comparisons.
        x = self.hidden_norm(x)
        x = self.output_projection(x)
        x = self.output_dropout(x)

        # Unit-normalize so downstream dot products against GO term vectors are
        # cosine similarities before the top-level model applies logit_scale.
        return F.normalize(x, p=2.0, dim=-1, eps=self.config.normalize_eps)


@dataclass(frozen=True)
class GoTermGCNConfig:
    """Configuration for the GO ontology graph encoder branch."""

    n_terms: int
    init_dim: int = 256
    hidden_dim: int = 512
    embed_dim: int = 1024
    normalize_eps: float = 1e-12


class GoTermGCN(nn.Module):
    """Encode GO terms with a two-layer GCN over a fixed ontology adjacency."""

    def __init__(self, adjacency: torch.Tensor, config: GoTermGCNConfig) -> None:
        super().__init__()
        self.config = config

        # Validate the graph/model contract before allocating the learnable GO
        # node features. The adjacency rows must align exactly with the
        # deterministic GO vocabulary index used for dense targets.
        if self.config.n_terms <= 0:
            raise ValueError("n_terms must be positive")
        if self.config.init_dim <= 0:
            raise ValueError("init_dim must be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.config.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if adjacency.shape != (self.config.n_terms, self.config.n_terms):
            raise ValueError(
                "GO adjacency shape mismatch: "
                f"expected {(self.config.n_terms, self.config.n_terms)}, "
                f"got {tuple(adjacency.shape)}"
            )
        if not adjacency.is_sparse:
            raise ValueError("GO adjacency must be a torch sparse tensor")

        # The normalized adjacency is fixed model structure, not a learnable
        # parameter. Registering it as a buffer makes model.to(device) move it
        # with the GCN weights before torch.sparse.mm is called.
        self.register_buffer("adjacency", adjacency.coalesce().to(dtype=torch.float32))

        # Each GO term starts with a learnable feature vector. These vectors do
        # not encode semantics by themselves; the graph aggregation below ties
        # them to ontology neighbors on every forward pass.
        self.node_features = nn.Embedding(self.config.n_terms, self.config.init_dim)
        nn.init.xavier_uniform_(self.node_features.weight)

        # First GCN block: aggregate one-hop ontology context, then map from the
        # initial node-feature space into the hidden graph representation.
        self.input_projection = nn.Linear(
            self.config.init_dim,
            self.config.hidden_dim,
            bias=False,
        )
        self.hidden_norm = nn.LayerNorm(self.config.hidden_dim)
        self.activation = nn.GELU()

        # Second GCN block: aggregate a second hop of ontology context and map
        # into the shared sequence/GO latent dimension.
        self.output_projection = nn.Linear(
            self.config.hidden_dim,
            self.config.embed_dim,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(self.config.embed_dim)

    def forward(self) -> torch.Tensor:
        """Return L2-normalized GO term representations for all vocabulary terms."""

        # Keep graph inputs in float32. This branch is cheap relative to Evo2
        # embedding generation, and the final normalization should not lose
        # precision if a training loop wraps the model in torch.autocast.
        adjacency = self.adjacency.to(dtype=torch.float32)
        h = self.node_features.weight.to(dtype=torch.float32)

        # First message-passing layer: each term averages information from
        # itself, is_a neighbors, and part_of neighbors according to the fixed
        # D^-1/2 A D^-1/2 weights.
        h = torch.sparse.mm(adjacency, h)
        h = self.input_projection(h)
        h = self.hidden_norm(h)
        h = self.activation(h)

        # Second message-passing layer gives a two-hop receptive field before
        # projecting into the same latent space used by the sequence encoder.
        h = torch.sparse.mm(adjacency, h)
        h = self.output_projection(h)
        h = self.output_norm(h)

        # Unit-normalize so downstream dot products against sequence vectors are
        # cosine similarities before the top-level model applies logit_scale.
        h = h.to(dtype=torch.float32)
        return F.normalize(h, p=2.0, dim=-1, eps=self.config.normalize_eps)


@dataclass(frozen=True)
class DualEncoderConfig:
    """Configuration for the sequence/GO dual encoder model."""

    go_encoder: GoTermGCNConfig
    sequence_encoder: SequenceEncoderConfig = field(
        default_factory=SequenceEncoderConfig
    )
    initial_logit_scale: float = 20.0
    min_logit_scale: float = 1.0
    max_logit_scale: float = 100.0


class DualEncoderModel(nn.Module):
    """Score Evo2 sequence embeddings against all GO vocabulary terms."""

    def __init__(self, adjacency: torch.Tensor, config: DualEncoderConfig) -> None:
        super().__init__()
        self.config = config

        if self.config.sequence_encoder.embed_dim != self.config.go_encoder.embed_dim:
            raise ValueError(
                "Sequence and GO encoder latent dimensions must match: "
                f"sequence embed_dim={self.config.sequence_encoder.embed_dim}, "
                f"GO embed_dim={self.config.go_encoder.embed_dim}"
            )
        if self.config.initial_logit_scale <= 0.0:
            raise ValueError("initial_logit_scale must be positive")
        if self.config.min_logit_scale <= 0.0:
            raise ValueError("min_logit_scale must be positive")
        if self.config.max_logit_scale < self.config.min_logit_scale:
            raise ValueError("max_logit_scale must be >= min_logit_scale")
        if not (
            self.config.min_logit_scale
            <= self.config.initial_logit_scale
            <= self.config.max_logit_scale
        ):
            raise ValueError(
                "initial_logit_scale must be inside the configured clamp range"
            )

        self.sequence_encoder = SequenceEncoder(self.config.sequence_encoder)
        self.go_encoder = GoTermGCN(adjacency, self.config.go_encoder)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(self.config.initial_logit_scale), dtype=torch.float32)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return raw logits for each sequence against every GO term.

        Args:
            embeddings: Tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Tensor of shape ``(batch_size, n_terms)``. Sigmoid is intentionally
            not applied here so training can use numerically stable logit losses.
        """

        seq_repr = self.sequence_encoder(embeddings)
        go_repr = self.go_encoder()
        scale = self.logit_scale.clamp(
            min=self.config.min_logit_scale,
            max=self.config.max_logit_scale,
        )
        return scale * torch.matmul(seq_repr, go_repr.transpose(0, 1))

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""

        return torch.sigmoid(self.forward(embeddings))


@dataclass(frozen=True)
class AsymmetricLossConfig:
    """Configuration for multi-label Asymmetric Loss."""

    gamma_neg: float = 4.0
    gamma_pos: float = 0.0
    clip: float = 0.05
    eps: float = 1e-8
    reduction: Literal["balanced", "mean"] = "balanced"
    positive_weight: float = 1.0
    negative_weight: float = 1.0


@dataclass(frozen=True)
class AsymmetricLossResult:
    """Loss tensor plus batch diagnostics for sparse multilabel training."""

    loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    positive_labels_per_batch: torch.Tensor
    negative_labels_per_batch: torch.Tensor
    positive_labels_per_example: torch.Tensor
    mean_positive_probability: torch.Tensor
    mean_negative_probability: torch.Tensor


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label GO prediction from raw logits."""

    def __init__(self, config: AsymmetricLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or AsymmetricLossConfig()

        if self.config.gamma_neg < 0.0:
            raise ValueError("gamma_neg must be non-negative")
        if self.config.gamma_pos < 0.0:
            raise ValueError("gamma_pos must be non-negative")
        if not 0.0 <= self.config.clip < 1.0:
            raise ValueError("clip must be in the interval [0, 1)")
        if self.config.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.config.reduction not in {"balanced", "mean"}:
            raise ValueError("reduction must be either 'balanced' or 'mean'")
        if self.config.positive_weight < 0.0:
            raise ValueError("positive_weight must be non-negative")
        if self.config.negative_weight < 0.0:
            raise ValueError("negative_weight must be non-negative")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return reduced ASL over the batch.

        Args:
            logits: Raw model outputs of shape ``(batch_size, n_terms)``.
            targets: Dense binary targets with the same shape as ``logits``.
        """
        return self.forward_with_diagnostics(logits, targets).loss

    def forward_with_diagnostics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> AsymmetricLossResult:
        """Return ASL plus diagnostics for sparse multilabel training."""

        if logits.shape != targets.shape:
            raise ValueError(
                "AsymmetricLoss shape mismatch: "
                f"logits shape {tuple(logits.shape)} != targets shape {tuple(targets.shape)}"
            )

        logits = logits.to(dtype=torch.float32)
        targets = targets.to(dtype=torch.float32)

        p = torch.sigmoid(logits)
        positive_mask = targets > 0.5
        negative_mask = ~positive_mask
        positive_mask_float = positive_mask.to(dtype=torch.float32)
        negative_mask_float = negative_mask.to(dtype=torch.float32)

        positive_log_loss = -torch.log(torch.clamp(p, min=self.config.eps))
        if self.config.gamma_pos == 0.0:
            positive_loss = positive_log_loss
        else:
            positive_weight = torch.pow(1.0 - p, self.config.gamma_pos)
            positive_loss = positive_weight * positive_log_loss

        negative_probability = torch.clamp(1.0 - p + self.config.clip, max=1.0)
        negative_focal_base = torch.clamp(p - self.config.clip, min=0.0)
        negative_log_loss = -torch.log(
            torch.clamp(negative_probability, min=self.config.eps)
        )
        if self.config.gamma_neg == 0.0:
            negative_loss = negative_log_loss
        else:
            negative_weight = torch.pow(negative_focal_base, self.config.gamma_neg)
            negative_loss = negative_weight * negative_log_loss

        positive_element_loss = positive_mask_float * positive_loss
        negative_element_loss = negative_mask_float * negative_loss

        positive_counts = positive_mask_float.sum(dim=1)
        negative_counts = negative_mask_float.sum(dim=1)
        positive_labels_per_batch = positive_counts.sum()
        negative_labels_per_batch = negative_counts.sum()
        positive_labels_per_example = positive_counts.mean()

        if self.config.reduction == "mean":
            loss = (
                self.config.positive_weight * positive_element_loss
                + self.config.negative_weight * negative_element_loss
            ).mean()
            positive_reduced = positive_element_loss.sum() / positive_labels_per_batch.clamp_min(1.0)
            negative_reduced = negative_element_loss.sum() / negative_labels_per_batch.clamp_min(1.0)
        else:
            positive_per_example = positive_element_loss.sum(dim=1) / positive_counts.clamp_min(1.0)
            negative_per_example = negative_element_loss.sum(dim=1) / negative_counts.clamp_min(1.0)
            positive_reduced = positive_per_example.mean()
            negative_reduced = negative_per_example.mean()
            loss = (
                self.config.positive_weight * positive_reduced
                + self.config.negative_weight * negative_reduced
            )

        zero = torch.zeros((), device=p.device, dtype=p.dtype)
        mean_positive_probability = (
            p[positive_mask].mean() if positive_mask.any() else zero
        )
        mean_negative_probability = (
            p[negative_mask].mean() if negative_mask.any() else zero
        )

        return AsymmetricLossResult(
            loss=loss,
            positive_loss=positive_reduced,
            negative_loss=negative_reduced,
            positive_labels_per_batch=positive_labels_per_batch,
            negative_labels_per_batch=negative_labels_per_batch,
            positive_labels_per_example=positive_labels_per_example,
            mean_positive_probability=mean_positive_probability,
            mean_negative_probability=mean_negative_probability,
        )


def load_sparse_adjacency_npz(path: str | Path) -> torch.Tensor:
    """Load a SciPy sparse adjacency artifact as a torch sparse COO tensor."""

    from scipy import sparse

    matrix = sparse.load_npz(path).tocoo()
    indices = torch.stack(
        (
            torch.as_tensor(matrix.row, dtype=torch.long),
            torch.as_tensor(matrix.col, dtype=torch.long),
        )
    )
    values = torch.as_tensor(matrix.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()
