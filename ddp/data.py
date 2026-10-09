"""Synthetic data and deterministic sharding.

The dataset is generated once from a fixed seed so every worker sees the
same data. Worker r owns every N-th sample starting at r (strided sharding),
which makes the union of all workers' batches at step s exactly the
single-worker batch: {r + s*B + k*N}.
"""

from __future__ import annotations

import torch

__all__ = ["make_regression_data", "shard_indices"]


def make_regression_data(
    num_samples: int = 4096,
    in_features: int = 20,
    noise_std: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y = X @ w_true + noise. Returns (X, y, w_true)."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(num_samples, in_features, generator=gen)
    w_true = torch.randn(in_features, 1, generator=gen)
    y = X @ w_true + noise_std * torch.randn(num_samples, 1, generator=gen)
    return X, y, w_true


def shard_indices(num_samples: int, rank: int, world_size: int) -> torch.Tensor:
    """Strided shard: indices rank, rank+N, rank+2N, ..."""
    return torch.arange(rank, num_samples, world_size)
