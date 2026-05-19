"""Bar-distribution bucket-border fitting from a corpus's `ys.npy`.

Ported from the `--ys_borders` branch of `pfns4hpo_v2/main.py:117-144`
on kislurm. The submitit-launched `estimate_buckets` and the
interactive `input(...)` prompt are not ported (the published recipes
always pass `--ys_borders <corpus>/ys.npy`, so a precomputed `ys.npy`
is always available).

Two surfaces:
  - `buckets_from_ys_npy(ys_path, num_borders, full_support=True)`:
    return a tensor of border edges that turns into the
    `FullSupportBarDistribution.borders` field at training time.
  - `linspace_buckets(num_borders)`: simple `linspace(0, 1, n+1)`
    fallback for ablations that pass `--linspace_borders`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from alphapfn.model import bar_distribution


def make_unique(values, epsilon: float = 1e-12) -> np.ndarray:
    """Add small epsilons to values to make them unique without changing order."""
    sorted_indices = np.argsort(values)
    values_sorted = np.array(values)[sorted_indices]
    for i in range(1, len(values_sorted)):
        if values_sorted[i] <= values_sorted[i - 1]:
            values_sorted[i] = values_sorted[i - 1] + epsilon
    unique_values = np.empty_like(values_sorted)
    unique_values[sorted_indices] = values_sorted
    return unique_values


def buckets_from_ys_npy(
    ys_path: Path | str,
    num_borders: int,
    full_support: bool = True,
) -> torch.Tensor:
    """Fit bucket borders from a precomputed `ys.npy`.

    The v1 published recipes pass `--full_support` (true), so the
    `bar_distribution.get_bucket_borders` path is exercised. The
    percentile-based limited-support path is preserved for ablations.

    Args:
      ys_path:     path to the `ys.npy` shipped inside each corpus.
      num_borders: number of bar-distribution borders to fit
                   (5000 for v1 → 5001 border values).
      full_support: if True (default + v1 recipe), use
                    `bar_distribution.get_bucket_borders`. If False,
                    use the percentile-based limited-support path.

    Returns:
      Tensor of border edges, shape `(num_borders + 1,)`.
    """
    ys = np.load(str(ys_path))
    if full_support:
        ys_bucket = torch.from_numpy(ys)
        return bar_distribution.get_bucket_borders(num_borders, ys=ys_bucket)
    # Limited-support: percentile-based, after de-duplicating
    unique_ys = make_unique(ys)
    percentiles = np.linspace(0, 100, num_borders + 1)
    return torch.from_numpy(np.percentile(unique_ys, percentiles)).float()


def linspace_buckets(num_borders: int) -> torch.Tensor:
    """Simple [0, 1] linspace fallback (used by `--linspace_borders`)."""
    return torch.linspace(0.0, 1.0, num_borders + 1)
