"""Minimal BO loop driven by alphapfn + botorch.optim.optimize_acqf.

Usage:
    python examples/bo_with_optimize_acqf.py [--acquisition JES] [--steps 20]

The script optimizes a 2D test function on the unit cube. Each BO step:
  1. Fit AlphaPFN on the observed (X, y).
  2. optimize_acqf finds argmax of the acquisition over the bounds.
  3. Evaluate the function at the new point; append.
"""
from __future__ import annotations

import argparse
import math

import torch
from botorch.optim import optimize_acqf

from alphapfn import AlphaPFN


def branin_normalized(X: torch.Tensor) -> torch.Tensor:
    """Branin on [0, 1]^2 (rescaled from its usual domain).

    Branin is a *minimization* benchmark, but AlphaPFN scores for
    *maximization* — so we negate the output here. Best y is the
    largest value; the true Branin minimum (≈ 0.398) corresponds to
    a maximum here of ≈ -0.398.
    """
    x1 = 15.0 * X[..., 0] - 5.0  # → [-5, 10]
    x2 = 15.0 * X[..., 1]         # → [0, 15]
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
    return -y


def run(acquisition: str, steps: int, n_init: int, seed: int) -> None:
    torch.manual_seed(seed)
    d = 2
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

    # Initial design (Sobol-ish via uniform random for simplicity).
    X = torch.rand(n_init, d, dtype=torch.double)
    y = branin_normalized(X)

    print(f"acquisition = {acquisition}")
    print(f"init: n={n_init}, best so far = {y.max().item():.4f}")

    acqf = AlphaPFN.from_pretrained(acquisition=acquisition)

    for step in range(steps):
        # The pretrained model assumes standardized y; strict=True (default)
        # would raise on raw Branin values, which span ~hundreds.
        y_std = (y - y.mean()) / (y.std() + 1e-8)
        acqf.fit(X, y_std)

        # optimize_acqf calls acqf(X_candidate) under the hood.
        X_next, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=128,
        )

        y_next = branin_normalized(X_next.squeeze(0))
        X = torch.cat([X, X_next.detach().double()], dim=0)
        y = torch.cat([y, y_next.detach().double().reshape(1)])

        print(
            f"  step {step + 1:>2}: x={X_next.squeeze(0).tolist()}  "
            f"y={y_next.item():.4f}  acq={acq_value.item():.4f}  "
            f"best={y.max().item():.4f}"
        )

    print(f"final best y = {y.max().item():.4f} (Branin optimum is ≈ -0.397887)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--acquisition",
        choices=["EI", "UCB", "PES", "MES", "JES"],
        default="JES",
    )
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.acquisition, args.steps, args.n_init, args.seed)
