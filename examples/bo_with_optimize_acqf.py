"""Minimal BO loop driven by alphapfn + botorch.optim.optimize_acqf.

Usage:
    python examples/bo_with_optimize_acqf.py [--acquisition JES] [--steps 30]

The script optimizes Hartmann 6D on the unit cube. Each BO step:
  1. Fit AlphaPFN on the observed (X, y).
  2. optimize_acqf finds argmax of the acquisition over the bounds.
  3. Evaluate the function at the new point; append.
"""
from __future__ import annotations

import argparse

import torch
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann

from alphapfn import AlphaPFN


def run(acquisition: str, steps: int, n_init: int, seed: int) -> None:
    torch.manual_seed(seed)
    d = 6
    bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
    hartmann = Hartmann(dim=d, negate=True)  # negate so AlphaPFN maximizes

    # Initial design (Sobol-ish via uniform random for simplicity).
    X = torch.rand(n_init, d, dtype=torch.double)
    y = hartmann(X)

    print(f"acquisition = {acquisition}")
    print(f"init: n={n_init}, best so far = {y.max().item():.4f}")

    acqf = AlphaPFN.from_pretrained(acquisition=acquisition)

    for step in range(steps):
        acqf.fit(X, y)  # fit() standardizes y internally

        # optimize_acqf calls acqf(X_candidate) under the hood.
        X_next, acq_value = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=128,
        )

        y_next = hartmann(X_next.squeeze(0))
        X = torch.cat([X, X_next.detach().double()], dim=0)
        y = torch.cat([y, y_next.detach().double().reshape(1)])

        print(
            f"  step {step + 1:>2}: y={y_next.item():.4f}  "
            f"acq={acq_value.item():.4f}  best={y.max().item():.4f}"
        )

    print(
        f"final best y = {y.max().item():.4f} "
        f"(Hartmann 6D optimum is ≈ {float(hartmann.optimal_value):.4f})"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--acquisition",
        choices=["EI", "UCB", "PES", "MES", "JES"],
        default="JES",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--n-init", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.acquisition, args.steps, args.n_init, args.seed)
