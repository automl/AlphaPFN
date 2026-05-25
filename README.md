# $\alpha$-PFN: Fast Entropy Search via In-Context Learning

Lightweight package for $\alpha$-PFN, a Prior-Fitted Network for fast entropy search. Supported acquisitions: Predictive Entropy Search (PES), Max Value Entropy Search (MES), Joint-Entropy Search (JES).

> To reproduce our ICML paper experiments, see branch
> [`icml2026`](https://github.com/automl/AlphaPFN/tree/icml2026).

## Install

```bash
git clone https://github.com/automl/AlphaPFN
cd AlphaPFN
uv sync
```

PyPI release: TODO.

## Quick start

A minimal BO loop driven by `botorch.optim.optimize_acqf`:

```python
import torch
from botorch.optim import optimize_acqf  # needs botorch
from alphapfn import AlphaPFN

bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
acqf = AlphaPFN.from_pretrained(acquisition="JES")

for step in range(num_steps):
    # $\alpha$-PFN expects standardized targets.
    y_std = (y - y.mean()) / (y.std() + 1e-8)
    acqf.fit(X, y_std)

    X_next, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=128,
    )
    y_next = objective(X_next.squeeze(0))
    X = torch.cat([X, X_next.detach().double()], dim=0)
    y = torch.cat([y, y_next.detach().double().reshape(1)])
```

A runnable version lives at
[`examples/bo_with_optimize_acqf.py`](examples/bo_with_optimize_acqf.py):

```bash
.venv/bin/python examples/bo_with_optimize_acqf.py --acquisition JES --steps 15
```

## API

```python
AlphaPFN.from_pretrained(
    acquisition: str | None = None,            # or "MES","JES"
    version: str = "v1",
    *,
    load_base_model: bool = False,
    ucb_beta: float = 2.0,
    strict: bool = True,
)
```

The pretrained models assume:

- **Maximization.** $f_\text{best} = $ `train_Y.max()`. To minimize $f$,
  fit on `-f(X)` and negate. *Not checked* — silently wrong if you
  forget.
- **`X ⊂ [0, 1]^d`.** Rescale your search space; results outside the
  cube are meaningless.
- **`y` approximately standardized.** `|mean(y)| ≲ 0.5`, `|std(y) - 1| ≲ 0.5`.
  Pre-standardize: `(y - y.mean()) / (y.std() + 1e-8)`.

The cube and standardization checks fire on every `fit` and `forward`
under `strict=True` (default); pass `strict=False` to disable.
