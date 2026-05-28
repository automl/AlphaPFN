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
    acquisition: str | None = None,   # or "MES", "JES"
    version: str = "v1",
    *,
    load_base_model: bool = False,
    ucb_beta: float = 2.0,
    strict: bool = True,              # Pass strict=False to skip the input-range and standardization checks.
)
```

Before fitting, prepare your data so that:
  - You are maximizing. To minimize instead, negate your objective.
    This is NOT checked, so forgetting it silently gives wrong results.
  - Each input feature is rescaled to lie between 0 and 1.
  - Targets are roughly standardized (subtract the mean, divide by the std).


# Cite

```latex
@inproceedings{
  rakotoarison2026alphapfn,
  title={{$\alpha$}-PFN: Fast Entropy Search via In-Context Learning},
  author={Rakotoarison, Herilalaina and Adriaensen, Steven and Viering, Tom and Hvarfner, Carl and M{\"u}ller, Samuel and Hutter, Frank and Bakshy, Eytan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=7Oonij8oLU}
}
```