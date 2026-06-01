# $\alpha$-PFN: Fast Entropy Search via In-Context Learning

[![PyPI version](https://img.shields.io/pypi/v/alphapfn.svg)](https://pypi.org/project/alphapfn/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=7Oonij8oLU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/AlphaPFN/blob/main/examples/quickstart.ipynb)

**$\alpha$-PFN** is a Prior-Fitted Network that amortizes information-theoretic acquisition functions. Supported acquisition functions: Predictive Entropy Search (PES), Max-value Entropy Search (MES), and Joint Entropy Search (JES). 

<p align="center">
  <img src="images/hero.gif" alt="Traditional GP-based Entropy Search samples optima via RFF and averages conditional entropies over N MC samples; α-PFN approximates the same acquisition in a single transformer forward pass.">
</p>

> To reproduce our ICML 2026 paper experiments, see branch
> [`icml2026`](https://github.com/automl/AlphaPFN/tree/icml2026).

## Install

```bash
pip install "alphapfn[botorch]"
```

Or from source:

```bash
git clone https://github.com/automl/AlphaPFN
cd AlphaPFN
uv sync --extra botorch
```

Pretrained checkpoints (~20 MB) download automatically on the first `from_pretrained` call and cache under `~/.cache/alphapfn/`.

## Quick start

A self-contained 2D BO loop on Branin, using `botorch.optim.optimize_acqf`:

```python
import math
import torch
from botorch.optim import optimize_acqf
from alphapfn import AlphaPFN

# 1. Define the objective on the unit cube (α-PFN maximizes — we negate Branin).
def branin(X):
    x1 = 15.0 * X[..., 0] - 5.0
    x2 = 15.0 * X[..., 1]
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return -(a * (x2 - b * x1**2 + c * x1 - r) ** 2
             + s * (1 - t) * torch.cos(x1) + s)

# 2. Initial design.
torch.manual_seed(0)
d, n_init, num_steps = 2, 5, 15
X = torch.rand(n_init, d, dtype=torch.double)
y = branin(X)
bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()

# 3. Load the pretrained acquisition; checkpoints download on first call.
acqf = AlphaPFN.from_pretrained(acquisition="JES")

# 4. BO loop.
for step in range(num_steps):
    y_std = (y - y.mean()) / (y.std() + 1e-8)   # standardize targets
    acqf.fit(X, y_std)
    X_next, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=5, raw_samples=128)
    y_next = branin(X_next.squeeze(0))
    X = torch.cat([X, X_next.detach().double()])
    y = torch.cat([y, y_next.detach().double().reshape(1)])
    print(f"step {step+1:>2}: best so far = {y.max().item():.4f}")
```

Runnable version: [`examples/bo_with_optimize_acqf.py`](examples/bo_with_optimize_acqf.py)
or open the [Colab notebook](https://colab.research.google.com/github/automl/AlphaPFN/blob/main/examples/quickstart.ipynb).

## API

```python
AlphaPFN.from_pretrained(
    acquisition: str | None = None,   # "PES" (default), "MES", or "JES"
    version: str = "v1",
    *,
    load_base_model: bool = False,
    ucb_beta: float = 2.0,
    strict: bool = True,              # pass strict=False to skip input checks
)
```

Before fitting, prepare your data so that:
- **You are maximizing.** To minimize instead, negate your objective.
  This is NOT checked, so forgetting it silently gives wrong results.
- **Each input feature lies in `[0, 1]`.** Rescale your search space accordingly.
- **Targets are roughly standardized** (subtract the mean, divide by the std).

`strict=True` (default) validates the cube and standardization on every `fit`/`forward`; pass `strict=False` to skip.

## Cite

```bibtex
@inproceedings{
  rakotoarison2026alphapfn,
  title={{$\alpha$}-PFN: Fast Entropy Search via In-Context Learning},
  author={Rakotoarison, Herilalaina and Adriaensen, Steven and Viering, Tom and Hvarfner, Carl and M{\"u}ller, Samuel and Hutter, Frank and Bakshy, Eytan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=7Oonij8oLU}
}
```
