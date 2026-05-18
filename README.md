# $\alpha$-PFN: Fast Entropy Search via In-Context Learning

Lightweight package for $\alpha$-PFN, a Prior-Fitted Network for fast entropy search. $\alpha$-PFN replaces the Gaussian Process surrogate in a Bayesian optimization loop with a single
transformer forward pass: condition on observed $(X, y)$, score candidate
points, optimize. Supported acquisitions:

- Predictive Entropy Search (`PES`), Max Value Entropy Search (`MES`), Joint-Entropy Search (`JES`).
- `EI`, `UCB` (computed analytically from the PPD bar-distribution),


## Install

```bash
git clone <repo>
cd alpha-pfn
uv sync
```

TODO: add on pypi.

## Quick start

See `examples/bo_with_optimize_acqf.py`. The relevant snippet:

```python
import torch
from botorch.optim import optimize_acqf # need to install Botorch
from alphapfn import AlphaPFN

bounds = torch.stack([torch.zeros(d), torch.ones(d)]).double()
acqf = AlphaPFN.from_pretrained(acquisition="JES")

for step in range(num_steps):
    # Standardize y; $\alpha$-PFN expects standardized targets.
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

Run it:

```bash
.venv/bin/python examples/bo_with_optimize_acqf.py --acquisition JES --steps 15
```

## API

```python
AlphaPFN.from_pretrained(
    acquisition: str | None = None,            # one of "EI","UCB","PES","MES","JES"
    version: str = "v1",
    *,
    load_base_model: bool = False,
    ucb_beta: float = 2.0,
    strict: bool = True,
)
```

Loading rules:

- `EI` and `UCB` implicitly load the PPD base model (the acquisition is
  computed in closed form from PPD logits).
- `PES` / `MES` / `JES` load a separately-trained acquisition head; the
  base model is loaded only if `load_base_model=True`.
- `load_base_model=True`: TBD.

The returned object is callable: `acqf(X)` with `X.shape == (b, 1, d)`
returns acquisition values of shape `(b,)`. Calling
`acqf.fit(train_X, train_Y)` must be called once before `forward`.

The pretrained models assume:

- **Maximization.** $\alpha$-PFN scores points for a *maximization*
  objective. `f_best` is `train_Y.max()`, EI/UCB/PES/MES/JES all
  return higher values for "better" inputs. To minimize $f$, pass
  `-f(X)` to `fit` and negate the result.
- **`X ⊂ [0, 1]^d`** — inputs are normalized to the unit cube. Out-of-cube
  inputs silently produce nonsense logits (the encoder has zero training
  signal outside that range). Rescale your search space to the unit
  cube and rescale back when reporting results.
- **`y` approximately standardized** — `|mean(y)| ≲ 0.5`, `|std(y) - 1| ≲ 0.5`.
  Standardize before `fit`: `y_std = (y - y.mean()) / (y.std() + 1e-8)`.

With `strict=True` (default) the cube and standardization conditions
are checked on every `fit` and every `forward`; violations raise
`ValueError`. The maximization assumption is **not** checked — it
silently produces wrong results if you fit on a minimization-shaped
objective. Pass `strict=False` if you intentionally use out-of-cube
inputs or non-standard targets.