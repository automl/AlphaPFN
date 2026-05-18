"""Public AlphaPFN model API.

The model itself is the acquisition function: `model(X_test)` returns
scalar acquisition values that an outer optimizer can maximize. When
botorch is installed AlphaPFN inherits `AcquisitionFunction` so it
plugs straight into `botorch.optim.optimize_acqf`. Without botorch,
`AcquisitionFunction` falls back to `nn.Module` and the decorator
becomes a no-op — the model is still callable; the user provides
their own optimizer.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from alphapfn.loader import load_predictor, ALLOWED_VERSIONS

try:
    from botorch.acquisition.acquisition import AcquisitionFunction
    from botorch.utils.transforms import t_batch_mode_transform
    _BOTORCH_AVAILABLE = True
except ImportError:
    AcquisitionFunction = nn.Module  # type: ignore[misc,assignment]
    _BOTORCH_AVAILABLE = False

    # Fallback that mirrors botorch's contract: accept X of shape
    # (b, q, d) or (q, d); assert q == expected_q if given; insert a
    # leading batch dim if missing so the wrapped method always
    # receives a 3-D tensor.
    def t_batch_mode_transform(expected_q=None, assert_output_shape=True):  # type: ignore[no-redef]
        def _decorator(fn):
            def _wrapper(self, X, *args, **kwargs):
                if not isinstance(X, torch.Tensor):
                    return fn(self, X, *args, **kwargs)
                if X.dim() < 2:
                    raise ValueError(
                        f"{type(self).__name__} requires X to have at least 2 "
                        f"dimensions, but received X with {X.dim()} dimensions."
                    )
                if expected_q is not None and X.shape[-2] != expected_q:
                    raise AssertionError(
                        f"Expected X to be `batch_shape x q={expected_q} x d`, "
                        f"but got X with shape {tuple(X.shape)}."
                    )
                if X.dim() == 2:
                    X = X.unsqueeze(0)
                return fn(self, X, *args, **kwargs)
            return _wrapper
        return _decorator


ALLOWED_ACQUISITIONS = ("EI", "UCB", "PES", "MES", "JES")
_DIRECT_HEADS = {"PES": "pes", "MES": "mes", "JES": "jes"}


class AlphaPFN(AcquisitionFunction):
    """Acquisition-function-shaped wrapper around the PFN model.

    Construct via `AlphaPFN.from_pretrained(acquisition=..., version=...)`,
    call `fit(train_X, train_Y)` once, then call as a function on
    candidate `X_test` tensors to get acquisition values.
    """

    def __init__(
        self,
        ppd_model: Optional[nn.Module],
        head_model: Optional[nn.Module],
        acquisition: Optional[str],
        ucb_beta: float = 2.0,
        is_base_model: bool = False,
        strict: bool = True,
        _registered_model: Optional[nn.Module] = None,
    ) -> None:
        # super().__init__() with model= only works under botorch's
        # AcquisitionFunction; under the nn.Module fallback we have to
        # avoid passing keyword args.
        if _BOTORCH_AVAILABLE:
            super().__init__(model=_registered_model or ppd_model or head_model)
        else:
            super().__init__()
        self._ppd_model = ppd_model
        self._head_model = head_model
        self._acquisition = acquisition
        self._is_base_model = bool(is_base_model)
        self._ucb_beta = float(ucb_beta)
        self._strict = bool(strict)
        # Training cache (set by fit())
        self._train_X: Optional[Tensor] = None
        self._train_Y: Optional[Tensor] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        acquisition: Optional[str] = None,
        version: str = "v1",
        *,
        load_base_model: bool = False,
        ucb_beta: float = 2.0,
        strict: bool = True,
    ) -> "AlphaPFN":
        """Load a pretrained AlphaPFN.

        Pass at least one of:
          - acquisition ∈ {"EI", "UCB", "PES", "MES", "JES"}
                → callable model returning acquisition values via forward().
          - load_base_model=True
                → loads the PPD base model. forward() not implemented;
                  access the underlying model via `acqf.base_model`.

        Loading rules:
          - EI / UCB always implicitly load the base (PPD) model
            — the acquisition is computed in closed form from PPD logits.
            Passing `load_base_model=True` alongside is redundant but
            permitted.
          - PES / MES / JES load a separately-trained head. The base
            model is loaded only if `load_base_model=True`.
          - Without `acquisition`, `load_base_model=True` is required.

        Input contract:
          - Maximization. AlphaPFN scores points for a *maximization*
            objective (f_best is train_Y.max(); EI/UCB/PES/MES/JES all
            return higher values for "better" inputs). For minimization,
            pass -f(X) to fit and negate the result. The maximization
            assumption is NOT checked at runtime.
          - X must lie in [0, 1]^d. Rescale your search space first.
          - train_Y must be approximately standardized
            (|mean(y)| <= 0.5, |std(y) - 1| <= 0.5).

          With `strict=True` (default), the cube and standardization
          conditions are checked on every fit/forward; violations raise
          ValueError. Pass `strict=False` if you intentionally use
          out-of-cube inputs or non-standard targets.
        """
        if acquisition is None and not load_base_model:
            raise ValueError(
                "Specify acquisition=... or load_base_model=True (or both)."
            )
        if acquisition is not None and acquisition not in ALLOWED_ACQUISITIONS:
            raise ValueError(
                f"acquisition={acquisition!r} is not supported. "
                f"Allowed: {ALLOWED_ACQUISITIONS}"
            )
        if version not in ALLOWED_VERSIONS:
            raise ValueError(
                f"version={version!r} is not supported. "
                f"Allowed: {sorted(ALLOWED_VERSIONS)}"
            )

        # PPD/base model is loaded if:
        #   - user asked via load_base_model=True, OR
        #   - acquisition needs it (EI/UCB compute on top of PPD).
        ppd_needed = load_base_model or acquisition in {"EI", "UCB"}
        ppd_model = load_predictor("ppd", version=version) if ppd_needed else None

        head_model = None
        if acquisition in _DIRECT_HEADS:
            head_model = load_predictor(_DIRECT_HEADS[acquisition], version=version)

        # For PES/MES/JES without load_base_model, _ppd_model stays None;
        # forward() uses the head only. The acquisition function still
        # needs an `nn.Module` to register under botorch — fall back to
        # the head model in that case.
        registered_model = ppd_model if ppd_model is not None else head_model
        return cls(
            ppd_model=ppd_model,
            head_model=head_model,
            acquisition=acquisition,
            ucb_beta=ucb_beta,
            is_base_model=load_base_model,
            strict=strict,
            _registered_model=registered_model,
        )

    @property
    def base_model(self) -> nn.Module:
        """Access the loaded PPD/base model directly.

        Useful when `load_base_model=True` — the conditioning interface
        (x*, f*) is not yet exposed on AlphaPFN, so callers have to
        invoke the underlying model themselves for now.

        Raises if no base model was loaded (PES/MES/JES without
        load_base_model=True).
        """
        if self._ppd_model is None:
            raise AttributeError(
                "No base model was loaded. Pass `load_base_model=True` to "
                "from_pretrained()."
            )
        return self._ppd_model

    # ------------------------------------------------------------------
    # Input-contract checks (strict-mode)
    # ------------------------------------------------------------------

    _X_BOUND_EPS = 1e-6
    _Y_MEAN_TOL = 0.5
    _Y_STD_TOL = 0.5

    def _check_X_in_cube(self, X: Tensor, *, where: str) -> None:
        """Assert X ⊂ [0, 1]^d (with a small tolerance). Honors strict."""
        if not self._strict:
            return
        lo = float(X.min().item())
        hi = float(X.max().item())
        if lo < -self._X_BOUND_EPS or hi > 1.0 + self._X_BOUND_EPS:
            raise ValueError(
                f"{where}: X must lie in [0, 1]^d but got min={lo:.4g}, "
                f"max={hi:.4g}. The pretrained model assumes inputs "
                f"are normalized to the unit cube. If you intentionally "
                f"pass out-of-cube inputs, construct AlphaPFN with "
                f"strict=False."
            )

    def _check_y_standardized(self, y: Tensor) -> None:
        """Assert y is approximately standardized. Honors strict."""
        if not self._strict:
            return
        if y.numel() < 2:
            return  # std() undefined / unstable with < 2 points
        mean = float(y.mean().item())
        std = float(y.std().item())
        if abs(mean) > self._Y_MEAN_TOL or abs(std - 1.0) > self._Y_STD_TOL:
            raise ValueError(
                f"fit: train_Y must be approximately standardized "
                f"(|mean| <= {self._Y_MEAN_TOL}, |std-1| <= {self._Y_STD_TOL}); "
                f"got mean={mean:.4g}, std={std:.4g}. The pretrained model "
                f"assumes roughly-standard targets. Standardize before "
                f"calling fit, e.g. `y = (y - y.mean()) / (y.std() + 1e-8)`. "
                f"If you intentionally pass non-standard targets, construct "
                f"AlphaPFN with strict=False."
            )

    # ------------------------------------------------------------------
    # Fit (one-shot; stores the train data; no real "fitting" happens)
    # ------------------------------------------------------------------

    def fit(self, train_X: Tensor, train_Y: Tensor) -> "AlphaPFN":
        """Provide training context. Required before forward()."""
        if train_X.ndim != 2:
            raise ValueError(
                f"train_X must be (n, d); got shape {tuple(train_X.shape)}"
            )
        if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)
        if train_Y.ndim != 1:
            raise ValueError(
                f"train_Y must be (n,) or (n, 1); got shape {tuple(train_Y.shape)}"
            )
        if train_X.shape[0] != train_Y.shape[0]:
            raise ValueError(
                f"train_X / train_Y leading-dim mismatch: "
                f"{train_X.shape[0]} vs {train_Y.shape[0]}"
            )
        self._check_X_in_cube(train_X, where="fit")
        self._check_y_standardized(train_Y)
        self._train_X = train_X.detach()
        self._train_Y = train_Y.detach()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Forward / __call__: returns acquisition values
    # ------------------------------------------------------------------

    def _run_ppd(self, X_test: Tensor) -> Tensor:
        """Returns PPD logits at X_test, shape (num_test, num_bars)."""
        assert self._train_X is not None and self._train_Y is not None
        x_train = self._train_X.to(X_test.dtype).unsqueeze(1)        # (n, 1, d)
        y_train = self._train_Y.to(X_test.dtype).unsqueeze(1)        # (n, 1)
        x_test = X_test.unsqueeze(1)                                 # (m, 1, d)

        n = x_train.shape[0]
        gp_dim = x_train.shape[-1]
        nan_style = torch.full((1, gp_dim, 1), float("nan"),
                               dtype=X_test.dtype, device=X_test.device)
        nan_y_style = torch.full((1, 1), float("nan"),
                                 dtype=X_test.dtype, device=X_test.device)

        logits = self._ppd_model(
            x=x_train.float(),
            y=y_train.float(),
            test_x=x_test.float(),
            style=nan_style.float(),
            y_style=nan_y_style.float(),
        )
        return logits.squeeze(1)  # (m, num_bars)

    def _run_head(self, X_test: Tensor) -> Tensor:
        """Returns direct-head scalar acquisition values at X_test, shape (m,)."""
        assert self._head_model is not None
        assert self._train_X is not None and self._train_Y is not None
        x_train = self._train_X.to(X_test.dtype).unsqueeze(1)
        y_train = self._train_Y.to(X_test.dtype).unsqueeze(1)
        x_test = X_test.unsqueeze(1)

        logits = self._head_model(
            x_train.float(), y_train.float(), x_test.float()
        )
        return self._head_model.criterion.mean(logits).flatten()

    def _ei_from_ppd(self, X_test: Tensor) -> Tensor:
        assert self._train_Y is not None
        f_best = float(self._train_Y.max().item())
        logits = self._run_ppd(X_test)
        return self._ppd_model.criterion.ei(logits, f_best).flatten()

    def _ucb_from_ppd(self, X_test: Tensor) -> Tensor:
        # UCB at quantile alpha = 0.5 * (1 + erf(beta / sqrt(2))) — i.e.
        # the beta-sigma upper bound. For a simple bar distribution
        # approximation we use the inverse-cdf at p = 1 - normal_tail(beta).
        import math
        p = 0.5 * (1.0 + math.erf(self._ucb_beta / math.sqrt(2.0)))
        logits = self._run_ppd(X_test)
        return self._ppd_model.criterion.icdf(logits, p).flatten()

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Returns acquisition values at X.

        Input contract (botorch-style, applied with or without botorch):
            X has shape (b, q=1, d) — or (q=1, d), in which case the
            decorator prepends a leading batch dim. The decorator
            asserts q == 1.

        Output: (b,) for scalar acquisitions (EI/UCB/PES/MES/JES).

        When `load_base_model=True` was passed (and no acquisition),
        forward() is not implemented — the base model has a different
        interface (conditioning on optimizer x* and optimum f*) which is
        not yet designed. Use `model.base_model` directly for now.
        """
        if not self._fitted:
            raise RuntimeError("call .fit(train_X, train_Y) before forward()")

        # After the decorator, X has shape (b, 1, d). Collapse the q dim.
        X = X.squeeze(-2)
        self._check_X_in_cube(X, where="forward")

        acq = self._acquisition
        if acq is None:
            raise NotImplementedError(
                "This model was loaded with load_base_model=True. "
                "The base-model forward interface (with x*/f* conditioning) "
                "is not designed yet. Access the underlying model via "
                "`acqf.base_model` for now."
            )
        if acq == "EI":
            return self._ei_from_ppd(X)
        if acq == "UCB":
            return self._ucb_from_ppd(X)
        if acq in {"PES", "MES", "JES"}:
            return self._run_head(X)
        raise ValueError(f"Unknown acquisition: {acq!r}")
