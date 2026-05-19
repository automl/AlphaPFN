"""Backward-compat stub for `pfns4hpo_v2.esmodel`.

The training-time direct prior (`alphapfn.priors.alphapfn_model`)
imports `Predictor`, `PFNAcq`, and `PFNAcqCombi` from the original
`pfns4hpo_v2.esmodel` module to load a trained PPD checkpoint and
compute entropy reductions as data-gen targets. The alphapfn public
inference path replaced `esmodel.py` with `alphapfn.api.AlphaPFN`.

For the published-corpus training recipe these classes are NOT needed
at runtime — the targets are already baked into the pickled chunks.
But the prior file imports them at module-import time. This stub keeps
that import resolvable.

If a downstream user re-runs the data-generation step (out of scope for
v1), they need the full `PFNAcq` / `PFNAcqCombi` machinery. The plan
defers that to a future datagen pass; until then, instantiating one
of these stubs raises a clear error.
"""
from __future__ import annotations


class Predictor:
    """Enum of predictor-key strings used by PFNAcqCombi."""

    PPD = "0"
    PES = "3"
    MES = "4"
    JES = "5"
    OnlyPPD = "6"
    OnlyPES = "7"
    OnlyMES = "8"
    OnlyJES = "9"


class _UnsupportedAtTraining:
    """Raises if instantiated. Datagen-only; v1 training doesn't need this."""

    _what = "PFNAcq/PFNAcqCombi"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._what} is not supported in alphapfn training. "
            "It was used only at data-generation time (computing entropy "
            "targets from a trained PPD). The published v1 corpora already "
            "carry pre-computed targets, so training does not load these "
            "classes. If you intend to re-generate corpora, port the "
            "full data-gen pipeline (out of scope for this release)."
        )


class PFNAcq(_UnsupportedAtTraining):
    _what = "PFNAcq"


class PFNAcqCombi(_UnsupportedAtTraining):
    _what = "PFNAcqCombi"
