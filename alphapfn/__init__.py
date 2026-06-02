"""alphapfn — fast entropy-search acquisition via in-context learning.

Public surface:
    from alphapfn import AlphaPFN
    model = AlphaPFN.from_pretrained(acquisition="JES")
    model.fit(train_X, train_Y)
    acq = model(X_test)
"""
from alphapfn.api import AlphaPFN, AlphaPFNPosteriorMean, ALLOWED_ACQUISITIONS

__version__ = "0.0.2"
__all__ = [
    "AlphaPFN",
    "AlphaPFNPosteriorMean",
    "ALLOWED_ACQUISITIONS",
    "__version__",
]
