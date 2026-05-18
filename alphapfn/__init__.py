"""alphapfn — fast entropy-search acquisition via in-context learning.

Public surface:
    from alphapfn import AlphaPFN
    model = AlphaPFN.from_pretrained(acquisition="JES")
    model.fit(train_X, train_Y)
    acq = model(X_test)
"""
from alphapfn.api import AlphaPFN, ALLOWED_ACQUISITIONS

__all__ = ["AlphaPFN", "ALLOWED_ACQUISITIONS"]
