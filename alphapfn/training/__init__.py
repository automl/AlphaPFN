"""alphapfn.training — training entry point + sys.modules shim.

Importing this subpackage installs aliases so that pickled training
batches (which reference `pfns4hpo_v2.priors.prior.Batch` in their
bytecode) resolve transparently onto alphapfn classes. The shim is
installed BEFORE any prior or data-loader module is imported, since
those modules' top-level imports themselves reach back into
`pfns4hpo_v2.*` paths.

The inference path (`from alphapfn import AlphaPFN`) does NOT import
this subpackage, so the shim is opt-in: regular inference users see
no namespace pollution.

After the shim is in place this module re-exports the training
entry-points (`train`, `PriorDataLoader`, `DistributedPriorDataLoader`,
bucket helpers, the gp-precompute helpers, and the live-prior
factories) for convenience.
"""
from __future__ import annotations

import sys


def _install_shim() -> None:
    """Install pfns4hpo_v2 → alphapfn module aliases.

    Idempotent: re-importing this subpackage is a no-op because we
    check whether the aliases are already in `sys.modules`.
    """
    import importlib

    # Resolve each target via importlib so we don't depend on attribute
    # access against the still-initializing `alphapfn.training` parent.
    alphapfn = importlib.import_module("alphapfn")
    priors = importlib.import_module("alphapfn.priors")
    prior_mod = importlib.import_module("alphapfn.priors.prior")
    priors_utils = importlib.import_module("alphapfn.priors.utils")
    base_model = importlib.import_module("alphapfn.priors.base_model")
    alphapfn_model = importlib.import_module("alphapfn.priors.alphapfn_model")
    gp_precompute = importlib.import_module("alphapfn.training.gp_precompute")
    esmodel_stub = importlib.import_module("alphapfn.training._esmodel_stub")

    aliases = {
        "pfns4hpo_v2": alphapfn,
        "pfns4hpo_v2.priors": priors,
        "pfns4hpo_v2.priors.prior": prior_mod,
        # The pickled batches reference Batch via the original
        # `pfns4hpo_v2.priors.utils` module path. Route to the
        # existing inference-side re-export.
        "pfns4hpo_v2.priors.utils": priors_utils,
        # Training-version v18 priors live in the same files as the
        # inference trims, augmented with extra functions.
        "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_ppd": base_model,
        "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_direct": alphapfn_model,
        # Imported at module-load time by the v18 priors.
        "pfns4hpo_v2.gp_precompute_sam_1905": gp_precompute,
        # Imported at module-load time by alphapfn_model (direct).
        # Datagen-only; the stub raises if instantiated.
        "pfns4hpo_v2.esmodel": esmodel_stub,
    }
    for name, mod in aliases.items():
        # Idempotent: don't clobber if the user has already mapped these
        # (e.g. from `scripts/export_checkpoints.py` which installs its
        # own subset of aliases).
        sys.modules.setdefault(name, mod)


_install_shim()


# Re-export training entry points (imported AFTER the shim so any prior
# import chain inside these modules resolves cleanly).
from alphapfn.training.prior_data_loader import (  # noqa: E402
    PriorDataLoader,
    DistributedPriorDataLoader,
    get_uniform_sampler,
    get_expon_sep_sampler,
    get_rank,
)

__all__ = [
    "PriorDataLoader",
    "DistributedPriorDataLoader",
    "get_uniform_sampler",
    "get_expon_sep_sampler",
    "get_rank",
]
