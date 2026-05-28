"""One-off conversion: old pfns4hpo_v2 .pt files → alphapfn L2 format.

For each input checkpoint:
  1. Load the pickled nn.Module under transient sys.modules aliases so
     the OLD module paths (`pfns4hpo_v2.*`) still resolve.
  2. Extract a config.json describing the architecture and the
     normalization stats that were baked into the encoders.
  3. Save the state_dict (only tensors, no pickled classes) as
     safetensors.

Output layout:
    <repo_root>/checkpoints/v1/<predictor>/
        config.json
        weights.safetensors

This script is meant to be run once. The runtime loader does not
install any sys.modules aliases.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

# Resolve repo root so `import alphapfn` works whether or not the
# package is pip-installed editable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default source directory: the repo's `final_models/` (where
# `model_training/train.py` writes by default). Set via `--source` to point
# at a different directory (e.g. an out-of-tree training output).
DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "final_models"

# Predictors and the filename patterns we will try to discover.
PREDICTORS = ("ppd", "pes", "mes", "jes")

# Per-predictor name patterns. The first match wins. Patterns cover:
#   1. The new `model_training/train.py` output: `<predictor>_seed<N>.pt` or
#      just `<predictor>.pt`.
#   2. The legacy kislurm v1 names: `es_pfn2_gp_scen7_v18_*.pt`.
# Each value is a list of glob patterns relative to the source dir.
_DISCOVERY_PATTERNS = {
    "ppd": [
        "ppd*.pt",
        "es_pfn2_gp_scen7_v18_ppd_*.pt",
    ],
    "pes": [
        "pes*.pt",
        "es_pfn2_gp_scen7_v18_direct_PES_*.pt",
    ],
    "mes": [
        "mes*.pt",
        "es_pfn2_gp_scen7_v18_direct_MES_*.pt",
    ],
    "jes": [
        "jes*.pt",
        "es_pfn2_gp_scen7_v18_direct_JES_*.pt",
    ],
}

VERSION = "v1"


def discover_predictor_file(source: Path, predictor: str, override: Path | None = None) -> Path:
    """Return the .pt file for `predictor` inside `source`.

    `override`, if set, is taken literally (no discovery). Otherwise,
    each pattern in `_DISCOVERY_PATTERNS[predictor]` is tried in order;
    the first one with exactly one match wins. Multiple matches raise.
    """
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--{predictor}-path file missing: {override}")
        return override
    for pattern in _DISCOVERY_PATTERNS[predictor]:
        matches = sorted(source.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous {predictor!r} files in {source}: "
                f"{[m.name for m in matches]} — pass --{predictor}-path to disambiguate."
            )
    raise FileNotFoundError(
        f"No .pt file found for predictor={predictor!r} in {source}. "
        f"Tried patterns: {_DISCOVERY_PATTERNS[predictor]}. "
        f"Pass --{predictor}-path to specify explicitly."
    )


# Keys installed by install_shim — uninstall_shim removes exactly these,
# so that a co-resident `alphapfn.training` shim (which installs a
# superset) is not collaterally wiped.
_SHIM_KEYS = (
    "pfns4hpo_v2",
    "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_ppd",
    "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_direct",
)
_PRE_SHIM_VALUES: dict[str, object] = {}


def install_shim() -> None:
    """Make old pfns4hpo_v2.* module paths resolve to the alphapfn rewrite."""
    import alphapfn
    import alphapfn.priors.base_model as base_model
    import alphapfn.priors.alphapfn_model as alphapfn_model

    targets = {
        "pfns4hpo_v2": alphapfn,
        "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_ppd": base_model,
        "pfns4hpo_v2.priors.es_pfn2_gp_scen_v18_direct": alphapfn_model,
    }
    for key, mod in targets.items():
        if key in sys.modules:
            _PRE_SHIM_VALUES[key] = sys.modules[key]
        sys.modules[key] = mod


def uninstall_shim() -> None:
    """Reverse install_shim. Preserves any pre-existing entries (e.g.
    from a co-resident `alphapfn.training` shim that ran first)."""
    for key in _SHIM_KEYS:
        if key in _PRE_SHIM_VALUES:
            sys.modules[key] = _PRE_SHIM_VALUES.pop(key)
        else:
            sys.modules.pop(key, None)


def _const_norm_stats(submodule) -> tuple[float, float]:
    return float(submodule.mean), float(submodule.std)


def extract_config(model: torch.nn.Module, predictor: str) -> dict[str, Any]:
    arch = {
        "nlayers": len(model.transformer_layers.layers),
        "emsize": int(model.ninp),
        "nhead": int(model.nhead),
        "nhid": int(model.nhid),
        "features_per_group": int(model.features_per_group),
        "attention_between_features": bool(model.attention_between_features),
        "activation": "gelu",
        "feature_positional_embedding": model.feature_positional_embedding,
        "num_bars": int(model.criterion.borders.shape[0] - 1),
    }

    # Encoders. "kind" strings are arch-level switches consumed by the loader.
    encoders: dict[str, Any] = {
        "x": {"kind": "linear_with_norm", "out_dim": arch["emsize"]},
        "y": {"kind": "linear_with_nan_handling", "out_dim": arch["emsize"]},
    }

    # y normalization stats come from the y_encoder's first step.
    if model.y_encoder is not None:
        y_mean, y_std = _const_norm_stats(model.y_encoder[0])
    else:
        y_mean, y_std = 0.0, 1.0

    # Style / y-style encoders are only present on the PPD checkpoint.
    if model.style_encoder is not None:
        encoders["style"] = {
            "kind": "alphapfn_style",
            "num_features": int(model.style_encoder.linear.in_features // 2),
            "emsize": int(model.style_encoder.linear.out_features),
        }
    else:
        encoders["style"] = None

    if model.y_style_encoder is not None:
        encoders["y_style"] = {
            "kind": "alphapfn_y_style",
            "num_features": 1,
            "emsize": int(model.y_style_encoder.linear.out_features),
            "y_mean": float(model.y_style_encoder.mean),
            "y_std": float(model.y_style_encoder.std),
        }
    else:
        encoders["y_style"] = None

    # Decoder.
    dec = model.decoder_dict["standard"]
    in_features = dec[0].in_features
    hidden = dec[0].out_features
    out_features = dec[2].out_features
    assert in_features == arch["emsize"], (in_features, arch["emsize"])
    assert out_features == arch["num_bars"], (out_features, arch["num_bars"])
    decoders = {
        "standard": {"hidden": int(hidden), "out": int(out_features), "activation": "gelu"}
    }

    normalization = {
        "y": {"mean": float(y_mean), "std": float(y_std)},
        "x": {"mean": 0.5, "std": 0.2886751345948129},  # sqrt(1/12)
    }

    return {
        "alphapfn_version": VERSION,
        "predictor": predictor,
        "architecture": arch,
        "encoders": encoders,
        "decoders": decoders,
        "normalization": normalization,
    }


def _arch_signature(cfg: dict[str, Any]) -> tuple:
    a = cfg["architecture"]
    return (
        a["nlayers"], a["emsize"], a["nhead"], a["nhid"],
        a["features_per_group"], a["attention_between_features"],
        a["activation"], a["feature_positional_embedding"], a["num_bars"],
    )


def export(
    source_dir: Path,
    output_dir: Path,
    *,
    overrides: dict[str, Path] | None = None,
    predictors: tuple[str, ...] = PREDICTORS,
) -> None:
    overrides = overrides or {}
    install_shim()
    try:
        configs: dict[str, dict] = {}
        for predictor in predictors:
            src = discover_predictor_file(
                source_dir, predictor, override=overrides.get(predictor)
            )

            print(f"[{predictor}] loading {src.name}")
            model = torch.load(src, map_location="cpu", weights_only=False)

            cfg = extract_config(model, predictor=predictor)
            configs[predictor] = cfg

            out_dir = output_dir / VERSION / predictor
            out_dir.mkdir(parents=True, exist_ok=True)

            state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, str(out_dir / "weights.safetensors"))
            with open(out_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2, sort_keys=False)

            num_params = sum(int(v.numel()) for v in state_dict.values())
            print(
                f"[{predictor}] wrote {out_dir}/weights.safetensors "
                f"({num_params:,} params) + config.json"
            )

        # Sanity-check architecture is identical across the 4 checkpoints.
        sigs = {p: _arch_signature(c) for p, c in configs.items()}
        if len(set(sigs.values())) != 1:
            print("ARCHITECTURE MISMATCH across checkpoints:", file=sys.stderr)
            for p, s in sigs.items():
                print(f"  {p}: {s}", file=sys.stderr)
            raise SystemExit(1)
        print(f"All {len(configs)} checkpoints share architecture: {next(iter(sigs.values()))}")
    finally:
        uninstall_shim()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            f"Directory containing the .pt files for each predictor. "
            f"Files are discovered by predictor name (e.g. ppd_seed0.pt) "
            f"or by the legacy kislurm pattern (es_pfn2_gp_scen7_v18_*). "
            f"Default: {DEFAULT_SOURCE}"
        ),
    )
    # `--output` and `--destination` are accepted as aliases.
    p.add_argument(
        "--output",
        "--destination",
        dest="output",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Output directory (default: <repo>/checkpoints)",
    )
    p.add_argument(
        "--predictor",
        choices=PREDICTORS,
        action="append",
        default=None,
        help=(
            "Export only this predictor (may be passed multiple times; "
            "default: all four)."
        ),
    )
    for predictor in PREDICTORS:
        p.add_argument(
            f"--{predictor}-path",
            type=Path,
            default=None,
            help=f"Explicit path to the {predictor.upper()} .pt file (overrides discovery).",
        )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        predictor: getattr(args, f"{predictor}_path")
        for predictor in PREDICTORS
        if getattr(args, f"{predictor}_path") is not None
    }
    predictors = tuple(args.predictor) if args.predictor else PREDICTORS
    export(args.source, args.output, overrides=overrides, predictors=predictors)


if __name__ == "__main__":
    main()
