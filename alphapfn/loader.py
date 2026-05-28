"""Runtime loader for AlphaPFN L2 checkpoints.

Reconstructs `PerFeatureTransformer` from `config.json` and a
state_dict stored as `weights.safetensors`. No pickled Python classes
ever cross the load boundary; the checkpoint is portable to any
environment that has `alphapfn` importable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from alphapfn.model.bar_distribution import FullSupportBarDistribution
from alphapfn.model.encoders import (
    SequentialEncoder,
    LinearInputEncoderStep,
    ConstantNormalizationInputEncoderStep,
    NanHandlingEncoderStep,
    VariableNumFeaturesEncoderStep,
)
from alphapfn.model.transformer import PerFeatureTransformer
from alphapfn.priors.base_model import StyleEncoder, StyleYEncoder


from alphapfn.checkpoints import ensure_checkpoints

REPO_ROOT = Path(__file__).resolve().parent.parent
# Dev convenience: if running from a clone with a populated
# checkpoints/<version>/ tree (e.g. produced by model_training/export_checkpoints.py),
# prefer it over the user cache. Falls back to the cache otherwise.
REPO_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

ALLOWED_PREDICTORS = {"ppd", "pes", "mes", "jes"}
ALLOWED_VERSIONS = {"v1"}


def checkpoint_dir(predictor: str, version: str = "v1") -> Path:
    """Resolve the directory for `(predictor, version)`.

    Order: dev repo `checkpoints/<version>/` if present, else user cache
    (downloading the bundle on first use).
    """
    repo_dir = REPO_CHECKPOINT_ROOT / version / predictor
    if (repo_dir / "weights.safetensors").exists():
        return repo_dir
    return ensure_checkpoints(version) / predictor


def _build_x_encoder(spec: dict[str, Any], features_per_group: int) -> SequentialEncoder:
    if spec["kind"] != "linear_with_norm":
        raise ValueError(f"Unsupported x encoder kind: {spec['kind']!r}")
    emsize = int(spec["out_dim"])
    import math
    return SequentialEncoder(
        ConstantNormalizationInputEncoderStep(mean=0.5, std=math.sqrt(1 / 12)),
        VariableNumFeaturesEncoderStep(num_features=features_per_group),
        LinearInputEncoderStep(
            num_features=features_per_group,
            emsize=emsize,
            in_keys=("main",),
        ),
    )


def _build_y_encoder(
    spec: dict[str, Any], y_norm: dict[str, float]
) -> SequentialEncoder:
    if spec["kind"] != "linear_with_nan_handling":
        raise ValueError(f"Unsupported y encoder kind: {spec['kind']!r}")
    emsize = int(spec["out_dim"])
    return SequentialEncoder(
        ConstantNormalizationInputEncoderStep(
            mean=float(y_norm["mean"]), std=float(y_norm["std"])
        ),
        NanHandlingEncoderStep(),
        LinearInputEncoderStep(
            num_features=2,
            emsize=emsize,
            out_keys=("output",),
            in_keys=("main", "nan_indicators"),
        ),
    )


def _build_style(spec: Optional[dict[str, Any]]) -> Optional[StyleEncoder]:
    if spec is None:
        return None
    if spec["kind"] != "alphapfn_style":
        raise ValueError(f"Unsupported style encoder kind: {spec['kind']!r}")
    return StyleEncoder(
        num_features=int(spec["num_features"]),
        emsize=int(spec["emsize"]),
    )


def _build_y_style(spec: Optional[dict[str, Any]]) -> Optional[StyleYEncoder]:
    if spec is None:
        return None
    if spec["kind"] != "alphapfn_y_style":
        raise ValueError(f"Unsupported y_style encoder kind: {spec['kind']!r}")
    enc = StyleYEncoder.__new__(StyleYEncoder)
    torch.nn.Module.__init__(enc)
    enc.linear = torch.nn.Linear(2, int(spec["emsize"]))
    enc.mean = float(spec["y_mean"])
    enc.std = float(spec["y_std"])
    return enc


def _build_decoder(spec: dict[str, Any], in_features: int) -> torch.nn.Sequential:
    if spec["activation"] != "gelu":
        raise ValueError(f"Unsupported decoder activation: {spec['activation']!r}")
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, int(spec["hidden"])),
        torch.nn.GELU(),
        torch.nn.Linear(int(spec["hidden"]), int(spec["out"])),
    )


def _build_model(cfg: dict[str, Any], borders: torch.Tensor) -> PerFeatureTransformer:
    arch = cfg["architecture"]
    encs = cfg["encoders"]

    x_enc = _build_x_encoder(encs["x"], features_per_group=int(arch["features_per_group"]))
    y_enc = _build_y_encoder(encs["y"], y_norm=cfg["normalization"]["y"])
    style = _build_style(encs.get("style"))
    y_style = _build_y_style(encs.get("y_style"))

    decoder_cfg = cfg["decoders"]["standard"]
    if decoder_cfg["hidden"] != int(arch["nhid"]):
        raise ValueError(
            f"decoder hidden ({decoder_cfg['hidden']}) does not match "
            f"transformer nhid ({arch['nhid']}); current loader assumes they share."
        )
    if decoder_cfg["activation"] != "gelu":
        raise ValueError(f"Unsupported decoder activation: {decoder_cfg['activation']!r}")

    model = PerFeatureTransformer(
        encoder=x_enc,
        y_encoder=y_enc,
        style_encoder=style,
        y_style_encoder=y_style,
        ninp=int(arch["emsize"]),
        nhead=int(arch["nhead"]),
        nhid=int(arch["nhid"]),
        nlayers=int(arch["nlayers"]),
        features_per_group=int(arch["features_per_group"]),
        attention_between_features=bool(arch["attention_between_features"]),
        activation=arch.get("activation", "gelu"),
        feature_positional_embedding=arch.get("feature_positional_embedding"),
        decoder_dict={"standard": (None, int(decoder_cfg["out"]))},
    )
    model.criterion = FullSupportBarDistribution(borders=borders)
    return model


def load_predictor(
    predictor: str,
    version: str = "v1",
    path: Optional[Path] = None,
) -> PerFeatureTransformer:
    """Load a single checkpoint (config + safetensors) and return the model."""
    if predictor not in ALLOWED_PREDICTORS:
        raise ValueError(
            f"Unknown predictor {predictor!r}. "
            f"Allowed: {sorted(ALLOWED_PREDICTORS)}"
        )
    if version not in ALLOWED_VERSIONS:
        raise ValueError(
            f"Unknown version {version!r}. "
            f"Allowed: {sorted(ALLOWED_VERSIONS)}"
        )

    ckpt_dir = Path(path) if path is not None else checkpoint_dir(predictor, version)
    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "weights.safetensors"

    with open(config_path) as f:
        cfg = json.load(f)
    # Lazy import: `safetensors` is only required at actual checkpoint
    # load time, not at `import alphapfn` time. This keeps the package
    # usable in environments (e.g. kislurm login node) where safetensors
    # isn't installed but the training/data path is exercised.
    from safetensors.torch import load_file
    state_dict = load_file(str(weights_path))

    borders = state_dict["criterion.borders"]
    model = _build_model(cfg, borders=borders)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch for {predictor!r}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    return model
