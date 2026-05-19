"""v18 direct (PES/MES/JES) prior — inference symbols + training scaffolding.

Renamed from `priors/es_pfn2_gp_scen_v18_direct.py` on kislurm (566
lines in the original).

What's here:
  - Inference symbols (`TASK_IDS`, `normalize`, `denormalize`,
    `get_encoder`, `get_y_encoder_by_scenario`, the StyleEncoder /
    StyleYEncoder classes re-exported from `base_model`). Pickled
    direct checkpoints (`alpha_pfn_v1_{pes,mes,jes}`) reference these
    via the module path at unpickle time. `style_encoder` /
    `y_style_encoder` themselves are `None` for direct checkpoints
    (confirmed empirically across PES/MES/JES corpora), so they only
    need to be importable, not instantiable.

What is NOT here (deliberate):
  - The training-time `get_batch(...)` (~280 lines in the kislurm
    source) was the live-datagen path that loaded a trained PPD
    checkpoint and computed entropy reductions as training targets.
    The published direct corpora ship those targets pre-computed in
    each chunk's `target_y` tensor, and the training recipes always
    use `--load_path`, so `prior.get_batch` is never called by the
    offline training path.
  - The kislurm `[:15_000_000]` cap on `dim2indices` (line 300 of the
    source) — went with the live-datagen code that's not ported.
  - The `from pfns4hpo_v2.esmodel import Predictor, PFNAcq, PFNAcqCombi`
    line (used at line 322 of the source by the unported `get_batch`
    to load a PPD model). The shim in `alphapfn/training/__init__.py`
    aliases `pfns4hpo_v2.esmodel` to a stub that raises on
    instantiation, so the import resolves at module-load time but any
    accidental use triggers a clear error.
"""
import math

from alphapfn.model.encoders import (
    SequentialEncoder,
    LinearInputEncoderStep,
    ConstantNormalizationInputEncoderStep,
    NanHandlingEncoderStep,
    VariableNumFeaturesEncoderStep,
)
from alphapfn.priors.base_model import (
    MAX_DIMS,
    StyleEncoder,
    StyleYEncoder,
    _Y_NORM_BY_SCENARIO,
)


TASK_IDS = {
    "ordinary": 0,
    "neither_given": 1,
    "optimizer_given": 2,
    "optimum_given": 3,
    "both_given": 4,
}


def normalize(y, scenario):
    return y


def denormalize(y, scenario):
    return y


def get_linear_x_encoder(emsize, features_per_group):
    return SequentialEncoder(
        ConstantNormalizationInputEncoderStep(mean=0.5, std=math.sqrt(1 / 12)),
        VariableNumFeaturesEncoderStep(num_features=features_per_group),
        LinearInputEncoderStep(
            num_features=features_per_group,
            emsize=emsize,
            in_keys=("main",),
        ),
    )


def get_encoder():
    return lambda num_features, emsize: get_linear_x_encoder(
        emsize, features_per_group=num_features
    )


def get_y_encoder_by_scenario(scenario):
    if scenario not in _Y_NORM_BY_SCENARIO:
        raise NotImplementedError(f"No y-normalization stats for scenario {scenario}")
    mean, std = _Y_NORM_BY_SCENARIO[scenario]

    def get_y_encoder(num_features):
        return lambda in_dim, emsize: SequentialEncoder(
            ConstantNormalizationInputEncoderStep(mean=mean, std=std),
            NanHandlingEncoderStep(),
            LinearInputEncoderStep(
                num_features=2,
                emsize=emsize,
                out_keys=("output",),
                in_keys=("main", "nan_indicators"),
            ),
        )

    return get_y_encoder
