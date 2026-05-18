"""Inference-only symbols extracted from the original v18 direct prior.

Renamed from `priors/es_pfn2_gp_scen_v18_direct.py`. The direct
checkpoints (PES/MES/JES) keep their pickled `SequentialEncoder`
instances pointing at `alphapfn.model.encoders`, so this module is
mostly a placeholder so that pickled `__module__` paths still resolve.
"""
import math

from alphapfn.model.encoders import (
    SequentialEncoder,
    LinearInputEncoderStep,
    ConstantNormalizationInputEncoderStep,
    NanHandlingEncoderStep,
    VariableNumFeaturesEncoderStep,
)
from alphapfn.priors.base_model import StyleEncoder, StyleYEncoder, _Y_NORM_BY_SCENARIO


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
