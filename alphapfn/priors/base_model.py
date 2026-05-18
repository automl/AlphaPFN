"""Inference-only symbols extracted from the original v18 PPD prior.

Renamed from `priors/es_pfn2_gp_scen_v18_ppd.py`. Training-time code
(get_batch, generate_trace, GP sampling, etc.) has been removed; this
module retains only the symbols that pickled checkpoints reference at
load time and the helpers needed at inference.
"""
import math

import torch
import torch.nn as nn

from alphapfn.model.encoders import (
    SequentialEncoder,
    LinearInputEncoderStep,
    ConstantNormalizationInputEncoderStep,
    NanHandlingEncoderStep,
    VariableNumFeaturesEncoderStep,
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


_Y_NORM_BY_SCENARIO = {
    0: (0.8508188, 3.3751173),
    1: (1.2269106, 3.5846262),
    6: (0.25, 1.177),
    7: (0.25, 1.177),
}


def get_y_encoder(num_features, scenario):
    if scenario not in _Y_NORM_BY_SCENARIO:
        raise NotImplementedError(f"No y-normalization stats for scenario {scenario}")
    mean, std = _Y_NORM_BY_SCENARIO[scenario]
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


class StyleEncoder(nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.linear = nn.Linear(num_features * 2, emsize)

    def forward(self, x):
        if x.ndim > 2:
            x = x.squeeze(-1)

        x_is_nan = x.isnan()
        x_no_nan = torch.nan_to_num(x, 0.0)
        normalized_x = (x_no_nan - 0.5) / math.sqrt(1 / 12)
        normalized_x[x_is_nan] = 0.0
        x_out = torch.cat((normalized_x, x_is_nan.float()), dim=-1)
        return self.linear(x_out)


_YSTYLE_NORM_BY_SCENARIO = {
    0: (5.846995, 1.8855166),
    1: (8.347659, 1.586083),
    6: (1.5, 1.12),
    7: (1.5, 1.12),
}


class StyleYEncoder(nn.Module):
    def __init__(self, emsize, num_features, scenario):
        super().__init__()
        self.linear = nn.Linear(2, emsize)
        if scenario not in _YSTYLE_NORM_BY_SCENARIO:
            raise NotImplementedError(
                f"No style-y normalization stats for scenario {scenario}"
            )
        self.mean, self.std = _YSTYLE_NORM_BY_SCENARIO[scenario]

    def forward(self, x):
        if x.ndim > 2:
            x = x.squeeze(-1)

        x_is_nan = x.isnan()
        x_no_nan = torch.nan_to_num(x, 0.0)
        normalized_x = (x_no_nan - self.mean) / self.std
        normalized_x[x_is_nan] = 0.0
        x_out = torch.cat((normalized_x, x_is_nan.float()), dim=-1)
        return self.linear(x_out)


def get_style_encoder():
    return lambda num_features, emsize: StyleEncoder(num_features, emsize)


def get_y_style_encoder(num_features, scenario):
    return lambda emsize: StyleYEncoder(emsize, num_features, scenario=scenario)
