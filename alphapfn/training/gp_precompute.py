"""GP sample-generation helpers used by the training-time priors.

Ported from `pfns4hpo_v2/gp_precompute_sam_1905.py` on kislurm. The
original module contained a slurm-driven datagen CLI (do_job / do_work
/ main / generate_optimizers / Struct / get_res_file / get_initial_points
/ optimize_points / __main__ block); those are out of scope for the
alphapfn training package and were dropped.

What remains:
  - `sample_GP` — random Fourier-feature representation of a GP sample.
  - `get_GP_value_*` family — evaluate the sampled GP at given inputs.
  - `vectorized_get_GP_value_with_noise2` — batched version with
    per-element noise, used by the v18 priors.
  - `get_scenario` / `get_max_dim_per_scenario` and the two
    `get_scenario_*_hyperpar` helpers — scenario-keyed config used by
    the prior's `get_batch`.
"""
from __future__ import annotations

import numpy as np
import torch


##################################################################
############################ GP LOGIC ############################
##################################################################

# samples a random GP, where the GP is determined by the seed
def sample_GP(num_dimensions=10, num_fourier_features=500, gamma=50, seed=42, dtype=torch.double):
    state = np.random.get_state(legacy=False)

    # compute random fourier features
    np.random.seed(seed)
    random_weights = (2.0 * gamma) ** 0.5 * np.random.normal(size=(num_dimensions, num_fourier_features))
    random_offset = np.random.uniform(0, 2 * np.pi, size=num_fourier_features)
    W_GP = np.random.normal(0, 1, (num_fourier_features, 1))  # this vector defines the GP in RFF space

    t_random_weights = torch.from_numpy(random_weights).to(dtype)
    t_random_offset = torch.from_numpy(random_offset).to(dtype)
    t_W_GP = torch.from_numpy(W_GP).to(dtype)

    np.random.set_state(state)

    return t_random_weights, t_random_offset, t_W_GP


def get_GP_value_noiseless(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, mean_function=0):
    cos_pos = torch.matmul(torch.clamp(X, min=0, max=1), t_random_weights) + t_random_offset
    X_RFF_torch = torch.cos(cos_pos) * (2.0 / len(t_random_offset)) ** 0.5
    F_GP_torch = torch.sum(torch.matmul(X_RFF_torch, t_W_GP), axis=1) * sigma_output + mean_function
    return F_GP_torch


def get_GP_value_with_noise(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, sigma_noise, mean_function=0):
    F_GP_torch = get_GP_value_noiseless(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, mean_function=mean_function)
    Y_GP_torch = F_GP_torch + torch.normal(mean=0, std=sigma_noise, size=(X.shape[0],))
    return Y_GP_torch


def vectorized_get_GP_value_noiseless(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, mean_function=0):
    # X: [batch_size, seq_len, num_dimensions]
    # t_random_weights: [batch_size, num_dimensions, num_fourier_features]
    # t_random_offset: [batch_size, num_fourier_features]
    # t_W_GP: [batch_size, num_fourier_features, 1]
    X_clamped = torch.clamp(X, min=0, max=1)
    cos_pos = torch.bmm(X_clamped, t_random_weights) + t_random_offset.unsqueeze(1)
    X_RFF_torch = torch.cos(cos_pos) * (2.0 / t_random_offset.shape[1]) ** 0.5
    F_GP_torch = torch.bmm(X_RFF_torch, t_W_GP).squeeze(-1) * sigma_output + mean_function
    return F_GP_torch


def vectorized_get_GP_value_with_noise(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, sigma_noise, mean_function=0):
    F_GP_torch = vectorized_get_GP_value_noiseless(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, mean_function=mean_function)
    Y_GP_torch = F_GP_torch + torch.normal(mean=0, std=sigma_noise, size=F_GP_torch.shape)
    return Y_GP_torch


def vectorized_get_GP_value_with_noise2(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, sigma_noise, mean_function=0):
    # Supports varying noise across batch elements.
    F_GP_torch = vectorized_get_GP_value_noiseless(X, t_random_weights, t_random_offset, t_W_GP, sigma_output, mean_function=mean_function)
    noise = torch.randn_like(F_GP_torch)
    scaled_noise = noise * sigma_noise
    Y_GP_torch = F_GP_torch + scaled_noise
    return Y_GP_torch


##################################################################
########################## SCENARIO CONFIG #######################
##################################################################

def get_scenario(scenario, seed):
    if scenario < 5:
        return get_scenario_fixed_hyperpar(scenario)
    else:
        return get_scenario_varying_hyperpar(scenario, seed)


def get_max_dim_per_scenario(scenario):
    table = {0: 1, 1: 2, 2: 4, 3: 6, 4: 12, 5: 1, 6: 2, 7: 6, 8: 18, 9: 7}
    if scenario not in table:
        raise ValueError(f"Scenario {scenario} is not recognized")
    return table[scenario]


def get_scenario_fixed_hyperpar(scenario):
    sigma_squared_noise = 0.01
    sigma_noise = sigma_squared_noise ** 0.5
    sigma_output = 10 ** 0.5

    if scenario == 0:
        num_dimensions, length_scale = 1, 0.05
    elif scenario == 1:
        num_dimensions, length_scale = 2, 0.1
    elif scenario == 2:
        num_dimensions, length_scale = 4, 0.2
    elif scenario == 3:
        num_dimensions, length_scale = 6, 0.3
    elif scenario == 4:
        num_dimensions, length_scale = 12, 0.6
    else:
        raise ValueError(f"Scenario {scenario} is not a fixed-hyperpar scenario")

    gamma = 1 / (2 * length_scale ** 2)
    mean_function = np.array([0.0])
    t_mean_function = torch.from_numpy(mean_function).double()

    return num_dimensions, gamma, sigma_noise, length_scale, sigma_output, t_mean_function


def get_scenario_varying_hyperpar(scenario, seed):
    state = np.random.get_state(legacy=False)

    np.random.seed(seed)

    sigma_noise = np.random.lognormal(-4, 1, size=1)
    sigma_noise = sigma_noise[0]
    sigma_output = 1 ** 0.5  # following Carl's suggestion
    sigma_mean = 0.25 ** 0.5

    if scenario == 5:
        num_dimensions = 1
    elif scenario == 6:
        num_dimensions = 2
    elif scenario == 7:
        num_dimensions = np.random.randint(low=1, high=7)  # uniform in [1, 6]
    elif scenario == 8:
        num_dimensions = np.random.randint(low=1, high=19)  # uniform in [1, 18]
    elif scenario == 9:
        num_dimensions = np.random.randint(low=1, high=8)
    else:
        raise ValueError(f"Scenario {scenario} is not a varying-hyperpar scenario")

    mu0 = -0.75
    sigma0 = 0.75

    length_scale = np.random.lognormal(mu0 + np.log(num_dimensions) / 2, sigma0, size=(num_dimensions, 1))

    gamma = 1 / (2 * length_scale ** 2)
    mean_function = np.random.normal(0, sigma_mean, size=(1))
    t_mean_function = torch.from_numpy(mean_function).double()

    np.random.set_state(state)

    return num_dimensions, gamma, sigma_noise, length_scale, sigma_output, t_mean_function
