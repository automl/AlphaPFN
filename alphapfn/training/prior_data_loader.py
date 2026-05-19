"""Training-time prior data loaders.

Ported from `pfns4hpo_v2/priors/utils.py` on kislurm. Datagen-only
machinery (`store_prior`, `store_prior_map_array`,
`store_prior_map_array_nemo`, plot helpers) has been removed; those
required submitit/matplotlib and are out of scope for alpha-pfn
training.

This module provides:
  - `Batch` (re-exported from `alphapfn.priors.prior`).
  - `PriorDataLoader` — single-process loader that streams pickled
    chunks from a corpus on disk.
  - `DistributedPriorDataLoader` — rank-aware version that shards the
    chunk space across ranks (reads `LOCAL_RANK` / `SLURM_PROCID`).
  - `get_uniform_sampler` / `get_expon_sep_sampler` — eval-pos samplers
    used by the priors at data-generation time. Kept so that any code
    still references them by import path.
  - `get_batch_to_dataloader`, `get_batch_sequence`, `StandardDataLoader`
    — utilities for live (online) priors. Live priors are not used in
    the published recipes; kept for completeness.
  - Several `nn.Module` activation classes used by live priors.

When training from the published corpora the only entry points exercised
are `PriorDataLoader.__init__` + `get_batch` + `get_single_eval_pos`
(or the `DistributedPriorDataLoader` equivalents). Everything else is
preserved as a faithful port for downstream code that may import it.
"""
from __future__ import annotations

import inspect
import math
import os
import random
import time
import types
from functools import partial

import cloudpickle
import numpy as np
import scipy.stats as stats
import torch
from torch import nn

from alphapfn.priors.prior import Batch, PriorDataLoader as _BasePriorDataLoader  # noqa: F401
from alphapfn.utils import set_locals_in_self, normalize_data


# Re-export `Batch` so this module can serve as the alias target for
# `pfns4hpo_v2.priors.utils` (where the source pickles' Batch class
# is recorded). See `alphapfn/training/__init__.py` for the shim.
__all__ = [
    "Batch",
    "PriorDataLoader",
    "DistributedPriorDataLoader",
    "get_rank",
    "get_uniform_sampler",
    "get_expon_sep_sampler",
    "get_batch_to_dataloader",
    "get_batch_sequence",
    "StandardDataLoader",
]


def get_uniform_sampler(min_eval_pos, seq_len):
    """eval-pos sampler used by both PPD and direct v18 priors.

    Returns a 0-arg callable sampling integers from
    [min_eval_pos, seq_len) (upper bound exclusive — matches the
    original `np.random.randint` convention).
    """
    def foo():
        return np.random.randint(min_eval_pos, seq_len)
    return foo


def get_expon_sep_sampler(base, min_eval_pos, seq_len):
    """Geometric (exponential-decay) eval-pos sampler.

    Faithful port of the original: builds a discrete distribution over
    `seq_len - min_eval_pos` integer offsets with probability
    proportional to `base ** i`, then samples and shifts by
    `min_eval_pos`.
    """
    p_levels = np.array([np.power(base, i) for i in range(seq_len - min_eval_pos)])
    p_levels /= p_levels.sum()

    def foo():
        return np.random.choice(seq_len - min_eval_pos, p=p_levels) + min_eval_pos
    return foo


class PriorDataLoader:
    """Stream pickled training batches from a partitioned corpus on disk.

    Layout expected:
        <load_path>/
            partition_0/chunk_0.pkl, chunk_1.pkl, ...
            partition_1/chunk_1000.pkl, ...
            ...
    Each `chunk_*.pkl` is a list of `(single_eval_pos, Batch)` tuples
    (typically 10 tuples per chunk, each Batch packs `batch_size=100`).

    The unpartitioned layout (`<load_path>/chunk_*.pkl`) is detected
    automatically if `partition_0/` is absent.
    """

    def _load_chunk(self, chunk_id):
        if self.partition:
            partition_id = chunk_id // 1000
            chunk_file = os.path.join(self.path, f"partition_{partition_id}", f"chunk_{chunk_id}.pkl")
        else:
            chunk_file = os.path.join(self.path, f"chunk_{chunk_id}.pkl")
        with open(chunk_file, "rb") as f:
            self.loaded_chunk = cloudpickle.load(f)
        self.loaded_chunk_id = chunk_id
        self.batch_counter = 0
        self.subsample_counter = 0

    def __init__(self, load_path, n_chunks=2_000, store=False, subsample=1, partition=None):
        self.path = load_path
        if store:
            self.partition = True  # legacy datagen flag; alpha-pfn never stores
        elif partition is None:
            self.partition = os.path.isdir(os.path.join(self.path, "partition_0"))
        else:
            self.partition = partition
        if not store:
            self._load_chunk(0)
        self.n_chunks = n_chunks
        self.subsample = subsample

    def get_batch(self, device):
        if self.subsample == 1:
            _, batch_data = self.loaded_chunk[self.batch_counter]
            batch_data.x = batch_data.x.to(device)
            batch_data.y = batch_data.y.to(device)
            batch_data.target_y = batch_data.target_y.to(device)
            if batch_data.style is not None:
                batch_data.style = batch_data.style.to(device)
            if batch_data.y_style is not None:
                batch_data.y_style = batch_data.y_style.to(device)
            self.batch_counter += 1
            if self.batch_counter >= len(self.loaded_chunk):
                self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)
            return batch_data
        else:
            raise NotImplementedError(
                "subsampling > 1 was experimental in the original loader "
                "and is not supported in alphapfn."
            )

    def get_single_eval_pos(self):
        single_eval_pos, _ = self.loaded_chunk[self.batch_counter]
        if single_eval_pos == 1000:
            # legacy hack from the original loader; documented as a TEMP correction
            single_eval_pos = 999
        return single_eval_pos


def get_rank():
    """Resolve the process rank for distributed-data sharding.

    Reads `LOCAL_RANK` (torchrun / torch.distributed.launch) first,
    then `SLURM_PROCID` (submitit-managed multi-task), and falls back
    to 0 for single-process runs.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    return 0


class DistributedPriorDataLoader(PriorDataLoader):
    """Rank-aware PriorDataLoader.

    Each rank starts reading chunks at an offset of
    `rank * n_chunks // n_gpus`, ensuring different ranks see disjoint
    portions of the corpus. The first chunk is loaded lazily on the
    first `get_batch` call (so that `init_process_group` has set
    `LOCAL_RANK` before we read it).
    """

    def __init__(self, load_path, n_gpus=1, n_chunks=2_000, store=False, subsample=1, partition=None):
        self.path = load_path
        if store:
            self.partition = True
        elif partition is None:
            self.partition = os.path.isdir(os.path.join(self.path, "partition_0"))
        else:
            self.partition = partition
        if not store:
            self.n_gpus = n_gpus
            self.loaded_chunk = None  # lazy
        self.n_chunks = n_chunks
        self.subsample = subsample
        self.rank = None

    def data_sync(self):
        # Lazy load on first call from the current rank.
        if self.loaded_chunk is None or self.rank != get_rank():
            self.rank = get_rank()
            offset = self.rank * self.n_chunks // self.n_gpus
            self._load_chunk(offset)

    def get_batch(self, device):
        self.data_sync()
        if self.subsample != 1:
            raise NotImplementedError(
                "subsampling > 1 was experimental and is not supported in alphapfn."
            )
        _, batch_data = self.loaded_chunk[self.batch_counter]
        batch_data.x = batch_data.x.to(device)
        batch_data.y = batch_data.y.to(device)
        batch_data.target_y = batch_data.target_y.to(device)
        if batch_data.style is not None:
            batch_data.style = batch_data.style.to(device)
        if batch_data.y_style is not None:
            batch_data.y_style = batch_data.y_style.to(device)
        self.batch_counter += 1
        if self.batch_counter >= len(self.loaded_chunk):
            self._load_chunk((self.loaded_chunk_id + 1) % self.n_chunks)
        return batch_data

    def get_single_eval_pos(self):
        self.data_sync()
        single_eval_pos, _ = self.loaded_chunk[self.batch_counter]
        if single_eval_pos == 1000:
            single_eval_pos = 999
        return single_eval_pos


# ---------------------------------------------------------------------------
# Below: live-prior utilities. Not exercised by the published recipes
# (offline corpus path is the only one we use), but kept for
# compatibility with code that may import them.
# ---------------------------------------------------------------------------


def get_batch_to_dataloader(get_batch_method_):
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        def __init__(self, num_steps, **get_batch_kwargs):
            set_locals_in_self(locals())
            self.num_features = get_batch_kwargs.get("num_features") or self.num_features
            self.epoch_count = 0

        @staticmethod
        def gbm(*args, eval_pos_seq_len_sampler, **kwargs):
            kwargs["single_eval_pos"], kwargs["seq_len"] = eval_pos_seq_len_sampler()
            if kwargs.get("dynamic_batch_size"):
                kwargs["batch_size"] = kwargs["batch_size"] * math.floor(
                    math.pow(kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"])
                    / math.pow(kwargs["seq_len"], kwargs["dynamic_batch_size"])
                )
            batch: Batch = get_batch_method_(*args, **kwargs)
            if batch.single_eval_pos is None:
                batch.single_eval_pos = kwargs["single_eval_pos"]
            return batch

        def __len__(self):
            return self.num_steps

        def get_test_batch(self, **kwargs):
            return self.gbm(
                **self.get_batch_kwargs,
                epoch=self.epoch_count,
                model=self.model if hasattr(self, "model") else None,
                **kwargs,
            )

        def __iter__(self):
            assert hasattr(self, "model"), "Please assign model with `dl.model = ...` before training."
            self.epoch_count += 1
            return iter(
                self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1, model=self.model)
                for _ in range(self.num_steps)
            )

    return DL


trunc_norm_sampler_f = lambda mu, sigma: lambda: stats.truncnorm(
    (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
).rvs(1)[0]
beta_sampler_f = lambda a, b: lambda: np.random.beta(a, b)
gamma_sampler_f = lambda a, b: lambda: np.random.gamma(a, b)
uniform_sampler_f = lambda a, b: lambda: np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name="bounded_zipf", values=(x, weights)).rvs(1)


scaled_beta_sampler_f = lambda a, b, scale, minimum: lambda: minimum + round(beta_sampler_f(a, b)() * (scale - minimum))


def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)
    x = x[order]
    y = y[order]
    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


@torch.no_grad()
def sample_num_feaetures_get_batch(batch_size, seq_len, num_features, hyperparameters, get_batch, **kwargs):
    if hyperparameters.get("sample_num_features", True) and kwargs["epoch"] > 0:
        num_features = random.randint(3, num_features)
    return get_batch(batch_size, seq_len, num_features, hyperparameters=hyperparameters, **kwargs)


class CategoricalActivation(nn.Module):
    def __init__(
        self,
        categorical_p=0.1,
        ordered_p=0.7,
        keep_activation_size=False,
        num_classes_sampler=zipf_sampler_f(0.8, 1, 10),
    ):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler
        super().__init__()

    def forward(self, x):
        # Apply x on output locally
        x_ = x.clone()
        sampled_classes = self.num_classes_sampler()
        for b in range(x.shape[1]):
            for f in range(x.shape[2]):
                if random.random() < self.categorical_p:
                    if random.random() < self.ordered_p:
                        x_[:, b, f] = (x[:, b, f] - x[:, b, f].min()) / (x[:, b, f].max() - x[:, b, f].min() + 1e-9)
                        x_[:, b, f] = (x_[:, b, f] * sampled_classes).floor()
                    else:
                        x_[:, b, f] = randomize_classes(x[:, b, f].unsqueeze(-1), sampled_classes)
                else:
                    x_[:, b, f] = x[:, b, f]

        x = x_
        if self.keep_activation_size:
            x = x * x.std(0, keepdim=True)

        return x


class QuantizationActivation(torch.nn.Module):
    def __init__(self, n_thresholds, reorder_p=0.5) -> None:
        super().__init__()
        self.n_thresholds = n_thresholds
        self.reorder_p = reorder_p
        self.thresholds = torch.nn.Parameter(torch.randn(self.n_thresholds))

    def forward(self, x):
        x = normalize_data(x).unsqueeze(-1)
        x = (x > self.thresholds).sum(-1)
        if random.random() < self.reorder_p:
            x = randomize_classes(x.unsqueeze(-1), self.n_thresholds)
        x = x / self.n_thresholds
        return x


class NormalizationActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = normalize_data(x)
        return x


class PowerActivation(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.exp = torch.nn.Parameter(0.5 * torch.ones(1))
        self.shared_exp_strength = 0.5

    def forward(self, x):
        shared_exp = torch.randn(1)
        exp = torch.nn.Parameter(
            (shared_exp * self.shared_exp_strength + shared_exp * torch.randn(x.shape[-1]) * (1 - self.shared_exp_strength)) * 2 + 0.5
        ).to(x.device)
        x_ = torch.pow(torch.nn.functional.softplus(x) + 0.001, exp)
        return x_


def lambda_time(f, name="", enabled=True):
    if not enabled:
        return f()
    start = time.time()
    r = f()
    print("Timing", name, time.time() - start)
    return r


def pretty_get_batch(get_batch):
    if isinstance(get_batch, types.FunctionType):
        return f"<{get_batch.__module__}.{get_batch.__name__} {inspect.signature(get_batch)}"
    else:
        return repr(get_batch)


class get_batch_sequence(list):
    """Chain a series of `get_batch` priors so that each wraps the previous."""

    def __init__(self, *get_batch_methods):
        if len(get_batch_methods) == 0:
            raise ValueError("Must have at least one get_batch method")
        super().__init__(get_batch_methods)

    def __repr__(self):
        s = ",\n\t".join([f"{pretty_get_batch(get_batch)}" for get_batch in self])
        return f"get_batch_sequence(\n\t{s}\n)"

    def __call__(self, *args, **kwargs):
        final_get_batch = self[0]
        for get_batch in self[1:]:
            final_get_batch = partial(get_batch, get_batch=final_get_batch)
        return final_get_batch(*args, **kwargs)


class StandardDataLoader(PriorDataLoader):
    """Live-prior DataLoader used when training without precomputed chunks.

    Not exercised by the published alpha-pfn recipes (those always
    pass `--load_path` and stream from disk), but kept for users who
    want to retrain with a live prior.
    """

    def __init__(self, get_batch_method, num_steps, **get_batch_kwargs):
        set_locals_in_self(locals())
        self.num_features = get_batch_kwargs.get("num_features") or self.num_features
        self.epoch_count = 0
        self.grad_magnitues_and_infos = None

    def gbm(self, *args, eval_pos_seq_len_sampler, **kwargs):
        kwargs["single_eval_pos"], kwargs["seq_len"] = eval_pos_seq_len_sampler()
        if kwargs.get("dynamic_batch_size"):
            kwargs["batch_size"] = kwargs["batch_size"] * math.floor(
                math.pow(kwargs["seq_len_maximum"], kwargs["dynamic_batch_size"])
                / math.pow(kwargs["seq_len"], kwargs["dynamic_batch_size"])
            )
        batch: Batch = self.get_batch_method(*args, **kwargs)
        if batch.single_eval_pos is None:
            batch.single_eval_pos = kwargs["single_eval_pos"]
        return batch

    def __len__(self):
        return self.num_steps

    def get_test_batch(self, **kwargs):
        return self.gbm(
            **self.get_batch_kwargs,
            epoch=self.epoch_count,
            model=self.model if hasattr(self, "model") else None,
            **kwargs,
        )

    def __iter__(self):
        assert hasattr(self, "model"), "Please assign model with `dl.model = ...` before training."
        self.epoch_count += 1
        return iter(
            self.gbm(**self.get_batch_kwargs, epoch=self.epoch_count - 1, model=self.model)
            for _ in range(self.num_steps)
        )
