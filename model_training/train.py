"""Public CLI for AlphaPFN training.

Usage:
    torchrun --nproc-per-node=$N model_training/train.py \\
        --acquisition {ppd|pes|mes|jes} \\
        [--epochs 2500] [--seed 0] \\
        --output-dir <dir> \\
        [--resume]

The CLI is a thin orchestrator:

  1. Imports `alphapfn.training` (installs the sys.modules shim so the
     pickled chunks under `pfns4hpo_v2.priors.prior.Batch` resolve onto
     the alphapfn rewrite).
  2. Seeds torch + numpy.
  3. Fetches the prior corpus via `alphapfn.data.ensure_prior_corpus`
     (downloads from the artifact host on first use).
  4. Fits bar-distribution borders from the corpus's `ys.npy`.
  5. Builds the encoder factories from the v18 prior module and
     constructs a `FullSupportBarDistribution` criterion.
  6. Builds a `DistributedPriorDataLoader` over the corpus.
  7. Calls `alphapfn.training.train.train(...)`.

For direct (PES/MES/JES) heads, the published v1 PPD checkpoint is
informational only — the published direct corpora ship pre-computed
targets so PPD inference is not required during training.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Importing `alphapfn.training` installs the sys.modules shim — must
# happen before any code path that may cloudpickle.load a chunk.
from alphapfn import training  # noqa: F401  (side-effect import)

from alphapfn import data as _data_module
from alphapfn.loader import checkpoint_dir as _checkpoint_dir
from alphapfn.model.bar_distribution import FullSupportBarDistribution
from alphapfn.priors import alphapfn_model as direct_prior
from alphapfn.priors import base_model as ppd_prior
from alphapfn.training.buckets import buckets_from_ys_npy
from alphapfn.training.prior_data_loader import DistributedPriorDataLoader
from alphapfn.training.train import train as run_train


# v1 recipe constants — match the kislurm shell scripts that produced
# the published checkpoints. Override on the CLI if you want to deviate.
_RECIPE = {
    "epochs": 2500,
    "emsize": 128,
    "nlayers": 8,
    "nhead": 4,
    "nhid": 512,
    "batch_size": 100,
    "seq_len": 150,
    "subsample": 1,
    "lr": 1e-4,
    "num_borders": 5000,
    "num_features": 6,
    "features_per_group": 1,
    "n_chunks": 100_000,
    "steps_per_epoch": 100,
    "aggregate_k_gradients": 1,
    "scenario": 7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Required:
    p.add_argument(
        "--acquisition",
        choices=["ppd", "pes", "mes", "jes"],
        required=True,
        help="Which model to train. 'ppd' is the base; 'pes/mes/jes' are direct heads.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to save the final .pt and resume checkpoint (must be writable).",
    )
    # Reproducibility:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/<name>.ckpt if present.",
    )
    # Recipe overrides:
    p.add_argument("--epochs", type=int, default=_RECIPE["epochs"])
    p.add_argument("--emsize", type=int, default=_RECIPE["emsize"])
    p.add_argument("--nlayers", type=int, default=_RECIPE["nlayers"])
    p.add_argument("--nhead", type=int, default=_RECIPE["nhead"])
    p.add_argument("--nhid", type=int, default=_RECIPE["nhid"])
    p.add_argument("--batch-size", type=int, default=_RECIPE["batch_size"])
    p.add_argument("--seq-len", type=int, default=_RECIPE["seq_len"])
    p.add_argument("--subsample", type=int, default=_RECIPE["subsample"])
    p.add_argument("--lr", type=float, default=_RECIPE["lr"])
    p.add_argument("--num-borders", type=int, default=_RECIPE["num_borders"])
    p.add_argument("--num-features", type=int, default=_RECIPE["num_features"])
    p.add_argument(
        "--features-per-group", type=int, default=_RECIPE["features_per_group"]
    )
    p.add_argument("--n-chunks", type=int, default=_RECIPE["n_chunks"])
    p.add_argument("--steps-per-epoch", type=int, default=_RECIPE["steps_per_epoch"])
    p.add_argument(
        "--aggregate-k-gradients", type=int, default=_RECIPE["aggregate_k_gradients"]
    )
    p.add_argument("--scenario", type=int, default=_RECIPE["scenario"])
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Final model file name (default: '<acquisition>_seed<seed>').",
    )
    # Optional ablations:
    p.add_argument(
        "--no-full-support",
        action="store_true",
        help="Use percentile-based limited-support bar distribution instead of full_support.",
    )
    p.add_argument(
        "--linspace-borders",
        action="store_true",
        help="Use linspace(0,1) borders instead of fitting from ys.npy.",
    )
    p.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable autocast/GradScaler.",
    )
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_prior_module(acquisition: str):
    """Return the alphapfn training-augmented prior module."""
    if acquisition == "ppd":
        return ppd_prior, "es_pfn2_gp_scen_v18_ppd"
    return direct_prior, "es_pfn2_gp_scen_v18_direct"


def _maybe_warn_ppd_checkpoint(acquisition: str) -> None:
    """Direct heads originally needed a trained PPD at data-gen time.

    The published v1 corpora ship target_y already computed, so PPD is
    no longer required at training time. This print is informational
    only.
    """
    if acquisition == "ppd":
        return
    ppd_dir = _checkpoint_dir("ppd", "v1")
    if not (ppd_dir / "weights.safetensors").exists():
        print(
            f"[note] direct head '{acquisition}': v1 PPD checkpoint not found at "
            f"{ppd_dir}. Published corpora ship pre-computed targets so this is "
            f"informational; ignore unless you intend to re-generate corpora."
        )


def main() -> None:
    args = parse_args()

    _seed_everything(args.seed)
    full_support = not args.no_full_support
    use_linspace = args.linspace_borders
    name = args.name or f"{args.acquisition}_seed{args.seed}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{name}.pt"
    snapshot_dir = args.output_dir / "snapshots"
    ckpt_path = args.output_dir / f"{name}.ckpt"

    print(f"alphapfn train: acquisition={args.acquisition}  seed={args.seed}")
    print(f"output_dir: {args.output_dir}")
    print(f"final model will land at: {output_path}")

    # 1. Resolve the corpus (downloads on first use).
    print(f"Resolving prior corpus for '{args.acquisition}' (v1) ...")
    corpus_dir = _data_module.ensure_prior_corpus(args.acquisition, version="v1")
    print(f"corpus at: {corpus_dir}")

    ys_path = corpus_dir / "ys.npy"
    if not ys_path.exists():
        raise FileNotFoundError(
            f"corpus missing ys.npy at {ys_path} — required for bucket fitting."
        )

    _maybe_warn_ppd_checkpoint(args.acquisition)

    # 2. Fit bar-distribution borders.
    if use_linspace:
        from alphapfn.training.buckets import linspace_buckets

        borders = linspace_buckets(args.num_borders)
        print(f"Using linspace borders: {tuple(borders.shape)}")
    else:
        borders = buckets_from_ys_npy(
            ys_path, num_borders=args.num_borders, full_support=full_support
        )
        print(
            f"Fitted borders from {ys_path.name}: shape={tuple(borders.shape)} "
            f"full_support={full_support}"
        )

    criterion = FullSupportBarDistribution(borders=borders)

    # 3. Encoder factories from the v18 prior module.
    prior_module, prior_name = _select_prior_module(args.acquisition)

    encoder_generator = prior_module.get_encoder()
    if args.acquisition == "ppd":
        y_encoder_generator = prior_module.get_y_encoder(1, args.scenario)
        style_encoder_generator = prior_module.get_style_encoder()
        y_style_encoder_generator = prior_module.get_y_style_encoder(
            1, args.scenario
        )
    else:
        y_encoder_generator = prior_module.get_y_encoder_by_scenario(args.scenario)(1)
        style_encoder_generator = None
        y_style_encoder_generator = None

    # 4. Data loader over the corpus.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(
        f"Building DistributedPriorDataLoader "
        f"(world_size={world_size}, n_chunks={args.n_chunks}) ..."
    )
    dl_factory = DistributedPriorDataLoader(
        str(corpus_dir),
        n_gpus=world_size,
        n_chunks=args.n_chunks,
        subsample=args.subsample,
    )

    def get_batch_method(*method_args, **method_kwargs):
        device = method_kwargs.get("device", "cpu")
        return dl_factory.get_batch(device)

    # 5. Resume?
    train_state_dict_load_path = None
    if args.resume and ckpt_path.exists():
        train_state_dict_load_path = str(ckpt_path)
        print(f"will resume from {ckpt_path}")

    # 6. Build prior_hyperparameters for provenance (passed through to
    # the inner training loop but not actually consumed in the offline
    # path).
    prior_hps = {
        "scenario": args.scenario,
        "standardize": False,
        "extra_scen7_data": True,
    }
    if args.acquisition != "ppd":
        prior_hps["af"] = args.acquisition.upper()
        prior_hps["base_model_class"] = "es_pfn2_gp_scen_v18_ppd"

    # 7. eval-pos sampler — uniform[1, 50] for both PPD and direct,
    # matching the kislurm corpus-generation recipe.
    from alphapfn.training.prior_data_loader import get_uniform_sampler

    single_eval_pos_gen = get_uniform_sampler(1, 50)

    # 8. Call the training loop.
    result, stats = run_train(
        get_batch_method=get_batch_method,
        criterion=criterion,
        encoder_generator=encoder_generator,
        y_encoder_generator=y_encoder_generator,
        style_encoder_generator=style_encoder_generator,
        y_style_encoder_generator=y_style_encoder_generator,
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        nhead=args.nhead,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        single_eval_pos_gen=single_eval_pos_gen,
        features_per_group=args.features_per_group,
        train_mixed_precision=not args.no_mixed_precision,
        aggregate_k_gradients=args.aggregate_k_gradients,
        extra_prior_kwargs_dict={"hyperparameters": prior_hps},
        train_state_dict_save_path=str(ckpt_path),
        train_state_dict_load_path=train_state_dict_load_path,
        snapshot_dir=str(snapshot_dir),
        output_path=str(output_path),
        final_model_name=name,
    )

    if result is not None:
        print(f"\nTraining complete. final loss={result.total_loss:.4f}")
        print(f"total time: {stats['total_time']:.1f}s")
        print(f"output: {output_path}")


if __name__ == "__main__":
    main()
