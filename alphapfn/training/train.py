"""Training loop for AlphaPFN, ported from `pfns4hpo_v2/train.py`.

Surface: `train(...)` is the long parameter-by-name function the
kislurm pipeline used. `model_training/train.py` is the public CLI that
calls it.

What changed during the port (vs the kislurm source):

  - All `from . / from pfns4hpo_v2.*` imports rewritten to `alphapfn.*`.
  - Default `dataloader_class` switched to
    `alphapfn.training.prior_data_loader.StandardDataLoader`.
  - The two hardcoded `/work/dlclarge2/...` paths (periodic snapshot
    and final save) replaced with config-driven `snapshot_dir` /
    `output_path`. Both default to None, in which case the periodic
    snapshot save is skipped and only the final `.pt` (named by
    `final_model_name`) is written via `train_state_dict_save_path`.
  - Dead code at the kislurm `try: ... initialize_with_model ...`
    block removed (referenced an undefined name and silently `pass`ed).
  - The `wandb_configs={}` parameter dropped — `train.py` itself
    never imports or calls `wandb` in the original; only `main.py`
    set the dict (and that's gone with submitit).
  - The `final_model_name` parameter is kept and now applies as the
    basename for the periodic snapshots under `snapshot_dir` and as
    the file name for the final save under `output_path`.

What is unchanged: the actual gradient steps, AMP/scaler handling,
DDP wrapping, scheduler, importance-sampling-infos tracking, and the
batch-shape pipeline (`einops.rearrange`).
"""
from __future__ import annotations

import itertools
import os
import time
import typing as tp
from contextlib import nullcontext

import einops
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from alphapfn import utils
from alphapfn.model import transformer
from alphapfn.model.bar_distribution import BarDistribution
from alphapfn.priors import prior
from alphapfn.training.prior_data_loader import StandardDataLoader
from alphapfn.utils import get_cosine_schedule_with_warmup, init_dist


class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = lambda num_classes: nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(num_classes)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution


class TrainingResult(tp.NamedTuple):
    # the mean loss in the last epoch across dataset sizes
    total_loss: tp.Optional[float]
    # the mean loss in the last epoch for each dataset size
    total_positional_losses: tp.Optional[tp.List[float]]
    # the trained model
    model: nn.Module
    # the dataloader used for training
    data_loader: tp.Optional[torch.utils.data.DataLoader]


def train(
    get_batch_method,
    criterion,
    encoder_generator=None,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    seq_len=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    train_state_dict_save_path=None,
    train_state_dict_load_path=None,
    y_encoder_generator=None,
    decoder_dict=None,
    extra_prior_kwargs_dict=None,
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_eval_pos_gen=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    style_encoder_generator=None,
    y_style_encoder_generator=None,
    epoch_callback=None,
    step_callback=None,
    continue_model=None,
    features_per_group=-1,
    train_mixed_precision=True,
    progress_bar=False,
    n_targets_per_input=1,
    dataloader_class=StandardDataLoader,
    num_workers=None,
    final_model_name=None,
    output_path=None,
    snapshot_dir=None,
    snapshot_every=100,
    **model_extra_args,
):
    """Run the training loop and return `(TrainingResult, stats)`.

    Output handling:
      - `train_state_dict_save_path`: per-epoch checkpoint dict
        (`{"model_state_dict", "optimizer_state_dict", "epoch"}`).
      - `snapshot_dir` (optional): if set, every `snapshot_every`
        epochs save `{snapshot_dir}/{final_model_name}_{epoch}.pt`
        with the raw `nn.Module` (cpu-mapped) for offline inspection.
      - `output_path` (optional): if set, also save the final model
        there at the end of training (raw `nn.Module`, cpu).
    """
    if decoder_dict is None:
        decoder_dict = {}
    if extra_prior_kwargs_dict is None:
        extra_prior_kwargs_dict = {}

    total_start_time = time.time()
    device: str = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, seq_len

    if num_workers is not None:
        extra_prior_kwargs_dict["num_workers"] = num_workers

    dl = dataloader_class(
        get_batch_method=get_batch_method,
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=seq_len,
        device=device,
        n_targets_per_input=n_targets_per_input,
        **extra_prior_kwargs_dict,
    )

    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution)
        or "BarDistribution" in criterion.__class__.__name__
    ):
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    if continue_model:
        model = continue_model
    else:
        decoder_dict = decoder_dict if decoder_dict else {"standard": (None, n_out)}

        assert (
            features_per_group > 0 or features_per_group == -1
        ), "features_per_group must be > 0 or -1"
        architectural_features_per_group = (
            dl.num_features if features_per_group == -1 else features_per_group
        )
        print(
            f"Using {architectural_features_per_group} architectural features per group"
        )

        encoder = (
            encoder_generator(architectural_features_per_group, emsize)
            if encoder_generator
            else None
        )
        y_encoder = y_encoder_generator(1, emsize) if y_encoder_generator else None
        model = transformer.PerFeatureTransformer(
            encoder=encoder,
            y_encoder=y_encoder,
            features_per_group=architectural_features_per_group,
            decoder_dict=decoder_dict,
            ninp=emsize,
            nhid=nhid,
            nlayers=nlayers,
            nhead=nhead,
            attention_between_features=features_per_group != -1,
            style_encoder=(
                style_encoder_generator(architectural_features_per_group, emsize)
                if style_encoder_generator is not None
                else None
            ),
            y_style_encoder=(
                y_style_encoder_generator(emsize)
                if y_style_encoder_generator is not None
                else None
            ),
            **model_extra_args,
        )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )

    start_epoch = 1

    if train_state_dict_load_path:
        print(f"Loading checkpoint from {train_state_dict_load_path}")
        checkpoint = torch.load(train_state_dict_load_path, map_location=device)
        if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict'. Got keys: "
                f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
        # Fast-forward the scheduler to the resumed epoch.
        for _ in range(start_epoch - 1):
            scheduler.step()

    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )
        # Use the local module from the data loader — DDP forwarding
        # would not work with our get_batch flow.
        dl.model = model.module
    else:
        dl.model = model

    scaler = GradScaler() if train_mixed_precision else None

    utils.check_compatibility(dl)

    def train_epoch(epoch):
        model.train()
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        tqdm_iter = (
            tqdm(range(len(dl)), desc="Training Epoch")
            if rank == 0 and progress_bar
            else None
        )
        importance_sampling_infos = []

        for batch, full_data in enumerate(dl):
            targets = full_data.target_y.to(device)
            single_eval_pos = full_data.single_eval_pos

            def get_metrics():
                return (
                    total_loss / steps_per_epoch,
                    (
                        total_positional_losses / total_positional_losses_recorded
                    ).tolist(),
                    time_to_get_batch,
                    forward_time,
                    step_time,
                    nan_steps.cpu().item() / (batch + 1),
                    ignore_steps.cpu().item() / (batch + 1),
                )

            tqdm_iter.update() if tqdm_iter is not None else None
            if using_dist and not (
                batch % aggregate_k_gradients == aggregate_k_gradients - 1
            ):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                metrics_to_log = {}
                with autocast(device.split(":")[0], enabled=scaler is not None):
                    style = (
                        full_data.style.to(device)
                        if full_data.style is not None
                        else None
                    )
                    if style is not None:
                        if style.dim() == 2:
                            broken = style.shape[0] != full_data.x.shape[1]
                        elif style.dim() == 3:
                            broken = (
                                style.shape[0] != full_data.x.shape[1]
                                or style.shape[1] != full_data.x.shape[2]
                            )
                        else:
                            raise ValueError(
                                f"style must have 2 or 3 dimensions, got {style.shape}"
                            )
                        if broken:
                            raise ValueError(
                                f"style must have the same batch size as x and, if 3D, "
                                f"the middle dim must match the feature count. "
                                f"style={style.shape} x={full_data.x.shape}"
                            )
                    y_style = (
                        full_data.y_style.to(device)
                        if full_data.y_style is not None
                        else None
                    )
                    if y_style is not None:
                        if y_style.dim() != 2:
                            raise ValueError(
                                f"y_style must have 2 dimensions, got {y_style.shape}"
                            )
                        if y_style.shape[0] != full_data.y.shape[1]:
                            raise ValueError(
                                f"y_style must have the same batch size as y; "
                                f"y_style={y_style.shape} y={full_data.y.shape}"
                            )
                    x = full_data.x.to(device)
                    y = full_data.y.to(device)
                    out = model(
                        x,
                        y[:single_eval_pos],
                        style=style,
                        y_style=y_style,
                        only_return_standard_out=False,
                    )

                    # Older models returned (out, output_once); newer return out only.
                    out, output_once = out if isinstance(out, tuple) else (out, None)
                    output = out["standard"] if isinstance(out, dict) else out

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]

                    output = output.unsqueeze(2).expand(
                        *output.shape[:2], n_targets_per_input, output.shape[-1]
                    )

                    assert targets.shape == output.shape[:-1], (
                        f"Target shape {targets.shape} does not match output shape "
                        f"{output.shape}. The trailing 1-dim might be missing in the target."
                    )
                    output = einops.rearrange(output, "s b t l -> s (b t) l")
                    targets = einops.rearrange(targets, "s b t -> s (b t)")

                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert (
                            output.shape[-1] == 2
                        ), "Multi-target Gaussian NLL not implemented."
                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(
                            mean_pred.flatten(),
                            targets.flatten(),
                            var=var_pred.flatten(),
                        )
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        targets[torch.isnan(targets)] = -100
                        losses = criterion(output.flatten(), targets.flatten())
                        losses = losses.view(*targets.shape)
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        targets[torch.isnan(targets)] = -100
                        losses = criterion(
                            output.reshape(-1, n_out), targets.long().flatten()
                        )
                    else:
                        losses = criterion(output, targets.unsqueeze(-1))
                    losses = einops.rearrange(
                        losses, "s (b t) -> s b t", t=n_targets_per_input
                    )
                    losses = losses.mean(2)
                    loss, nan_share = utils.torch_nanmean(
                        losses.mean(0), return_nanshare=True
                    )
                    loss_scaled = loss / aggregate_k_gradients

                if scaler:
                    loss_scaled = scaler.scale(loss_scaled)
                loss_scaled.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        scaler.unscale_(optimizer)

                    squared_grad_magnitudes = {
                        name: (w.grad ** 2).sum().cpu().item()
                        for name, w in model.named_parameters()
                        if w.grad is not None
                    }
                    total_grad_magnitude = sum(squared_grad_magnitudes.values())

                    normalized_squared_grad_magnitudes = {}
                    total_normalized_grad_magnitude = 0.0
                    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                        beta2 = optimizer.param_groups[0]["betas"][1]
                        for name, param in model.named_parameters():
                            if param.grad is None:
                                continue
                            state = optimizer.state.get(param, {})
                            if "exp_avg_sq" in state:
                                normalized_grad_magnitude = (
                                    (
                                        (param.grad ** 2)
                                        / (
                                            state["exp_avg_sq"]
                                            * (1 - beta2 ** state.get("step", 1))
                                            + 1e-8
                                        )
                                    )
                                    .sum()
                                    .cpu()
                                    .item()
                                )
                                normalized_squared_grad_magnitudes[name] = (
                                    normalized_grad_magnitude
                                )
                        total_normalized_grad_magnitude = sum(
                            normalized_squared_grad_magnitudes.values()
                        )

                    importance_sampling_infos.append(
                        (
                            total_grad_magnitude,
                            full_data.info_used_with_gradient_magnitudes,
                            loss.cpu().item(),
                            total_normalized_grad_magnitude,
                        )
                    )
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    if full_data.gradient_multipliers is not None:
                        assert (
                            aggregate_k_gradients == 1
                        ), "Gradient scaling is only supported when aggregate_k_gradients == 1."
                        assert all(
                            full_data.gradient_multipliers.view(-1)[0]
                            == full_data.gradient_multipliers.view(-1)[i]
                            for i in range(full_data.gradient_multipliers.numel())
                        ), (
                            "We only support a single common scaler — interactions with "
                            "gradient clipping for per-element scalers are untested."
                        )
                        with torch.no_grad():
                            for w in model.parameters():
                                w.grad = (
                                    w.grad
                                    * full_data.gradient_multipliers.view(-1)[0]
                                )

                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += loss.cpu().detach().item()
                    total_positional_losses += (
                        losses.mean(1).cpu().detach()
                        if single_eval_pos is None
                        else nn.functional.one_hot(
                            torch.tensor(single_eval_pos), seq_len
                        )
                        * utils.torch_nanmean(
                            losses[: seq_len - single_eval_pos].mean(0)
                        )
                        .cpu()
                        .detach()
                    )

                    total_positional_losses_recorded += (
                        torch.ones(seq_len)
                        if single_eval_pos is None
                        else nn.functional.one_hot(
                            torch.tensor(single_eval_pos), seq_len
                        )
                    )

                    metrics_to_log = {
                        **metrics_to_log,
                        "loss": loss,
                        "single_eval_pos": single_eval_pos,
                    }
                    if step_callback is not None and rank == 0:
                        step_callback(metrics_to_log)
                    nan_steps += nan_share
                    ignore_steps += (targets == -100).float().mean()
                else:
                    print("Invalid step encountered (NaN loss), skipping...")

            if tqdm_iter:
                tqdm_iter.set_postfix(
                    {
                        "data_time": time_to_get_batch,
                        "step_time": step_time,
                        "mean_loss": total_loss / (batch + 1),
                    }
                )

            before_get_batch = time.time()
        return get_metrics(), importance_sampling_infos

    total_loss = float("inf")
    total_positional_losses = float("inf")
    importance_sampling_infos = None
    try:
        if epoch_callback is not None and rank == 0 and start_epoch == 1:
            epoch_callback(
                model, 0, loss=total_loss, data_loader=dl, scheduler=scheduler
            )

        epoch_iterator = (
            range(start_epoch, epochs + 1)
            if epochs is not None
            else itertools.count(start_epoch)
        )

        for epoch in epoch_iterator:
            epoch_start_time = time.time()
            (
                total_loss,
                total_positional_losses,
                time_to_get_batch,
                forward_time,
                step_time,
                nan_share,
                ignore_share,
            ), importance_sampling_infos = train_epoch(epoch)
            dl.importance_sampling_infos = importance_sampling_infos

            if hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} "
                    f"| time: {(time.time() - epoch_start_time):5.2f}s "
                    f"| mean loss {total_loss:5.2f} "
                    f"| pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])} "
                    f"| lr {scheduler.get_last_lr()[0]} "
                    f"| data {time_to_get_batch:5.2f} "
                    f"| step {step_time:5.2f} "
                    f"| forward {forward_time:5.2f} "
                    f"| nan {nan_share:5.2f} "
                    f"| ignore {ignore_share:5.4f}"
                    + (f" | val {val_score}" if val_score is not None else "")
                )
                print("-" * 89)

            if epoch_callback is not None and rank == 0:
                epoch_callback(
                    model, epoch, loss=total_loss, data_loader=dl, scheduler=scheduler
                )
            scheduler.step()

            # Per-epoch resume-checkpoint save (only on rank 0).
            if train_state_dict_save_path is not None and rank == 0:
                save_model = (
                    model.module
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    else model
                )
                print(
                    f"Saving checkpoint to {train_state_dict_save_path} (epoch {epoch})"
                )
                os.makedirs(
                    os.path.dirname(train_state_dict_save_path) or ".", exist_ok=True
                )
                try:
                    checkpoint = {
                        "model_state_dict": save_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(checkpoint, train_state_dict_save_path)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")

                # Optional periodic snapshot of the raw model — used
                # for offline inspection. Replaces the hardcoded
                # /work/dlclarge2/... path in the kislurm original.
                if (
                    snapshot_dir is not None
                    and final_model_name is not None
                    and epoch % snapshot_every == 0
                ):
                    os.makedirs(snapshot_dir, exist_ok=True)
                    snapshot_path = os.path.join(
                        snapshot_dir, f"{final_model_name}_{epoch}.pt"
                    )
                    try:
                        torch.save(save_model.to("cpu"), snapshot_path)
                        save_model.to(device)
                    except Exception as e:
                        print(f"Error saving snapshot: {e}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    if rank == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        # Final save under output_path (replaces the kislurm hardcoded
        # /work/dlclarge2/.../final_models/<name>.pt).
        if output_path is not None:
            model.eval()
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(model.to("cpu"), output_path)
        elif final_model_name is not None and snapshot_dir is not None:
            # Backward-compat: if only snapshot_dir is set, save the final
            # model there under <name>.pt.
            model.eval()
            torch.save(
                model.to("cpu"),
                os.path.join(snapshot_dir, f"{final_model_name}.pt"),
            )
        return TrainingResult(
            total_loss,
            total_positional_losses,
            model.to("cpu"),
            dl,
        ), {"total_time": time.time() - total_start_time}

    return None, {"total_time": time.time() - total_start_time}
