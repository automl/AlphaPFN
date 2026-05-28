"""Plot all the figures from the paper.

Usage:
    python plot_from_csv.py
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from tueplots import bundles, figsizes

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR
COMBINED_CSV = SCRIPT_DIR / "all_results.csv.gz"

ACQ_FUNCTIONS_MAIN: tuple[str, ...] = ("JES", "MES", "PES")
ACQ_FUNCTIONS_FULL: tuple[str, ...] = ("JES", "MES", "PES", "EI")
ACQ_FUNCTIONS_CLUSTERING: tuple[str, ...] = ("JES", "MES", "PES")

ACQ_COLORS: dict[str, str] = {
    "JES": "#4040f8",  
    "MES": "#cf4040",  
    "PES": "#409d40",  
    "EI":  "#7f7f7f",
}

METHOD_STYLE: dict[str, dict] = {
    "PFN":         {"linestyle": "-",  "linewidth": 1.1, "z": 5, "label": r"$\alpha$-PFN"},
    "GP":          {"linestyle": ":",  "linewidth": 1.2, "z": 0, "label": "GP"},
    "PFN_uniform": {"linestyle": ":",  "linewidth": 1.3, "z": 0, "label": "Uniform"},
}
FLOOR = 1e-20

# Per-dataset config:
#   kind = "per_benchmark_curves"     -> plot one curve per (acq, method) per benchmark
#                                        (mean ± SEM over seeds, optional log10 transform)
#   kind = "per_search_space_ranking" -> ranks computed per-seed within each dataset,
#                                        averaged across (datasets × seeds), one panel per search space
DATASETS: dict[str, dict] = {
    "synthetic": {
        "kind": "per_benchmark_curves",
        "required_iters": 100,
        "transform": "log10",
        "winrate_direction": "lower",  # PFN wins if it has lower regret
        "y_label": r"$\log_{10}$ Inference Regret",
        "title_suffix": "",
        "y_limits": {},
        "benchmarks": [
            ("Branin2D_noise0.316",   "Branin (2D)"),
            ("Hartmann4D_noise0.316", "Hartmann (4D)"),
            ("Ackley5D_noise0.316",   "Ackley (5D)"),
            ("Hartmann6D_noise0.316", "Hartmann (6D)"),
            ("Ackley8D_noise0.316",   "Ackley (8D)"),
        ],
    },
    "lcbench": {
        "kind": "per_benchmark_curves",
        "required_iters": 100,
        "transform": "raw",
        "winrate_direction": "higher",  # PFN wins if it has higher accuracy
        "y_label": "Accuracy",
        "title_suffix": " (7D)",
        "y_limits": {
            "car_noise0":          (80,   None),
            "Fashion-MNIST_noise0":(83.5, None),
            "MiniBooNE_noise0":    (84.6, None),
            "higgs_noise0":        (63.5, None),
            "segment_noise0":      (81,   84.5),
        },
        "benchmarks": [
            ("car_noise0",           "Car"),
            ("Fashion-MNIST_noise0", "FashionMNIST"),
            ("MiniBooNE_noise0",     "MiniBooNE"),
            ("higgs_noise0",         "Higgs"),
            ("segment_noise0",       "Segment"),
        ],
    },
    "noise_ablation": {
        "kind": "noise_ablation_pairs",
        "required_iters": 100,
        "transform": "log10",
        "y_label": r"$\log_{10}$ Inference Regret",
        # Panels are (benchmark, noise, display_label, share_y_group)
        # share_y_group: panels with the same int share a y-axis
        "panels": [
            ("Hartmann4D", "0.316", r"Hartmann (4D), $\sigma_{n}=0.316$", 0),
            ("Hartmann4D", "0.5",   r"Hartmann (4D), $\sigma_{n}=0.5$",   0),
            ("Hartmann6D", "0.316", r"Hartmann (6D), $\sigma_{n}=0.316$", 1),
            ("Hartmann6D", "0.5",   r"Hartmann (6D), $\sigma_{n}=0.5$",   1),
        ],
    },
    "hpob": {
        "kind": "per_search_space_ranking",
        "required_iters": 50,
        "winrate_direction": "higher",  # PFN wins if it has higher accuracy
        "y_label": "Average Rank",
        "title_suffix": "",
        "panels": [
            ("5527", "ID=5527 (8D)"),
            ("5891", "ID=5891 (8D)"),
            ("7609", "ID=7609 (9D)"),
            ("5965", "ID=5965 (10D)"),
            ("5971", "ID=5971 (16D)"),
        ],
    },
}


def apply_icml_style(ncols: int) -> None:
    plt.rcParams.update(
        bundles.icml2024(column="full", nrows=1, ncols=ncols, usetex=False)
    )
    plt.rcParams.update(
        figsizes.icml2024_full(nrows=1, ncols=ncols, height_to_width_ratio=1.0)
    )
    plt.rcParams.update({
        "axes.grid": False,
        "grid.alpha": 0.4,
        "grid.linestyle": "-",
        "grid.linewidth": 0.3,
    })


def _open_combined():
    """Open the combined gzipped CSV for text reading."""
    return gzip.open(COMBINED_CSV, "rt", newline="")


def load_curves_csv(
    dataset_filter: str,
    required_iters: int,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Return {(bench, acq, method): array of shape (n_seeds, required_iters)}."""
    raw: dict[tuple[str, str, str], dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with _open_combined() as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset_filter:
                continue
            key = (row["benchmark"], row["acq"], row["method"])
            raw[key][int(row["seed"])][int(row["iter"])] = float(row["value"])

    out: dict[tuple[str, str, str], np.ndarray] = {}
    for key, by_seed in raw.items():
        seeds = sorted(by_seed)
        mat = np.full((len(seeds), required_iters), np.nan)
        for i, s in enumerate(seeds):
            for it, v in by_seed[s].items():
                if 0 <= it < required_iters:
                    mat[i, it] = v
        if np.isnan(mat).any():
            raise ValueError(f"missing iterations for {key}")
        out[key] = mat
    return out


def load_noise_ablation_csv(
    dataset_filter: str,
    required_iters: int,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Return {(benchmark, noise, acq, method): (n_seeds, required_iters)}."""
    raw: dict[tuple[str, str, str, str], dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with _open_combined() as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset_filter:
                continue
            key = (row["benchmark"], row["noise"], row["acq"], row["method"])
            raw[key][int(row["seed"])][int(row["iter"])] = float(row["value"])

    out: dict[tuple[str, str, str, str], np.ndarray] = {}
    for key, by_seed in raw.items():
        seeds = sorted(by_seed)
        mat = np.full((len(seeds), required_iters), np.nan)
        for i, s in enumerate(seeds):
            for it, v in by_seed[s].items():
                if 0 <= it < required_iters:
                    mat[i, it] = v
        if np.isnan(mat).any():
            raise ValueError(f"missing iterations for {key}")
        out[key] = mat
    return out


def load_hpob_csv(
    dataset_filter: str,
    required_iters: int,
) -> dict[tuple[str, str, str, str], np.ndarray]:
    """Return {(search_space, dataset_id, acq, method): (n_seeds, required_iters)}."""
    raw: dict[tuple[str, str, str, str], dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with _open_combined() as f:
        for row in csv.DictReader(f):
            if row["dataset"] != dataset_filter:
                continue
            key = (row["search_space"], row["dataset_id"], row["acq"], row["method"])
            raw[key][int(row["seed"])][int(row["iter"])] = float(row["value"])

    out: dict[tuple[str, str, str, str], np.ndarray] = {}
    for key, by_seed in raw.items():
        seeds = sorted(by_seed)
        mat = np.full((len(seeds), required_iters), np.nan)
        for i, s in enumerate(seeds):
            for it, v in by_seed[s].items():
                if 0 <= it < required_iters:
                    mat[i, it] = v
        if np.isnan(mat).any():
            raise ValueError(f"missing iterations for {key}")
        out[key] = mat
    return out


def mean_sem(values: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray]:
    if transform == "log10":
        values = np.log10(np.maximum(values, FLOOR))
    elif transform != "raw":
        raise ValueError(f"unknown transform {transform!r}")
    mean = values.mean(axis=0)
    sem = (
        values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        if values.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def compute_search_space_ranks(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    search_space: str,
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    required_iters: int,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """For each (acq, method) compute mean ± SEM rank across datasets × seeds.

    Higher predicted_accuracy = better = rank 1 (per iteration, per seed, per dataset).
    """
    method_keys = [(acq, m) for acq in acq_functions for m in methods]
    per_method_ranks: dict[tuple[str, str], list[np.ndarray]] = {k: [] for k in method_keys}

    dataset_ids = sorted({k[1] for k in raw if k[0] == search_space})
    if not dataset_ids:
        raise ValueError(f"no datasets for search_space={search_space}")

    for ds in dataset_ids:
        arrays = [raw.get((search_space, ds, acq, m)) for acq, m in method_keys]
        if any(a is None for a in arrays):
            continue  # skip datasets that lack a (acq, method) combination
        n_seeds = min(a.shape[0] for a in arrays)
        for s in range(n_seeds):
            # Stack values at this seed across all method_keys: (n_methods, n_iters)
            seed_vals = np.vstack([a[s, :] for a in arrays])
            seed_ranks = np.empty_like(seed_vals)
            for t in range(required_iters):
                # negate so higher accuracy => lower (better) rank
                seed_ranks[:, t] = stats.rankdata(-seed_vals[:, t], method="average")
            for i, k in enumerate(method_keys):
                per_method_ranks[k].append(seed_ranks[i, :])

    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for k, lst in per_method_ranks.items():
        stacked = np.vstack(lst)  # (n_datasets × n_seeds, n_iters)
        m = stacked.mean(axis=0)
        if stacked.shape[0] > 1:
            sem = stacked.std(axis=0, ddof=1) / math.sqrt(stacked.shape[0])
        else:
            sem = np.zeros_like(m)
        out[k] = (m, sem)
    return out


def _pairwise_wins(
    pfn: np.ndarray,
    gp: np.ndarray,
    direction: str,
) -> np.ndarray:
    """Per (seed, iter) PFN-vs-GP win indicator: 1, 0, or 0.5 for ties."""
    if pfn.shape != gp.shape:
        raise ValueError(f"shape mismatch {pfn.shape} vs {gp.shape}")
    if direction == "lower":
        win = (pfn < gp).astype(float)
    elif direction == "higher":
        win = (pfn > gp).astype(float)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    tie = (pfn == gp).astype(float)
    return win + 0.5 * tie


def compute_winrate_per_benchmark(
    raw: dict[tuple[str, str, str], np.ndarray],
    bench_key: str,
    acq: str,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± SEM PFN-vs-GP win rate over seeds, for one (bench, acq) cell."""
    pfn = raw[(bench_key, acq, "PFN")]
    gp = raw[(bench_key, acq, "GP")]
    wins = _pairwise_wins(pfn, gp, direction)  # (n_seeds, n_iters)
    mean = wins.mean(axis=0)
    sem = (
        wins.std(axis=0, ddof=1) / math.sqrt(wins.shape[0])
        if wins.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def compute_winrate_per_search_space(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    search_space: str,
    acq: str,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean PFN-vs-GP win rate per dataset (avg over seeds), then mean ± SEM across datasets."""
    dataset_ids = sorted({k[1] for k in raw if k[0] == search_space and k[2] == acq})
    per_dataset_means: list[np.ndarray] = []
    for ds in dataset_ids:
        pfn = raw.get((search_space, ds, acq, "PFN"))
        gp = raw.get((search_space, ds, acq, "GP"))
        if pfn is None or gp is None:
            continue
        wins = _pairwise_wins(pfn, gp, direction)
        per_dataset_means.append(wins.mean(axis=0))
    if not per_dataset_means:
        raise ValueError(f"no data for {search_space=} {acq=}")
    stacked = np.vstack(per_dataset_means)  # (n_datasets, n_iters)
    mean = stacked.mean(axis=0)
    sem = (
        stacked.std(axis=0, ddof=1) / math.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def plot_winrate_panel(
    ax: Axes,
    title: str,
    winrates: dict[str, tuple[np.ndarray, np.ndarray]],
    acq_functions: tuple[str, ...],
    show_ylabel: bool,
    required_iters: int,
) -> None:
    iterations = np.arange(1, required_iters + 1)
    for acq in acq_functions:
        mean, sem = winrates[acq]
        color = ACQ_COLORS[acq]
        ax.fill_between(
            iterations, mean - sem, mean + sem,
            color=color, alpha=0.2, linewidth=0, zorder=5,
        )
        ax.plot(
            iterations, mean,
            color=color, linestyle="-", linewidth=1.0, zorder=10,
        )
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, zorder=1)
    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel("Win Rate (PFN vs GP)")
    title_size = float(plt.rcParams.get("axes.titlesize", 10))
    ax.set_title(title, fontsize=title_size * 0.9)
    ax.set_xlim(1, required_iters)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_acq_only_legend(fig: Figure, acq_functions: tuple[str, ...]) -> None:
    handles = [
        mlines.Line2D([], [], color=ACQ_COLORS[a], linestyle="-", linewidth=1.4, label=a)
        for a in acq_functions
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(acq_functions), frameon=True, edgecolor="0.7",
    )
    legend.set_in_layout(True)
    fig.add_artist(legend)


def plot_panel(
    ax: Axes,
    title: str,
    stats_dict: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    y_label: str,
    show_ylabel: bool,
    required_iters: int,
    y_limits: tuple[float, float | None] | None = None,
) -> None:
    iterations = np.arange(1, required_iters + 1)
    upper_envelope: list[np.ndarray] = []
    for method in reversed(methods):
        style = METHOD_STYLE[method]
        for acq in reversed(acq_functions):
            mean, sem = stats_dict[(acq, method)]
            color = ACQ_COLORS[acq]
            ax.fill_between(
                iterations, mean - sem, mean + sem,
                color=color, alpha=0.18, linewidth=0,
                zorder=style["z"],
            )
            ax.plot(
                iterations, mean,
                color=color,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=style["z"] + 1,
            )
            upper_envelope.append(mean + sem)

    if y_limits is not None:
        y_min, y_max = y_limits
        if y_max is None:
            y_max_data = max(arr.max() for arr in upper_envelope)
            y_max = y_max_data + (y_max_data - y_min) * 0.05
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Iteration")
    if show_ylabel:
        ax.set_ylabel(y_label)
    title_size = float(plt.rcParams.get("axes.titlesize", 10))
    ax.set_title(title, fontsize=title_size * 0.9)
    ax.set_xlim(1, required_iters)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_legend(
    fig: Figure,
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    method_legend_title: str,
    layout: str,
    show_titles: bool,
    method_labels: dict[str, str] | None = None,
) -> None:
    if layout == "none":
        return
    method_labels = method_labels or {}
    acq_handles = [
        mlines.Line2D([], [], color=ACQ_COLORS[a], linestyle="-", linewidth=1.4, label=a)
        for a in acq_functions
    ]
    method_handles = [
        mlines.Line2D(
            [], [], color="black",
            linestyle=METHOD_STYLE[m]["linestyle"],
            linewidth=1.4,
            label=method_labels.get(m, METHOD_STYLE[m]["label"]),
        )
        for m in methods
    ]
    acq_title = "Acquisition" if show_titles else None
    method_title = method_legend_title if show_titles else None

    if layout == "horizontal":
        legend_acq = fig.legend(
            handles=acq_handles,
            loc="lower center", bbox_to_anchor=(0.30, 1.02),
            ncol=len(acq_functions), frameon=True, edgecolor="0.7",
            title=acq_title,
        )
        legend_method = fig.legend(
            handles=method_handles,
            loc="lower center", bbox_to_anchor=(0.75, 1.02),
            ncol=len(methods), frameon=True, edgecolor="0.7",
            title=method_title,
        )
    elif layout == "stacked":
        legend_method = fig.legend(
            handles=method_handles,
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=len(methods), frameon=True, edgecolor="0.7",
            title=method_title,
        )
        legend_acq = fig.legend(
            handles=acq_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.32 if show_titles else 1.20),
            ncol=len(acq_functions), frameon=True, edgecolor="0.7",
            title=acq_title,
        )
    else:
        raise ValueError(f"unknown legend layout {layout!r}")

    legend_acq.set_in_layout(True)
    legend_method.set_in_layout(True)
    fig.add_artist(legend_acq)
    fig.add_artist(legend_method)


def render_curves(
    raw: dict[tuple[str, str, str], np.ndarray],
    dataset: dict,
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    method_legend_title: str,
    out_pdf: Path,
    legend_layout: str,
    legend_show_titles: bool,
    method_labels: dict[str, str] | None = None,
) -> None:
    benchmarks = dataset["benchmarks"]
    apply_icml_style(ncols=len(benchmarks))
    fig, axes = plt.subplots(1, len(benchmarks), constrained_layout=True)
    for idx, (bench_key, bench_label) in enumerate(benchmarks):
        stats_dict = {
            (acq, method): mean_sem(raw[(bench_key, acq, method)], dataset["transform"])
            for acq in acq_functions
            for method in methods
        }
        plot_panel(
            axes[idx],
            bench_label + dataset["title_suffix"],
            stats_dict, acq_functions, methods,
            y_label=dataset["y_label"],
            show_ylabel=(idx == 0),
            required_iters=dataset["required_iters"],
            y_limits=dataset["y_limits"].get(bench_key),
        )
    add_legend(
        fig, acq_functions, methods, method_legend_title,
        layout=legend_layout, show_titles=legend_show_titles,
        method_labels=method_labels,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def render_ranking(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    dataset: dict,
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    method_legend_title: str,
    out_pdf: Path,
    legend_layout: str,
    legend_show_titles: bool,
) -> None:
    panels = dataset["panels"]
    apply_icml_style(ncols=len(panels))
    fig, axes = plt.subplots(1, len(panels), constrained_layout=True)
    for idx, (search_space, label) in enumerate(panels):
        stats_dict = compute_search_space_ranks(
            raw, search_space, acq_functions, methods, dataset["required_iters"],
        )
        plot_panel(
            axes[idx],
            label + dataset["title_suffix"],
            stats_dict, acq_functions, methods,
            y_label=dataset["y_label"],
            show_ylabel=(idx == 0),
            required_iters=dataset["required_iters"],
        )
    add_legend(
        fig, acq_functions, methods, method_legend_title,
        layout=legend_layout, show_titles=legend_show_titles,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def render_noise_ablation(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    dataset: dict,
    acq_functions: tuple[str, ...],
    methods: tuple[str, ...],
    method_legend_title: str,
    out_pdf: Path,
    legend_layout: str,
    legend_show_titles: bool,
    method_labels: dict[str, str] | None = None,
) -> None:
    panels = dataset["panels"]
    apply_icml_style(ncols=len(panels))
    # Build sharey groups: panels with the same group share a y-axis (first axis
    # in each group is the "anchor"; subsequent panels share its y).
    fig = plt.figure(constrained_layout=True)
    axes: list[Axes] = []
    group_anchor: dict[int, Axes] = {}
    for idx, (_, _, _, group) in enumerate(panels):
        ax = fig.add_subplot(
            1, len(panels), idx + 1,
            sharey=group_anchor.get(group),
        )
        if group not in group_anchor:
            group_anchor[group] = ax
        axes.append(ax)

    for idx, (bench, noise, label, group) in enumerate(panels):
        stats_dict = {
            (acq, method): mean_sem(
                raw[(bench, noise, acq, method)], dataset["transform"]
            )
            for acq in acq_functions
            for method in methods
        }
        # show ylabel only on the first panel of each group
        show_ylabel = (axes[idx] is group_anchor[group])
        plot_panel(
            axes[idx], label, stats_dict, acq_functions, methods,
            y_label=dataset["y_label"],
            show_ylabel=show_ylabel,
            required_iters=dataset["required_iters"],
        )
        # Hide y tick labels on shared (non-anchor) panels to keep things tidy
        if not show_ylabel:
            for tick in axes[idx].get_yticklabels():
                tick.set_visible(False)

    add_legend(
        fig, acq_functions, methods, method_legend_title,
        layout=legend_layout, show_titles=legend_show_titles,
        method_labels=method_labels,
    )
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def render_winrate_per_benchmark(
    raw: dict[tuple[str, str, str], np.ndarray],
    dataset: dict,
    acq_functions: tuple[str, ...],
    out_pdf: Path,
) -> None:
    benchmarks = dataset["benchmarks"]
    apply_icml_style(ncols=len(benchmarks))
    fig, axes = plt.subplots(1, len(benchmarks), constrained_layout=True)
    for idx, (bench_key, bench_label) in enumerate(benchmarks):
        winrates = {
            acq: compute_winrate_per_benchmark(
                raw, bench_key, acq, dataset["winrate_direction"]
            )
            for acq in acq_functions
        }
        plot_winrate_panel(
            axes[idx],
            bench_label + dataset["title_suffix"],
            winrates, acq_functions,
            show_ylabel=(idx == 0),
            required_iters=dataset["required_iters"],
        )
    add_acq_only_legend(fig, acq_functions)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def render_winrate_per_search_space(
    raw: dict[tuple[str, str, str, str], np.ndarray],
    dataset: dict,
    acq_functions: tuple[str, ...],
    out_pdf: Path,
) -> None:
    panels = dataset["panels"]
    apply_icml_style(ncols=len(panels))
    fig, axes = plt.subplots(1, len(panels), constrained_layout=True)
    for idx, (search_space, label) in enumerate(panels):
        winrates = {
            acq: compute_winrate_per_search_space(
                raw, search_space, acq, dataset["winrate_direction"]
            )
            for acq in acq_functions
        }
        plot_winrate_panel(
            axes[idx],
            label + dataset["title_suffix"],
            winrates, acq_functions,
            show_ylabel=(idx == 0),
            required_iters=dataset["required_iters"],
        )
    add_acq_only_legend(fig, acq_functions)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


FIGURES: tuple[dict, ...] = (
    {
        "filename": "synthetic_v1_inference_regret_main_text_crc.pdf",
        "dataset": "synthetic",
        "acqs": ACQ_FUNCTIONS_MAIN,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "horizontal",
        "legend_show_titles": False,
    },
    {
        "filename": "synthetic_v1_inference_regret_appendix_crc.pdf",
        "dataset": "synthetic",
        "acqs": ACQ_FUNCTIONS_FULL,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "horizontal",
        "legend_show_titles": False,
    },
    {
        "filename": "clustering_ablation_inference_regret_crc.pdf",
        "dataset": "synthetic",
        "acqs": ACQ_FUNCTIONS_CLUSTERING,
        "methods": ("PFN", "PFN_uniform"),
        "method_legend_title": "Synthetic traces",
        "legend_layout": "horizontal",
        "legend_show_titles": True,
        "method_labels": {"PFN": "Clustering"},
    },
    {
        "filename": "lcbench_accuracy_v1_main_text_crc.pdf",
        "dataset": "lcbench",
        "acqs": ACQ_FUNCTIONS_MAIN,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "none",
        "legend_show_titles": False,
    },
    {
        "filename": "lcbench_accuracy_v1_appendix_crc.pdf",
        "dataset": "lcbench",
        "acqs": ACQ_FUNCTIONS_FULL,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "none",
        "legend_show_titles": False,
    },
    {
        "filename": "hpob_ranking_main_text_crc.pdf",
        "dataset": "hpob",
        "acqs": ACQ_FUNCTIONS_MAIN,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "none",
        "legend_show_titles": False,
    },
    {
        "filename": "hpob_ranking_appendix_crc.pdf",
        "dataset": "hpob",
        "acqs": ACQ_FUNCTIONS_FULL,
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "none",
        "legend_show_titles": False,
    },
    {
        "filename": "noise_ablation_hartmann_inference_regret_crc.pdf",
        "dataset": "noise_ablation",
        "acqs": ("JES", "MES", "PES"),
        "methods": ("PFN", "GP"),
        "method_legend_title": "Model",
        "legend_layout": "horizontal",
        "legend_show_titles": False,
    },
    # Win rate figures (single curve per acq, PFN vs GP fixed)
    {
        "filename": "synthetic_v1_inference_regret_win_rate_crc.pdf",
        "dataset": "synthetic",
        "acqs": ACQ_FUNCTIONS_FULL,
        "kind_override": "winrate_curves",
    },
    {
        "filename": "lcbench_v1_win_rate_crc.pdf",
        "dataset": "lcbench",
        "acqs": ACQ_FUNCTIONS_FULL,
        "kind_override": "winrate_curves",
    },
    {
        "filename": "hpob_win_rate_crc.pdf",
        "dataset": "hpob",
        "acqs": ACQ_FUNCTIONS_FULL,
        "kind_override": "winrate_ranking",
    },
)


def main() -> None:
    if not COMBINED_CSV.is_file():
        raise FileNotFoundError(f"CSV not found: {COMBINED_CSV}")
    loaded: dict[str, dict] = {}
    for name, ds in DATASETS.items():
        if ds["kind"] == "per_benchmark_curves":
            loaded[name] = load_curves_csv(name, ds["required_iters"])
        elif ds["kind"] == "per_search_space_ranking":
            loaded[name] = load_hpob_csv(name, ds["required_iters"])
        elif ds["kind"] == "noise_ablation_pairs":
            loaded[name] = load_noise_ablation_csv(name, ds["required_iters"])
        else:
            raise ValueError(f"unknown kind {ds['kind']!r}")

    for spec in FIGURES:
        dataset = DATASETS[spec["dataset"]]
        raw = loaded[spec["dataset"]]
        out_pdf = OUTPUT_DIR / spec["filename"]
        kind = spec.get("kind_override", dataset["kind"])
        if kind == "per_benchmark_curves":
            render_curves(
                raw, dataset=dataset,
                acq_functions=spec["acqs"], methods=spec["methods"],
                method_legend_title=spec["method_legend_title"],
                out_pdf=out_pdf,
                legend_layout=spec["legend_layout"],
                legend_show_titles=spec["legend_show_titles"],
                method_labels=spec.get("method_labels"),
            )
        elif kind == "per_search_space_ranking":
            render_ranking(
                raw, dataset=dataset,
                acq_functions=spec["acqs"], methods=spec["methods"],
                method_legend_title=spec["method_legend_title"],
                out_pdf=out_pdf,
                legend_layout=spec["legend_layout"],
                legend_show_titles=spec["legend_show_titles"],
            )
        elif kind == "noise_ablation_pairs":
            render_noise_ablation(
                raw, dataset=dataset,
                acq_functions=spec["acqs"], methods=spec["methods"],
                method_legend_title=spec["method_legend_title"],
                out_pdf=out_pdf,
                legend_layout=spec["legend_layout"],
                legend_show_titles=spec["legend_show_titles"],
                method_labels=spec.get("method_labels"),
            )
        elif kind == "winrate_curves":
            render_winrate_per_benchmark(
                raw, dataset=dataset, acq_functions=spec["acqs"], out_pdf=out_pdf,
            )
        elif kind == "winrate_ranking":
            render_winrate_per_search_space(
                raw, dataset=dataset, acq_functions=spec["acqs"], out_pdf=out_pdf,
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")
        print(f"wrote {out_pdf} ({out_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
