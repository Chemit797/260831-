#!/usr/bin/env python3
"""Plot real descriptors against zero and row-shuffled controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]
SCENARIO_LABELS = ["New\ncompound", "New\nstrain", "Double\nunseen", "Temporal\nvalidation"]
METRICS = [
    ("rmse_log2", "RMSE", "lower is better"),
    ("mae_log2", "MAE", "lower is better"),
    ("global_r2", "Global R²", "higher is better"),
    ("sample_pcc_median", "Sample PCC median", "higher is better"),
    ("protein_pcc_median", "Protein PCC median", "higher is better"),
    ("protein_r2_median", "Protein R² median", "higher is better"),
]
VARIANTS = {
    "zero": ("No descriptors", "#B8B8B8", ".."),
    "shuffle": ("Shuffled descriptors", "#E3A41A", "//"),
    "real": ("New descriptors", "#3C78A8", ""),
}


def _latest_run(runs_root: Path, variant: str, seed: int) -> Path:
    matches = sorted(runs_root.glob(f"{variant}-seed{seed}-*"))
    if not matches:
        raise FileNotFoundError(f"No {variant} run found for seed {seed}")
    return matches[-1]


def _load_metrics(runs_root: Path, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for seed in seeds:
            run = _latest_run(runs_root, variant, seed)
            frame = pd.read_csv(run / "metrics.csv")
            frame = frame.loc[frame["subset"].eq("treatment_only")].copy()
            frame["variant"] = variant
            frame["seed"] = seed
            frame["source"] = str(run)
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(VARIANTS) * len(seeds) * len(SCENARIOS)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} metric rows, found {len(result)}")
    return result


def _axis_limit(metric: str, values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values) & np.isfinite(errors)
    low = float(np.min(values[finite] - errors[finite]))
    high = float(np.max(values[finite] + errors[finite]))
    if metric == "protein_r2_median":
        padding = max((high - low) * 0.14, 0.04)
        return min(low - padding, -0.05), max(high + padding, 0.05)
    upper = max(high * 1.14, 0.10)
    return 0.0, upper


def make_plot(runs_root: Path, output: Path, seeds: list[int]) -> None:
    metrics = _load_metrics(runs_root, seeds)
    metric_names = [metric for metric, _, _ in METRICS]
    grouped = metrics.groupby(["variant", "split"], sort=False)[metric_names].agg(["mean", "std"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#1F1F1F",
    })
    fig, axes = plt.subplots(2, 3, figsize=(20, 11.2), facecolor="white")
    axes = axes.ravel()
    x = np.arange(len(SCENARIOS))
    width = 0.23
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * width

    for ax, (metric, title, direction) in zip(axes, METRICS, strict=True):
        panel_values: list[float] = []
        panel_errors: list[float] = []
        plotted: list[tuple[object, np.ndarray, np.ndarray]] = []
        for offset, (variant, (label, color, hatch)) in zip(offsets, VARIANTS.items(), strict=True):
            values = np.asarray(
                [grouped.loc[(variant, split), (metric, "mean")] for split in SCENARIOS],
                dtype=float,
            )
            errors = np.asarray(
                [grouped.loc[(variant, split), (metric, "std")] for split in SCENARIOS],
                dtype=float,
            )
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=label,
                color=color,
                edgecolor="#383838",
                linewidth=0.7,
                hatch=hatch,
                yerr=errors,
                error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9, "ecolor": "#303030"},
                zorder=3,
            )
            panel_values.extend(values.tolist())
            panel_errors.extend(errors.tolist())
            plotted.append((bars, values, errors))

        ymin, ymax = _axis_limit(metric, np.asarray(panel_values), np.asarray(panel_errors))
        ax.set_ylim(ymin, ymax)
        span = ymax - ymin
        for bars, values, errors in plotted:
            for bar, value, error in zip(bars, values, errors, strict=True):
                if value >= 0:
                    y = value + error + span * 0.018
                    va = "bottom"
                else:
                    y = value - error - span * 0.018
                    va = "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    fontsize=7.2,
                    rotation=90,
                    clip_on=False,
                )

        ax.axhline(0, color="#333333", linewidth=0.9, zorder=2)
        ax.set_title(f"{title}\n{direction}", fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xticks(x, SCENARIO_LABELS)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, linestyle=":", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.915),
        fontsize=11.5,
    )
    seed_text = ", ".join(str(seed) for seed in seeds)
    fig.suptitle(
        "GOAI descriptor MLP: real vs control inputs\n"
        f"Released validation, treatment rows | mean ± SD over seeds {seed_text}",
        fontsize=17,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.018,
        "Bars show seed means; error bars show ±1 SD. Protein R² retains negative values. Source: basic_descriptor_mlp run metrics.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(top=0.81, bottom=0.10, left=0.055, right=0.985, hspace=0.38, wspace=0.21)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/basic_descriptor_mlp"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    make_plot(args.runs_root, args.output, args.seeds)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
