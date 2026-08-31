#!/usr/bin/env python3
"""Render the six-model released-validation comparison with BCR and V7 together."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]
SCENARIO_LABELS = ["New\ncompound", "New\nstrain", "Double\nunseen", "Temporal\nvalidation"]
METRICS = [
    ("rmse_log2", "RMSE (lower is better)"),
    ("mae_log2", "MAE (lower is better)"),
    ("global_r2", "Global R² (higher is better)"),
    ("sample_pcc_median", "Sample PCC median (higher is better)"),
    ("protein_pcc_median", "Protein PCC median (higher is better)"),
    ("protein_r2_median", "Protein R² median (higher is better)"),
]
VARIANTS = {
    "mean": ("Protein mean", "#8C8C8C", ".."),
    "matched_control_new": ("Matched control", "#E69F00", "\\\\"),
    "zero": ("MLP without descriptors", "#5B8CC0", "xx"),
    "real": ("Descriptor MLP", "#2A9D8F", ""),
    "semantic_bcr_v1_released_refit": ("Upgraded BCR (OP3 + CalV2)", "#D55E00", "//"),
    "biostate_v7_proteome": ("BioState-Readout V7", "#7E57C2", "oo"),
}


def grouped_value(
    grouped: pd.DataFrame, model: str, split: str, metric: str, statistic: str
) -> float:
    return float(grouped.loc[(model, split), (metric, statistic)])


def make_plot(metrics_csv: Path, output: Path, seeds: list[int]) -> None:
    frame = pd.read_csv(metrics_csv)
    if set(frame["model"]) != set(VARIANTS):
        raise RuntimeError("six-model plot surface differs")
    if set(frame["split"]) != set(SCENARIOS):
        raise RuntimeError("released-validation scenario surface differs")
    metric_names = [name for name, _ in METRICS]
    grouped = frame.groupby(["model", "split"])[metric_names].agg(["mean", "std"])

    fig, axes = plt.subplots(2, 3, figsize=(24, 13), constrained_layout=False)
    axes = axes.ravel()
    x = np.arange(len(SCENARIOS))
    width = 0.13
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * width

    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        for offset, model in zip(offsets, VARIANTS, strict=True):
            values = np.asarray(
                [grouped_value(grouped, model, split, metric, "mean") for split in SCENARIOS],
                dtype=np.float64,
            )
            errors = np.asarray(
                [grouped_value(grouped, model, split, metric, "std") for split in SCENARIOS],
                dtype=np.float64,
            )
            shown = np.clip(values, 0.0, 1.0)
            label, color, hatch = VARIANTS[model]
            bars = axis.bar(
                x + offset,
                np.nan_to_num(shown, nan=0.0),
                width,
                label=label,
                color=color,
                edgecolor="#333333",
                linewidth=0.55,
                hatch=hatch,
                yerr=np.where(np.isfinite(errors), errors, 0.0),
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
                alpha=0.93,
                zorder=3,
            )
            for bar, raw, clipped in zip(bars, values, shown, strict=True):
                if not np.isfinite(raw):
                    label_y, text, vertical = 0.018, "N/A", "bottom"
                elif raw < 0:
                    label_y, text, vertical = 0.035, f"{raw:.3f}", "bottom"
                elif raw > 1:
                    label_y, text, vertical = 0.985, f"{raw:.3f}", "top"
                elif clipped > 0.92:
                    label_y, text, vertical = clipped - 0.012, f"{raw:.3f}", "top"
                else:
                    label_y, text, vertical = min(clipped + 0.018, 0.985), f"{raw:.3f}", "bottom"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    text,
                    ha="center",
                    va=vertical,
                    fontsize=5.8,
                    rotation=90,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.62, "pad": 0.25},
                    clip_on=False,
                )
        axis.set_title(title, fontsize=12.5, fontweight="bold", pad=9)
        axis.set_xticks(x, SCENARIO_LABELS)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
        fontsize=10.2,
    )
    seed_text = ", ".join(str(seed) for seed in seeds)
    fig.suptitle(
        "GOAI Upgraded BCR vs BioState-Readout V7\n"
        f"Released validation, treatment rows | mean over seeds {seed_text} | all y-axes: 0–1",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        -0.004,
        "Same released-validation rows and matched-control common mask. Bars show seed means; error bars show seed SD where defined. "
        "Values outside [0, 1] are clipped but retain raw labels. Predictive comparison is valid; architecture attribution is not "
        "parameter/info matched (BCR: OP3+CalV2, no plate input; V7: chemical-512 plus instrument/plate observer).",
        ha="center",
        fontsize=9.3,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(top=0.79, bottom=0.10, left=0.05, right=0.985, hspace=0.34, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    make_plot(args.metrics_csv, args.output, args.seeds)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
