"""Plot six metrics for protein mean, new matched control, and two MLPs."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    "matched_control_new": ("Matched control (new)", "#E69F00", "//"),
    "zero": ("MLP without descriptors", "#4C78A8", "xx"),
    "real": ("Descriptor MLP", "#2A9D8F", ""),
}


def _format_value(value: float) -> str:
    return f"{value:.3f}"


def make_plot(metrics_csv: Path, output: Path, seeds: list[int]) -> None:
    all_frames = pd.read_csv(metrics_csv)
    grouped = all_frames.groupby(["model", "split"], as_index=False)[[metric[0] for metric in METRICS]].agg(["mean", "std"])

    fig, axes = plt.subplots(2, 3, figsize=(21, 11.5), constrained_layout=False)
    axes = axes.ravel()
    x = np.arange(len(SCENARIOS))
    width = 0.19
    offsets = (np.arange(len(VARIANTS)) - (len(VARIANTS) - 1) / 2) * width
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for offset, variant in zip(offsets, VARIANTS, strict=True):
            values = np.array([
                float(grouped.loc[(grouped[("model", "")] == variant) & (grouped[("split", "")] == split), (metric, "mean")].iloc[0])
                for split in SCENARIOS
            ])
            errors = np.array([
                float(grouped.loc[(grouped[("model", "")] == variant) & (grouped[("split", "")] == split), (metric, "std")].iloc[0])
                for split in SCENARIOS
            ])
            plotted = np.clip(values, 0.0, 1.0)
            shown_errors = np.where(np.isfinite(errors), errors, 0.0)
            bars = ax.bar(
                x + offset,
                np.nan_to_num(plotted, nan=0.0),
                width,
                label=VARIANTS[variant][0],
                color=VARIANTS[variant][1],
                edgecolor="#333333",
                linewidth=0.55,
                hatch=VARIANTS[variant][2],
                yerr=shown_errors,
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
                alpha=0.92,
            )
            for bar, raw, shown in zip(bars, values, plotted, strict=True):
                if not np.isfinite(raw):
                    label_y = 0.018
                    label = "N/A"
                    va = "bottom"
                elif raw < 0:
                    label_y = 0.035
                    label = f"{raw:.3f}"
                    va = "bottom"
                elif raw > 1:
                    label_y = 0.985
                    label = f"{raw:.3f}"
                    va = "top"
                elif shown > 0.92:
                    label_y = shown - 0.012
                    label = f"{raw:.3f}"
                    va = "top"
                else:
                    label_y = min(shown + 0.018, 0.985)
                    label = f"{raw:.3f}"
                    va = "bottom"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=6.6,
                    rotation=90,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4},
                    clip_on=False,
                )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=9)
        ax.set_xticks(x, SCENARIO_LABELS)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.935), fontsize=11)
    seed_text = ", ".join(str(seed) for seed in seeds)
    fig.suptitle(
        "GOAI basic descriptor MLP comparison\n"
        f"Released validation, treatment rows | MLP mean over seeds {seed_text} | all y-axes: 0–1",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        -0.012,
        "Matched control uses the new regime-specific definition. Values outside [0, 1] are clipped but retain raw labels; N/A is undefined.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.05, right=0.985, hspace=0.34, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
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
