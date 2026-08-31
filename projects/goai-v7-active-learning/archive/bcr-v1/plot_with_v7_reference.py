#!/usr/bin/env python3
"""Plot the formal BCR head-to-head beside the separate-protocol V7 reference.

The two panels deliberately remain separate.  The BCR result is pooled internal
OOF on split=train, while V7 was fitted on all train rows and evaluated on the
released validation surface.  Sharing a y-axis makes the metric scale legible;
it does not make cross-panel model ranking valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BCR_SCENARIOS = ["S1", "Time"]
BCR_MODELS = [
    "GOAI-SEMANTIC-BCR-V1",
    "FLAT-MLP-SAME-INFO",
    "PROTEIN-MEAN",
    "MATCHED-CONTROL-ORACLE-DIAGNOSTIC",
]
BCR_STYLES = {
    "GOAI-SEMANTIC-BCR-V1": ("BCR-V1", "#D55E00", "//"),
    "FLAT-MLP-SAME-INFO": ("Flat MLP (same info)", "#4472C4", "xx"),
    "PROTEIN-MEAN": ("Protein mean", "#8C8C8C", ".."),
    "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": (
        "Matched control (oracle)",
        "#E69F00",
        "\\\\",
    ),
}

V7_SCENARIOS = ["val_chem_only", "val_strain_only", "val_both", "val_time"]
V7_SCENARIO_LABELS = [
    "New\ncompound",
    "New\nstrain",
    "Double\nunseen",
    "Released\ntemporal",
]
V7_MODELS = ["mean", "matched_control_new", "zero", "real", "biostate_v7_proteome"]
V7_STYLES = {
    "mean": ("Protein mean", "#8C8C8C", ".."),
    "matched_control_new": ("Matched control", "#E69F00", "\\\\"),
    "zero": ("MLP, no descriptors", "#6C8EBF", "xx"),
    "real": ("Descriptor MLP", "#2A9D8F", ""),
    "biostate_v7_proteome": ("BioState-Readout V7", "#7E57C2", "oo"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bcr(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["kind"].eq("primary")
        & frame["status"].eq("OK")
        & frame["scenario"].isin(BCR_SCENARIOS)
        & frame["model"].isin(BCR_MODELS)
    ].copy()
    expected = {(scenario, model) for scenario in BCR_SCENARIOS for model in BCR_MODELS}
    observed = set(zip(selected["scenario"], selected["model"], strict=True))
    if observed != expected:
        raise ValueError(f"BCR result surface mismatch: missing={expected - observed}, extra={observed - expected}")
    return selected


def load_v7(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["split"].isin(V7_SCENARIOS) & frame["model"].isin(V7_MODELS)
    ].copy()
    expected = {(scenario, model) for scenario in V7_SCENARIOS for model in V7_MODELS}
    observed = set(zip(selected["split"], selected["model"], strict=True))
    if observed != expected:
        raise ValueError(f"V7 result surface mismatch: missing={expected - observed}, extra={observed - expected}")
    return selected


def build_protocol_table(bcr: pd.DataFrame, v7: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in bcr.itertuples(index=False):
        rows.append(
            {
                "protocol_id": "internal_frozen_oof",
                "protocol_role": "HEAD_TO_HEAD_PRIMARY",
                "scenario": record.scenario,
                "model": record.model,
                "protein_r2_median_mean": record.pooled_oof_median_protein_r2,
                "protein_r2_median_sd": np.nan,
                "n_seeds": 1,
                "seed_definition": "seed42; four OOF folds pooled before metric",
                "n_samples": int(record.n_samples),
                "n_scored_rows": int(record.n_scored_rows),
                "n_observed_values": int(record.n_observed_values),
                "comparison_rule": "rank only against rows with the same protocol_id",
            }
        )

    grouped = v7.groupby(["split", "model"], sort=False)
    for (split, model), group in grouped:
        values = group["protein_r2_median"].astype(float)
        rows.append(
            {
                "protocol_id": "released_validation_reference",
                "protocol_role": "CONTEXT_ONLY_NOT_BCR_HEAD_TO_HEAD",
                "scenario": split,
                "model": model,
                "protein_r2_median_mean": values.mean(),
                "protein_r2_median_sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                "n_seeds": int(len(values)),
                "seed_definition": (
                    "mean over seeds 42,43,44" if len(values) == 3 else "deterministic/single available row"
                ),
                "n_samples": int(group["n_samples"].iloc[0]),
                "n_scored_rows": int(group["n_samples"].iloc[0]),
                "n_observed_values": int(group["n_observed_values"].iloc[0]),
                "comparison_rule": "rank only against rows with the same protocol_id",
            }
        )
    return pd.DataFrame(rows)


def add_value_labels(axis: plt.Axes, bars, values: np.ndarray, *, fontsize: float) -> None:
    for bar, value in zip(bars, values, strict=True):
        offset = 0.018 if value >= 0 else -0.018
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            rotation=90,
            fontsize=fontsize,
            color="#222222",
            clip_on=False,
        )


def plot_bcr(axis: plt.Axes, bcr: pd.DataFrame) -> None:
    axis.set_facecolor("#F7FAFD")
    x = np.arange(len(BCR_SCENARIOS))
    width = 0.19
    offsets = (np.arange(len(BCR_MODELS)) - 1.5) * width
    for offset, model in zip(offsets, BCR_MODELS, strict=True):
        values = np.array(
            [
                float(
                    bcr.loc[
                        bcr["scenario"].eq(scenario) & bcr["model"].eq(model),
                        "pooled_oof_median_protein_r2",
                    ].iloc[0]
                )
                for scenario in BCR_SCENARIOS
            ]
        )
        label, color, hatch = BCR_STYLES[model]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#3A3A3A",
            linewidth=0.7,
            hatch=hatch,
            alpha=0.94,
            zorder=3,
        )
        add_value_labels(axis, bars, values, fontsize=8.6)

    axis.set_xticks(x, ["S1 / new compound", "Time-forward"])
    axis.set_title(
        "A  BCR experiment — internal frozen OOF",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    axis.text(
        0.0,
        1.01,
        "HEAD-TO-HEAD  •  seed 42  •  identical eval rows and common mask",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#355C7D",
        fontweight="bold",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
    )
    axis.text(
        0.5,
        -0.245,
        "Within-panel result:  BCR − Flat MLP = −0.063 (S1), −0.046 (Time)",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color="#A33A12",
    )


def plot_v7(axis: plt.Axes, v7: pd.DataFrame) -> None:
    axis.set_facecolor("#FBF8FE")
    x = np.arange(len(V7_SCENARIOS))
    width = 0.145
    offsets = (np.arange(len(V7_MODELS)) - 2.0) * width
    for offset, model in zip(offsets, V7_MODELS, strict=True):
        values = np.array(
            [
                float(v7.loc[v7["split"].eq(split) & v7["model"].eq(model), "protein_r2_median"].mean())
                for split in V7_SCENARIOS
            ]
        )
        errors = np.array(
            [
                float(v7.loc[v7["split"].eq(split) & v7["model"].eq(model), "protein_r2_median"].std(ddof=1))
                for split in V7_SCENARIOS
            ]
        )
        label, color, hatch = V7_STYLES[model]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#3A3A3A",
            linewidth=0.65,
            hatch=hatch,
            alpha=0.94,
            yerr=np.where(np.isfinite(errors), errors, 0.0),
            error_kw={"elinewidth": 0.9, "capsize": 2.2, "capthick": 0.9},
            zorder=3,
        )
        add_value_labels(axis, bars, values, fontsize=7.2)

    axis.set_xticks(x, V7_SCENARIO_LABELS)
    axis.set_title(
        "B  BioState-Readout V7 — released validation",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    axis.text(
        0.0,
        1.01,
        "CONTEXT ONLY  •  mean ± seed SD (42/43/44)  •  separate evaluation surface",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#6A3D9A",
        fontweight="bold",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        frameon=False,
        fontsize=9.0,
        handlelength=2.0,
    )
    axis.text(
        0.5,
        -0.245,
        "Within-panel result:  V7 − Descriptor MLP = +0.763, +0.594, +0.731, +0.454",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color="#6030A0",
    )


def make_plot(bcr: pd.DataFrame, v7: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(22, 9.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.65]},
    )
    plot_bcr(axes[0], bcr)
    plot_v7(axes[1], v7)

    for axis in axes:
        axis.axhline(0.0, color="#333333", linewidth=1.0, zorder=2)
        axis.set_ylim(-0.22, 0.92)
        axis.set_yticks(np.arange(-0.2, 1.0, 0.2))
        axis.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.55, zorder=0)
        axis.tick_params(axis="x", labelsize=11)
        axis.tick_params(axis="y", labelsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median per-protein R²", fontsize=13, fontweight="bold")

    fig.suptitle(
        "BCR Head-to-Head, with V7 Added as a Separate-Protocol Reference",
        fontsize=21,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "Same metric, different rows and training contracts — compare models within each panel only",
        ha="center",
        va="center",
        fontsize=13,
        color="#444444",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "A: split=train treatment rows; four canonical folds pooled before scoring.  "
        "B: all 5,920 train rows fitted, then released validation scored.  "
        "V7 has 73.95M parameters and adds a chemical-512 branch plus instrument/plate observer; "
        "BCR/Flat use OP3-64 + fold-local CalV2. No BCR↔V7 delta is valid without fold-local V7 retraining.",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#555555",
        wrap=True,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.27, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bcr-results", type=Path, required=True)
    parser.add_argument("--v7-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bcr = load_bcr(args.bcr_results)
    v7 = load_v7(args.v7_results)
    table = build_protocol_table(bcr, v7)
    table_path = args.output.with_suffix(".csv")
    table.to_csv(table_path, index=False)
    make_plot(bcr, v7, args.output)

    provenance = {
        "format": "goai.semantic_bcr_v1.v7_reference_figure.v1",
        "interpretation": "panels are separate protocols; cross-panel ranking is forbidden",
        "generator_script": str(Path(__file__).resolve()),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "bcr_input": str(args.bcr_results.resolve()),
        "bcr_input_sha256": sha256_file(args.bcr_results),
        "v7_input": str(args.v7_results.resolve()),
        "v7_input_sha256": sha256_file(args.v7_results),
        "png": str(args.output.resolve()),
        "png_sha256": sha256_file(args.output),
        "svg": str(args.output.with_suffix('.svg').resolve()),
        "svg_sha256": sha256_file(args.output.with_suffix('.svg')),
        "comparison_csv": str(table_path.resolve()),
        "comparison_csv_sha256": sha256_file(table_path),
    }
    provenance_path = args.output.with_name(f"{args.output.stem}_PROVENANCE.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    print(table_path.resolve())
    print(provenance_path.resolve())


if __name__ == "__main__":
    main()
