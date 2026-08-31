#!/usr/bin/env python3
"""Paired released-validation diagnostics for upgraded BCR versus V7."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SPLIT_TO_REGIME = {
    "val_chem_only": "R10",
    "val_strain_only": "R01",
    "val_both": "R00",
    "val_time": "RT",
}
SPLIT_LABELS = {
    "val_chem_only": "New compound",
    "val_strain_only": "New strain",
    "val_both": "Double unseen",
    "val_time": "Temporal validation",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz_prediction(
    run_dir: Path, split: str, split_ids: pd.Index, *, require_ids: bool
) -> np.ndarray:
    with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as payload:
        values = payload[split].astype(np.float32)
        if require_ids:
            ids = payload[f"{split}_sample_ids"].astype(str)
            if not np.array_equal(ids, split_ids.astype(str).to_numpy()):
                raise RuntimeError(f"prediction sample IDs/order differ: {run_dir} {split}")
    if values.shape[0] != len(split_ids) or not np.isfinite(values).all():
        raise RuntimeError(f"prediction surface differs: {run_dir} {split}")
    return values


def per_protein_r2(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    observed = mask.astype(bool)
    count = observed.sum(axis=0)
    safe_truth = np.where(observed, truth, 0.0)
    mean = np.divide(
        safe_truth.sum(axis=0),
        count,
        out=np.full(truth.shape[1], np.nan, np.float64),
        where=count > 0,
    )
    centered = np.where(observed, truth - mean[None, :], 0.0)
    tss = np.square(centered).sum(axis=0)
    residual = np.where(observed, prediction - truth, 0.0)
    sse = np.square(residual).sum(axis=0)
    return np.divide(
        tss - sse,
        tss,
        out=np.full(truth.shape[1], np.nan, np.float64),
        where=(count >= 2) & (tss > 0),
    )


def quartiles(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)


def finite_mean(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values)
    finite = np.isfinite(stacked)
    count = finite.sum(axis=0)
    return np.divide(
        np.where(finite, stacked, 0.0).sum(axis=0),
        count,
        out=np.full(stacked.shape[1], np.nan, np.float64),
        where=count > 0,
    )


def build_diagnostics(
    comparison_metrics: Path,
    bcr_runs_root: Path,
    v7_runs_root: Path,
    base_runs: Path,
    base_builder_path: Path,
    senior_script: Path,
    output_dir: Path,
    seeds: list[int],
) -> None:
    base_builder = load_module("goai_gap_base_builder", base_builder_path)
    senior = base_builder._load_senior_metric_module(senior_script)
    mean_dir = base_builder._latest_run(base_runs, "mean", 42)
    config = json.loads((mean_dir / "config.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(
        config["data"]["metadata_train_val"], low_memory=False
    ).set_index("sample_ID", verify_integrity=True)
    raw = (
        pd.read_csv(config["data"]["proteome_train_val"], low_memory=False)
        .set_index("sample_ID", verify_integrity=True)
        .reindex(metadata.index)
    )
    proteins = senior.retained_proteins(
        raw, metadata, float(config["data"]["missing_rate_threshold"])
    )
    truth_frame = np.log2(raw.loc[:, proteins].astype(np.float32))
    baselines, _ = senior.build_baselines(metadata, truth_frame, "released-validation")

    per_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    for split, regime in SPLIT_TO_REGIME.items():
        split_ids = metadata.index[metadata["split_final"].eq(split)]
        target_ids = baselines[regime]["target_ids"]
        target_positions = split_ids.get_indexer(target_ids)
        if (target_positions < 0).any():
            raise RuntimeError(f"target IDs are not a subset of split IDs: {split}")
        truth = truth_frame.loc[target_ids].to_numpy(np.float64)
        control_mask = baselines[regime]["prediction"].notna().to_numpy(bool)
        mask = np.isfinite(truth) & control_mask
        bcr_r2_seeds: list[np.ndarray] = []
        v7_r2_seeds: list[np.ndarray] = []
        for seed in seeds:
            bcr = load_npz_prediction(
                bcr_runs_root / f"seed_{seed}", split, split_ids, require_ids=True
            )[target_positions].astype(np.float64)
            v7_candidates = sorted(v7_runs_root.glob(f"biostate-seed{seed}-*"))
            if len(v7_candidates) != 1:
                raise RuntimeError(f"expected one V7 run for seed {seed}: {v7_candidates}")
            v7 = load_npz_prediction(
                v7_candidates[0], split, split_ids, require_ids=False
            )[target_positions].astype(np.float64)
            bcr_r2_seeds.append(per_protein_r2(bcr, truth, mask))
            v7_r2_seeds.append(per_protein_r2(v7, truth, mask))

        bcr_r2 = finite_mean(bcr_r2_seeds)
        v7_r2 = finite_mean(v7_r2_seeds)
        observed_count = mask.sum(axis=0)
        safe_truth = np.where(mask, truth, 0.0)
        truth_mean = np.divide(
            safe_truth.sum(axis=0),
            observed_count,
            out=np.full(len(proteins), np.nan),
            where=observed_count > 0,
        )
        truth_variance = np.divide(
            np.where(mask, np.square(truth - truth_mean[None, :]), 0.0).sum(axis=0),
            observed_count,
            out=np.full(len(proteins), np.nan),
            where=observed_count > 0,
        )
        frame = pd.DataFrame(
            {
                "split": split,
                "scenario": SPLIT_LABELS[split],
                "protein": proteins,
                "n_observed": observed_count,
                "observed_fraction": observed_count / len(target_ids),
                "truth_variance": truth_variance,
                "bcr_r2_mean": bcr_r2,
                "v7_r2_mean": v7_r2,
                "bcr_minus_v7_r2": bcr_r2 - v7_r2,
            }
        )
        finite = np.isfinite(frame[["bcr_r2_mean", "v7_r2_mean"]]).all(axis=1)
        frame["bcr_wins"] = finite & frame["bcr_minus_v7_r2"].gt(0)
        frame["observed_fraction_quartile"] = quartiles(frame["observed_fraction"])
        frame["truth_variance_quartile"] = quartiles(frame["truth_variance"])
        per_rows.append(frame)

        delta = frame.loc[finite, "bcr_minus_v7_r2"].to_numpy(np.float64)
        summary_rows.append(
            {
                "split": split,
                "scenario": SPLIT_LABELS[split],
                "n_evaluable_proteins": int(finite.sum()),
                "fraction_proteins_bcr_wins": float(np.mean(delta > 0)),
                "median_paired_protein_delta_r2": float(np.median(delta)),
                "mean_paired_protein_delta_r2": float(np.mean(delta)),
                "p10_paired_protein_delta_r2": float(np.quantile(delta, 0.10)),
                "p90_paired_protein_delta_r2": float(np.quantile(delta, 0.90)),
            }
        )
        for dimension in ("observed_fraction_quartile", "truth_variance_quartile"):
            for quartile, group in frame.loc[finite].groupby(dimension, sort=True):
                values = group["bcr_minus_v7_r2"].to_numpy(np.float64)
                slice_rows.append(
                    {
                        "split": split,
                        "scenario": SPLIT_LABELS[split],
                        "dimension": dimension,
                        "quartile": quartile,
                        "n_proteins": len(group),
                        "median_bcr_minus_v7_r2": float(np.median(values)),
                        "fraction_proteins_bcr_wins": float(np.mean(values > 0)),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_rows, ignore_index=True).to_csv(
        output_dir / "BCR_V7_PER_PROTEIN_PAIRED.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "BCR_V7_PAIRED_GAP_SUMMARY.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(slice_rows).to_csv(
        output_dir / "BCR_V7_GAP_BY_PROTEIN_SLICE.csv", index=False, lineterminator="\n"
    )

    histories: list[pd.DataFrame] = []
    for seed in seeds:
        history = pd.read_csv(bcr_runs_root / f"seed_{seed}" / "training_history.csv")
        for stage in ("A", "B", "C"):
            row = history.loc[history["stage"].eq(stage)].iloc[-1].copy()
            row["seed"] = seed
            histories.append(pd.DataFrame([row]))
    history_summary = pd.concat(histories, ignore_index=True)
    history_summary.to_csv(
        output_dir / "BCR_FINAL_STAGE_LOSSES.csv", index=False, lineterminator="\n"
    )

    metrics = pd.read_csv(comparison_metrics)
    focal = metrics.loc[
        metrics["model"].isin(
            ["semantic_bcr_v1_released_refit", "biostate_v7_proteome"]
        )
    ].copy()
    focal.groupby(["model", "split"], as_index=False)[
        [
            "rmse_log2",
            "mae_log2",
            "global_r2",
            "sample_pcc_median",
            "protein_pcc_median",
            "protein_r2_median",
        ]
    ].agg(["mean", "std"]).to_csv(output_dir / "BCR_V7_SIX_METRIC_SUMMARY.csv")
    print(output_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-metrics", type=Path, required=True)
    parser.add_argument("--bcr-runs-root", type=Path, required=True)
    parser.add_argument("--v7-runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-runs", type=Path, default=Path("runs/basic_descriptor_mlp"))
    parser.add_argument(
        "--base-builder",
        type=Path,
        default=Path("experiments/basic_descriptor_mlp/scripts/build_four_model_metrics.py"),
    )
    parser.add_argument(
        "--senior-script",
        type=Path,
        default=Path("/home/chenyuming/Project/go-ai/scripts/evaluate_senior_matched_control_r2.py"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    build_diagnostics(
        args.comparison_metrics,
        args.bcr_runs_root,
        args.v7_runs_root,
        args.base_runs,
        args.base_builder,
        args.senior_script,
        args.output_dir,
        args.seeds,
    )


if __name__ == "__main__":
    main()
