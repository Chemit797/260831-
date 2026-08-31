#!/usr/bin/env python3
"""Build one comparable six-metric table for the four requested baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_TO_REGIME = {
    "val_chem_only": "R10",
    "val_strain_only": "R01",
    "val_both": "R00",
    "val_time": "RT",
}
METRICS = [
    "rmse_log2",
    "mae_log2",
    "global_r2",
    "sample_pcc_median",
    "protein_pcc_median",
    "protein_r2_median",
]


def _latest_run(runs_root: Path, variant: str, seed: int) -> Path:
    matches = sorted(runs_root.glob(f"{variant}-seed{seed}-*"))
    if not matches:
        raise FileNotFoundError(f"No run found for {variant}, seed {seed}")
    return matches[-1]


def _load_senior_metric_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("senior_matched_control", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load matched-control implementation: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _median_pcc(prediction: np.ndarray, truth: np.ndarray, axis: int) -> float:
    if axis == 1:
        prediction, truth = prediction.T, truth.T
    mask = np.isfinite(prediction) & np.isfinite(truth)
    count = mask.sum(axis=0)
    safe_prediction = np.where(mask, prediction, 0.0)
    safe_truth = np.where(mask, truth, 0.0)
    pred_mean = np.divide(safe_prediction.sum(axis=0), count, out=np.full(mask.shape[1], np.nan), where=count > 0)
    truth_mean = np.divide(safe_truth.sum(axis=0), count, out=np.full(mask.shape[1], np.nan), where=count > 0)
    pred_centered = np.where(mask, prediction - pred_mean, 0.0)
    truth_centered = np.where(mask, truth - truth_mean, 0.0)
    numerator = (pred_centered * truth_centered).sum(axis=0)
    denominator = np.sqrt((pred_centered**2).sum(axis=0) * (truth_centered**2).sum(axis=0))
    pcc = np.divide(numerator, denominator, out=np.full(mask.shape[1], np.nan), where=(count >= 2) & (denominator > 0))
    return float(np.nanmedian(pcc)) if np.isfinite(pcc).any() else float("nan")


def _six_metrics(prediction: pd.DataFrame, truth: pd.DataFrame, senior) -> dict[str, float | int]:
    pred = prediction.to_numpy(dtype=np.float64)
    actual = truth.to_numpy(dtype=np.float64)
    common = np.isfinite(pred) & np.isfinite(actual)
    error = pred[common] - actual[common]
    target = actual[common]
    total = np.sum((target - target.mean()) ** 2)
    proteins = senior.protein_metrics(pred, actual)
    protein_pcc = proteins["protein_pcc"].to_numpy()
    return {
        "n_samples": int(len(truth)),
        "n_observed_values": int(common.sum()),
        "coverage": float(common.mean()),
        "rmse_log2": float(np.sqrt(np.mean(error**2))),
        "mae_log2": float(np.mean(np.abs(error))),
        "global_r2": float(1.0 - np.sum(error**2) / total),
        "sample_pcc_median": _median_pcc(pred.T, actual.T, axis=0),
        "protein_pcc_median": float(np.nanmedian(protein_pcc)) if np.isfinite(protein_pcc).any() else float("nan"),
        "protein_r2_median": float(np.nanmedian(proteins["protein_r2"])),
        "n_evaluable_proteins": int(proteins["protein_r2"].notna().sum()),
    }


def _prediction_frame(run_dir: Path, split: str, split_ids: pd.Index, proteins: list[str]) -> pd.DataFrame:
    with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as payload:
        values = payload[split].astype(np.float32)
    if values.shape != (len(split_ids), len(proteins)):
        raise ValueError(f"Prediction shape mismatch in {run_dir}: {values.shape}")
    return pd.DataFrame(values, index=split_ids, columns=proteins)


def _build_comparable_rows(
    runs_root: Path,
    config_path: Path,
    senior_script: Path,
    seeds: list[int],
) -> pd.DataFrame:
    senior = _load_senior_metric_module(senior_script)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    metadata = pd.read_csv(data_config["metadata_train_val"], low_memory=False).set_index("sample_ID", verify_integrity=True)
    raw = pd.read_csv(data_config["proteome_train_val"], low_memory=False).set_index("sample_ID", verify_integrity=True).reindex(metadata.index)
    proteins = senior.retained_proteins(raw, metadata, float(data_config["missing_rate_threshold"]))
    truth = np.log2(raw.loc[:, proteins].astype(np.float32))
    baselines, _ = senior.build_baselines(metadata, truth, "released-validation")

    rows: list[dict[str, object]] = []
    for split, regime in SPLIT_TO_REGIME.items():
        baseline = baselines[regime]
        target_ids = baseline["target_ids"]
        control_prediction = baseline["prediction"]
        comparable_truth = truth.loc[target_ids].where(control_prediction.notna())
        metrics = _six_metrics(control_prediction, comparable_truth, senior)
        rows.append(
            {
                "model": "matched_control_new",
                "seed": np.nan,
                "source": str(senior_script),
                "split": split,
                "subset": "new_definition_common_mask",
                **metrics,
            }
        )

        split_ids = metadata.index[metadata["split_final"].eq(split)]
        for variant, variant_seeds in (("mean", [42]), ("zero", seeds), ("real", seeds)):
            for seed in variant_seeds:
                run_dir = _latest_run(runs_root, variant, seed)
                prediction = _prediction_frame(run_dir, split, split_ids, proteins).loc[target_ids]
                metrics = _six_metrics(prediction, comparable_truth, senior)
                rows.append(
                    {
                        "model": variant,
                        "seed": seed,
                        "source": str(run_dir),
                        "split": split,
                        "subset": "new_definition_common_mask",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_table(runs_root: Path, senior_script: Path, output: Path, seeds: list[int]) -> None:
    mean_dir = _latest_run(runs_root, "mean", 42)
    table = _build_comparable_rows(runs_root, mean_dir / "config.json", senior_script, seeds)
    columns = ["model", "seed", "source", "split", "subset", "n_samples", "n_observed_values", "coverage", *METRICS, "n_evaluable_proteins"]
    output.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, columns].to_csv(output, index=False)
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/basic_descriptor_mlp"))
    parser.add_argument(
        "--senior-script",
        type=Path,
        default=Path("/home/chenyuming/Project/go-ai/scripts/evaluate_senior_matched_control_r2.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    build_table(args.runs_root, args.senior_script, args.output, args.seeds)


if __name__ == "__main__":
    main()
