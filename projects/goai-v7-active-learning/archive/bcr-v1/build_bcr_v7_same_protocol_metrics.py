#!/usr/bin/env python3
"""Append released-validation BCR refits to the exact V7 common-mask table."""

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
MODEL_NAME = "semantic_bcr_v1_released_refit"
EXPECTED_EXISTING_MODELS = {
    "mean",
    "matched_control_new",
    "zero",
    "real",
    "biostate_v7_proteome",
}
METRIC_COLUMNS = [
    "rmse_log2",
    "mae_log2",
    "global_r2",
    "sample_pcc_median",
    "protein_pcc_median",
    "protein_r2_median",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prediction_frame(
    run_dir: Path, split: str, expected_ids: pd.Index, proteins: list[str]
) -> pd.DataFrame:
    completion_path = run_dir / "completed.json"
    if not completion_path.is_file():
        raise FileNotFoundError(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE":
        raise RuntimeError(f"BCR run is not complete: {run_dir}")
    with np.load(run_dir / "validation_predictions.npz", allow_pickle=False) as payload:
        values = payload[split].astype(np.float32)
        ids = payload[f"{split}_sample_ids"].astype(str)
    expected = expected_ids.astype(str).to_numpy()
    if not np.array_equal(ids, expected):
        raise RuntimeError(f"BCR prediction sample IDs/order differ: {run_dir} {split}")
    if values.shape != (len(expected_ids), len(proteins)):
        raise RuntimeError(f"BCR prediction shape differs: {run_dir} {split} {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"BCR prediction contains non-finite values: {run_dir} {split}")
    return pd.DataFrame(values, index=expected_ids, columns=proteins)


def build(
    existing_metrics: Path,
    bcr_runs_root: Path,
    base_runs: Path,
    base_builder_path: Path,
    senior_script: Path,
    output: Path,
    seeds: list[int],
) -> None:
    existing = pd.read_csv(existing_metrics)
    if len(existing) != 44 or set(existing["model"]) != EXPECTED_EXISTING_MODELS:
        raise RuntimeError("existing five-model V7 metric surface differs")
    if set(existing["split"]) != set(SPLIT_TO_REGIME):
        raise RuntimeError("existing V7 validation splits differ")

    base_builder = load_module("goai_base_metric_builder_for_bcr", base_builder_path)
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
    truth = np.log2(raw.loc[:, proteins].astype(np.float32))
    baselines, _ = senior.build_baselines(metadata, truth, "released-validation")

    rows: list[dict[str, object]] = []
    for split, regime in SPLIT_TO_REGIME.items():
        baseline = baselines[regime]
        target_ids = baseline["target_ids"]
        comparable_truth = truth.loc[target_ids].where(
            baseline["prediction"].notna()
        )
        split_ids = metadata.index[metadata["split_final"].eq(split)]
        reference = existing.loc[existing["split"].eq(split)].iloc[0]
        expected_observed = int(reference["n_observed_values"])
        expected_samples = int(reference["n_samples"])
        for seed in seeds:
            run_dir = bcr_runs_root / f"seed_{seed}"
            prediction = prediction_frame(run_dir, split, split_ids, proteins).loc[
                target_ids
            ]
            metrics = base_builder._six_metrics(prediction, comparable_truth, senior)
            if int(metrics["n_samples"]) != expected_samples:
                raise RuntimeError(f"BCR common-mask sample count differs: {split} seed={seed}")
            if int(metrics["n_observed_values"]) != expected_observed:
                raise RuntimeError(f"BCR common-mask observed count differs: {split} seed={seed}")
            rows.append(
                {
                    "model": MODEL_NAME,
                    "seed": seed,
                    "source": str(run_dir.resolve()),
                    "split": split,
                    "subset": "new_definition_common_mask",
                    **metrics,
                }
            )

    combined = pd.concat((existing, pd.DataFrame(rows)), ignore_index=True)
    if len(combined) != 56:
        raise RuntimeError(f"six-model metric table row count differs: {len(combined)}")
    columns = [
        "model",
        "seed",
        "source",
        "split",
        "subset",
        "n_samples",
        "n_observed_values",
        "coverage",
        *METRIC_COLUMNS,
        "n_evaluable_proteins",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.loc[:, columns].to_csv(output, index=False, lineterminator="\n")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-metrics", type=Path, required=True)
    parser.add_argument("--bcr-runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    build(
        args.existing_metrics,
        args.bcr_runs_root,
        args.base_runs,
        args.base_builder,
        args.senior_script,
        args.output,
        args.seeds,
    )


if __name__ == "__main__":
    main()
