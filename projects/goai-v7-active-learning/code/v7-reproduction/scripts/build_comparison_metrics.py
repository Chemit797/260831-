#!/usr/bin/env python3
"""Add BioState-Readout runs to the exact existing four-model comparison mask."""

from __future__ import annotations

import argparse
import importlib.util
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latest_run(runs_root: Path, seed: int) -> Path:
    matches = sorted(runs_root.glob(f"biostate-seed{seed}-*"))
    if not matches:
        raise FileNotFoundError(f"no BioState run for seed {seed}")
    return matches[-1]


def build(
    base_runs: Path,
    biostate_runs: Path,
    base_builder_path: Path,
    senior_script: Path,
    output: Path,
    seeds: list[int],
) -> None:
    base_builder = load_module("goai_four_model_builder", base_builder_path)
    mean_dir = base_builder._latest_run(base_runs, "mean", 42)
    base = base_builder._build_comparable_rows(base_runs, mean_dir / "config.json", senior_script, seeds)
    senior = base_builder._load_senior_metric_module(senior_script)
    config = __import__("json").loads((mean_dir / "config.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(config["data"]["metadata_train_val"], low_memory=False).set_index(
        "sample_ID", verify_integrity=True
    )
    raw = pd.read_csv(config["data"]["proteome_train_val"], low_memory=False).set_index(
        "sample_ID", verify_integrity=True
    ).reindex(metadata.index)
    proteins = senior.retained_proteins(raw, metadata, float(config["data"]["missing_rate_threshold"]))
    truth = np.log2(raw.loc[:, proteins].astype(np.float32))
    baselines, _ = senior.build_baselines(metadata, truth, "released-validation")

    rows: list[dict[str, object]] = []
    for split, regime in SPLIT_TO_REGIME.items():
        target_ids = baselines[regime]["target_ids"]
        comparable_truth = truth.loc[target_ids].where(baselines[regime]["prediction"].notna())
        split_ids = metadata.index[metadata["split_final"].eq(split)]
        for seed in seeds:
            run_dir = latest_run(biostate_runs, seed)
            prediction = base_builder._prediction_frame(run_dir, split, split_ids, proteins).loc[target_ids]
            metrics = base_builder._six_metrics(prediction, comparable_truth, senior)
            rows.append(
                {
                    "model": "biostate_v7_proteome",
                    "seed": seed,
                    "source": str(run_dir),
                    "split": split,
                    "subset": "new_definition_common_mask",
                    **metrics,
                }
            )
    combined = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runs", type=Path, default=Path("runs/basic_descriptor_mlp"))
    parser.add_argument(
        "--biostate-runs", type=Path, default=Path("runs/proteome_biostate_readout_v7_reproduction")
    )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    build(args.base_runs, args.biostate_runs, args.base_builder, args.senior_script, args.output, args.seeds)


if __name__ == "__main__":
    main()

