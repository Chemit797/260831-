#!/usr/bin/env python3
"""Zero-retrain released-validation BCR component ablations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

import released_validation as released
import run as core
from model import GOAISemanticBCRV1


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


def evaluate(
    runs_root: Path,
    base_runs: Path,
    base_builder_path: Path,
    senior_script: Path,
    output: Path,
    device_name: str,
    seeds: list[int],
) -> None:
    released.validate_inputs()
    prepared = released.prepare_released_validation()
    device = core.resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    base_builder = load_module("goai_ablation_base_builder", base_builder_path)
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
    eval_position = {sample_id: index for index, sample_id in enumerate(prepared.eval_ids)}

    rows: list[dict[str, object]] = []
    for seed in seeds:
        checkpoint = torch.load(
            runs_root / f"seed_{seed}" / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = GOAISemanticBCRV1(
            n_medium=len(prepared.feature_state.medium_categories),
            chemical_dim=prepared.fit_features.chemical.shape[1],
            n_proteins=len(prepared.proteins),
            cal_center=torch.from_numpy(prepared.cal_center),
            cal_decoder_scaled=torch.from_numpy(prepared.cal_decoder_scaled),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
        surfaces = core.predict_bcr(
            model,
            prepared.eval_features,
            target_mean=prepared.target_mean,
            target_std=prepared.target_std,
            device=device,
        )
        model = model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for split, regime in SPLIT_TO_REGIME.items():
            split_ids = metadata.index[metadata["split_final"].eq(split)]
            target_ids = baselines[regime]["target_ids"]
            selected = np.asarray([eval_position[str(sample_id)] for sample_id in target_ids])
            comparable_truth = truth.loc[target_ids].where(
                baselines[regime]["prediction"].notna()
            )
            for surface_name, values in surfaces.items():
                prediction = pd.DataFrame(
                    values[selected], index=target_ids, columns=proteins
                )
                metrics = base_builder._six_metrics(prediction, comparable_truth, senior)
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "surface": surface_name.upper().replace("_", "-"),
                        **metrics,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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
    evaluate(
        args.runs_root,
        args.base_runs,
        args.base_builder,
        args.senior_script,
        args.output,
        args.device,
        args.seeds,
    )


if __name__ == "__main__":
    main()
