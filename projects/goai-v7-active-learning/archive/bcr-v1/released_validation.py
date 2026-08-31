#!/usr/bin/env python3
"""Train upgraded BCR on all released train rows and predict released validation.

This is the same fixed BCR recipe used by the internal-OOF experiment, refitted
on all ``split_final == train`` rows so it can be scored on the exact released
validation surface used by the BioState-Readout V7 reproduction.  The frozen
production CalV2 source+instrument artifact is legal here because released
validation is an external holdout; it remains forbidden for internal OOF.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

import run as core


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_SHA256_AT_START = core.sha256_file(SCRIPT_PATH)
DEFAULT_RUN_ROOT = ROOT / "runs/goai_semantic_bcr_v1/released_validation_v7_comparison"
VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")
PRODUCTION_CALV2_ROOT = (
    ROOT
    / "runs/calibration_embedding_v2/v2-fullstate-final-20260822-040500/"
    "backbone_SI/final_refit/seed_42/production_artifact"
)
PRODUCTION_CALV2_CHECKPOINT = PRODUCTION_CALV2_ROOT / "checkpoint.pt"
PRODUCTION_CALV2_HASHES = {
    PRODUCTION_CALV2_CHECKPOINT: "4bff772ece15ac8afa80e0c6f06c9224bf328ea2dd699851dc6d0c971564f5ed",
    PRODUCTION_CALV2_ROOT / "canonical_decoder.npy": "4457e306089109a35e91f1cad7957d558d88c28a5eb169353b53986d4cb89fd0",
    PRODUCTION_CALV2_ROOT / "raw_decoder.npy": "225e9f1a1fdd94511784fd5c27034f1f692458374ad030bfb808a8a6074e1d4d",
    PRODUCTION_CALV2_ROOT / "singular_values.npy": "3b83b744c48ef2d24971dd4029c41b4e0cedeef1d429748f78cc96baac590657",
    PRODUCTION_CALV2_ROOT / "protein_standardization.csv": "c1ffc3f6acdaa05e64a1a273f19df907f7dc8f51a6db195e42e1c6b0a7d1a485",
    PRODUCTION_CALV2_ROOT / "artifact_manifest.json": "aafaca94fdef2b74dad3f00560fed072330c3f24199598f3440709071a291032",
}


def assert_source_unchanged() -> None:
    core.assert_code_unchanged()
    if core.sha256_file(SCRIPT_PATH) != SCRIPT_SHA256_AT_START:
        raise RuntimeError("released_validation.py changed after process start")


def validate_inputs() -> dict[str, str]:
    observed = core.validate_expected_hashes()
    for path, expected in PRODUCTION_CALV2_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = core.sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"production CalV2 hash mismatch: {path}")
        observed[str(path)] = digest
    observed[str(Path(core.__file__).resolve())] = core.sha256_file(Path(core.__file__).resolve())
    observed[str(Path(core.__file__).with_name("model.py").resolve())] = core.sha256_file(
        Path(core.__file__).with_name("model.py").resolve()
    )
    observed[str(SCRIPT_PATH)] = SCRIPT_SHA256_AT_START
    return observed


def prepare_released_validation() -> core.FoldPrepared:
    metadata = core.load_metadata()
    proteins = core.load_proteins()
    checkpoint = torch.load(
        PRODUCTION_CALV2_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if checkpoint.get("artifact_version") != "calibration_embedding_v2_artifact_v1":
        raise RuntimeError("production CalV2 artifact version differs")
    if tuple(map(str, checkpoint.get("proteins", ()))) != proteins:
        raise RuntimeError("production CalV2/main protein order differs")

    split = metadata["split_final"].astype(str)
    fit_ids = tuple(metadata.index[split.eq("train")].astype(str))
    eval_ids = tuple(metadata.index[split.isin(VALIDATION_SPLITS)].astype(str))
    if len(fit_ids) != 5920 or len(eval_ids) != 3038:
        raise RuntimeError(
            f"released train/validation row counts differ: fit={len(fit_ids)} eval={len(eval_ids)}"
        )
    if set(fit_ids) & set(eval_ids):
        raise RuntimeError("released train/validation overlap is nonzero")
    fit_metadata = metadata.loc[list(fit_ids)].copy()
    eval_metadata = metadata.loc[list(eval_ids)].copy()

    y_log2 = core.load_log2_proteome(metadata, proteins)
    fit_y = y_log2.loc[list(fit_ids)].to_numpy(np.float32)
    eval_y = y_log2.loc[list(eval_ids)].to_numpy(np.float32)
    target_mean, target_std = core.fold_target_statistics(fit_y)
    fit_observed = np.isfinite(fit_y)
    fit_target = np.zeros_like(fit_y, np.float32)
    standardized = (fit_y - target_mean[None, :]) / target_std[None, :]
    fit_target[fit_observed] = standardized[fit_observed]

    all_metadata = pd.concat((fit_metadata, eval_metadata), axis=0)
    z_all, cal_decoder, cal_protein_std = core.encode_calv2(all_metadata, checkpoint)
    z_fit = z_all[: len(fit_metadata)]
    z_eval = z_all[len(fit_metadata) :]
    cal_center = z_fit.astype(np.float64).mean(axis=0).astype(np.float32)
    centered_error = float(
        np.max(np.abs((z_fit.astype(np.float64) - cal_center).mean(axis=0)))
    )
    if centered_error > 2e-5:
        raise RuntimeError(f"full-train CalV2 gauge failed: {centered_error}")
    cal_decoder_scaled = (
        cal_decoder * (cal_protein_std / target_std)[:, None]
    ).astype(np.float32)

    feature_state = core.fit_feature_state(fit_metadata)
    strain_table, chemical_table = core.load_descriptors()
    fit_features = core.build_feature_arrays(
        fit_metadata,
        state=feature_state,
        strain_table=strain_table,
        chemical_table=chemical_table,
        z_cal=z_fit,
        cal_center=cal_center,
    )
    eval_features = core.build_feature_arrays(
        eval_metadata,
        state=feature_state,
        strain_table=strain_table,
        chemical_table=chemical_table,
        z_cal=z_eval,
        cal_center=cal_center,
    )

    control_map = core.build_control_map(fit_metadata, fit_y)
    response_positions, local_response, local_mask, correction_max = (
        core.build_response_targets(
            fit_metadata,
            fit_y,
            target_std,
            z_fit,
            cal_decoder,
            cal_protein_std,
            control_map,
        )
    )
    local_oracle, local_has_match = core.build_oracle_predictions(
        eval_metadata, control_map
    )

    core._calv2_python_paths()
    from semantic_feature_engineering_v1.controls import (  # type: ignore
        exact_control_predictions,
        training_fc_targets,
    )

    reference_fc = training_fc_targets(metadata, y_log2, fit_ids, target_std)
    reference_positions = np.flatnonzero(reference_fc.mask.any(axis=1))
    if not np.array_equal(reference_positions, response_positions):
        raise RuntimeError("full-train matched-response row surface differs from frozen helper")
    if not np.array_equal(reference_fc.mask[response_positions].astype(bool), local_mask):
        raise RuntimeError("full-train matched-response mask differs from frozen helper")
    if not np.allclose(
        reference_fc.values[response_positions], local_response, rtol=0.0, atol=1e-5
    ):
        raise RuntimeError("full-train matched-response target differs from frozen helper")
    reference_oracle, reference_has_match = exact_control_predictions(
        metadata, y_log2, eval_ids, fit_ids
    )
    if not np.array_equal(reference_has_match.to_numpy(bool), local_has_match):
        raise RuntimeError("full-train matched-control row surface differs from frozen helper")
    if not np.allclose(
        reference_oracle.to_numpy(np.float32),
        local_oracle,
        equal_nan=True,
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError("full-train matched-control values differ from frozen helper")

    response_target = reference_fc.values[response_positions].astype(np.float32)
    response_mask = reference_fc.mask[response_positions].astype(bool)
    oracle = reference_oracle.to_numpy(np.float32)
    oracle_has_match = reference_has_match.to_numpy(bool)
    control_fit_positions = np.flatnonzero(
        fit_metadata["perturbation_no_concentration"]
        .astype(str)
        .isin(core.CONTROL_CHEMICALS)
        .to_numpy()
    ).astype(np.int64)
    if len(control_fit_positions) != 751:
        raise RuntimeError(f"released train control count differs: {len(control_fit_positions)}")

    return core.FoldPrepared(
        scenario="released_validation",
        fold=-1,
        fit_ids=fit_ids,
        eval_ids=eval_ids,
        proteins=proteins,
        fit_features=fit_features,
        eval_features=eval_features,
        response_fit_positions=response_positions,
        response_target=response_target,
        response_mask=response_mask,
        target_mean=target_mean,
        target_std=target_std,
        fit_target=fit_target,
        fit_mask=fit_observed,
        eval_truth=eval_y,
        eval_truth_mask=np.isfinite(eval_y),
        oracle_prediction=oracle,
        oracle_has_match=oracle_has_match,
        control_fit_positions=control_fit_positions,
        cal_center=cal_center,
        cal_decoder_scaled=cal_decoder_scaled,
        feature_state=feature_state,
        calv2_checkpoint_path=str(PRODUCTION_CALV2_CHECKPOINT),
        calv2_checkpoint_sha256=PRODUCTION_CALV2_HASHES[PRODUCTION_CALV2_CHECKPOINT],
        response_correction_max_abs=correction_max,
    )


def checkpoint_contract(
    prepared: core.FoldPrepared, seed: int, observed_inputs: dict[str, str]
) -> dict[str, Any]:
    return {
        "format": "goai.semantic_bcr_v1.released_validation_checkpoint.v1",
        "model": "GOAI-SEMANTIC-BCR-V1-RELEASED-REFIT",
        "seed": seed,
        "fit_surface": "split_final == train",
        "evaluation_surface": list(VALIDATION_SPLITS),
        "fit_sample_count": len(prepared.fit_ids),
        "eval_sample_count": len(prepared.eval_ids),
        "fit_sample_ids_sha256": core.ordered_json_hash(prepared.fit_ids),
        "eval_sample_ids_sha256": core.ordered_json_hash(prepared.eval_ids),
        "protein_count": len(prepared.proteins),
        "protein_order_sha256": core.ordered_json_hash(prepared.proteins),
        "training_recipe": "same fixed Stage A/B/C recipe as internal OOF BCR V1",
        "released_validation_used_for_fitting_or_selection": False,
        "production_calv2_usage": "LEGAL_EXTERNAL_HOLDOUT_REFIT_ONLY",
        "production_calv2_seed": 42,
        "calv2_checkpoint_path": prepared.calv2_checkpoint_path,
        "calv2_checkpoint_sha256": prepared.calv2_checkpoint_sha256,
        "op3_frozen": True,
        "strain_descriptor_frozen": True,
        "strain_descriptor_status": (
            "HISTORICAL_RAW4096_EXECUTION_ASSUMPTION_NOT_FORMALLY_VALIDATED"
        ),
        "plate_well_model_inputs": False,
        "plate_used_in_response_matching_key_only": True,
        "response_contract": (
            "TASK_LOCAL_RESOLUTION_ATTACHED_WATER_DMSO_PLUS_"
            "HASH_LOCKED_EXECUTED_8KEY_CONTRACT"
        ),
        "feature_state": asdict(prepared.feature_state),
        "target_standardization": "train-only per-protein log2 mean/std; std floor 0.1",
        "target_mean": prepared.target_mean,
        "target_std": prepared.target_std,
        "calibration_center": prepared.cal_center,
        "response_calv2_correction_max_abs": prepared.response_correction_max_abs,
        "source_hashes": observed_inputs,
    }


def completed_artifacts_valid(output: Path, completion: dict[str, Any], seed: int) -> bool:
    if completion.get("seed") != seed:
        return False
    required = {
        "checkpoint.pt": completion.get("checkpoint_sha256"),
        "training_history.csv": completion.get("training_history_sha256"),
        "validation_predictions.npz": completion.get("validation_predictions_sha256"),
        "contract.json": completion.get("contract_sha256"),
    }
    return all(
        isinstance(expected, str)
        and (output / name).is_file()
        and core.sha256_file(output / name) == expected
        for name, expected in required.items()
    )


def train(seed: int, device_name: str, output: Path) -> None:
    if seed not in (42, 43, 44):
        raise ValueError("released-validation comparison is frozen to seeds 42/43/44")
    assert_source_unchanged()
    observed_inputs = validate_inputs()
    completion_path = output / "completed.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completed_artifacts_valid(output, completion, seed):
            print(f"seed={seed} exact completed released-validation refit reused", flush=True)
            return
        raise RuntimeError("released-validation completion exists but artifacts differ")

    free_gb = shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free / 2**30
    if free_gb < 20:
        raise RuntimeError(f"training requires >=20 GiB free; observed={free_gb:.1f}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    prepared = prepare_released_validation()
    device = core.resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    core.SEED = seed
    core.set_seed(seed)
    print(
        f"seed={seed} fit={len(prepared.fit_ids)} eval={len(prepared.eval_ids)} "
        f"controls={len(prepared.control_fit_positions)} "
        f"matched_response={len(prepared.response_fit_positions)} device={device}",
        flush=True,
    )
    model, history = core.train_bcr(prepared, device=device)
    assert_source_unchanged()
    predictions = core.predict_bcr(
        model,
        prepared.eval_features,
        target_mean=prepared.target_mean,
        target_std=prepared.target_std,
        device=device,
    )["full"]
    model = model.cpu()

    contract = checkpoint_contract(prepared, seed, observed_inputs)
    contract_path = output / "contract.json"
    core.write_json(contract_path, contract)
    checkpoint_path = output / "checkpoint.pt"
    core.torch_save_atomic(
        checkpoint_path,
        {
            "contract": contract,
            "architecture": (
                "CellState(RAW4096,medium16,temp16,time16)->128; "
                "Background 128-512-4422; frozen production CalV2 S+I 12D; "
                "OP3 64-128 adapter; Response 256-512-512-4422"
            ),
            "model_state_dict": model.state_dict(),
        },
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    history_path = output / "training_history.csv"
    core.write_csv(history_path, history)
    position = {sample_id: index for index, sample_id in enumerate(prepared.eval_ids)}
    metadata = core.load_metadata()
    payload: dict[str, np.ndarray] = {}
    for split in VALIDATION_SPLITS:
        ids = tuple(metadata.index[metadata["split_final"].astype(str).eq(split)].astype(str))
        selected = np.asarray([position[sample_id] for sample_id in ids], np.int64)
        payload[split] = predictions[selected].astype(np.float32)
        payload[f"{split}_sample_ids"] = np.asarray(ids, dtype=np.str_)
    predictions_path = output / "validation_predictions.npz"
    core.npz_save_atomic(predictions_path, **payload)

    assert_source_unchanged()
    completion = {
        "format": "goai.semantic_bcr_v1.released_validation_completion.v1",
        "status": "COMPLETE",
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "fit_rows": len(prepared.fit_ids),
        "eval_rows": len(prepared.eval_ids),
        "checkpoint_sha256": core.sha256_file(checkpoint_path),
        "training_history_sha256": core.sha256_file(history_path),
        "validation_predictions_sha256": core.sha256_file(predictions_path),
        "contract_sha256": core.sha256_file(contract_path),
        "runner_sha256": SCRIPT_SHA256_AT_START,
        "base_run_sha256": core.RUN_CODE_SHA256_AT_START,
        "model_sha256": core.MODEL_CODE_SHA256_AT_START,
    }
    core.write_json(completion_path, completion)
    print(
        f"seed={seed} COMPLETE elapsed={completion['elapsed_seconds']:.1f}s "
        f"checkpoint={completion['checkpoint_sha256'][:12]}",
        flush=True,
    )


def preflight() -> None:
    observed = validate_inputs()
    prepared = prepare_released_validation()
    result = {
        "status": "PASS",
        "fit_rows": len(prepared.fit_ids),
        "eval_rows": len(prepared.eval_ids),
        "controls": len(prepared.control_fit_positions),
        "matched_response_rows": len(prepared.response_fit_positions),
        "proteins": len(prepared.proteins),
        "flat_input_dim": prepared.fit_features.flat.shape[1],
        "production_calv2_checkpoint_sha256": prepared.calv2_checkpoint_sha256,
        "response_correction_max_abs": prepared.response_correction_max_abs,
        "input_file_count": len(observed),
    }
    print(json.dumps(core.safe_json(result), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        preflight()
        return
    output = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (DEFAULT_RUN_ROOT / f"seed_{args.seed}").resolve()
    )
    train(args.seed, args.device, output)


if __name__ == "__main__":
    main()
