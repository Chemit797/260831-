"""Executable controller for the frozen GOAI active-learning v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .acquisition import VALID_STRATEGIES, select_batch
from .audit import write_audit_outputs
from .data import (
    CONDITION_ID,
    GROUP_FIELDS,
    INTERPOLATION_SPLIT,
    PROTOCOL_VERSION,
    VALIDATION_SPLITS,
    GroupedDataset,
    PoolFeatureEncoder,
    load_grouped_dataset,
)
from .metrics import budget_to_target, normalized_trapezoidal_aulc, score_response
from .model import ModelSettings, ResponseFit, fit_response_model
from .simulator import (
    BudgetSchedule,
    PoolState,
    RetrospectiveOracle,
    RoundReceipt,
)


MODEL_ID = "GOAI-AL-V2-PILOT-01"
GLOBAL_SEED = 42
FORMAL_INITIAL_BUDGET = 128
FORMAL_BATCH_SIZE = 128
FORMAL_CHECKPOINTS = (128, 256, 512, 1024)
FORMAL_EPOCHS = 80
FORMAL_MC_PASSES = 8
SMOKE_INITIAL_BUDGET = 32
SMOKE_BATCH_SIZE = 32
SMOKE_CHECKPOINTS = (32, 64, 96)
SMOKE_EPOCHS = 2
SMOKE_MC_PASSES = 2
STRATEGIES = ("random", "coreset", "uncertainty")
TARGET_FRACTION = 0.80
CURVE_METRICS = (
    "delta_skill_zero",
    "condition_pcc_median",
    "protein_r2_median",
)
METRIC_DIRECTIONS = {
    "delta_rmse": False,
    "delta_mae": False,
    "delta_skill_zero": True,
    "pooled_delta_pcc": True,
    "condition_pcc_median": True,
    "protein_pcc_median": True,
    "protein_r2_median": True,
    "protein_r2_mean": True,
    "protein_r2_positive_fraction": True,
}
EXPECTED_ARTIFACTS = (
    "active_metrics.csv",
    "acquisitions.csv",
    "full_reference_metrics.csv",
    "model_fit_receipts.csv",
    "representation_metrics.csv",
    "split_assignments.csv",
    "analysis_summary.json",
    "data_audit.json",
    "control_vehicle_sensitivity.csv",
    "tensor_coverage.csv",
    "low_rank_spectrum.csv",
    "learning_curve_delta_skill_zero.png",
    "learning_curve_condition_pcc_median.png",
    "learning_curve_protein_r2_median.png",
    "round_receipts/",
    "manifest.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_ids(ids: Sequence[str]) -> str:
    return _hash_payload(tuple(ids))


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary_name, format="png", dpi=180)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(figure)


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping")
    for section in ("data", "model", "protocol", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
    return config


def _configured_path(config_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_output(
    config_path: Path,
    config: Mapping[str, object],
    output_suffix: str | None,
    output_dir: Path | None,
) -> Path:
    if (output_suffix is None) == (output_dir is None):
        raise ValueError("Specify exactly one of --output-suffix or --output-dir")
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    assert output_suffix is not None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_suffix):
        raise ValueError("--output-suffix must be a nonempty path-safe identifier")
    runtime = dict(config["runtime"])
    base = _configured_path(config_path, runtime["output_dir"])
    return base.parent / f"{base.name}_{output_suffix}"


def _reserve_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output target exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing nonempty existing output target: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    _fsync_directory(path.parent)


def _model_seed(global_seed: int, current_budget: int) -> int:
    """A strategy-independent seed derived only from seed and current budget."""

    if global_seed < 0 or current_budget <= 0:
        raise ValueError("global_seed must be nonnegative and current_budget positive")
    return int((global_seed * 1_000_003 + current_budget) % (2**31 - 1))


def _acquisition_seed(global_seed: int, current_budget: int) -> int:
    return int((global_seed * 2_000_033 + current_budget) % (2**63 - 1))


def _deterministic_nested_ids(pool_ids: Sequence[str], seed: int = GLOBAL_SEED) -> tuple[str, ...]:
    unique = tuple(sorted(str(row_id) for row_id in pool_ids))
    if len(set(unique)) != len(unique):
        raise ValueError("pool_ids contains duplicates")
    order = np.random.default_rng(seed).permutation(len(unique))
    return tuple(unique[int(position)] for position in order)


def _deterministic_initial_ids(
    pool_ids: Sequence[str], size: int, seed: int = GLOBAL_SEED
) -> tuple[str, ...]:
    if size <= 0 or size > len(pool_ids):
        raise ValueError("initial size must be positive and no larger than the pool")
    return _deterministic_nested_ids(pool_ids, seed)[:size]


def _protocol(config: Mapping[str, object], smoke: bool) -> dict[str, object]:
    declared = dict(config["protocol"])
    if str(declared.get("version")) != PROTOCOL_VERSION:
        raise ValueError(f"protocol.version must be {PROTOCOL_VERSION!r}")
    if int(declared.get("seed", -1)) != GLOBAL_SEED:
        raise ValueError("The frozen v2 protocol requires seed 42")
    if tuple(declared.get("strategies", ())) != STRATEGIES:
        raise ValueError(f"The frozen strategies are exactly {STRATEGIES}")
    settings = dict(declared["smoke" if smoke else "formal"])
    expected = {
        "initial_budget": SMOKE_INITIAL_BUDGET if smoke else FORMAL_INITIAL_BUDGET,
        "acquisition_batch_size": SMOKE_BATCH_SIZE if smoke else FORMAL_BATCH_SIZE,
        "checkpoints": list(SMOKE_CHECKPOINTS if smoke else FORMAL_CHECKPOINTS),
        "epochs": SMOKE_EPOCHS if smoke else FORMAL_EPOCHS,
        "mc_passes": SMOKE_MC_PASSES if smoke else FORMAL_MC_PASSES,
    }
    for key, value in expected.items():
        actual = settings.get(key)
        if key == "checkpoints":
            actual = list(actual or ())
        if actual != value:
            raise ValueError(f"Frozen {'smoke' if smoke else 'formal'} setting {key} must be {value}")
    return {
        "mode": "smoke" if smoke else "formal",
        "scientific": not smoke,
        "seed": GLOBAL_SEED,
        "strategies": STRATEGIES,
        "initial_budget": expected["initial_budget"],
        "batch_size": expected["acquisition_batch_size"],
        "checkpoints": tuple(expected["checkpoints"]),
        "epochs": expected["epochs"],
        "mc_passes": expected["mc_passes"],
        "target_fraction": TARGET_FRACTION,
    }


def _model_settings(config: Mapping[str, object], protocol: Mapping[str, object]) -> ModelSettings:
    values = dict(config["model"])
    values["epochs"] = int(protocol["epochs"])
    settings = ModelSettings(**values)
    if settings.kind != "low_rank" or settings.response_rank != 64:
        raise ValueError("The formal v2 predictor is fixed to low_rank rank 64")
    return settings


def _source_hashes() -> dict[str, dict[str, object]]:
    source_dir = Path(__file__).resolve().parent
    return {
        str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(source_dir.glob("*.py"))
    }


def _data_paths(config_path: Path, config: Mapping[str, object]) -> dict[str, Path]:
    data = dict(config["data"])
    return {
        "metadata": _configured_path(config_path, data["metadata"]),
        "proteome": _configured_path(config_path, data["proteome"]),
    }


def _environment() -> dict[str, object]:
    return {
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }


def _initial_manifest(
    config_path: Path,
    config: Mapping[str, object],
    protocol: Mapping[str, object],
    command: Sequence[str],
) -> dict[str, object]:
    data_files = _data_paths(config_path, config)
    data_hashes = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in data_files.items()
    }
    return {
        "model_id": MODEL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "command": shlex.join(command),
        "argv": list(command),
        "mode": protocol["mode"],
        "scientific": protocol["scientific"],
        "seed": GLOBAL_SEED,
        "hashes": {
            "config": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": _sha256(config_path),
            },
            "data": data_hashes,
            "source": _source_hashes(),
        },
        "environment": _environment(),
        "query_contract": {
            "unit": "biological condition",
            "fields": list(GROUP_FIELDS),
            "label": "mean matched-control natural log2-delta response across available measurement replicates",
            "response_width": None,
            "concentration_claim": False,
        },
        "control_contract": {
            "controls": "Water/DMSO exact measurement-context mean",
            "role": "assay_overhead",
            "query_candidate": False,
            "predictor_input": False,
            "acquisition_input": False,
        },
        "split_contract": {
            "candidate_source": "official train minus metadata-only interpolation holdout",
            "interpolation": "deterministic level-preserving 20% official-train holdout",
            "official_validation": list(VALIDATION_SPLITS),
            "official_train_overlap_removed": True,
            "candidate_pool_count": None,
            "evaluation_counts": None,
        },
        "information_boundary": {
            "oracle_scope": "candidate-pool responses only; one isolated oracle per formal strategy",
            "label_access": "RetrospectiveOracle.reveal only",
            "acquisition_input": "AcquisitionContext(public IDs, descriptors, optional predictor uncertainty)",
            "evaluation_ids_revealable": False,
            "response_preprocessing_fit_scope": "current revealed labels only",
        },
        "protocol": {
            **dict(protocol),
            "model_seed_function": "(global_seed * 1000003 + current_budget) mod (2^31 - 1)",
            "predictor": "fresh rank-64 low-rank dropout MLP at every fixed batch",
            "primary_split": INTERPOLATION_SPLIT,
            "metrics": "score_response v2 exact natural-delta panel",
            "full_reference": "same-backbone full candidate-pool fit; acquisition_input=false",
            "representation_comparison": "direct versus rank-64 low-rank at nested random 128, 512, full pool; formal only",
        },
        "artifact_inventory": {
            name: {"status": "pending"} for name in EXPECTED_ARTIFACTS
        },
    }


def _load_dataset_and_features(
    config_path: Path,
    config: Mapping[str, object],
) -> tuple[GroupedDataset, PoolFeatureEncoder, np.ndarray]:
    data = dict(config["data"])
    paths = _data_paths(config_path, config)
    cache_value = data.get("cache_dir")
    cache_dir = None if cache_value in (None, "") else _configured_path(config_path, cache_value)
    dataset = load_grouped_dataset(
        paths["metadata"],
        paths["proteome"],
        missing_rate_threshold=float(data["missing_rate_threshold"]),
        cache_dir=cache_dir,
        interpolation_fraction=float(data.get("interpolation_fraction", 0.20)),
        split_seed=GLOBAL_SEED,
    )
    if dataset.benchmark_split is None:
        raise ValueError("GroupedDataset must contain the condition-atomic BenchmarkSplit")
    pool_ids = tuple(dataset.candidate_pool_ids.astype(str))
    pool_set = set(pool_ids)
    seen_evaluation: set[str] = set()
    for split_name, ids in dataset.validation_ids.items():
        current = set(ids.astype(str))
        if pool_set & current:
            raise ValueError(f"Candidate pool overlaps evaluation split {split_name}")
        if seen_evaluation & current:
            raise ValueError(f"Evaluation split {split_name} overlaps another evaluation split")
        seen_evaluation.update(current)
    encoder = PoolFeatureEncoder().fit(dataset.metadata.loc[list(pool_ids)])
    features = encoder.transform(dataset.metadata)
    return dataset, encoder, features


def _positions(dataset: GroupedDataset, ids: Sequence[str]) -> np.ndarray:
    result = dataset.metadata.index.astype(str).get_indexer(list(ids))
    if (result < 0).any():
        missing = [ids[index] for index in np.flatnonzero(result < 0)[:5]]
        raise ValueError(f"Condition IDs do not align with metadata: {missing}")
    return result.astype(np.int64, copy=False)


def _masked_model_features(
    encoder: PoolFeatureEncoder,
    all_features: np.ndarray,
    labelled_positions: np.ndarray,
) -> np.ndarray:
    if len(labelled_positions) == 0:
        raise ValueError("At least one labelled condition is required")
    return encoder.mask_unsupported(all_features, all_features[labelled_positions])


def _evaluate_fit(
    dataset: GroupedDataset,
    fit: ResponseFit,
    model_features: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, split_ids_index in dataset.validation_ids.items():
        split_ids = tuple(split_ids_index.astype(str))
        positions = _positions(dataset, split_ids)
        truth = dataset.response.loc[list(split_ids)].to_numpy(dtype=np.float32)
        prediction = fit.predict(model_features[positions])
        rows.append(
            {
                "split": split_name,
                "primary_split": split_name == INTERPOLATION_SPLIT,
                **score_response(prediction, truth),
            }
        )
    return rows


def _fit_receipt_row(
    fit: ResponseFit,
    *,
    role: str,
    budget: int,
    labelled_ids: Sequence[str],
    seconds: float,
    evaluated: bool,
    strategy: str | None = None,
    round_index: int | None = None,
) -> dict[str, object]:
    return {
        "role": role,
        "strategy": strategy,
        "round": round_index,
        "budget": budget,
        "evaluated": evaluated,
        "labelled_ids_sha256": _hash_ids(labelled_ids),
        "train_seconds": seconds,
        **fit.fit_summary(),
    }


def _fit_from_revealed(
    dataset: GroupedDataset,
    encoder: PoolFeatureEncoder,
    all_features: np.ndarray,
    labelled_ids: Sequence[str],
    labels: np.ndarray,
    settings: ModelSettings,
) -> tuple[ResponseFit, np.ndarray, float]:
    labelled_positions = _positions(dataset, labelled_ids)
    model_features = _masked_model_features(encoder, all_features, labelled_positions)
    started = perf_counter()
    fit = fit_response_model(
        model_features[labelled_positions],
        labels,
        settings,
        seed=_model_seed(GLOBAL_SEED, len(labelled_ids)),
    )
    return fit, model_features, perf_counter() - started


def _write_split_assignments(dataset: GroupedDataset, output_dir: Path) -> pd.DataFrame:
    assignments: list[dict[str, object]] = []
    pool = set(dataset.candidate_pool_ids.astype(str))
    evaluation_by_id = {
        str(row_id): split_name
        for split_name, ids in dataset.validation_ids.items()
        for row_id in ids.astype(str)
    }
    for condition_id, row in dataset.metadata.iterrows():
        row_id = str(condition_id)
        if row_id in pool:
            assignment = "candidate_pool"
            query_allowed = True
        elif row_id in evaluation_by_id:
            assignment = evaluation_by_id[row_id]
            query_allowed = False
        else:
            assignment = "excluded_official_overlap"
            query_allowed = False
        assignments.append(
            {
                CONDITION_ID: row_id,
                "assignment": assignment,
                "query_allowed": query_allowed,
                "official_split_provenance": str(row.get("split_provenance", row.get("split_final", ""))),
            }
        )
    frame = pd.DataFrame(assignments).sort_values(["assignment", CONDITION_ID])
    _atomic_csv(output_dir / "split_assignments.csv", frame)
    return frame


def _fit_full_reference(
    dataset: GroupedDataset,
    encoder: PoolFeatureEncoder,
    all_features: np.ndarray,
    settings: ModelSettings,
    output_dir: Path,
) -> tuple[ResponseFit, pd.DataFrame, dict[str, object]]:
    pool_ids = tuple(dataset.candidate_pool_ids.astype(str))
    oracle = RetrospectiveOracle(
        pool_ids,
        dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
        response_ids=pool_ids,
    )
    revealed = oracle.reveal(pool_ids)
    fit, model_features, seconds = _fit_from_revealed(
        dataset, encoder, all_features, revealed.ids, revealed.labels, settings
    )
    rows = [
        {
            "role": "full_pool_reference",
            "seed": GLOBAL_SEED,
            "budget": len(pool_ids),
            "model_seed": _model_seed(GLOBAL_SEED, len(pool_ids)),
            **metrics,
        }
        for metrics in _evaluate_fit(dataset, fit, model_features)
    ]
    frame = pd.DataFrame(rows)
    _atomic_csv(output_dir / "full_reference_metrics.csv", frame)
    receipt = _fit_receipt_row(
        fit,
        role="full_pool_reference",
        budget=len(pool_ids),
        labelled_ids=pool_ids,
        seconds=seconds,
        evaluated=True,
    )
    return fit, frame, receipt


def _run_active_learning(
    dataset: GroupedDataset,
    encoder: PoolFeatureEncoder,
    all_features: np.ndarray,
    settings: ModelSettings,
    protocol: Mapping[str, object],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    pool_ids = tuple(dataset.candidate_pool_ids.astype(str))
    schedule = BudgetSchedule(
        initial_budget=int(protocol["initial_budget"]),
        batch_size=int(protocol["batch_size"]),
        checkpoints=tuple(int(value) for value in protocol["checkpoints"]),
        pool_size=len(pool_ids),
    )
    initial_ids = _deterministic_initial_ids(pool_ids, schedule.initial_budget)
    descriptor_positions = _positions(dataset, pool_ids)
    pool_descriptors = all_features[descriptor_positions]
    metric_rows: list[dict[str, object]] = []
    acquisition_rows: list[dict[str, object]] = []
    fit_receipts: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        oracle = RetrospectiveOracle(
            pool_ids,
            dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
            response_ids=pool_ids,
        )
        empty_state = PoolState(pool_ids, pool_descriptors, descriptor_ids=pool_ids)
        state = empty_state.select(initial_ids)
        initial_reveal = oracle.reveal(initial_ids)
        labelled_response = initial_reveal.labels.copy()
        transition_before = empty_state
        transition_selected = initial_ids
        transition_acquisition_seed: int | None = None
        transition_hashes: dict[str, str] = {}
        for rank, condition_id in enumerate(initial_ids, start=1):
            acquisition_rows.append(
                {
                    "strategy": strategy,
                    "seed": GLOBAL_SEED,
                    "round": 0,
                    "selection_type": "deterministic_initial",
                    "rank_in_batch": rank,
                    "budget_before": 0,
                    "budget_after": schedule.initial_budget,
                    CONDITION_ID: condition_id,
                }
            )

        round_index = 0
        while True:
            budget = state.budget
            fit, model_features, seconds = _fit_from_revealed(
                dataset,
                encoder,
                all_features,
                state.selected_ids,
                labelled_response,
                settings,
            )
            evaluated = schedule.is_checkpoint(budget)
            fit_receipts.append(
                _fit_receipt_row(
                    fit,
                    role="active_learning",
                    budget=budget,
                    labelled_ids=state.selected_ids,
                    seconds=seconds,
                    evaluated=evaluated,
                    strategy=strategy,
                    round_index=round_index,
                )
            )
            split_metrics = _evaluate_fit(dataset, fit, model_features) if evaluated else []
            if evaluated:
                for metrics in split_metrics:
                    metric_rows.append(
                        {
                            "role": "active_learning",
                            "strategy": strategy,
                            "seed": GLOBAL_SEED,
                            "round": round_index,
                            "budget": budget,
                            "model_seed": _model_seed(GLOBAL_SEED, budget),
                            "train_seconds": seconds,
                            **metrics,
                        }
                    )
            receipt = RoundReceipt.from_transition(
                round_index,
                strategy,
                transition_before,
                state,
                transition_selected,
                hashes=transition_hashes,
                global_seed=GLOBAL_SEED,
                acquisition_seed=transition_acquisition_seed,
                model_seed=_model_seed(GLOBAL_SEED, budget),
                checkpoint=evaluated,
                labelled_ids=state.selected_ids,
                model_fit_summary=fit.fit_summary(),
                split_metrics=split_metrics,
                train_seconds=seconds,
            )
            receipt.write_json_atomic(
                output_dir
                / "round_receipts"
                / strategy
                / f"round_{round_index:03d}.json"
            )
            print(
                f"strategy={strategy} round={round_index} budget={budget} "
                f"evaluated={evaluated} seconds={seconds:.2f}",
                flush=True,
            )
            if budget == schedule.final_budget:
                break

            candidate_ids = state.candidate_ids
            if strategy == "uncertainty":
                candidate_positions = _positions(dataset, candidate_ids)
                uncertainty = fit.uncertainty(
                    model_features[candidate_positions], passes=int(protocol["mc_passes"])
                )
                context = state.acquisition_context(
                    uncertainty_ids=candidate_ids, uncertainty=uncertainty
                )
            else:
                context = state.acquisition_context()
            acquisition_seed = _acquisition_seed(GLOBAL_SEED, budget)
            selected = select_batch(
                strategy,
                context,
                schedule.batch_size,
                seed=acquisition_seed,
            )
            before = state
            state = schedule.advance(before, selected)
            revealed = oracle.reveal(selected)
            labelled_response = np.concatenate(
                [labelled_response, revealed.labels], axis=0
            )
            round_index += 1
            transition_before = before
            transition_selected = selected
            transition_acquisition_seed = acquisition_seed
            transition_hashes = {
                "acquisition_model_fit_sha256": _hash_payload(fit.fit_summary())
            }
            for rank, condition_id in enumerate(selected, start=1):
                acquisition_rows.append(
                    {
                        "strategy": strategy,
                        "seed": GLOBAL_SEED,
                        "round": round_index,
                        "selection_type": strategy,
                        "rank_in_batch": rank,
                        "budget_before": before.budget,
                        "budget_after": state.budget,
                        CONDITION_ID: condition_id,
                    }
                )
            if len(oracle.revealed_ids) != state.budget:
                raise AssertionError("Oracle reveal count and conserved budget differ")

    metrics = pd.DataFrame(metric_rows)
    acquisitions = pd.DataFrame(acquisition_rows)
    forbidden = {"label", "labels", "response", "oracle_impact", "impact"}
    if forbidden & set(acquisitions.columns):
        raise AssertionError("Acquisition artifact contains a forbidden hidden-label field")
    _atomic_csv(output_dir / "active_metrics.csv", metrics)
    _atomic_csv(output_dir / "acquisitions.csv", acquisitions)
    return metrics, acquisitions, fit_receipts


def _run_representation_comparison(
    dataset: GroupedDataset,
    encoder: PoolFeatureEncoder,
    all_features: np.ndarray,
    settings: ModelSettings,
    formal: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    columns = [
        "role",
        "representation",
        "seed",
        "budget",
        "model_seed",
        "split",
        "primary_split",
        *score_response(np.zeros((0, 1)), np.zeros((0, 1))).keys(),
    ]
    if not formal:
        frame = pd.DataFrame(columns=columns)
        _atomic_csv(output_dir / "representation_metrics.csv", frame)
        return frame, []

    pool_ids = tuple(dataset.candidate_pool_ids.astype(str))
    nested = _deterministic_nested_ids(pool_ids)
    budgets = (128, 512, len(pool_ids))
    oracle = RetrospectiveOracle(
        pool_ids,
        dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
        response_ids=pool_ids,
    )
    revealed_ids: tuple[str, ...] = ()
    revealed_labels = np.empty((0, len(dataset.proteins)), dtype=np.float32)
    rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for budget in budgets:
        additions = nested[len(revealed_ids) : budget]
        batch = oracle.reveal(additions)
        revealed_ids = (*revealed_ids, *batch.ids)
        revealed_labels = np.concatenate([revealed_labels, batch.labels], axis=0)
        if tuple(revealed_ids) != nested[:budget]:
            raise AssertionError("Representation budgets are not nested")
        for kind in ("direct", "low_rank"):
            comparison_settings = replace(settings, kind=kind)
            fit, model_features, seconds = _fit_from_revealed(
                dataset,
                encoder,
                all_features,
                revealed_ids,
                revealed_labels,
                comparison_settings,
            )
            receipts.append(
                _fit_receipt_row(
                    fit,
                    role="representation_comparison",
                    budget=budget,
                    labelled_ids=revealed_ids,
                    seconds=seconds,
                    evaluated=True,
                    strategy=kind,
                )
            )
            for metrics in _evaluate_fit(dataset, fit, model_features):
                rows.append(
                    {
                        "role": "representation_comparison",
                        "representation": kind,
                        "seed": GLOBAL_SEED,
                        "budget": budget,
                        "model_seed": _model_seed(GLOBAL_SEED, budget),
                        **metrics,
                    }
                )
    frame = pd.DataFrame(rows)
    _atomic_csv(output_dir / "representation_metrics.csv", frame)
    return frame, receipts


def _target_summary(
    budgets: np.ndarray,
    values: np.ndarray,
    full_reference: float,
    higher_is_better: bool,
) -> dict[str, object]:
    direction = 1.0 if higher_is_better else -1.0
    if direction * full_reference <= direction * float(values[0]):
        return {
            "status": "not_reached",
            "not_reached": True,
            "budget": None,
            "reason": "full_reference_not_better_than_initial",
            "target_fraction": TARGET_FRACTION,
            "initial_value": float(values[0]),
            "full_reference": float(full_reference),
        }
    return budget_to_target(
        budgets,
        values,
        full_reference,
        target_fraction=TARGET_FRACTION,
        higher_is_better=higher_is_better,
    )


def _analysis_summary(
    active_metrics: pd.DataFrame,
    full_reference_metrics: pd.DataFrame,
    representation_metrics: pd.DataFrame,
    protocol: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    reference_lookup = {
        str(row["split"]): row for _, row in full_reference_metrics.iterrows()
    }
    curves: list[dict[str, object]] = []
    for (strategy, split_name), values in active_metrics.groupby(
        ["strategy", "split"], sort=True
    ):
        values = values.sort_values("budget")
        budgets = values["budget"].to_numpy(dtype=np.float64)
        for metric, higher_is_better in METRIC_DIRECTIONS.items():
            metric_values = values[metric].to_numpy(dtype=np.float64)
            reference = float(reference_lookup[split_name][metric])
            record: dict[str, object] = {
                "strategy": strategy,
                "split": split_name,
                "primary_split": split_name == INTERPOLATION_SPLIT,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "budgets": budgets.tolist(),
                "values": metric_values.tolist(),
                "full_reference": reference,
            }
            if not np.isfinite(metric_values).all() or not np.isfinite(reference):
                record.update(
                    {
                        "aulc_status": "unavailable_nonfinite_metric",
                        "normalized_trapezoidal_aulc": None,
                        "budget_to_80_percent_target": {
                            "status": "not_reached",
                            "not_reached": True,
                            "budget": None,
                            "reason": "nonfinite_metric",
                        },
                    }
                )
            else:
                record.update(
                    {
                        "aulc_status": "complete",
                        "normalized_trapezoidal_aulc": normalized_trapezoidal_aulc(
                            budgets, metric_values, higher_is_better=higher_is_better
                        ),
                        "budget_to_80_percent_target": _target_summary(
                            budgets, metric_values, reference, higher_is_better
                        ),
                    }
                )
            curves.append(record)
    payload = {
        "model_id": MODEL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "mode": protocol["mode"],
        "scientific": protocol["scientific"],
        "primary_split": INTERPOLATION_SPLIT,
        "metric_disclaimer": (
            "All scores are GOAI-AL local matched-control natural-delta diagnostics; "
            "delta_skill_zero is a local proxy and none is an official GOAI leaderboard score."
        ),
        "impact_and_hit_metrics": "deprecated and absent",
        "aulc_definition": "trapezoidal integration on actual budget x values divided by budget span",
        "target_definition": "80% of achievable improvement from the strategy initial value to the same-backbone full-pool reference; monotone envelope and linear interpolation; no extrapolation",
        "learning_curves": curves,
        "representation_comparison": {
            "formal_only": True,
            "rank_sweep": False,
            "rows": _json_safe(representation_metrics.to_dict(orient="records")),
        },
    }
    _atomic_json(output_dir / "analysis_summary.json", payload)
    return payload


def _plot_learning_curves(active_metrics: pd.DataFrame, output_dir: Path) -> None:
    primary = active_metrics[active_metrics["split"].eq(INTERPOLATION_SPLIT)]
    palette = {"random": "#2F7DBA", "coreset": "#D97706", "uncertainty": "#7A5AA6"}
    for metric in CURVE_METRICS:
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for strategy in STRATEGIES:
            values = primary[primary["strategy"].eq(strategy)].sort_values("budget")
            axis.plot(
                values["budget"],
                values[metric],
                marker="o",
                linewidth=2.0,
                color=palette[strategy],
                label=strategy,
            )
        axis.set_xlabel("Labelled biological conditions")
        axis.set_ylabel(metric)
        axis.set_title(f"GOAI-AL v2 interpolation learning curve: {metric}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        _atomic_figure(output_dir / f"learning_curve_{metric}.png", figure)


def _artifact_inventory(output_dir: Path) -> dict[str, object]:
    inventory: dict[str, object] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(output_dir))
        if relative == "manifest.json":
            continue
        inventory[relative] = {
            "status": "complete",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    inventory["manifest.json"] = {
        "status": "self_describing",
        "bytes": None,
        "sha256": None,
    }
    return inventory


def run_controller(
    config_path: Path,
    *,
    smoke: bool,
    output_dir: Path,
    command: Sequence[str],
) -> Path:
    """Run one immutable v2 attempt in a previously reserved empty directory."""

    config_path = config_path.resolve()
    config = _load_config(config_path)
    protocol = _protocol(config, smoke)
    _reserve_output(output_dir)
    manifest = _initial_manifest(config_path, config, protocol, command)
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    try:
        dataset, encoder, all_features = _load_dataset_and_features(config_path, config)
        manifest["query_contract"]["response_width"] = len(dataset.proteins)
        manifest["split_contract"]["candidate_pool_count"] = len(dataset.candidate_pool_ids)
        manifest["split_contract"]["evaluation_counts"] = {
            name: len(ids) for name, ids in dataset.validation_ids.items()
        }
        manifest["split_contract"]["removed_validation_overlap_counts"] = {
            name: len(ids) for name, ids in dataset.removed_validation_overlap.items()
        }
        manifest["feature_contract"] = {
            "fit_scope": "candidate-pool metadata only",
            "dimension": encoder.output_dim,
            "categorical_fields": list(encoder.categorical_fields),
            "continuous_time_column_retained_during_support_masking": True,
        }
        _atomic_json(manifest_path, manifest)

        _write_split_assignments(dataset, output_dir)
        write_audit_outputs(dataset, output_dir)
        manifest["posthoc_audit"] = {
            "completed_before_experiment": True,
            "acquisition_input": False,
        }
        _atomic_json(manifest_path, manifest)

        settings = _model_settings(config, protocol)
        _, full_reference, reference_receipt = _fit_full_reference(
            dataset, encoder, all_features, settings, output_dir
        )
        active_metrics, _, active_receipts = _run_active_learning(
            dataset, encoder, all_features, settings, protocol, output_dir
        )
        representation_metrics, representation_receipts = _run_representation_comparison(
            dataset,
            encoder,
            all_features,
            settings,
            formal=not smoke,
            output_dir=output_dir,
        )
        fit_receipts = pd.DataFrame(
            [reference_receipt, *active_receipts, *representation_receipts]
        )
        _atomic_csv(output_dir / "model_fit_receipts.csv", fit_receipts)
        _analysis_summary(
            active_metrics, full_reference, representation_metrics, protocol, output_dir
        )
        _plot_learning_curves(active_metrics, output_dir)

        manifest["status"] = "complete"
        manifest["completed_at"] = _utc_now()
        manifest["artifact_inventory"] = _artifact_inventory(output_dir)
        _atomic_json(manifest_path, manifest)
        return output_dir
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["completed_at"] = _utc_now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        manifest["artifact_inventory"] = _artifact_inventory(output_dir)
        _atomic_json(manifest_path, manifest)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-suffix")
    output.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_args)
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    output_dir = _resolve_output(
        config_path, config, args.output_suffix, args.output_dir
    )
    command = [sys.executable, "-m", "goai_al.experiment", *raw_args]
    completed = run_controller(
        config_path,
        smoke=bool(args.smoke),
        output_dir=output_dir,
        command=command,
    )
    print(f"Completed GOAI-AL v2 attempt: {completed}")


if __name__ == "__main__":
    main()
