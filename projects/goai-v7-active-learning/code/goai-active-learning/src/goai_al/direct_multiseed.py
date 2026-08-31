"""Executable GOAI v2.2 Direct semantic multi-seed experiment runner.

The runner keeps the retrospective oracle behind the public acquisition
boundary, trains a fresh Direct predictor at every dense acquisition budget,
evaluates only registered checkpoints, and commits each seed atomically after
its complete artifact inventory has been independently verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .acquisition import AcquisitionContext, select_batch
from .audit import write_audit_outputs
from .data import (
    CONDITION_ID,
    DEFAULT_CONTROL_POLICY,
    GROUP_FIELDS,
    INTERPOLATION_SPLIT,
    PROTOCOL_VERSION,
    VALIDATION_SPLITS,
    GroupedDataset,
    PoolFeatureEncoder,
    load_grouped_dataset,
)
from .metrics import budget_to_target, normalized_trapezoidal_aulc, score_response
from .model import ModelSettings, fit_response_model
from .semantics import (
    DEFAULT_CHEMICAL_EMBEDDINGS_PATH,
    DEFAULT_CHEMICAL_MANIFEST_PATH,
    DEFAULT_CHEMICAL_RISK_MANIFEST_PATH,
    DEFAULT_STRAIN_MANIFEST_PATH,
    DEFAULT_STRAIN_SEMANTICS_PATH,
    STRAIN_IDENTITY_WARNING,
    load_feature_bundle as load_current_semantic_feature_bundle,
)
from .simulator import BudgetSchedule, PoolState, RetrospectiveOracle, RoundReceipt


EXPERIMENT_ID = "goai-al-direct-semantic-multiseed-v2.2"
DATA_PROTOCOL = "goai-condition-atomic-v2.1"
MODEL_ID = "GOAI-AL-V22-DIRECT-SEMANTIC-01"
FORMAL_SEEDS = (42, 43, 44, 45, 46)
SMOKE_SEEDS = (42, 43)
SPLIT_SEED = 42
STRATEGIES = ("random", "coreset", "uncertainty")

FORMAL_INITIAL_BUDGET = 128
FORMAL_ACQUISITION_BATCH_SIZE = 128
FORMAL_CHECKPOINTS = (128, 256, 512, 1024)
FORMAL_EPOCHS = 80
FORMAL_MC_PASSES = 8

SMOKE_INITIAL_BUDGET = 32
SMOKE_ACQUISITION_BATCH_SIZE = 32
SMOKE_CHECKPOINTS = (32, 64, 96)
SMOKE_EPOCHS = 2
SMOKE_MC_PASSES = 2

# Friendly aliases for consumers that use ``*_VERSION`` or ``*_SEED`` names.
EXPERIMENT_VERSION = EXPERIMENT_ID
DATA_PROTOCOL_VERSION = DATA_PROTOCOL
FORMAL_RUN_SEEDS = FORMAL_SEEDS
SMOKE_RUN_SEEDS = SMOKE_SEEDS

T_CRITICAL_95 = MappingProxyType({2: 12.706205, 5: 2.776445})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_SPEC_PATH = PROJECT_ROOT / "FRAMEWORK_SPEC_V22.md"
FROZEN_METADATA_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_metadata_train_val.csv"
)
FROZEN_PROTEOME_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_proteome_raw_train_val.csv"
)
FROZEN_CACHE_PATH = PROJECT_ROOT / "results" / "cache_v22"
MISSING_RATE_THRESHOLD = 0.80
INTERPOLATION_FRACTION = 0.20
TARGET_FRACTION = 0.80
PRIMARY_METRIC = "delta_skill_zero"
SEED_MANIFEST_SCHEMA = "goai.direct_multiseed.seed_manifest.v2"
ROOT_MANIFEST_SCHEMA = "goai.direct_multiseed.root_manifest.v2"
SOURCE_SNAPSHOT_SCHEMA = "goai.direct_multiseed.source_snapshot.v1"

CONTROL_CONTRACT = MappingProxyType(
    {
        "control_policy": DEFAULT_CONTROL_POLICY,
        "default_policy": True,
        "vehicle_column": None,
        "vehicle_mapping_state": "not_applicable_pooled_policy",
        "control_aggregation": "direct_measurement_mean_in_log2_space",
        "control_type_means_averaged_equally": False,
        "selected_control_rule": "all_exact_context_water_dmso_measurements",
        "pooled_across_types": True,
        "vehicle_inference": False,
    }
)

SCORE_METRICS = (
    "delta_rmse",
    "delta_mae",
    "delta_skill_zero",
    "pooled_delta_pcc",
    "condition_pcc_median",
    "protein_pcc_median",
    "protein_r2_median",
    "protein_r2_mean",
    "protein_r2_positive_fraction",
)
COUNT_METRICS = (
    "n_conditions",
    "n_proteins",
    "n_observed_values",
    "n_evaluable_conditions_pcc",
    "n_evaluable_proteins_pcc",
    "n_evaluable_proteins_r2",
)
METRIC_DIRECTIONS = MappingProxyType(
    {
        "delta_rmse": False,
        "delta_mae": False,
        **{name: True for name in SCORE_METRICS if name not in {"delta_rmse", "delta_mae"}},
    }
)
SEED_REQUIRED_FILES = (
    "acquisitions.csv",
    "active_metrics.csv",
    "full_reference_metrics.csv",
    "ablation_metrics.csv",
    "fit_receipts.json",
)
ROOT_TABLE_FILES = (
    "active_metrics.csv",
    "full_reference_metrics.csv",
    "ablation_metrics.csv",
    "acquisitions.csv",
    "per_seed_curve_summary.csv",
    "aggregate_metrics.csv",
    "paired_policy_comparisons.csv",
    "representation_ablation_summary.csv",
)

# The architecture/optimizer contract is shared by formal and smoke runs; only
# epochs differ between the two profiles.
DIRECT_MODEL_DEFAULTS = MappingProxyType(
    {
        "kind": "direct",
        "hidden_dim": 128,
        "dropout": 0.10,
        "learning_rate": 0.001,
        "weight_decay": 0.0002,
        "batch_size": 512,
        "target_scale_floor": 0.05,
        "device": "cuda",
        "response_rank": 64,
        "svd_niter": 2,
    }
)

RECEIPT_FORBIDDEN_KEYS = frozenset(
    {
        "descriptor_matrix",
        "descriptor_features",
        "descriptors",
        "encoder",
        "features",
        "ground_truth",
        "ground_truths",
        "hidden_value",
        "hidden_values",
        "impact",
        "impacts",
        "label",
        "labels",
        "model_features",
        "model_matrix",
        "model_state",
        "model_state_dict",
        "optimizer_state",
        "optimizer_state_dict",
        "oracle_impact",
        "oracle_impacts",
        "oracle_response",
        "oracle_responses",
        "prediction",
        "predictions",
        "response",
        "responses",
        "target",
        "targets",
        "truth",
        "truths",
        "y_true",
    }
)

SEED_MANIFEST_FILENAME = "manifest.json"


def _json_safe_copy(value: Any, *, role: str = "value") -> Any:
    """Return a detached strict-JSON copy or raise a useful error."""

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{role} must be strictly JSON-safe") from error


def canonical_json(value: Any) -> str:
    """Serialize a JSON-safe value in the experiment's canonical form."""

    safe = _json_safe_copy(value)
    return json.dumps(
        safe,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    """Return the SHA256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _matrix(values: Any, role: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{role} must be a two-dimensional matrix")
    if array.shape[1] == 0:
        raise ValueError(f"{role} must contain at least one column")
    try:
        finite = np.isfinite(array)
    except TypeError as error:
        raise ValueError(f"{role} must be numeric and finite") from error
    if not finite.all():
        raise ValueError(f"{role} must contain only finite values")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{role} must remain finite as float32")
    canonical = np.array(result, dtype=np.float32, order="C", copy=True)
    canonical.setflags(write=False)
    return canonical


def _matrix_identity(values: Any) -> dict[str, Any]:
    canonical = np.array(values, dtype=np.float32, order="C", copy=True)
    header = {
        "shape": [int(value) for value in canonical.shape],
        "dtype": str(canonical.dtype),
        "bytes": int(canonical.nbytes),
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(canonical_json(header).encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return {**header, "sha256": digest.hexdigest()}


def _row_ids(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Feature row_ids must be a one-dimensional ID sequence")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError("Feature row_ids must be a one-dimensional ID sequence") from error
    result = tuple(str(value) for value in items)
    if len(set(result)) != len(result):
        raise ValueError("Feature row_ids must be unique")
    return result


@dataclass(frozen=True)
class PublicFeatureDatasetView:
    """Target-free dataset view exposed to feature loaders."""

    metadata: pd.DataFrame
    candidate_pool_ids: pd.Index

    @classmethod
    def from_dataset(cls, dataset: GroupedDataset) -> PublicFeatureDatasetView:
        if not isinstance(dataset.metadata, pd.DataFrame):
            raise TypeError("dataset.metadata must be a pandas DataFrame")
        metadata_ids = tuple(str(value) for value in dataset.metadata.index)
        response_ids = tuple(str(value) for value in dataset.response.index)
        if response_ids != metadata_ids:
            raise ValueError("Dataset response rows must exactly align with metadata rows")
        metadata = dataset.metadata.copy(deep=True)
        candidate_pool_ids = pd.Index(
            [str(value) for value in dataset.candidate_pool_ids],
            name=CONDITION_ID,
        )
        return cls(metadata=metadata, candidate_pool_ids=candidate_pool_ids)

    @property
    def train_ids(self) -> pd.Index:
        """Compatibility alias for feature loaders using ``train_ids``."""

        return self.candidate_pool_ids


@dataclass(frozen=True)
class FeatureBundle:
    """Validated, row-aligned descriptor and predictor feature matrices."""

    row_ids: tuple[str, ...]
    descriptor_matrix: np.ndarray
    model_matrix: np.ndarray
    masker: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    summary: Mapping[str, Any]
    asset_hashes: Mapping[str, Any]
    encoder: object | None = None
    row_ids_sha256: str = ""
    descriptor_matrix_identity: Mapping[str, Any] | None = None
    model_matrix_identity: Mapping[str, Any] | None = None

    @property
    def descriptor_features(self) -> np.ndarray:
        return self.descriptor_matrix

    @property
    def model_features(self) -> np.ndarray:
        return self.model_matrix


def validate_feature_bundle(
    bundle: FeatureBundle,
    dataset: Any,
) -> FeatureBundle:
    """Validate matrices against the dataset's exact metadata row order."""

    if not isinstance(bundle, FeatureBundle):
        raise TypeError("bundle must be a FeatureBundle")
    if not isinstance(dataset.metadata, pd.DataFrame):
        raise TypeError("dataset.metadata must be a pandas DataFrame")
    expected_ids = tuple(str(value) for value in dataset.metadata.index)
    actual_ids = _row_ids(bundle.row_ids)
    if actual_ids != expected_ids:
        raise ValueError("Feature row_ids must exactly match dataset.metadata row order")
    descriptor = _matrix(bundle.descriptor_matrix, "descriptor_matrix")
    model = _matrix(bundle.model_matrix, "model_matrix")
    if descriptor.shape[0] != len(expected_ids) or model.shape[0] != len(expected_ids):
        raise ValueError("Feature matrices must have one row per dataset metadata row")
    if bundle.masker is not None and not callable(bundle.masker):
        raise ValueError("Feature masker must be callable or None")
    summary = _json_safe_copy(bundle.summary, role="Feature summary")
    hashes = _json_safe_copy(bundle.asset_hashes, role="Feature asset_hashes")
    if not isinstance(summary, dict):
        raise ValueError("Feature summary must be a JSON object")
    if not isinstance(hashes, dict):
        raise ValueError("Feature asset_hashes must be a JSON object")
    if summary.get("response_used") is not False:
        raise ValueError("Feature summary must explicitly declare response_used=false")
    row_ids_sha256 = canonical_hash(list(actual_ids))
    descriptor_identity = MappingProxyType(_matrix_identity(descriptor))
    model_identity = MappingProxyType(_matrix_identity(model))
    return FeatureBundle(
        row_ids=actual_ids,
        descriptor_matrix=descriptor,
        model_matrix=model,
        masker=bundle.masker,
        summary=summary,
        asset_hashes=hashes,
        encoder=bundle.encoder,
        row_ids_sha256=row_ids_sha256,
        descriptor_matrix_identity=descriptor_identity,
        model_matrix_identity=model_identity,
    )


def coerce_feature_bundle(
    value: FeatureBundle | Mapping[str, Any],
    dataset: Any,
) -> FeatureBundle:
    """Coerce the current semantics mapping (or a bundle) to ``FeatureBundle``."""

    if isinstance(value, FeatureBundle):
        return validate_feature_bundle(value, dataset)
    if not isinstance(value, Mapping):
        raise TypeError("Feature loader output must be a mapping or FeatureBundle")
    descriptor = value.get(
        "descriptor_matrix",
        value.get("descriptor_features", value.get("descriptors")),
    )
    model = value.get("model_matrix", value.get("model_features"))
    missing = [
        name
        for name, item in (
            ("row_ids", value.get("row_ids")),
            ("descriptor_matrix", descriptor),
            ("model_matrix", model),
        )
        if item is None
    ]
    if missing:
        raise ValueError(f"Feature loader output is missing required values: {missing}")
    bundle = FeatureBundle(
        row_ids=_row_ids(value["row_ids"]),
        descriptor_matrix=np.asarray(descriptor),
        model_matrix=np.asarray(model),
        masker=value.get("masker"),
        summary=value.get("summary", {}),
        asset_hashes=value.get("asset_hashes", {}),
        encoder=value.get("encoder"),
    )
    return validate_feature_bundle(bundle, dataset)


def _slice_summary(values: Mapping[str, slice]) -> dict[str, dict[str, int]]:
    return {
        name: {"start": int(block.start), "stop": int(block.stop)}
        for name, block in values.items()
    }


def load_identity_feature_bundle(
    dataset: Any,
    encoder: PoolFeatureEncoder | None = None,
) -> FeatureBundle:
    """Build the target-free identity/time ablation feature bundle."""

    if not isinstance(dataset.metadata, pd.DataFrame):
        raise TypeError("dataset.metadata must be a pandas DataFrame")
    pool_ids = dataset.candidate_pool_ids
    missing = pool_ids.difference(dataset.metadata.index)
    if len(missing):
        raise ValueError(f"Candidate-pool IDs are missing from metadata: {missing[:5].tolist()}")
    identity = PoolFeatureEncoder() if encoder is None else encoder
    identity.fit(dataset.metadata.loc[pool_ids])
    features = identity.transform(dataset.metadata)
    summary = {
        "feature_mode": "identity",
        "fit_scope": "candidate_pool_metadata",
        "response_used": False,
        "row_count": int(len(dataset.metadata)),
        "candidate_pool_count": int(len(pool_ids)),
        "descriptor_width": int(features.shape[1]),
        "model_width": int(features.shape[1]),
        "categorical_column_slices": _slice_summary(
            identity.categorical_column_slices
        ),
        "masking": "categorical_one_hot_only; continuous_time_preserved",
    }
    return coerce_feature_bundle(
        {
            "row_ids": dataset.metadata.index,
            "descriptor_matrix": features,
            "model_matrix": features,
            "masker": identity.mask_unsupported,
            "summary": summary,
            "asset_hashes": {},
            "encoder": identity,
        },
        dataset,
    )


def load_semantic_feature_bundle(
    dataset: Any,
    loader: Callable[[Any], Mapping[str, Any]] | None = None,
) -> FeatureBundle:
    """Load and validate the current combined identity/semantic mapping."""

    selected_loader = load_current_semantic_feature_bundle if loader is None else loader
    return coerce_feature_bundle(selected_loader(dataset), dataset)


def _nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{role} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{role} must be nonnegative")
    return result


def _derived_seed(namespace: str, run_seed: int, budget: int) -> int:
    run = _nonnegative_int(run_seed, "run_seed")
    step = _nonnegative_int(budget, "budget")
    digest = hashlib.sha256(f"{namespace}\0{run}\0{step}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def model_seed(run_seed: int, budget: int) -> int:
    """Derive a model-fit seed using only run seed and labeled budget."""

    return _derived_seed("model", run_seed, budget)


def acquisition_seed(run_seed: int, budget: int) -> int:
    """Derive an acquisition seed using only run seed and labeled budget."""

    return _derived_seed("acquisition", run_seed, budget)


def deterministic_initial_ids(
    candidate_ids: Iterable[object],
    initial_budget: int,
    run_seed: int,
) -> tuple[str, ...]:
    """Select an order-independent deterministic initial labeled set."""

    budget = _nonnegative_int(initial_budget, "initial_budget")
    run = _nonnegative_int(run_seed, "run_seed")
    ids = tuple(str(value) for value in candidate_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    if budget > len(ids):
        raise ValueError("initial_budget exceeds the candidate pool size")

    def rank(sample_id: str) -> tuple[bytes, str]:
        payload = canonical_json(
            {"namespace": "initial", "run_seed": run, "sample_id": sample_id}
        )
        return hashlib.sha256(payload.encode("utf-8")).digest(), sample_id

    selected = sorted(ids, key=rank)[:budget]
    return tuple(sorted(selected))


def build_direct_model_settings(
    config: Mapping[str, Any] | None = None,
    *,
    epochs: int,
    device_override: str | None = None,
) -> ModelSettings:
    """Build the frozen Direct-only model settings for one run profile."""

    supplied = dict(DIRECT_MODEL_DEFAULTS if config is None else config)
    unknown = sorted(set(supplied) - set(DIRECT_MODEL_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown model setting(s): {unknown}")
    missing = sorted(set(DIRECT_MODEL_DEFAULTS) - set(supplied))
    merged = {**DIRECT_MODEL_DEFAULTS, **supplied}
    mismatches = {
        key: (DIRECT_MODEL_DEFAULTS[key], merged[key])
        for key in DIRECT_MODEL_DEFAULTS
        if merged[key] != DIRECT_MODEL_DEFAULTS[key]
    }
    if missing or mismatches:
        raise ValueError(
            "Model settings differ from the frozen Direct contract: "
            f"missing={missing}, mismatches={mismatches}"
        )
    epoch_count = _nonnegative_int(epochs, "epochs")
    if epoch_count == 0:
        raise ValueError("epochs must be positive")
    if device_override is not None:
        if device_override not in {"cpu", "cuda"}:
            raise ValueError("device_override must be 'cpu' or 'cuda'")
        merged["device"] = device_override
    return ModelSettings(epochs=epoch_count, **merged)


def frozen_config() -> dict[str, Any]:
    """Return a detached copy of the exact registered v2.2 configuration."""

    return {
        "experiment_id": EXPERIMENT_ID,
        "data_protocol": DATA_PROTOCOL,
        "model_id": MODEL_ID,
        "split_seed": SPLIT_SEED,
        "data": {
            "metadata_path": str(FROZEN_METADATA_PATH),
            "proteome_path": str(FROZEN_PROTEOME_PATH),
            "cache_dir": "results/cache_v22",
            "missing_rate_threshold": MISSING_RATE_THRESHOLD,
            "interpolation_fraction": INTERPOLATION_FRACTION,
            "control_policy": DEFAULT_CONTROL_POLICY,
            "vehicle_column": None,
        },
        "evaluation": {"target_fraction": TARGET_FRACTION},
        "semantic_assets": {
            "chemical_embeddings_path": str(DEFAULT_CHEMICAL_EMBEDDINGS_PATH),
            "chemical_manifest_path": str(DEFAULT_CHEMICAL_MANIFEST_PATH),
            "strain_semantics_path": str(DEFAULT_STRAIN_SEMANTICS_PATH),
            "strain_manifest_path": str(DEFAULT_STRAIN_MANIFEST_PATH),
            "chemical_risk_manifest_path": str(DEFAULT_CHEMICAL_RISK_MANIFEST_PATH),
        },
        "features": {"primary": "semantic", "ablations": ["identity", "semantic"]},
        "strategies": list(STRATEGIES),
        "model": dict(DIRECT_MODEL_DEFAULTS),
        "profiles": {
            "formal": {
                "seeds": list(FORMAL_SEEDS),
                "initial_budget": FORMAL_INITIAL_BUDGET,
                "acquisition_batch_size": FORMAL_ACQUISITION_BATCH_SIZE,
                "checkpoints": list(FORMAL_CHECKPOINTS),
                "epochs": FORMAL_EPOCHS,
                "mc_passes": FORMAL_MC_PASSES,
            },
            "smoke": {
                "seeds": list(SMOKE_SEEDS),
                "initial_budget": SMOKE_INITIAL_BUDGET,
                "acquisition_batch_size": SMOKE_ACQUISITION_BATCH_SIZE,
                "checkpoints": list(SMOKE_CHECKPOINTS),
                "epochs": SMOKE_EPOCHS,
                "mc_passes": SMOKE_MC_PASSES,
            },
        },
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach the complete frozen multi-seed configuration."""

    safe = _json_safe_copy(config, role="Configuration")
    if not isinstance(safe, dict):
        raise ValueError("Configuration must be a JSON object")
    expected = frozen_config()
    if set(safe) != set(expected):
        raise ValueError("Configuration top-level fields differ from the frozen contract")
    for key in ("experiment_id", "data_protocol", "model_id", "split_seed", "strategies"):
        if safe.get(key) != expected[key]:
            raise ValueError(f"Configuration field {key!r} differs from the frozen contract")
    if safe.get("data") != expected["data"]:
        raise ValueError("Configuration data settings differ from the frozen contract")
    if safe.get("evaluation") != expected["evaluation"]:
        raise ValueError("Configuration evaluation settings differ from the frozen contract")
    if safe.get("semantic_assets") != expected["semantic_assets"]:
        raise ValueError("Configuration semantic assets differ from the frozen contract")
    features = safe.get("features")
    if features != expected["features"]:
        raise ValueError("Configuration must enable semantic primary and identity-vs-semantic ablation")
    model = safe.get("model")
    if not isinstance(model, dict):
        raise ValueError("Configuration model must be an object")
    build_direct_model_settings(model, epochs=FORMAL_EPOCHS)
    profiles = safe.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Configuration profiles must be an object")
    frozen_profiles = expected["profiles"]
    if profiles != frozen_profiles:
        raise ValueError("Configuration profiles differ from the frozen schedules")
    if safe != expected:
        raise ValueError("Configuration differs from the complete frozen contract")
    return safe


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and validate the frozen configuration."""

    import yaml

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not load configuration: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError("Configuration file must contain a mapping")
    return validate_config(value)


def _atomic_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically write strict canonical JSON with a trailing newline."""

    return _atomic_text(path, canonical_json(value) + "\n")


def atomic_write_csv(
    path: str | Path,
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically write a DataFrame or a homogeneous sequence of row mappings."""

    if isinstance(rows, pd.DataFrame):
        text = rows.to_csv(index=False, lineterminator="\n")
    else:
        records = [_json_safe_copy(row, role="CSV row") for row in rows]
        if any(not isinstance(row, dict) for row in records):
            raise ValueError("CSV rows must be mappings")
        fieldnames = list(records[0]) if records else []
        if any(list(row) != fieldnames for row in records):
            raise ValueError("CSV rows must have identical keys in identical order")
        from io import StringIO

        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)
        text = buffer.getvalue()
    return _atomic_text(path, text)


def validate_receipt(
    receipt: Mapping[str, Any],
    forbidden_keys: Iterable[str] = RECEIPT_FORBIDDEN_KEYS,
) -> dict[str, Any]:
    """Reject forbidden payload keys at any nesting depth and require strict JSON."""

    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    forbidden = {str(key).strip().casefold() for key in forbidden_keys}

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"Receipt key at {location} must be a string")
                if key.strip().casefold() in forbidden:
                    raise ValueError(f"Receipt contains forbidden key at {location}.{key}")
                walk(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")

    walk(receipt, "receipt")
    safe = _json_safe_copy(receipt, role="Receipt")
    if not isinstance(safe, dict):
        raise AssertionError("Receipt JSON copy unexpectedly changed type")
    return safe


def seed_directory(output_root: str | Path, run_seed: int) -> Path:
    """Return the canonical directory for one formal/smoke run seed."""

    return Path(output_root) / f"seed_{_nonnegative_int(run_seed, 'run_seed')}"


def seed_manifest_path(seed_dir: str | Path) -> Path:
    return Path(seed_dir) / SEED_MANIFEST_FILENAME


def make_seed_manifest(run_seed: int, run_identity: Mapping[str, Any]) -> dict[str, Any]:
    identity = _json_safe_copy(run_identity, role="Run identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Run identity must be a nonempty JSON object")
    return {
        "schema": SEED_MANIFEST_SCHEMA,
        "run_seed": _nonnegative_int(run_seed, "run_seed"),
        "run_identity": identity,
        "run_identity_sha256": canonical_hash(identity),
    }


def validate_run_identity(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Require exact canonical equality between two run identities."""

    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("Existing run identity does not match the requested run")


def validate_seed_manifest(
    path_or_directory: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and validate an existing seed manifest and its identity digest."""

    path = Path(path_or_directory)
    if path.is_dir():
        path = seed_manifest_path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Seed manifest is missing or invalid: {path}") from error
    expected = make_seed_manifest(run_seed, run_identity)
    if not isinstance(manifest, dict):
        raise ValueError("Seed manifest must contain a JSON object")
    if manifest.get("run_identity_sha256") != canonical_hash(
        manifest.get("run_identity")
    ):
        raise ValueError("Seed manifest run identity hash is invalid")
    validate_run_identity(manifest, expected)
    return expected


def ensure_seed_directory(
    output_root: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
) -> Path:
    """Create a new seed directory or validate an existing run identity."""

    directory = seed_directory(output_root, run_seed)
    manifest_path = seed_manifest_path(directory)
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Seed path exists and is not a directory: {directory}")
        if manifest_path.exists():
            validate_seed_manifest(manifest_path, run_seed, run_identity)
            return directory
        if any(directory.iterdir()):
            raise ValueError("Existing nonempty seed directory has no manifest")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, make_seed_manifest(run_seed, run_identity))
    return directory


def _seed_values(
    values: Mapping[int, float] | Sequence[Mapping[str, Any]],
) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    if isinstance(values, Mapping):
        source = values.items()
    else:
        source = []
        for record in values:
            if not isinstance(record, Mapping) or "seed" not in record or "value" not in record:
                raise ValueError("Per-seed records must contain seed and value")
            source.append((record["seed"], record["value"]))
    seen: set[int] = set()
    for raw_seed, raw_value in source:
        seed = _nonnegative_int(raw_seed, "seed")
        if seed in seen:
            raise ValueError(f"Duplicate seed: {seed}")
        seen.add(seed)
        if isinstance(raw_value, bool):
            raise ValueError("Metric values must be finite numbers")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Metric values must be finite numbers") from error
        if not math.isfinite(value):
            raise ValueError("Metric values must be finite numbers")
        pairs.append((seed, value))
    pairs.sort()
    return pairs


def aggregate_metrics(
    values: Mapping[int, float] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly two or five unique per-seed metric observations."""

    pairs = _seed_values(values)
    n = len(pairs)
    if n not in T_CRITICAL_95:
        raise ValueError("Metric aggregation requires exactly 2 or 5 unique seeds")
    array = np.asarray([value for _, value in pairs], dtype=np.float64)
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    q25, median, q75 = (float(value) for value in np.quantile(array, [0.25, 0.5, 0.75]))
    t_critical = T_CRITICAL_95[n]
    half_width = float(t_critical * sample_sd / math.sqrt(n))
    return {
        "per_seed": [{"seed": seed, "value": value} for seed, value in pairs],
        "raw": [value for _, value in pairs],
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "t_critical_95": t_critical,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
        "ci95": [float(mean - half_width), float(mean + half_width)],
    }


def paired_policy_differences(
    policy: Mapping[int, float] | Sequence[Mapping[str, Any]],
    reference: Mapping[int, float] | Sequence[Mapping[str, Any]],
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Summarize paired policy-minus-reference differences over aligned seeds."""

    left = _seed_values(policy)
    right = _seed_values(reference)
    left_seeds = [seed for seed, _ in left]
    right_seeds = [seed for seed, _ in right]
    if left_seeds != right_seeds:
        raise ValueError("Paired policies must contain exactly aligned unique seeds")
    direction = 1.0 if higher_is_better else -1.0
    rows = []
    directed: dict[int, float] = {}
    for (seed, policy_value), (_, reference_value) in zip(left, right, strict=True):
        raw_difference = policy_value - reference_value
        directed_difference = direction * raw_difference
        directed[seed] = directed_difference
        rows.append(
            {
                "seed": seed,
                "policy": policy_value,
                "reference": reference_value,
                "raw_difference": raw_difference,
                "directed_difference": directed_difference,
            }
        )
    aggregate = aggregate_metrics(directed)
    aggregate["per_seed"] = rows
    aggregate["raw_difference"] = [row["raw_difference"] for row in rows]
    aggregate["directed_difference"] = [row["directed_difference"] for row in rows]
    aggregate["wins"] = int(sum(value > 0.0 for value in directed.values()))
    aggregate["ties"] = int(sum(value == 0.0 for value in directed.values()))
    aggregate["losses"] = int(sum(value < 0.0 for value in directed.values()))
    aggregate["win_fraction"] = float(aggregate["wins"] / aggregate["n"])
    aggregate["higher_is_better"] = bool(higher_is_better)
    return aggregate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Hash a stable regular file and return its resolved identity record."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Identity input is not a regular file: {resolved}")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": int(after.st_size),
        "sha256": digest,
    }


def json_null_nonfinite(value: Any, *, role: str = "value") -> Any:
    """Convert nonfinite scalar metrics to null and reject unsafe objects."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{role} mapping keys must be strings")
            result[key] = json_null_nonfinite(item, role=f"{role}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            json_null_nonfinite(item, role=f"{role}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_null_nonfinite(value.item(), role=role)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        raise ValueError(f"{role} cannot serialize an array")
    raise ValueError(f"{role} contains a value that is not JSON-safe")


def atomic_write_json_nullable(path: str | Path, value: Any) -> Path:
    """Write strict JSON after replacing metric NaN/Inf scalars with null."""

    return atomic_write_json(path, json_null_nonfinite(value))


def write_root_manifest(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Normalize and atomically write a self-hashed root manifest."""

    payload = json_null_nonfinite(dict(value), role="Root manifest")
    if not isinstance(payload, dict):
        raise AssertionError("Root manifest normalization changed its type")
    payload.pop("manifest_payload_sha256", None)
    payload["manifest_payload_sha256"] = canonical_hash(payload)
    return atomic_write_json(path, payload)


def _atomic_binary_copy(source: str | Path, destination: str | Path) -> Path:
    """Atomically copy one regular file byte-for-byte and fsync it."""

    source_path = Path(source).resolve(strict=True)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise ValueError(f"Snapshot source is not a regular file: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if sha256_file(source_path) != sha256_file(temporary):
            raise ValueError(f"Atomic snapshot copy hash mismatch: {source_path}")
        os.replace(temporary, destination_path)
        _fsync_directory(destination_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


def _read_strict_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"JSON artifact is missing or invalid: {path}") from error


def _profile_spec(config: Mapping[str, Any], profile: str) -> dict[str, Any]:
    if profile not in {"smoke", "formal"}:
        raise ValueError("profile must be 'smoke' or 'formal'")
    values = dict(config["profiles"][profile])
    return {
        "name": profile,
        "scientific": profile == "formal",
        "diagnostic_only": profile == "smoke",
        "seeds": tuple(int(value) for value in values["seeds"]),
        "initial_budget": int(values["initial_budget"]),
        "batch_size": int(values["acquisition_batch_size"]),
        "checkpoints": tuple(int(value) for value in values["checkpoints"]),
        "epochs": int(values["epochs"]),
        "mc_passes": int(values["mc_passes"]),
        "ablation_fixed_budgets": (128, 512) if profile == "formal" else (32, 64),
    }


def deterministic_nested_ids(
    candidate_ids: Iterable[object], run_seed: int
) -> tuple[str, ...]:
    """Return a deterministic, order-independent nested permutation of pool IDs."""

    run = _nonnegative_int(run_seed, "run_seed")
    ids = tuple(str(value) for value in candidate_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")

    def rank(sample_id: str) -> tuple[bytes, str]:
        payload = canonical_json(
            {"namespace": "ablation", "run_seed": run, "sample_id": sample_id}
        )
        return hashlib.sha256(payload.encode("utf-8")).digest(), sample_id

    return tuple(sorted(ids, key=rank))


def validate_dataset_partitions(
    dataset: GroupedDataset,
    *,
    require_registered_splits: bool = True,
) -> dict[str, Any]:
    """Validate immutable candidate/evaluation membership and response alignment."""

    metadata_ids = tuple(str(value) for value in dataset.metadata.index)
    response_ids = tuple(str(value) for value in dataset.response.index)
    if len(set(metadata_ids)) != len(metadata_ids):
        raise ValueError("Dataset metadata condition IDs must be unique")
    if response_ids != metadata_ids:
        raise ValueError("Dataset response rows must exactly align with metadata rows")
    pool_ids = tuple(sorted(str(value) for value in dataset.candidate_pool_ids))
    if len(set(pool_ids)) != len(pool_ids):
        raise ValueError("Candidate-pool IDs must be unique")
    metadata_set = set(metadata_ids)
    if not set(pool_ids) <= metadata_set:
        raise ValueError("Candidate-pool IDs must be present in metadata")
    expected_splits = (INTERPOLATION_SPLIT, *VALIDATION_SPLITS)
    if require_registered_splits and set(dataset.validation_ids) != set(expected_splits):
        raise ValueError(
            "Evaluation panel must contain interpolation and every fixed validation split"
        )
    ordered_names = [name for name in expected_splits if name in dataset.validation_ids]
    ordered_names.extend(sorted(set(dataset.validation_ids) - set(ordered_names)))
    seen = set(pool_ids)
    evaluation_counts: dict[str, int] = {}
    evaluation_ids: dict[str, list[str]] = {}
    for split_name in ordered_names:
        ids = tuple(str(value) for value in dataset.validation_ids[split_name])
        if len(set(ids)) != len(ids):
            raise ValueError(f"Evaluation split {split_name} contains duplicate IDs")
        if not set(ids) <= metadata_set:
            raise ValueError(f"Evaluation split {split_name} contains unknown IDs")
        overlap = seen & set(ids)
        if overlap:
            raise ValueError(
                f"Evaluation split {split_name} overlaps candidate or evaluation IDs: "
                f"{sorted(overlap)[:5]}"
            )
        seen.update(ids)
        evaluation_counts[split_name] = len(ids)
        evaluation_ids[split_name] = list(ids)
    return {
        "candidate_pool_count": len(pool_ids),
        "candidate_pool_ids_sha256": canonical_hash(list(sorted(pool_ids))),
        "evaluation_counts": evaluation_counts,
        "evaluation_ids_sha256": {
            name: canonical_hash(values) for name, values in evaluation_ids.items()
        },
        "evaluation_ids_revealable": False,
        "pairwise_disjoint": True,
    }


def validate_control_contract(dataset: GroupedDataset) -> dict[str, Any]:
    """Validate the preregistered pooled-control policy and return its identity."""

    summary = _json_safe_copy(
        dataset.control_policy_summary, role="Dataset control_policy_summary"
    )
    if not isinstance(summary, dict):
        raise ValueError("Dataset control_policy_summary must be a JSON object")
    mismatches = {
        key: {"expected": expected, "actual": summary.get(key)}
        for key, expected in CONTROL_CONTRACT.items()
        if key not in summary or summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Dataset control_policy_summary differs from the frozen pooled contract: "
            f"{mismatches}"
        )
    contract = dict(CONTROL_CONTRACT)
    return {
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "summary": summary,
        "summary_sha256": canonical_hash(summary),
    }


def feature_bundle_identity(bundle: FeatureBundle) -> dict[str, Any]:
    """Return every stable target-free feature identity needed for resume."""

    summary = _json_safe_copy(bundle.summary, role="Feature summary")
    assets = _json_safe_copy(bundle.asset_hashes, role="Feature asset hashes")
    return {
        "row_count": len(bundle.row_ids),
        "row_ids_sha256": bundle.row_ids_sha256,
        "descriptor_matrix": dict(bundle.descriptor_matrix_identity or {}),
        "model_matrix": dict(bundle.model_matrix_identity or {}),
        "summary": summary,
        "summary_sha256": canonical_hash(summary),
        "asset_hashes": assets,
        "asset_hashes_sha256": canonical_hash(assets),
    }


def _row_positions(row_ids: Sequence[str], requested_ids: Sequence[str]) -> np.ndarray:
    lookup = {str(row_id): position for position, row_id in enumerate(row_ids)}
    missing = [str(row_id) for row_id in requested_ids if str(row_id) not in lookup]
    if missing:
        raise ValueError(f"Condition IDs are missing from the feature row index: {missing[:5]}")
    return np.asarray([lookup[str(row_id)] for row_id in requested_ids], dtype=np.int64)


def prepare_masked_model_features(
    bundle: FeatureBundle,
    labelled_ids: Sequence[str],
) -> np.ndarray:
    """Mask predictor features using exactly the current labelled support rows."""

    labelled = tuple(str(value) for value in labelled_ids)
    if not labelled:
        raise ValueError("At least one labelled ID is required for support masking")
    if len(set(labelled)) != len(labelled):
        raise ValueError("labelled_ids must be unique")
    positions = _row_positions(bundle.row_ids, labelled)
    values = np.array(bundle.model_matrix, dtype=np.float32, copy=True)
    support = np.array(values[positions], dtype=np.float32, copy=True)
    if bundle.masker is None:
        raise ValueError("Runner feature bundles must provide a support masker")
    masked = np.asarray(bundle.masker(values, support), dtype=np.float32)
    if masked.shape != values.shape or not np.isfinite(masked).all():
        raise ValueError("Feature masker returned an invalid predictor matrix")

    slice_summary = bundle.summary.get("categorical_column_slices", {})
    if isinstance(slice_summary, Mapping) and slice_summary:
        mutable = np.zeros(values.shape[1], dtype=bool)
        for name, block in slice_summary.items():
            if not isinstance(block, Mapping):
                raise ValueError(f"Categorical slice {name!r} must be an object")
            start, stop = int(block["start"]), int(block["stop"])
            if not 0 <= start <= stop <= values.shape[1]:
                raise ValueError(f"Categorical slice {name!r} is invalid")
            mutable[start:stop] = True
        if not np.array_equal(masked[:, ~mutable], values[:, ~mutable]):
            raise ValueError(
                "Feature masker changed continuous time or semantic feature blocks"
            )
    return np.array(masked, dtype=np.float32, copy=True)


def curve_summary(
    budgets: Sequence[float],
    values: Sequence[float],
    full_reference: float,
    *,
    higher_is_better: bool = True,
    target_fraction: float = TARGET_FRACTION,
) -> dict[str, Any]:
    """Summarize an irregular learning curve without extrapolating B80."""

    budget_array = np.asarray(budgets, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    base = {
        "budgets": budget_array.tolist(),
        "values": value_array.tolist(),
        "target_fraction": float(target_fraction),
    }
    if (
        len(budget_array) < 2
        or not np.isfinite(budget_array).all()
        or not np.isfinite(value_array).all()
        or not math.isfinite(float(full_reference))
    ):
        return {
            **base,
            "normalized_aulc": None,
            "b80": None,
            "b80_status": "not_reached",
            "b80_reason": "nonfinite_or_insufficient_curve",
        }
    aulc = normalized_trapezoidal_aulc(
        budget_array, value_array, higher_is_better=higher_is_better
    )
    direction = 1.0 if higher_is_better else -1.0
    if direction * float(full_reference) <= direction * float(value_array[0]):
        target = {
            "status": "not_reached",
            "budget": None,
            "reason": "full_reference_not_better_than_initial",
        }
    else:
        target = budget_to_target(
            budget_array,
            value_array,
            float(full_reference),
            target_fraction=target_fraction,
            higher_is_better=higher_is_better,
        )
    return {
        **base,
        "normalized_aulc": float(aulc),
        "b80": target.get("budget"),
        "b80_status": str(target["status"]),
        "b80_reason": target.get("reason"),
    }


_ALL_T_CRITICAL_95 = MappingProxyType(
    {2: 12.706205, 3: 4.302653, 4: 3.182446, 5: 2.776445}
)


def finite_metric_summary(values: Sequence[float]) -> dict[str, Any]:
    """Aggregate one to five finite values with sample statistics and a t CI."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = int(len(array))
    if n == 0:
        raise ValueError("At least one finite metric value is required")
    if n > 5:
        raise ValueError("At most five registered seed values are supported")
    q25, median, q75 = (
        float(value) for value in np.quantile(array, [0.25, 0.5, 0.75])
    )
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1)) if n > 1 else None
    t_critical = _ALL_T_CRITICAL_95.get(n)
    if sample_sd is None or t_critical is None:
        ci_low = ci_high = None
    else:
        half_width = float(t_critical * sample_sd / math.sqrt(n))
        ci_low, ci_high = float(mean - half_width), float(mean + half_width)
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sample_sd,
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": float(q75 - q25),
        "t_critical_95": t_critical,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def policy_superiority_decision(
    comparison: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Apply the preregistered directed paired-mean/CI/four-win rule."""

    if profile not in {"smoke", "formal"}:
        raise ValueError("profile must be 'smoke' or 'formal'")
    if profile == "smoke":
        return {
            "status": "diagnostic_only",
            "beats_random": None,
            "required_formal_seed_count": 5,
            "required_wins": 4,
        }
    n = int(comparison.get("n", 0))
    mean = comparison.get("mean")
    ci_low = comparison.get("ci95_low")
    wins = int(comparison.get("wins", 0))
    beats = bool(
        n == 5
        and mean is not None
        and ci_low is not None
        and float(mean) > 0.0
        and float(ci_low) > 0.0
        and wins >= 4
    )
    return {
        "status": "beats_random" if beats else "retain_random",
        "beats_random": beats,
        "required_formal_seed_count": 5,
        "required_wins": 4,
        "checks": {
            "five_aligned_seeds": n == 5,
            "directed_mean_gt_zero": mean is not None and float(mean) > 0.0,
            "directed_ci95_low_gt_zero": ci_low is not None and float(ci_low) > 0.0,
            "wins_at_least_four": wins >= 4,
        },
    }


def _csv_shape(path: Path) -> tuple[int, list[str]]:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"CSV artifact is unreadable: {path}") from error
    return int(len(frame)), [str(column) for column in frame.columns]


def artifact_inventory(
    directory: str | Path,
    *,
    exclude: Iterable[str] = (SEED_MANIFEST_FILENAME,),
) -> dict[str, dict[str, Any]]:
    """Inventory every regular artifact beneath a directory deterministically."""

    root = Path(directory)
    excluded = {str(value) for value in exclude}
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda value: str(value.relative_to(root))):
        if path.is_symlink():
            raise ValueError(f"Artifact inventories do not permit symlinks: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in excluded:
            continue
        record: dict[str, Any] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "rows": None,
            "columns": None,
        }
        if path.suffix.casefold() == ".csv":
            rows, columns = _csv_shape(path)
            record.update({"rows": rows, "columns": columns})
        inventory[relative] = record
    return inventory


def validate_artifact_inventory(
    directory: str | Path,
    inventory: Mapping[str, Mapping[str, Any]],
    *,
    exclude: Iterable[str] = (SEED_MANIFEST_FILENAME,),
) -> None:
    """Fail closed on missing, extra, tampered, or schema-changed artifacts."""

    root = Path(directory)
    expected = {str(key): dict(value) for key, value in inventory.items()}
    actual = artifact_inventory(root, exclude=exclude)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"Artifact inventory file set mismatch: missing={missing}, extra={extra}")
    for name in sorted(expected):
        if actual[name] != expected[name]:
            raise ValueError(f"Artifact inventory mismatch for {name}")


def create_seed_staging(output_root: str | Path, run_seed: int) -> Path:
    """Create a unique hidden staging directory for one seed."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = Path(
        tempfile.mkdtemp(
            prefix=f".seed_{_nonnegative_int(run_seed, 'run_seed')}.staging-",
            dir=root,
        )
    )
    _fsync_directory(root)
    return path


def _expected_round_count(run_identity: Mapping[str, Any]) -> int | None:
    schedule = run_identity.get("schedule")
    if not isinstance(schedule, Mapping):
        return None
    try:
        initial = int(schedule["initial_budget"])
        batch = int(schedule["batch_size"])
        final = int(tuple(schedule["checkpoints"])[-1])
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError("Run identity contains an invalid schedule")
    if initial <= 0 or batch <= 0 or final < initial or (final - initial) % batch:
        raise ValueError("Run identity contains an invalid dense round schedule")
    return 1 + (final - initial) // batch


def _integer_series(frame: pd.DataFrame, column: str, role: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"{role} is missing column {column!r}")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{role}.{column} must contain exact integers")
    return numeric.astype(np.int64)


def _require_exact_columns(frame: pd.DataFrame, expected: Sequence[str], role: str) -> None:
    actual = [str(value) for value in frame.columns]
    if actual != list(expected):
        raise ValueError(
            f"{role} columns differ from the exact contract: "
            f"expected={list(expected)}, actual={actual}"
        )


def _require_exact_grid(
    frame: pd.DataFrame,
    columns: Sequence[str],
    expected: set[tuple[Any, ...]],
    role: str,
) -> None:
    actual_rows = [tuple(row) for row in frame[list(columns)].itertuples(index=False, name=None)]
    actual = set(actual_rows)
    if len(actual_rows) != len(actual) or actual != expected:
        missing = sorted(expected - actual, key=str)
        extra = sorted(actual - expected, key=str)
        raise ValueError(
            f"{role} exact grid mismatch: duplicates={len(actual_rows) - len(actual)}, "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )


def _canonical_split_metric_record(
    value: Mapping[str, Any], *, role: str
) -> str:
    """Canonicalize one split metric row across JSON and CSV scalar types."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    expected_keys = {"split", "primary_split", *COUNT_METRICS, *SCORE_METRICS}
    if set(value) != expected_keys:
        raise ValueError(f"{role} fields differ from the exact metric contract")
    primary = value["primary_split"]
    if not isinstance(primary, (bool, np.bool_)):
        raise ValueError(f"{role}.primary_split must be boolean")
    normalized: dict[str, Any] = {
        "split": str(value["split"]),
        "primary_split": bool(primary),
    }
    for name in COUNT_METRICS:
        raw = json_null_nonfinite(value[name], role=f"{role}.{name}")
        if raw is None or isinstance(raw, bool):
            raise ValueError(f"{role}.{name} must be a finite integer")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{role}.{name} must be a finite integer") from error
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{role}.{name} must be a finite integer")
        normalized[name] = int(numeric)
    for name in SCORE_METRICS:
        raw = json_null_nonfinite(value[name], role=f"{role}.{name}")
        if raw is None:
            normalized[name] = None
            continue
        if isinstance(raw, bool):
            raise ValueError(f"{role}.{name} must be numeric or null")
        try:
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{role}.{name} must be numeric or null") from error
        normalized[name] = (
            (0.0 if numeric == 0.0 else numeric) if math.isfinite(numeric) else None
        )
    return canonical_json(normalized)


def validate_seed_semantics(
    directory: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
) -> None:
    """Reconcile one seed against the exact registered schedule and dataset contract."""

    root = Path(directory)
    schedule = run_identity.get("schedule")
    partition = run_identity.get("partition_contract")
    if not isinstance(schedule, Mapping) or not isinstance(partition, Mapping):
        raise ValueError("Run identity must contain strict schedule and partition_contract")
    try:
        initial = int(schedule["initial_budget"])
        batch = int(schedule["batch_size"])
        checkpoints = tuple(int(value) for value in schedule["checkpoints"])
        fixed = tuple(int(value) for value in schedule["ablation_fixed_budgets"])
        candidate_count = int(partition["candidate_pool_count"])
        split_counts = partition["evaluation_counts"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Run identity schedule or partition contract is invalid") from error
    if not isinstance(split_counts, Mapping) or not split_counts:
        raise ValueError("Run identity has no registered evaluation split contract")
    splits = tuple(str(value) for value in split_counts)
    if (
        initial <= 0
        or batch <= 0
        or not checkpoints
        or checkpoints[0] != initial
        or tuple(sorted(set(checkpoints))) != checkpoints
        or checkpoints[-1] > candidate_count
        or (checkpoints[-1] - initial) % batch
    ):
        raise ValueError("Run identity contains an invalid exact schedule")
    final_budget = checkpoints[-1]
    dense_budgets = tuple(range(initial, final_budget + 1, batch))
    expected_rounds = {budget: index for index, budget in enumerate(dense_budgets)}
    ablation_budgets = tuple(dict.fromkeys((*fixed, candidate_count)))
    if any(value <= 0 or value > candidate_count for value in ablation_budgets):
        raise ValueError("Run identity contains an invalid ablation schedule")

    active = pd.read_csv(root / "active_metrics.csv", float_precision="round_trip")
    full = pd.read_csv(
        root / "full_reference_metrics.csv", float_precision="round_trip"
    )
    ablation = pd.read_csv(
        root / "ablation_metrics.csv", float_precision="round_trip"
    )
    acquisitions = pd.read_csv(
        root / "acquisitions.csv", float_precision="round_trip"
    )
    _require_exact_columns(active, _active_columns(), "active_metrics")
    _require_exact_columns(full, _active_columns(), "full_reference_metrics")
    _require_exact_columns(ablation, _active_columns(), "ablation_metrics")
    _require_exact_columns(acquisitions, _acquisition_columns(), "acquisitions")
    seed = _nonnegative_int(run_seed, "run_seed")

    for frame, role in (
        (active, "active_metrics"),
        (full, "full_reference_metrics"),
        (ablation, "ablation_metrics"),
        (acquisitions, "acquisitions"),
    ):
        if not _integer_series(frame, "seed", role).eq(seed).all():
            raise ValueError(f"{role} contains a wrong seed")

    active["budget"] = _integer_series(active, "budget", "active_metrics")
    active["round"] = _integer_series(active, "round", "active_metrics")
    active["model_seed"] = _integer_series(active, "model_seed", "active_metrics")
    expected_active = {
        (strategy, budget, split)
        for strategy in STRATEGIES
        for budget in checkpoints
        for split in splits
    }
    _require_exact_grid(
        active, ("strategy", "budget", "split"), expected_active, "active_metrics"
    )
    if (
        set(active["role"]) != {"active_learning"}
        or set(active["feature_mode"]) != {"semantic"}
        or any(bool(row.primary_split) is not (str(row.split) == INTERPOLATION_SPLIT) for row in active.itertuples())
        or any(
            int(row.round) != expected_rounds[int(row.budget)]
            or int(row.model_seed) != model_seed(seed, int(row.budget))
            for row in active.itertuples()
        )
    ):
        raise ValueError("active_metrics semantic identity fields are invalid")

    full["budget"] = _integer_series(full, "budget", "full_reference_metrics")
    full["model_seed"] = _integer_series(full, "model_seed", "full_reference_metrics")
    _require_exact_grid(
        full,
        ("strategy", "budget", "split"),
        {("full_reference", candidate_count, split) for split in splits},
        "full_reference_metrics",
    )
    if (
        set(full["role"]) != {"full_pool_reference"}
        or set(full["feature_mode"]) != {"semantic"}
        or any(bool(row.primary_split) is not (str(row.split) == INTERPOLATION_SPLIT) for row in full.itertuples())
        or not full["model_seed"].eq(model_seed(seed, candidate_count)).all()
        or not full["round"].isna().all()
    ):
        raise ValueError("full_reference_metrics semantic identity fields are invalid")

    ablation["budget"] = _integer_series(ablation, "budget", "ablation_metrics")
    ablation["model_seed"] = _integer_series(
        ablation, "model_seed", "ablation_metrics"
    )
    _require_exact_grid(
        ablation,
        ("feature_mode", "budget", "split"),
        {
            (mode, budget, split)
            for mode in ("identity", "semantic")
            for budget in ablation_budgets
            for split in splits
        },
        "ablation_metrics",
    )
    if (
        set(ablation["role"]) != {"representation_ablation"}
        or set(ablation["strategy"]) != {"representation_ablation"}
        or any(bool(row.primary_split) is not (str(row.split) == INTERPOLATION_SPLIT) for row in ablation.itertuples())
        or not ablation["round"].isna().all()
        or any(
            int(row.model_seed) != model_seed(seed, int(row.budget))
            for row in ablation.itertuples()
        )
    ):
        raise ValueError("ablation_metrics semantic identity fields are invalid")

    metric_fields = ("split", "primary_split", *COUNT_METRICS, *SCORE_METRICS)
    full_metric_rows = {
        str(row["split"]): _canonical_split_metric_record(
            {key: row[key] for key in metric_fields},
            role=f"full_reference_metrics[{row['split']}]",
        )
        for _, row in full.iterrows()
    }
    semantic_full = ablation[
        ablation["feature_mode"].eq("semantic")
        & ablation["budget"].eq(candidate_count)
    ]
    semantic_full_metric_rows = {
        str(row["split"]): _canonical_split_metric_record(
            {key: row[key] for key in metric_fields},
            role=f"semantic full ablation[{row['split']}]",
        )
        for _, row in semantic_full.iterrows()
    }
    if semantic_full_metric_rows != full_metric_rows:
        raise ValueError(
            "Semantic full-pool ablation metrics differ from reused full-reference metrics"
        )

    acquisitions["round"] = _integer_series(acquisitions, "round", "acquisitions")
    for name in ("rank_in_batch", "budget_before", "budget_after", "model_seed"):
        acquisitions[name] = _integer_series(acquisitions, name, "acquisitions")
    actual_strategies = set(acquisitions["strategy"].astype(str))
    if (
        actual_strategies != set(STRATEGIES)
        or len(acquisitions) != len(STRATEGIES) * final_budget
    ):
        raise ValueError(
            "acquisitions must contain exactly final_budget rows for every registered "
            "strategy and no other strategies"
        )
    pool_selected: dict[str, list[str]] = {}
    initial_order: tuple[str, ...] | None = None
    for strategy in STRATEGIES:
        rows = acquisitions[acquisitions["strategy"].eq(strategy)].sort_values(
            ["round", "rank_in_batch"], kind="stable"
        )
        if len(rows) != final_budget or rows[CONDITION_ID].astype(str).duplicated().any():
            raise ValueError(f"{strategy} acquisitions must contain final_budget unique IDs")
        labelled: list[str] = []
        for round_index, budget_after in enumerate(dense_budgets):
            group = rows[rows["round"].eq(round_index)]
            size = initial if round_index == 0 else batch
            budget_before = 0 if round_index == 0 else dense_budgets[round_index - 1]
            if (
                len(group) != size
                or tuple(group["rank_in_batch"]) != tuple(range(1, size + 1))
                or not group["budget_before"].eq(budget_before).all()
                or not group["budget_after"].eq(budget_after).all()
            ):
                raise ValueError(f"{strategy} acquisition round {round_index} is invalid")
            if round_index == 0:
                if set(group["selection_type"]) != {"deterministic_initial"} or not group[
                    "acquisition_seed"
                ].isna().all():
                    raise ValueError(f"{strategy} initial acquisition contract is invalid")
                expected_model_seed = model_seed(seed, initial)
            else:
                if set(group["selection_type"]) != {strategy}:
                    raise ValueError(f"{strategy} selection_type is invalid")
                actual_acquisition = pd.to_numeric(
                    group["acquisition_seed"], errors="coerce"
                )
                if actual_acquisition.isna().any() or not actual_acquisition.astype(
                    np.int64
                ).eq(acquisition_seed(seed, budget_before)).all():
                    raise ValueError(f"{strategy} acquisition seed continuity is invalid")
                expected_model_seed = model_seed(seed, budget_before)
            if not group["model_seed"].eq(expected_model_seed).all():
                raise ValueError(f"{strategy} acquisition model seed is invalid")
            labelled.extend(group[CONDITION_ID].astype(str))
        current_initial = tuple(labelled[:initial])
        if initial_order is None:
            initial_order = current_initial
        elif current_initial != initial_order:
            raise ValueError("All strategies must use the same ordered initial IDs")
        pool_selected[strategy] = labelled

    metric_keys = {"split", "primary_split", *COUNT_METRICS, *SCORE_METRICS}
    for strategy in STRATEGIES:
        labelled: list[str] = []
        for round_index, budget_after in enumerate(dense_budgets):
            receipt = _read_strict_json(
                root / "round_receipts" / strategy / f"round_{round_index:03d}.json"
            )
            group = acquisitions[
                acquisitions["strategy"].eq(strategy)
                & acquisitions["round"].eq(round_index)
            ].sort_values("rank_in_batch", kind="stable")
            selected = group[CONDITION_ID].astype(str).tolist()
            labelled.extend(selected)
            budget_before = 0 if round_index == 0 else dense_budgets[round_index - 1]
            checkpoint = budget_after in checkpoints
            expected_acquisition_seed = (
                None if round_index == 0 else acquisition_seed(seed, budget_before)
            )
            if (
                receipt.get("round_index") != round_index
                or receipt.get("strategy") != strategy
                or receipt.get("budget_before") != budget_before
                or receipt.get("budget_after") != budget_after
                or receipt.get("selected_ids") != selected
                or receipt.get("labelled_ids") != labelled
                or receipt.get("global_seed") != seed
                or receipt.get("acquisition_seed") != expected_acquisition_seed
                or receipt.get("model_seed") != model_seed(seed, budget_after)
                or receipt.get("checkpoint") is not checkpoint
                or receipt.get("labelled_ids_sha256") != canonical_hash(labelled)
            ):
                raise ValueError(f"Round receipt reconciliation failed for {strategy} round {round_index}")
            hashes = receipt.get("hashes")
            if not isinstance(hashes, Mapping) or hashes.get(
                "selected_ids_sha256"
            ) != canonical_hash(selected):
                raise ValueError("Round receipt selected_ids hash is invalid")
            split_metrics = receipt.get("split_metrics")
            if not isinstance(split_metrics, list):
                raise ValueError("Round receipt split_metrics must be a list")
            if checkpoint:
                if len(split_metrics) != len(splits) or {
                    str(value.get("split")) for value in split_metrics
                } != set(splits) or any(
                    set(value) != metric_keys
                    or value.get("primary_split") is not (
                        value.get("split") == INTERPOLATION_SPLIT
                    )
                    for value in split_metrics
                ):
                    raise ValueError("Round receipt checkpoint split metric names are invalid")
                active_rows = active[
                    active["strategy"].eq(strategy)
                    & active["budget"].eq(budget_after)
                ]
                expected_metric_rows = {
                    str(row["split"]): _canonical_split_metric_record(
                        {
                            key: row[key]
                            for key in (
                                "split",
                                "primary_split",
                                *COUNT_METRICS,
                                *SCORE_METRICS,
                            )
                        },
                        role=(
                            f"active_metrics[{strategy}, {budget_after}, "
                            f"{row['split']}]"
                        ),
                    )
                    for _, row in active_rows.iterrows()
                }
                actual_metric_rows = {
                    str(value["split"]): _canonical_split_metric_record(
                        value,
                        role=(
                            f"round receipt split_metrics[{strategy}, "
                            f"{round_index}, {value.get('split')}]"
                        ),
                    )
                    for value in split_metrics
                }
                if actual_metric_rows != expected_metric_rows:
                    raise ValueError(
                        "Round receipt checkpoint split metric values do not reconcile "
                        f"with active_metrics for {strategy} round {round_index}"
                    )
            elif split_metrics:
                raise ValueError("Non-checkpoint round receipt contains split metrics")

    fit_payload = _read_strict_json(root / "fit_receipts.json")
    if (
        not isinstance(fit_payload, dict)
        or fit_payload.get("schema") != "goai.direct_multiseed.fit_receipts.v1"
        or fit_payload.get("run_seed") != seed
        or not isinstance(fit_payload.get("receipts"), list)
    ):
        raise ValueError("fit_receipts envelope differs from the exact contract")
    receipts = fit_payload["receipts"]
    expected_fit_grid = {
        ("active_learning", strategy, "semantic", budget)
        for strategy in STRATEGIES
        for budget in dense_budgets
    } | {("full_pool_reference", "full_reference", "semantic", candidate_count)} | {
        ("representation_ablation", mode, mode, budget)
        for mode in ("identity", "semantic")
        for budget in ablation_budgets
    }
    actual_fit_grid = [
        (
            value.get("role"),
            value.get("strategy"),
            value.get("feature_mode"),
            value.get("budget"),
        )
        for value in receipts
        if isinstance(value, Mapping)
    ]
    if len(actual_fit_grid) != len(receipts) or len(set(actual_fit_grid)) != len(
        actual_fit_grid
    ) or set(actual_fit_grid) != expected_fit_grid:
        raise ValueError("fit_receipts exact role/budget/strategy/mode grid mismatch")
    for receipt in receipts:
        budget = int(receipt["budget"])
        if (
            receipt.get("run_seed") != seed
            or receipt.get("support_count") != budget
            or receipt.get("model_seed") != model_seed(seed, budget)
        ):
            raise ValueError("fit_receipts seed, support, or model seed is invalid")
        if receipt["role"] == "active_learning":
            strategy = str(receipt["strategy"])
            round_index = expected_rounds[budget]
            expected_ids = pool_selected[strategy][:budget]
            if (
                receipt.get("round") != round_index
                or receipt.get("evaluated") is not (budget in checkpoints)
                or receipt.get("fit_reused") is not False
                or receipt.get("labelled_ids_sha256") != canonical_hash(expected_ids)
            ):
                raise ValueError("Active fit receipt does not reconcile with acquisitions")
        elif receipt["role"] == "full_pool_reference":
            if (
                receipt.get("round") is not None
                or receipt.get("evaluated") is not True
                or receipt.get("fit_reused") is not False
            ):
                raise ValueError("Full-reference fit receipt fields are invalid")
        else:
            expected_reused = (
                receipt.get("feature_mode") == "semantic" and budget == candidate_count
            )
            if (
                receipt.get("round") is not None
                or receipt.get("evaluated") is not True
                or receipt.get("fit_reused") is not expected_reused
            ):
                raise ValueError("Ablation fit receipt fields are invalid")
    ablation_receipts = [
        value for value in receipts if value["role"] == "representation_ablation"
    ]
    for budget in ablation_budgets:
        hashes = {
            value.get("labelled_ids_sha256")
            for value in ablation_receipts
            if int(value["budget"]) == budget
        }
        if len(hashes) != 1:
            raise ValueError("Ablation fit receipts do not use identical nested IDs")
    full_hash = next(
        value.get("labelled_ids_sha256")
        for value in receipts
        if value["role"] == "full_pool_reference"
    )
    semantic_full_hash = next(
        value.get("labelled_ids_sha256")
        for value in ablation_receipts
        if value["feature_mode"] == "semantic" and int(value["budget"]) == candidate_count
    )
    if full_hash != semantic_full_hash:
        raise ValueError("Full-reference and semantic full-pool fit receipts disagree")


def _validate_seed_payload_files(
    directory: Path, run_identity: Mapping[str, Any] | None = None
) -> None:
    missing = [name for name in SEED_REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Complete seed is missing required artifacts: {missing}")
    for strategy in STRATEGIES:
        round_dir = directory / "round_receipts" / strategy
        paths = sorted(round_dir.glob("round_*.json")) if round_dir.is_dir() else []
        if not paths:
            raise ValueError(f"Complete seed has no round receipts for {strategy}")
        if run_identity is not None:
            expected_count = _expected_round_count(run_identity)
            if expected_count is not None:
                expected_names = [
                    f"round_{index:03d}.json" for index in range(expected_count)
                ]
                if [path.name for path in paths] != expected_names:
                    raise ValueError(
                        f"Complete seed has an invalid round receipt set for {strategy}"
                    )

    acquisitions = pd.read_csv(directory / "acquisitions.csv")
    validate_receipt(
        {"acquisition_records": json_null_nonfinite(acquisitions.to_dict("records"))}
    )
    fit_receipts = _read_strict_json(directory / "fit_receipts.json")
    validate_receipt({"fit_receipts": fit_receipts})
    for path in sorted((directory / "round_receipts").rglob("round_*.json")):
        validate_receipt(_read_strict_json(path))
    if run_identity is not None:
        validate_seed_semantics(
            directory, _seed_from_directory(directory), run_identity
        )


def _seed_from_directory(directory: Path) -> int:
    name = directory.name
    if name.startswith(".seed_"):
        name = name[1:].split(".staging-", 1)[0]
    if not name.startswith("seed_"):
        raise ValueError(f"Cannot derive run seed from directory name: {directory}")
    try:
        return _nonnegative_int(int(name.removeprefix("seed_")), "run_seed")
    except ValueError as error:
        raise ValueError(f"Cannot derive run seed from directory name: {directory}") from error


def validate_complete_seed(
    path_or_root: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate completion, exact identity, required files, and every seed hash."""

    path = Path(path_or_root)
    directory = path if path.name.startswith("seed_") or path.name.startswith(".seed_") else seed_directory(path, run_seed)
    manifest_path = directory / SEED_MANIFEST_FILENAME
    manifest = _read_strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Seed manifest must contain a JSON object")
    if manifest.get("schema") != SEED_MANIFEST_SCHEMA:
        raise ValueError("Seed manifest schema differs")
    if manifest.get("run_seed") != _nonnegative_int(run_seed, "run_seed"):
        raise ValueError("Seed manifest run_seed differs")
    if manifest.get("status") != "complete" or manifest.get("completion_state") != "complete":
        raise ValueError("Seed manifest is not complete")
    recorded_manifest_hash = manifest.get("manifest_payload_sha256")
    unhashed_manifest = dict(manifest)
    unhashed_manifest.pop("manifest_payload_sha256", None)
    if recorded_manifest_hash != canonical_hash(unhashed_manifest):
        raise ValueError("Seed manifest payload hash is invalid")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Seed manifest run identity is missing")
    if manifest.get("run_identity_sha256") != canonical_hash(identity):
        raise ValueError("Seed manifest run identity hash is invalid")
    validate_run_identity(identity, run_identity)
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Seed manifest artifact inventory is missing")
    _validate_seed_payload_files(directory, run_identity)
    validate_artifact_inventory(directory, inventory)
    return manifest


def finalize_seed_staging(
    staging_directory: str | Path,
    output_root: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
    manifest_details: Mapping[str, Any],
) -> Path:
    """Write a complete seed manifest, verify it, then atomically publish seed_N."""

    staging = Path(staging_directory)
    root = Path(output_root)
    destination = seed_directory(root, run_seed)
    if not staging.is_dir() or staging.parent.resolve() != root.resolve():
        raise ValueError("Seed staging directory must be an immediate child of output root")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing seed directory: {destination}")
    _validate_seed_payload_files(staging, run_identity)
    inventory = artifact_inventory(staging)
    manifest = {
        "schema": SEED_MANIFEST_SCHEMA,
        "run_seed": _nonnegative_int(run_seed, "run_seed"),
        "run_identity": _json_safe_copy(run_identity, role="Run identity"),
        "run_identity_sha256": canonical_hash(run_identity),
        "status": "complete",
        "completion_state": "complete",
        **_json_safe_copy(manifest_details, role="Seed manifest details"),
        "artifact_inventory": inventory,
    }
    if manifest.get("status") != "complete" or manifest.get("completion_state") != "complete":
        raise ValueError("Manifest details cannot override seed completion state")
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_write_json(staging / SEED_MANIFEST_FILENAME, manifest)
    validate_complete_seed(staging, run_seed, run_identity)
    os.replace(staging, destination)
    _fsync_directory(root)
    validate_complete_seed(destination, run_seed, run_identity)
    return destination


def _reserve_output(output_root: Path, *, resume: bool) -> None:
    if not output_root.is_absolute():
        raise ValueError("--output-dir must be an absolute path")
    if resume:
        if not output_root.is_dir():
            raise ValueError("--resume requires an existing output directory")
        return
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"Output target is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise FileExistsError(f"Fresh output directory must be empty: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(output_root.parent)


@dataclass(frozen=True)
class RunnerDependencies:
    """Injectable real-API boundary used by lightweight runner tests."""

    load_dataset: Callable[..., GroupedDataset]
    load_semantic: Callable[[PublicFeatureDatasetView], FeatureBundle | Mapping[str, Any]]
    load_identity: Callable[[PublicFeatureDatasetView], FeatureBundle | Mapping[str, Any]]
    fit_model: Callable[[np.ndarray, np.ndarray, ModelSettings, int], Any]
    select: Callable[..., tuple[str, ...]]
    score: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]]
    write_audits: Callable[[GroupedDataset, str | Path], Mapping[str, Path]]
    oracle_factory: Callable[..., RetrospectiveOracle]


def default_runner_dependencies() -> RunnerDependencies:
    return RunnerDependencies(
        load_dataset=load_grouped_dataset,
        load_semantic=load_semantic_feature_bundle,
        load_identity=load_identity_feature_bundle,
        fit_model=fit_response_model,
        select=select_batch,
        score=score_response,
        write_audits=write_audit_outputs,
        oracle_factory=RetrospectiveOracle,
    )


def configure_determinism(
    effective_device: str,
    *,
    profile: str,
    injected_dependencies: bool,
) -> dict[str, Any]:
    """Enforce the formal process gates and enable deterministic torch execution."""

    if injected_dependencies and profile != "smoke":
        raise ValueError("Injected runner dependencies are allowed only for smoke runs")
    python_hash_seed = os.environ.get("PYTHONHASHSEED")
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not injected_dependencies and python_hash_seed != "0":
        raise ValueError("Default runner dependencies require PYTHONHASHSEED=0")
    if effective_device == "cuda" and cublas_workspace not in {":4096:8", ":16:8"}:
        raise ValueError(
            "CUDA default runs require CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "dependency_mode": "injected_smoke" if injected_dependencies else "default_real",
        "injected_dependencies": injected_dependencies,
        "formal_bypass_allowed": False,
        "PYTHONHASHSEED": python_hash_seed,
        "CUBLAS_WORKSPACE_CONFIG": cublas_workspace,
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _environment_metadata(effective_device: str) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available else None
        ),
        "effective_device": effective_device,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _fit_summary(fit: Any) -> dict[str, Any]:
    if callable(getattr(fit, "fit_summary", None)):
        raw = fit.fit_summary()
    elif isinstance(getattr(fit, "summary", None), Mapping):
        raw = fit.summary
    else:
        raise TypeError("Fitted predictor must provide fit_summary() or summary")
    if not isinstance(raw, Mapping):
        raise ValueError("Fitted predictor summary must be a mapping")
    safe = json_null_nonfinite(dict(raw), role="Fit summary")
    return validate_receipt({"fit_summary": safe})["fit_summary"]


def _fit_with_revealed(
    dataset: GroupedDataset,
    bundle: FeatureBundle,
    labelled_ids: Sequence[str],
    revealed_values: np.ndarray,
    settings: ModelSettings,
    run_seed: int,
    fit_model: Callable[[np.ndarray, np.ndarray, ModelSettings, int], Any],
) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    ids = tuple(str(value) for value in labelled_ids)
    values = np.asarray(revealed_values)
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError("Revealed values must have one matrix row per labelled ID")
    positions = _row_positions(bundle.row_ids, ids)
    masked = prepare_masked_model_features(bundle, ids)
    seed = model_seed(run_seed, len(ids))
    started = perf_counter()
    fit = fit_model(masked[positions], values, settings, seed)
    seconds = float(perf_counter() - started)
    summary = _fit_summary(fit)
    summary_seed = summary.get("seed")
    if summary_seed is not None and int(summary_seed) != seed:
        raise ValueError("Fitted predictor summary reports the wrong model seed")
    return fit, masked, seconds, summary


def _ordered_evaluation_splits(dataset: GroupedDataset) -> tuple[str, ...]:
    registered = (INTERPOLATION_SPLIT, *VALIDATION_SPLITS)
    names = [name for name in registered if name in dataset.validation_ids]
    names.extend(sorted(set(dataset.validation_ids) - set(names)))
    return tuple(names)


def _evaluate_fit(
    dataset: GroupedDataset,
    bundle: FeatureBundle,
    fit: Any,
    masked_features: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in _ordered_evaluation_splits(dataset):
        split_ids = tuple(str(value) for value in dataset.validation_ids[split_name])
        positions = _row_positions(bundle.row_ids, split_ids)
        prediction = np.asarray(fit.predict(masked_features[positions]))
        observed = dataset.response.loc[list(split_ids)].to_numpy(dtype=np.float32)
        metrics = dict(score(prediction, observed))
        rows.append(
            {
                "split": split_name,
                "primary_split": split_name == INTERPOLATION_SPLIT,
                **metrics,
            }
        )
    return rows


def _fit_receipt_record(
    *,
    role: str,
    run_seed: int,
    budget: int,
    feature_mode: str,
    labelled_ids: Sequence[str],
    train_seconds: float,
    fit_summary: Mapping[str, Any],
    strategy: str | None = None,
    round_index: int | None = None,
    evaluated: bool = True,
    reused: bool = False,
) -> dict[str, Any]:
    receipt = {
        "role": role,
        "strategy": strategy,
        "feature_mode": feature_mode,
        "run_seed": int(run_seed),
        "round": round_index,
        "budget": int(budget),
        "model_seed": model_seed(run_seed, budget),
        "evaluated": bool(evaluated),
        "support_count": int(len(labelled_ids)),
        "labelled_ids_sha256": canonical_hash(list(labelled_ids)),
        "train_seconds": float(train_seconds),
        "fit_reused": bool(reused),
        "fit_summary": dict(fit_summary),
    }
    return validate_receipt(json_null_nonfinite(receipt, role="Fit receipt"))


def _active_columns() -> list[str]:
    return [
        "role",
        "feature_mode",
        "strategy",
        "seed",
        "round",
        "budget",
        "model_seed",
        "train_seconds",
        "split",
        "primary_split",
        *COUNT_METRICS,
        *SCORE_METRICS,
    ]


def _acquisition_columns() -> list[str]:
    return [
        "seed",
        "strategy",
        "round",
        "selection_type",
        "rank_in_batch",
        "budget_before",
        "budget_after",
        "acquisition_seed",
        "model_seed",
        CONDITION_ID,
    ]


def run_policy_trajectories(
    dataset: GroupedDataset,
    bundle: FeatureBundle,
    settings: ModelSettings,
    profile_spec: Mapping[str, Any],
    run_seed: int,
    receipt_root: str | Path,
    *,
    dependencies: RunnerDependencies | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run isolated candidate-only active-learning trajectories for all policies."""

    deps = default_runner_dependencies() if dependencies is None else dependencies
    pool_ids = tuple(sorted(str(value) for value in dataset.candidate_pool_ids))
    schedule = BudgetSchedule(
        initial_budget=int(profile_spec["initial_budget"]),
        batch_size=int(profile_spec["batch_size"]),
        checkpoints=tuple(int(value) for value in profile_spec["checkpoints"]),
        pool_size=len(pool_ids),
    )
    initial_ids = deterministic_initial_ids(pool_ids, schedule.initial_budget, run_seed)
    pool_positions = _row_positions(bundle.row_ids, pool_ids)
    descriptors = bundle.descriptor_matrix[pool_positions]
    metric_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    fit_receipts: list[dict[str, Any]] = []

    for strategy in STRATEGIES:
        oracle = deps.oracle_factory(
            pool_ids,
            dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
            response_ids=pool_ids,
        )
        empty = PoolState(pool_ids, descriptors, descriptor_ids=pool_ids)
        state = empty.select(initial_ids)
        first = oracle.reveal(initial_ids)
        revealed_values = np.array(first.labels, copy=True)
        transition_before = empty
        transition_selected = initial_ids
        transition_acquisition_seed: int | None = None
        for rank, condition_id in enumerate(initial_ids, start=1):
            acquisition_rows.append(
                {
                    "seed": int(run_seed),
                    "strategy": strategy,
                    "round": 0,
                    "selection_type": "deterministic_initial",
                    "rank_in_batch": rank,
                    "budget_before": 0,
                    "budget_after": schedule.initial_budget,
                    "acquisition_seed": np.nan,
                    "model_seed": model_seed(run_seed, schedule.initial_budget),
                    CONDITION_ID: condition_id,
                }
            )

        round_index = 0
        while True:
            budget = state.budget
            fit, masked, train_seconds, fit_summary = _fit_with_revealed(
                dataset,
                bundle,
                state.selected_ids,
                revealed_values,
                settings,
                run_seed,
                deps.fit_model,
            )
            evaluated = schedule.is_checkpoint(budget)
            fit_receipts.append(
                _fit_receipt_record(
                    role="active_learning",
                    run_seed=run_seed,
                    budget=budget,
                    feature_mode="semantic",
                    labelled_ids=state.selected_ids,
                    train_seconds=train_seconds,
                    fit_summary=fit_summary,
                    strategy=strategy,
                    round_index=round_index,
                    evaluated=evaluated,
                )
            )
            split_metrics = (
                _evaluate_fit(dataset, bundle, fit, masked, deps.score)
                if evaluated
                else []
            )
            for split_row in split_metrics:
                metric_rows.append(
                    {
                        "role": "active_learning",
                        "feature_mode": "semantic",
                        "strategy": strategy,
                        "seed": int(run_seed),
                        "round": round_index,
                        "budget": budget,
                        "model_seed": model_seed(run_seed, budget),
                        "train_seconds": train_seconds,
                        **split_row,
                    }
                )

            receipt = RoundReceipt.from_transition(
                round_index,
                strategy,
                transition_before,
                state,
                transition_selected,
                global_seed=run_seed,
                acquisition_seed=transition_acquisition_seed,
                model_seed=model_seed(run_seed, budget),
                checkpoint=evaluated,
                labelled_ids=state.selected_ids,
                model_fit_summary=fit_summary,
                split_metrics=split_metrics,
                train_seconds=train_seconds,
            )
            receipt_payload = validate_receipt(receipt.to_dict())
            atomic_write_json(
                Path(receipt_root)
                / strategy
                / f"round_{round_index:03d}.json",
                receipt_payload,
            )
            if budget == schedule.final_budget:
                break

            candidates = state.candidate_ids
            current_acquisition_seed = acquisition_seed(run_seed, budget)
            if strategy == "uncertainty":
                context = AcquisitionContext.from_predictor(
                    candidates,
                    pool_ids,
                    descriptors,
                    labelled_ids=state.selected_ids,
                    predictor=fit,
                    model_feature_ids=pool_ids,
                    model_features=masked[pool_positions],
                    mc_passes=int(profile_spec["mc_passes"]),
                )
            else:
                context = state.acquisition_context()
            selected = tuple(
                deps.select(
                    strategy,
                    context,
                    schedule.batch_size,
                    seed=current_acquisition_seed,
                )
            )
            if len(selected) != schedule.batch_size:
                raise ValueError("Acquisition did not return the exact fixed batch size")
            if len(set(selected)) != len(selected) or not set(selected) <= set(candidates):
                raise ValueError("Acquisition selected duplicate, reselected, or out-of-pool IDs")
            before = state
            state = schedule.advance(before, selected)
            revealed = oracle.reveal(selected)
            revealed_values = np.concatenate(
                [revealed_values, np.asarray(revealed.labels)], axis=0
            )
            if tuple(oracle.revealed_ids) != state.selected_ids:
                raise AssertionError("Oracle reveal order differs from conserved pool state")
            round_index += 1
            transition_before = before
            transition_selected = selected
            transition_acquisition_seed = current_acquisition_seed
            for rank, condition_id in enumerate(selected, start=1):
                acquisition_rows.append(
                    {
                        "seed": int(run_seed),
                        "strategy": strategy,
                        "round": round_index,
                        "selection_type": strategy,
                        "rank_in_batch": rank,
                        "budget_before": before.budget,
                        "budget_after": state.budget,
                        "acquisition_seed": current_acquisition_seed,
                        "model_seed": model_seed(run_seed, before.budget),
                        CONDITION_ID: condition_id,
                    }
                )

    metrics = pd.DataFrame(metric_rows, columns=_active_columns()).sort_values(
        ["strategy", "budget", "split"], kind="stable", ignore_index=True
    )
    acquisitions = pd.DataFrame(
        acquisition_rows, columns=_acquisition_columns()
    ).sort_values(
        ["strategy", "round", "rank_in_batch"], kind="stable", ignore_index=True
    )
    validate_receipt(
        {"acquisition_records": json_null_nonfinite(acquisitions.to_dict("records"))}
    )
    return metrics, acquisitions, fit_receipts


def _fit_full_reference(
    dataset: GroupedDataset,
    bundle: FeatureBundle,
    settings: ModelSettings,
    run_seed: int,
    nested_ids: Sequence[str],
    dependencies: RunnerDependencies,
) -> tuple[Any, np.ndarray, pd.DataFrame, dict[str, Any]]:
    pool_ids = tuple(sorted(str(value) for value in dataset.candidate_pool_ids))
    ids = tuple(str(value) for value in nested_ids)
    if set(ids) != set(pool_ids) or len(ids) != len(pool_ids):
        raise ValueError("Full reference IDs must contain exactly the candidate pool")
    oracle = dependencies.oracle_factory(
        pool_ids,
        dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
        response_ids=pool_ids,
    )
    revealed = oracle.reveal(ids)
    fit, masked, seconds, summary = _fit_with_revealed(
        dataset,
        bundle,
        revealed.ids,
        revealed.labels,
        settings,
        run_seed,
        dependencies.fit_model,
    )
    rows = [
        {
            "role": "full_pool_reference",
            "feature_mode": "semantic",
            "strategy": "full_reference",
            "seed": int(run_seed),
            "round": np.nan,
            "budget": len(pool_ids),
            "model_seed": model_seed(run_seed, len(pool_ids)),
            "train_seconds": seconds,
            **split_row,
        }
        for split_row in _evaluate_fit(
            dataset, bundle, fit, masked, dependencies.score
        )
    ]
    frame = pd.DataFrame(rows, columns=_active_columns()).sort_values(
        ["split"], kind="stable", ignore_index=True
    )
    receipt = _fit_receipt_record(
        role="full_pool_reference",
        run_seed=run_seed,
        budget=len(pool_ids),
        feature_mode="semantic",
        labelled_ids=ids,
        train_seconds=seconds,
        fit_summary=summary,
        strategy="full_reference",
    )
    return fit, masked, frame, receipt


def _run_representation_ablation(
    dataset: GroupedDataset,
    bundles: Mapping[str, FeatureBundle],
    settings: ModelSettings,
    profile_spec: Mapping[str, Any],
    run_seed: int,
    nested_ids: Sequence[str],
    full_fit: Any,
    full_masked: np.ndarray,
    full_metrics: pd.DataFrame,
    dependencies: RunnerDependencies,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    pool_ids = tuple(sorted(str(value) for value in dataset.candidate_pool_ids))
    fixed = tuple(int(value) for value in profile_spec["ablation_fixed_budgets"])
    if any(value <= 0 or value > len(pool_ids) for value in fixed):
        raise ValueError("Representation ablation budget exceeds candidate pool")
    budgets = tuple(dict.fromkeys((*fixed, len(pool_ids))))
    oracle = dependencies.oracle_factory(
        pool_ids,
        dataset.response.loc[list(pool_ids)].to_numpy(dtype=np.float32),
        response_ids=pool_ids,
    )
    revealed_ids: tuple[str, ...] = ()
    revealed_values = np.empty((0, len(dataset.proteins)), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for budget in budgets:
        additions = tuple(nested_ids[len(revealed_ids) : budget])
        if additions:
            batch = oracle.reveal(additions)
            revealed_ids = (*revealed_ids, *batch.ids)
            revealed_values = np.concatenate(
                [revealed_values, np.asarray(batch.labels)], axis=0
            )
        if revealed_ids != tuple(nested_ids[:budget]):
            raise AssertionError("Representation ablation IDs are not nested")
        for feature_mode in ("identity", "semantic"):
            bundle = bundles[feature_mode]
            reused = feature_mode == "semantic" and budget == len(pool_ids)
            if reused:
                fit = full_fit
                masked = full_masked
                seconds = float(
                    full_metrics["train_seconds"].iloc[0]
                    if len(full_metrics)
                    else 0.0
                )
                summary = _fit_summary(fit)
                split_rows = [
                    {
                        key: row[key]
                        for key in ("split", "primary_split", *COUNT_METRICS, *SCORE_METRICS)
                    }
                    for _, row in full_metrics.iterrows()
                ]
            else:
                fit, masked, seconds, summary = _fit_with_revealed(
                    dataset,
                    bundle,
                    revealed_ids,
                    revealed_values,
                    settings,
                    run_seed,
                    dependencies.fit_model,
                )
                split_rows = _evaluate_fit(
                    dataset, bundle, fit, masked, dependencies.score
                )
            receipts.append(
                _fit_receipt_record(
                    role="representation_ablation",
                    run_seed=run_seed,
                    budget=budget,
                    feature_mode=feature_mode,
                    labelled_ids=revealed_ids,
                    train_seconds=seconds,
                    fit_summary=summary,
                    strategy=feature_mode,
                    reused=reused,
                )
            )
            for split_row in split_rows:
                rows.append(
                    {
                        "role": "representation_ablation",
                        "feature_mode": feature_mode,
                        "strategy": "representation_ablation",
                        "seed": int(run_seed),
                        "round": np.nan,
                        "budget": budget,
                        "model_seed": model_seed(run_seed, budget),
                        "train_seconds": seconds,
                        **split_row,
                    }
                )
    frame = pd.DataFrame(rows, columns=_active_columns()).sort_values(
        ["budget", "feature_mode", "split"], kind="stable", ignore_index=True
    )
    return frame, receipts


def _model_settings_summary(settings: ModelSettings) -> dict[str, Any]:
    return {
        "kind": settings.kind,
        "hidden_dim": settings.hidden_dim,
        "dropout": settings.dropout,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "target_scale_floor": settings.target_scale_floor,
        "device": settings.device,
        "response_rank": settings.response_rank,
        "svd_niter": settings.svd_niter,
    }


def _seed_data_summary(dataset: GroupedDataset) -> dict[str, Any]:
    response = dataset.response.to_numpy()
    return {
        "protocol_version": dataset.protocol_version,
        "query_unit": "biological_condition",
        "condition_fields": list(GROUP_FIELDS),
        "metadata_rows": int(len(dataset.metadata)),
        "candidate_pool_count": int(len(dataset.candidate_pool_ids)),
        "evaluation_counts": {
            name: int(len(ids)) for name, ids in dataset.validation_ids.items()
        },
        "protein_count": int(len(dataset.proteins)),
        "observed_value_count": int(np.isfinite(response).sum()),
        "cache_key": dataset.cache_key,
        "cache_hit": bool(dataset.cache_hit),
        "source_hashes": dict(dataset.source_hashes),
    }


def _append_failure_attempt(
    output_root: Path, run_seed: int, error: BaseException
) -> dict[str, Any]:
    directory = output_root / "failure_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("attempt_*.json"))
    record = {
        "attempt": len(existing) + 1,
        "at_utc": _utc_now(),
        "run_seed": int(run_seed),
        "type": type(error).__name__,
        "message": str(error),
    }
    atomic_write_json(directory / f"attempt_{len(existing) + 1:04d}.json", record)
    return record


def _append_root_failure_attempt(
    output_root: Path, phase: str, error: BaseException
) -> dict[str, Any]:
    """Atomically append a target-free root-level failure receipt."""

    directory = output_root / "failure_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (directory / f"root_attempt_{attempt:04d}.json").exists():
        attempt += 1
    record = validate_receipt(
        {
            "schema": "goai.direct_multiseed.root_failure_attempt.v1",
            "scope": "root",
            "attempt": attempt,
            "phase": str(phase),
            "at_utc": _utc_now(),
            "type": type(error).__name__,
            "message": str(error),
        }
    )
    atomic_write_json(directory / f"root_attempt_{attempt:04d}.json", record)
    return record


def _collect_failure_attempts(
    output_root: Path, preserved: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Merge manifest and on-disk seed/root failure history without duplicates."""

    attempts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        try:
            safe = validate_receipt(value)
            digest = canonical_hash(safe)
        except (TypeError, ValueError):
            return
        if digest not in seen:
            seen.add(digest)
            attempts.append(safe)

    for value in preserved:
        add(value)
    directory = output_root / "failure_attempts"
    try:
        files = (
            sorted(
                (
                    *directory.glob("attempt_*.json"),
                    *directory.glob("root_attempt_*.json"),
                ),
                key=lambda path: path.name,
            )
            if directory.is_dir()
            else []
        )
    except OSError:
        files = []
    for path in files:
        try:
            add(_read_strict_json(path))
        except (OSError, ValueError):
            continue
    return attempts


def run_seed_atomic(
    output_root: str | Path,
    run_seed: int,
    run_identity: Mapping[str, Any],
    dataset: GroupedDataset,
    bundles: Mapping[str, FeatureBundle],
    settings: ModelSettings,
    profile_spec: Mapping[str, Any],
    *,
    command: Sequence[str],
    dependencies: RunnerDependencies | None = None,
) -> Path:
    """Execute one seed in hidden staging and publish only a verified completion."""

    deps = default_runner_dependencies() if dependencies is None else dependencies
    started_at = _utc_now()
    started = perf_counter()
    staging = create_seed_staging(output_root, run_seed)
    try:
        nested_ids = deterministic_nested_ids(dataset.candidate_pool_ids, run_seed)
        active, acquisitions, active_receipts = run_policy_trajectories(
            dataset,
            bundles["semantic"],
            settings,
            profile_spec,
            run_seed,
            staging / "round_receipts",
            dependencies=deps,
        )
        full_fit, full_masked, full_metrics, full_receipt = _fit_full_reference(
            dataset,
            bundles["semantic"],
            settings,
            run_seed,
            nested_ids,
            deps,
        )
        ablation, ablation_receipts = _run_representation_ablation(
            dataset,
            bundles,
            settings,
            profile_spec,
            run_seed,
            nested_ids,
            full_fit,
            full_masked,
            full_metrics,
            deps,
        )
        atomic_write_csv(staging / "active_metrics.csv", active)
        atomic_write_csv(staging / "acquisitions.csv", acquisitions)
        atomic_write_csv(staging / "full_reference_metrics.csv", full_metrics)
        atomic_write_csv(staging / "ablation_metrics.csv", ablation)
        all_receipts = [full_receipt, *active_receipts, *ablation_receipts]
        validated_receipts = [validate_receipt(value) for value in all_receipts]
        atomic_write_json(
            staging / "fit_receipts.json",
            {
                "schema": "goai.direct_multiseed.fit_receipts.v1",
                "run_seed": int(run_seed),
                "receipts": validated_receipts,
            },
        )
        ended_at = _utc_now()
        details = {
            "profile": profile_spec["name"],
            "started_at_utc": started_at,
            "completed_at_utc": ended_at,
            "wall_time_seconds": float(perf_counter() - started),
            "command": shlex.join(command),
            "argv": list(command),
            "environment": _environment_metadata(settings.device),
            "model_summary": _model_settings_summary(settings),
            "data_summary": _seed_data_summary(dataset),
            "feature_summary": {
                name: dict(bundle.summary) for name, bundle in sorted(bundles.items())
            },
            "feature_asset_hashes": dict(bundles["semantic"].asset_hashes),
            "control_summary": dict(dataset.control_policy_summary),
            "strain_identity_contract": {
                "organizer_verified": False,
                "public_candidate_identities": True,
                "warning": STRAIN_IDENTITY_WARNING,
                "DHY210_semantics": "zero_as_provided",
            },
            "information_boundary": {
                "oracle_instances": "one isolated candidate-only oracle per policy",
                "evaluation_ids_revealable": False,
                "predictor_support": "current revealed candidate rows only",
                "acquisition_descriptors": "target_free",
                "serialized_hidden_values": False,
            },
        }
        return finalize_seed_staging(
            staging, output_root, run_seed, run_identity, details
        )
    except BaseException as error:
        _append_failure_attempt(Path(output_root), run_seed, error)
        if staging.exists():
            shutil.rmtree(staging)
            _fsync_directory(Path(output_root))
        raise


def _source_inventory(config_path: Path) -> dict[str, dict[str, Any]]:
    source_dir = Path(__file__).resolve().parent
    names = (
        "direct_multiseed.py",
        "data.py",
        "semantics.py",
        "simulator.py",
        "acquisition.py",
        "model.py",
        "metrics.py",
        "audit.py",
    )
    records = {
        f"src/goai_al/{name}": file_identity(source_dir / name) for name in names
    }
    records["configs/direct_multiseed.yaml"] = file_identity(config_path)
    records["FRAMEWORK_SPEC_V22.md"] = file_identity(FRAMEWORK_SPEC_PATH)
    return records


def _validate_source_snapshot(
    output_root: Path, source_inventory: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    snapshot_root = output_root / "source_snapshot"
    manifest = _read_strict_json(snapshot_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("Source snapshot manifest schema differs")
    recorded_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if recorded_hash != canonical_hash(payload):
        raise ValueError("Source snapshot manifest payload hash is invalid")
    expected_sources = {
        name: {"bytes": int(record["bytes"]), "sha256": str(record["sha256"])}
        for name, record in source_inventory.items()
    }
    if manifest.get("source_identities") != expected_sources:
        raise ValueError("Source snapshot identities differ from the run identity")
    inventory = manifest.get("snapshot_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Source snapshot inventory is missing")
    validate_artifact_inventory(
        snapshot_root, inventory, exclude=("manifest.json",)
    )
    return manifest


def write_source_snapshot(
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Create or validate the immutable exact source/config/spec snapshot."""

    root = Path(output_root)
    configured_path = (
        PROJECT_ROOT / "configs" / "direct_multiseed.yaml"
        if config_path is None
        else Path(config_path)
    ).resolve(strict=True)
    validate_config(config)
    sources = _source_inventory(configured_path)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_root = root / "source_snapshot"
    if snapshot_root.exists():
        _validate_source_snapshot(root, sources)
        return snapshot_root
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".source_snapshot.staging-", dir=root)
    )
    try:
        for relative, record in sources.items():
            _atomic_binary_copy(record["path"], temporary_root / relative)
        inventory = artifact_inventory(temporary_root, exclude=("manifest.json",))
        source_identities = {
            name: {
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
            for name, record in sources.items()
        }
        manifest = {
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "source_identities": source_identities,
            "snapshot_inventory": inventory,
        }
        manifest["manifest_payload_sha256"] = canonical_hash(manifest)
        atomic_write_json(temporary_root / "manifest.json", manifest)
        child_directories = sorted(
            (path for path in temporary_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in child_directories:
            _fsync_directory(directory)
        _fsync_directory(temporary_root)
        try:
            os.replace(temporary_root, snapshot_root)
        except OSError:
            if not snapshot_root.exists():
                raise
            _validate_source_snapshot(root, sources)
            return snapshot_root
        _fsync_directory(root)
        _validate_source_snapshot(root, sources)
        return snapshot_root
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def build_run_identity(
    config_path: str | Path,
    config: Mapping[str, Any],
    profile_spec: Mapping[str, Any],
    effective_device: str,
    semantic_bundle: FeatureBundle | Mapping[str, FeatureBundle],
    partition_contract: Mapping[str, Any] | None = None,
    control_contract: Mapping[str, Any] | None = None,
    determinism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical resume identity from protocol and live file hashes."""

    path = Path(config_path).resolve(strict=True)
    data = dict(config["data"])
    assets = dict(config["semantic_assets"])
    input_records = {
        "metadata": file_identity(data["metadata_path"]),
        "proteome": file_identity(data["proteome_path"]),
    }
    asset_records = {
        name: file_identity(value) for name, value in sorted(assets.items())
    }
    cache_path = (PROJECT_ROOT / str(data["cache_dir"])).resolve()
    if cache_path != FROZEN_CACHE_PATH.resolve():
        raise ValueError("Configured cache path is outside the frozen repository cache")
    bundles = (
        dict(semantic_bundle)
        if isinstance(semantic_bundle, Mapping)
        else {"semantic": semantic_bundle}
    )
    sources = _source_inventory(path)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "data_protocol": DATA_PROTOCOL,
        "model_id": MODEL_ID,
        "profile": profile_spec["name"],
        "effective_device": effective_device,
        "seeds": list(profile_spec["seeds"]),
        "schedule": {
            "initial_budget": profile_spec["initial_budget"],
            "batch_size": profile_spec["batch_size"],
            "checkpoints": list(profile_spec["checkpoints"]),
            "epochs": profile_spec["epochs"],
            "mc_passes": profile_spec["mc_passes"],
            "ablation_fixed_budgets": list(
                profile_spec["ablation_fixed_budgets"]
            ),
        },
        "strategies": list(STRATEGIES),
        "primary_decision": {
            "split": INTERPOLATION_SPLIT,
            "metric": PRIMARY_METRIC,
            "summary": "normalized_trapezoidal_aulc",
            "target_fraction": float(config["evaluation"]["target_fraction"]),
        },
        "config": {
            "canonical_sha256": canonical_hash(config),
            "file": file_identity(path),
        },
        "inputs": input_records,
        "assets": asset_records,
        "feature_bundles": {
            name: feature_bundle_identity(bundle)
            for name, bundle in sorted(bundles.items())
        },
        "feature_asset_hashes": dict(bundles["semantic"].asset_hashes),
        "partition_contract": _json_safe_copy(
            partition_contract or {}, role="Partition contract"
        ),
        "control_contract": _json_safe_copy(
            control_contract or {}, role="Control contract"
        ),
        "determinism": _json_safe_copy(determinism or {}, role="Determinism contract"),
        "sources": sources,
        "framework_spec": sources["FRAMEWORK_SPEC_V22.md"],
        "cache_path": str(cache_path),
    }
    return _json_safe_copy(identity, role="Run identity")


def _load_runtime_inputs(
    config: Mapping[str, Any],
    dependencies: RunnerDependencies,
) -> tuple[GroupedDataset, dict[str, FeatureBundle], dict[str, Any], dict[str, Any]]:
    data = dict(config["data"])
    dataset = dependencies.load_dataset(
        metadata_path=data["metadata_path"],
        proteome_path=data["proteome_path"],
        missing_rate_threshold=data["missing_rate_threshold"],
        cache_dir=(PROJECT_ROOT / str(data["cache_dir"])).resolve(),
        interpolation_fraction=data["interpolation_fraction"],
        split_seed=config["split_seed"],
        control_policy=data["control_policy"],
        vehicle_column=data["vehicle_column"],
    )
    if dataset.protocol_version != PROTOCOL_VERSION:
        raise ValueError("Loaded dataset protocol differs from the frozen protocol")
    partition_summary = validate_dataset_partitions(dataset)
    control_contract = validate_control_contract(dataset)
    public_view = PublicFeatureDatasetView.from_dataset(dataset)
    semantic = coerce_feature_bundle(dependencies.load_semantic(public_view), dataset)
    identity = coerce_feature_bundle(dependencies.load_identity(public_view), dataset)
    bundles = {"identity": identity, "semantic": semantic}
    for feature_mode, bundle in bundles.items():
        if bundle.summary.get("response_used") is not False:
            raise ValueError(
                f"{feature_mode} feature bundle must explicitly declare response_used=false"
            )
    return dataset, bundles, partition_summary, control_contract


def _write_split_assignments(dataset: GroupedDataset, output_root: Path) -> pd.DataFrame:
    pool = set(str(value) for value in dataset.candidate_pool_ids)
    evaluation = {
        str(condition_id): split_name
        for split_name in _ordered_evaluation_splits(dataset)
        for condition_id in dataset.validation_ids[split_name]
    }
    rows: list[dict[str, Any]] = []
    for condition_id, metadata_row in dataset.metadata.iterrows():
        row_id = str(condition_id)
        if row_id in pool:
            assignment, revealable = "candidate_pool", True
        elif row_id in evaluation:
            assignment, revealable = evaluation[row_id], False
        else:
            assignment, revealable = "excluded_official_overlap", False
        rows.append(
            {
                CONDITION_ID: row_id,
                "assignment": assignment,
                "revealable": revealable,
                "query_allowed": revealable,
                "official_split_provenance": str(
                    metadata_row.get(
                        "split_provenance", metadata_row.get("split_final", "")
                    )
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["assignment", CONDITION_ID], kind="stable", ignore_index=True
    )
    atomic_write_csv(output_root / "split_assignments.csv", frame)
    return frame


def _write_feature_summary(
    bundles: Mapping[str, FeatureBundle],
    output_root: Path,
    *,
    control_contract: Mapping[str, Any] | None = None,
    partition_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "primary": "semantic",
        "ablations": ["identity", "semantic"],
        "target_free": True,
        "response_used": False,
        "bundles": {
            name: {
                "descriptor_width": int(bundle.descriptor_matrix.shape[1]),
                "model_width": int(bundle.model_matrix.shape[1]),
                "summary": dict(bundle.summary),
                "asset_hashes": dict(bundle.asset_hashes),
                "identity": feature_bundle_identity(bundle),
            }
            for name, bundle in sorted(bundles.items())
        },
        "control_contract": _json_safe_copy(
            control_contract or {}, role="Control contract"
        ),
        "partition_contract": _json_safe_copy(
            partition_contract or {}, role="Partition contract"
        ),
        "strain_identity_contract": {
            "public_candidate_identities": True,
            "organizer_verified": False,
            "warning": STRAIN_IDENTITY_WARNING,
            "DHY210_semantics": "zero_as_provided",
        },
    }
    atomic_write_json(output_root / "feature_summary.json", summary)
    return summary


def _validate_root_identity(
    output_root: Path, run_identity: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _read_strict_json(output_root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != ROOT_MANIFEST_SCHEMA:
        raise ValueError("Existing root manifest schema differs")
    recorded_payload_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if recorded_payload_hash != canonical_hash(payload):
        raise ValueError("Existing root manifest payload hash is invalid")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Existing root manifest has no run identity")
    if manifest.get("run_identity_sha256") != canonical_hash(identity):
        raise ValueError("Existing root manifest run identity hash is invalid")
    validate_run_identity(identity, run_identity)
    sources = run_identity.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("Run identity source inventory is missing")
    _validate_source_snapshot(output_root, sources)
    inventory = manifest.get("artifact_inventory")
    if manifest.get("status") == "complete":
        if not isinstance(inventory, Mapping):
            raise ValueError("Complete root manifest artifact inventory is missing or invalid")
        validate_artifact_inventory(output_root, inventory, exclude=("manifest.json",))
    return manifest


def _seed_tables(
    output_root: Path,
    seeds: Sequence[int],
    run_identity: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "active_metrics.csv",
            "full_reference_metrics.csv",
            "ablation_metrics.csv",
            "acquisitions.csv",
        )
    }
    for run_seed in sorted(int(value) for value in seeds):
        directory = seed_directory(output_root, run_seed)
        validate_complete_seed(directory, run_seed, run_identity)
        for name in tables:
            frame = pd.read_csv(directory / name)
            if "seed" not in frame or not frame["seed"].eq(run_seed).all():
                raise ValueError(f"Seed artifact {name} contains a wrong seed value")
            tables[name].append(frame)
    return {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in tables.items()
    }


def _per_seed_curve_summary(
    active: pd.DataFrame, full_reference: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "role",
        "seed",
        "strategy",
        "split",
        "metric",
        "higher_is_better",
        "checkpoint_count",
        "budget_start",
        "budget_end",
        "budgets_json",
        "values_json",
        "same_seed_full_reference",
        "normalized_aulc",
        "target_fraction",
        "b80",
        "b80_status",
        "b80_reason",
    ]
    rows: list[dict[str, Any]] = []
    for run_seed in sorted(active["seed"].astype(int).unique()):
        reference_rows = full_reference[
            full_reference["seed"].eq(run_seed)
            & full_reference["split"].eq(INTERPOLATION_SPLIT)
        ]
        if len(reference_rows) != 1:
            raise ValueError("Every seed must have exactly one interpolation full reference")
        reference = float(reference_rows.iloc[0][PRIMARY_METRIC])
        for strategy in STRATEGIES:
            curve = active[
                active["seed"].eq(run_seed)
                & active["strategy"].eq(strategy)
                & active["split"].eq(INTERPOLATION_SPLIT)
            ].sort_values("budget", kind="stable")
            budgets = curve["budget"].to_numpy(dtype=np.float64)
            values = curve[PRIMARY_METRIC].to_numpy(dtype=np.float64)
            summary = curve_summary(
                budgets,
                values,
                reference,
                higher_is_better=True,
                target_fraction=TARGET_FRACTION,
            )
            rows.append(
                {
                    "role": "active_learning_curve",
                    "seed": int(run_seed),
                    "strategy": strategy,
                    "split": INTERPOLATION_SPLIT,
                    "metric": PRIMARY_METRIC,
                    "higher_is_better": True,
                    "checkpoint_count": int(len(curve)),
                    "budget_start": float(budgets[0]) if len(budgets) else np.nan,
                    "budget_end": float(budgets[-1]) if len(budgets) else np.nan,
                    "budgets_json": json.dumps(summary["budgets"], separators=(",", ":")),
                    "values_json": json.dumps(
                        json_null_nonfinite(summary["values"]), separators=(",", ":")
                    ),
                    "same_seed_full_reference": reference,
                    "normalized_aulc": summary["normalized_aulc"],
                    "target_fraction": TARGET_FRACTION,
                    "b80": summary["b80"],
                    "b80_status": summary["b80_status"],
                    "b80_reason": summary["b80_reason"],
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["seed", "strategy"], kind="stable", ignore_index=True
    )


def _aggregate_metric_tables(
    active: pd.DataFrame,
    full_reference: pd.DataFrame,
    ablation: pd.DataFrame,
    curves: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    table_specs = (
        (active, (*COUNT_METRICS, *SCORE_METRICS, "train_seconds")),
        (full_reference, (*COUNT_METRICS, *SCORE_METRICS, "train_seconds")),
        (ablation, (*COUNT_METRICS, *SCORE_METRICS, "train_seconds")),
    )
    group_columns = ("role", "strategy", "feature_mode", "budget", "split")
    for frame, metrics in table_specs:
        if frame.empty:
            continue
        for keys, group in frame.groupby(list(group_columns), sort=True, dropna=False):
            dimensions = dict(zip(group_columns, keys, strict=True))
            raw_seed_count = int(group["seed"].nunique())
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce")
                finite = np.isfinite(values.to_numpy(dtype=np.float64))
                if not finite.any():
                    continue
                summary = finite_metric_summary(values.to_numpy(dtype=np.float64)[finite])
                seed_values = [
                    {"seed": int(seed), "value": float(value)}
                    for seed, value in sorted(
                        zip(
                            group.loc[finite, "seed"].astype(int),
                            values.loc[finite].astype(float),
                        )
                    )
                ]
                records.append(
                    {
                        **dimensions,
                        "metric": metric,
                        "raw_seed_count": raw_seed_count,
                        "finite_seed_count": summary["n"],
                        "seed_values_json": json.dumps(seed_values, separators=(",", ":")),
                        **{key: value for key, value in summary.items() if key != "n"},
                    }
                )

    if not curves.empty:
        for (role, strategy, split), group in curves.groupby(
            ["role", "strategy", "split"], sort=True
        ):
            for metric in (
                "normalized_aulc",
                "b80",
                "checkpoint_count",
                "same_seed_full_reference",
            ):
                values = pd.to_numeric(group[metric], errors="coerce")
                finite = np.isfinite(values.to_numpy(dtype=np.float64))
                if not finite.any():
                    continue
                summary = finite_metric_summary(values.to_numpy(dtype=np.float64)[finite])
                seed_values = [
                    {"seed": int(seed), "value": float(value)}
                    for seed, value in sorted(
                        zip(
                            group.loc[finite, "seed"].astype(int),
                            values.loc[finite].astype(float),
                        )
                    )
                ]
                records.append(
                    {
                        "role": role,
                        "strategy": strategy,
                        "feature_mode": "semantic",
                        "budget": np.nan,
                        "split": split,
                        "metric": metric,
                        "raw_seed_count": int(group["seed"].nunique()),
                        "finite_seed_count": summary["n"],
                        "seed_values_json": json.dumps(seed_values, separators=(",", ":")),
                        **{key: value for key, value in summary.items() if key != "n"},
                    }
                )
    columns = [
        "role",
        "strategy",
        "feature_mode",
        "budget",
        "split",
        "metric",
        "raw_seed_count",
        "finite_seed_count",
        "seed_values_json",
        "mean",
        "sample_sd",
        "median",
        "q25",
        "q75",
        "iqr",
        "t_critical_95",
        "ci95_low",
        "ci95_high",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["role", "strategy", "feature_mode", "budget", "split", "metric"],
        kind="stable",
        ignore_index=True,
    )


def _paired_policy_comparisons(
    curves: pd.DataFrame, profile: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    random_rows = curves[curves["strategy"].eq("random")].set_index("seed")
    for policy in ("coreset", "uncertainty"):
        policy_rows = curves[curves["strategy"].eq(policy)].set_index("seed")
        if set(policy_rows.index) != set(random_rows.index):
            raise ValueError("Paired policy AULCs are not aligned by seed")
        paired: list[dict[str, float | int]] = []
        for run_seed in sorted(int(value) for value in random_rows.index):
            policy_raw = policy_rows.at[run_seed, "normalized_aulc"]
            random_raw = random_rows.at[run_seed, "normalized_aulc"]
            policy_value = float(policy_raw) if policy_raw is not None else float("nan")
            random_value = float(random_raw) if random_raw is not None else float("nan")
            if math.isfinite(policy_value) and math.isfinite(random_value):
                paired.append(
                    {
                        "seed": run_seed,
                        "policy": policy_value,
                        "random": random_value,
                        "directed_difference": policy_value - random_value,
                    }
                )
        differences = [float(value["directed_difference"]) for value in paired]
        summary = finite_metric_summary(differences) if differences else {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "t_critical_95": None,
            "ci95_low": None,
            "ci95_high": None,
        }
        comparison = {
            **summary,
            "wins": int(sum(value > 0.0 for value in differences)),
            "ties": int(sum(value == 0.0 for value in differences)),
            "losses": int(sum(value < 0.0 for value in differences)),
        }
        decision = policy_superiority_decision(comparison, profile=profile)
        decisions.append({"policy": policy, **decision})
        rows.append(
            {
                "role": "paired_policy_comparison",
                "policy": policy,
                "reference": "random",
                "split": INTERPOLATION_SPLIT,
                "metric": PRIMARY_METRIC,
                "summary_metric": "normalized_aulc",
                "higher_is_better": True,
                "raw_seed_count": int(len(random_rows)),
                "finite_pair_count": int(summary["n"]),
                "paired_values_json": json.dumps(paired, separators=(",", ":")),
                "directed_mean": summary["mean"],
                "sample_sd": summary["sample_sd"],
                "median": summary["median"],
                "q25": summary["q25"],
                "q75": summary["q75"],
                "iqr": summary["iqr"],
                "t_critical_95": summary["t_critical_95"],
                "ci95_low": summary["ci95_low"],
                "ci95_high": summary["ci95_high"],
                "wins": comparison["wins"],
                "ties": comparison["ties"],
                "losses": comparison["losses"],
                "decision_status": decision["status"],
                "beats_random": decision["beats_random"],
            }
        )
    return pd.DataFrame(rows), decisions


def _representation_ablation_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget, split_name), group in ablation.groupby(
        ["budget", "split"], sort=True
    ):
        for metric in (*COUNT_METRICS, *SCORE_METRICS):
            pivot = group.pivot(index="seed", columns="feature_mode", values=metric)
            if not {"identity", "semantic"} <= set(pivot.columns):
                raise ValueError("Representation ablation pairs are incomplete")
            pairs: list[dict[str, Any]] = []
            differences: list[float] = []
            directed: list[float] = []
            higher = METRIC_DIRECTIONS.get(metric, True)
            direction = 1.0 if higher else -1.0
            for run_seed, values in pivot.sort_index().iterrows():
                semantic = float(values["semantic"])
                identity = float(values["identity"])
                if not math.isfinite(semantic) or not math.isfinite(identity):
                    continue
                difference = semantic - identity
                pairs.append(
                    {
                        "seed": int(run_seed),
                        "semantic": semantic,
                        "identity": identity,
                        "semantic_minus_identity": difference,
                    }
                )
                differences.append(difference)
                directed.append(direction * difference)
            if not differences:
                continue
            raw_summary = finite_metric_summary(differences)
            directed_summary = finite_metric_summary(directed)
            rows.append(
                {
                    "role": "representation_ablation_summary",
                    "budget": int(budget),
                    "split": split_name,
                    "metric": metric,
                    "higher_is_better": higher,
                    "raw_seed_count": int(pivot.shape[0]),
                    "finite_pair_count": raw_summary["n"],
                    "paired_values_json": json.dumps(pairs, separators=(",", ":")),
                    "semantic_minus_identity_mean": raw_summary["mean"],
                    "semantic_minus_identity_sample_sd": raw_summary["sample_sd"],
                    "semantic_minus_identity_median": raw_summary["median"],
                    "semantic_minus_identity_q25": raw_summary["q25"],
                    "semantic_minus_identity_q75": raw_summary["q75"],
                    "semantic_minus_identity_iqr": raw_summary["iqr"],
                    "semantic_minus_identity_ci95_low": raw_summary["ci95_low"],
                    "semantic_minus_identity_ci95_high": raw_summary["ci95_high"],
                    "directed_mean": directed_summary["mean"],
                    "directed_ci95_low": directed_summary["ci95_low"],
                    "directed_ci95_high": directed_summary["ci95_high"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["budget", "split", "metric"], kind="stable", ignore_index=True
    )


def _plot_root_artifacts(
    active: pd.DataFrame,
    ablation: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def save(figure: Any, path: Path) -> None:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".png", dir=path.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                figure.savefig(temporary, format="png", dpi=180)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
                plt.close(figure)

        primary = active[active["split"].eq(INTERPOLATION_SPLIT)]
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for strategy in STRATEGIES:
            values = primary[primary["strategy"].eq(strategy)]
            summary = (
                values.groupby("budget", sort=True)[PRIMARY_METRIC]
                .agg(["mean", "std"])
                .reset_index()
            )
            axis.plot(summary["budget"], summary["mean"], marker="o", label=strategy)
            if summary["std"].notna().any():
                axis.fill_between(
                    summary["budget"],
                    summary["mean"] - summary["std"].fillna(0.0),
                    summary["mean"] + summary["std"].fillna(0.0),
                    alpha=0.15,
                )
        axis.set_xlabel("Labelled candidate conditions")
        axis.set_ylabel(PRIMARY_METRIC)
        axis.set_title("GOAI v2.2 Direct interpolation learning curve")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        save(figure, output_root / "learning_curve.png")

        primary_ablation = ablation[ablation["split"].eq(INTERPOLATION_SPLIT)]
        figure, axis = plt.subplots(figsize=(7.4, 4.8))
        for feature_mode in ("identity", "semantic"):
            values = primary_ablation[
                primary_ablation["feature_mode"].eq(feature_mode)
            ]
            summary = (
                values.groupby("budget", sort=True)[PRIMARY_METRIC]
                .agg(["mean", "std"])
                .reset_index()
            )
            axis.plot(
                summary["budget"], summary["mean"], marker="o", label=feature_mode
            )
            if summary["std"].notna().any():
                axis.fill_between(
                    summary["budget"],
                    summary["mean"] - summary["std"].fillna(0.0),
                    summary["mean"] + summary["std"].fillna(0.0),
                    alpha=0.15,
                )
        axis.set_xlabel("Nested labelled candidate conditions")
        axis.set_ylabel(PRIMARY_METRIC)
        axis.set_title("Combined semantic vs identity + time")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        save(figure, output_root / "representation_ablation.png")
        return {
            "status": "complete",
            "files": ["learning_curve.png", "representation_ablation.png"],
            "uncertainty_band": "sample_sd",
        }
    except Exception as error:
        for name in ("learning_curve.png", "representation_ablation.png"):
            (output_root / name).unlink(missing_ok=True)
        return {
            "status": "unavailable",
            "files": [],
            "uncertainty_band": "sample_sd",
            "reason": f"{type(error).__name__}: {error}",
        }


def _validate_root_required_files(
    output_root: Path,
    seeds: Sequence[int],
    plot_status: Mapping[str, Any],
) -> None:
    required = {
        "analysis_summary.json",
        "split_assignments.csv",
        "feature_summary.json",
        "data_audit.json",
        "tensor_coverage.csv",
        "low_rank_spectrum.csv",
        "control_vehicle_sensitivity.csv",
        *ROOT_TABLE_FILES,
        "source_snapshot/manifest.json",
    }
    missing = sorted(name for name in required if not (output_root / name).is_file())
    if missing:
        raise ValueError(f"Root run is missing required artifacts: {missing}")
    for run_seed in seeds:
        if not seed_directory(output_root, int(run_seed)).is_dir():
            raise ValueError(f"Root run is missing completed seed_{run_seed}")
    if plot_status.get("status") == "complete":
        plot_files = tuple(str(value) for value in plot_status.get("files", ()))
        expected_plots = ("learning_curve.png", "representation_ablation.png")
        if plot_files != expected_plots or any(
            not (output_root / name).is_file() for name in expected_plots
        ):
            raise ValueError("Plot completion status does not match plot artifacts")


def rebuild_root_artifacts(
    output_root: str | Path,
    seeds: Sequence[int],
    run_identity: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Rebuild every aggregate strictly from hash-valid completed seed outputs."""

    root = Path(output_root)
    tables = _seed_tables(root, seeds, run_identity)
    active = tables["active_metrics.csv"].sort_values(
        ["seed", "strategy", "budget", "split"], kind="stable", ignore_index=True
    )
    full = tables["full_reference_metrics.csv"].sort_values(
        ["seed", "split"], kind="stable", ignore_index=True
    )
    ablation = tables["ablation_metrics.csv"].sort_values(
        ["seed", "budget", "feature_mode", "split"],
        kind="stable",
        ignore_index=True,
    )
    acquisitions = tables["acquisitions.csv"].sort_values(
        ["seed", "strategy", "round", "rank_in_batch"],
        kind="stable",
        ignore_index=True,
    )
    curves = _per_seed_curve_summary(active, full)
    aggregates = _aggregate_metric_tables(active, full, ablation, curves)
    comparisons, decisions = _paired_policy_comparisons(curves, profile)
    representation = _representation_ablation_summary(ablation)
    for name, frame in (
        ("active_metrics.csv", active),
        ("full_reference_metrics.csv", full),
        ("ablation_metrics.csv", ablation),
        ("acquisitions.csv", acquisitions),
        ("per_seed_curve_summary.csv", curves),
        ("aggregate_metrics.csv", aggregates),
        ("paired_policy_comparisons.csv", comparisons),
        ("representation_ablation_summary.csv", representation),
    ):
        atomic_write_csv(root / name, frame)

    qualifying = [
        value for value in decisions if value.get("beats_random") is True
    ]
    if profile == "smoke":
        recommendation = "diagnostic_only"
        decision_status = "diagnostic_only"
    elif qualifying:
        qualifying_policies = sorted(str(value["policy"]) for value in qualifying)
        if len(qualifying_policies) == 1:
            recommendation = qualifying_policies[0]
            decision_status = "policy_beats_random"
        else:
            recommendation = "multiple_qualified"
            decision_status = "multiple_qualified"
    else:
        qualifying_policies = []
        recommendation = "random"
        decision_status = "retain_random"
    if profile == "smoke":
        qualifying_policies = []
    analysis = {
        "schema": "goai.direct_multiseed.analysis_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "profile": profile,
        "diagnostic_only": profile == "smoke",
        "primary_decision": {
            "split": INTERPOLATION_SPLIT,
            "metric": PRIMARY_METRIC,
            "summary_metric": "normalized_trapezoidal_aulc",
            "rule": (
                "directed paired mean > 0, paired 95% t-CI lower > 0, and "
                "at least 4 wins among exactly 5 formal seeds"
            ),
            "policy_decisions": decisions,
            "status": decision_status,
            "recommendation": recommendation,
            "qualifying_policies": qualifying_policies,
        },
        "b80": {
            "target_fraction": TARGET_FRACTION,
            "same_seed_full_reference": True,
            "monotone_envelope": True,
            "extrapolation": False,
            "not_reached_serialization": None,
        },
        "seed_count": len(seeds),
        "uncertainty_summary": "sample_sd",
        "curve_summaries": json_null_nonfinite(curves.to_dict("records")),
        "paired_policy_comparisons": json_null_nonfinite(
            comparisons.to_dict("records")
        ),
        "representation_ablation": {
            "contrast": "semantic_minus_identity",
            "identical_nested_ids_and_model_seeds": True,
            "strain_identity_warning": STRAIN_IDENTITY_WARNING,
            "DHY210_semantics": "zero_as_provided",
            "rows": json_null_nonfinite(representation.to_dict("records")),
        },
    }
    atomic_write_json(root / "analysis_summary.json", analysis)
    plot_status = _plot_root_artifacts(active, ablation, root)
    return {
        "analysis": analysis,
        "plot_status": plot_status,
        "tables": {name: int(len(frame)) for name, frame in (
            ("active_metrics.csv", active),
            ("full_reference_metrics.csv", full),
            ("ablation_metrics.csv", ablation),
            ("acquisitions.csv", acquisitions),
            ("per_seed_curve_summary.csv", curves),
            ("aggregate_metrics.csv", aggregates),
            ("paired_policy_comparisons.csv", comparisons),
            ("representation_ablation_summary.csv", representation),
        )},
    }


def run_direct_multiseed(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    resume: bool = False,
    device: str | None = None,
    command: Sequence[str] | None = None,
    dependencies: RunnerDependencies | None = None,
) -> Path:
    """Run or exactly resume the complete registered v2.2 experiment."""

    started_at = _utc_now()
    started = perf_counter()
    config_file = Path(config_path).expanduser().resolve(strict=True)
    output_root = Path(output_dir).expanduser()
    if not output_root.is_absolute():
        raise ValueError("--output-dir must be an absolute path")
    output_root = output_root.resolve()
    _reserve_output(output_root, resume=resume)
    config = load_config(config_file)
    spec = _profile_spec(config, profile)
    effective_device = str(config["model"]["device"] if device is None else device)
    if effective_device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    settings = build_direct_model_settings(
        config["model"], epochs=spec["epochs"], device_override=effective_device
    )
    injected_dependencies = dependencies is not None
    determinism = configure_determinism(
        effective_device,
        profile=profile,
        injected_dependencies=injected_dependencies,
    )
    deps = default_runner_dependencies() if dependencies is None else dependencies
    actual_command = tuple(
        command
        if command is not None
        else (
            sys.executable,
            "-m",
            "goai_al.direct_multiseed",
            "--config",
            str(config_file),
            "--profile",
            profile,
            "--output-dir",
            str(output_root),
            *(("--resume",) if resume else ()),
            *(("--device", effective_device) if device is not None else ()),
        )
    )

    dataset: GroupedDataset
    bundles: dict[str, FeatureBundle]
    partition_summary: dict[str, Any]
    control_contract: dict[str, Any]
    existing_manifest: dict[str, Any] | None = None
    root_identity_accepted = False
    phase = "runtime_inputs"
    try:
        dataset, bundles, partition_summary, control_contract = _load_runtime_inputs(
            config, deps
        )
        phase = "build_run_identity"
        run_identity = build_run_identity(
            config_file,
            config,
            spec,
            effective_device,
            bundles,
            partition_summary,
            control_contract,
            determinism,
        )
        for source_name, dataset_key in (
            ("metadata", "metadata_sha256"),
            ("proteome", "proteome_sha256"),
        ):
            dataset_digest = dataset.source_hashes.get(dataset_key)
            identity_digest = run_identity["inputs"][source_name]["sha256"]
            if dataset_digest is not None and dataset_digest != identity_digest:
                raise ValueError(
                    f"Loaded dataset {source_name} hash differs from the run identity"
                )
        phase = "source_snapshot"
        write_source_snapshot(output_root, config, config_path=config_file)
        if resume:
            phase = "resume_identity_validation"
            existing_manifest = _validate_root_identity(output_root, run_identity)
            root_identity_accepted = True
        elif (output_root / "manifest.json").exists():
            raise FileExistsError("Fresh run cannot overwrite a root manifest")
        else:
            root_identity_accepted = True
        phase = "staging_validation"
        unfinished_staging = sorted(
            path.name for path in output_root.glob(".seed_*.staging-*")
        )
        if unfinished_staging:
            raise ValueError(
                "Output contains unfinished seed staging directories; refusing to "
                f"treat them as complete: {unfinished_staging}"
            )

        command_history = []
        failure_attempts: list[dict[str, Any]] = []
        original_started_at = started_at
        if existing_manifest is not None:
            previous = existing_manifest.get("command_history", [])
            if isinstance(previous, list):
                command_history.extend(previous)
            previous_failures = existing_manifest.get("failure_attempts", [])
            if isinstance(previous_failures, list):
                failure_attempts.extend(
                    value for value in previous_failures if isinstance(value, dict)
                )
            failure_attempts = _collect_failure_attempts(
                output_root, failure_attempts
            )
            original_started_at = str(
                existing_manifest.get("started_at_utc", started_at)
            )
        command_history.append(
            {
                "at_utc": started_at,
                "resume": bool(resume),
                "command": shlex.join(actual_command),
                "argv": list(actual_command),
            }
        )
        root_manifest: dict[str, Any] = {
            "schema": ROOT_MANIFEST_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "model_id": MODEL_ID,
            "status": "running",
            "completion_state": "running",
            "profile": profile,
            "diagnostic_only": profile == "smoke",
            "started_at_utc": original_started_at,
            "current_attempt_started_at_utc": started_at,
            "completed_at_utc": None,
            "wall_time_seconds": None,
            "run_identity": run_identity,
            "run_identity_sha256": canonical_hash(run_identity),
            "command_history": command_history,
            "failure_attempts": failure_attempts,
            "environment": {
                **_environment_metadata(effective_device),
                "determinism": determinism,
            },
            "model_summary": _model_settings_summary(settings),
            "partition_summary": partition_summary,
            "data_summary": _seed_data_summary(dataset),
            "control_summary": dict(dataset.control_policy_summary),
            "control_contract": control_contract,
            "artifact_inventory": {},
        }
        phase = "running_manifest_publication"
        write_root_manifest(output_root / "manifest.json", root_manifest)

        phase = "root_setup_artifacts"
        _write_split_assignments(dataset, output_root)
        feature_summary = _write_feature_summary(
            bundles,
            output_root,
            control_contract=control_contract,
            partition_contract=partition_summary,
        )
        phase = "write_audits"
        deps.write_audits(dataset, output_root)
        root_manifest["feature_summary"] = feature_summary
        root_manifest["posthoc_data_audits"] = {
            "acquisition_input": False,
            "training_input": False,
        }
        phase = "running_manifest_update"
        write_root_manifest(output_root / "manifest.json", root_manifest)

        phase = "seed_execution"
        requested_seed_names = {f"seed_{seed}" for seed in spec["seeds"]}
        unexpected_seed_dirs = sorted(
            path.name
            for path in output_root.glob("seed_*")
            if path.name not in requested_seed_names
        )
        if unexpected_seed_dirs:
            raise ValueError(
                f"Output contains seed directories outside this run identity: {unexpected_seed_dirs}"
            )
        completed_seeds: list[int] = []
        skipped_seeds: list[int] = []
        for run_seed in spec["seeds"]:
            destination = seed_directory(output_root, run_seed)
            if destination.exists():
                if not resume:
                    raise FileExistsError(
                        f"Fresh run cannot overwrite existing seed: {destination}"
                    )
                validate_complete_seed(destination, run_seed, run_identity)
                skipped_seeds.append(run_seed)
            else:
                run_seed_atomic(
                    output_root,
                    run_seed,
                    run_identity,
                    dataset,
                    bundles,
                    settings,
                    spec,
                    command=actual_command,
                    dependencies=deps,
                )
            completed_seeds.append(run_seed)

        phase = "aggregation"
        aggregate_result = rebuild_root_artifacts(
            output_root,
            spec["seeds"],
            run_identity,
            profile=profile,
        )
        ended_at = _utc_now()
        root_manifest.update(
            {
                "status": "complete",
                "completion_state": "complete",
                "completed_at_utc": ended_at,
                "wall_time_seconds": float(perf_counter() - started),
                "completed_seeds": completed_seeds,
                "resume_skipped_seeds": skipped_seeds,
                "aggregate_summary": aggregate_result["tables"],
                "plot_status": aggregate_result["plot_status"],
            }
        )
        phase = "final_root_validation"
        _validate_root_required_files(
            output_root, spec["seeds"], aggregate_result["plot_status"]
        )
        inventory = artifact_inventory(output_root, exclude=("manifest.json",))
        root_manifest["artifact_inventory"] = inventory
        write_root_manifest(output_root / "manifest.json", root_manifest)
        validate_artifact_inventory(
            output_root, inventory, exclude=("manifest.json",)
        )
        _validate_root_identity(output_root, run_identity)
        return output_root
    except BaseException as error:
        manifest_path = output_root / "manifest.json"
        if root_identity_accepted and manifest_path.exists():
            root_attempt: dict[str, Any] | None = None
            try:
                root_attempt = _append_root_failure_attempt(output_root, phase, error)
            except BaseException:
                pass
            try:
                manifest = _read_strict_json(manifest_path)
                if isinstance(manifest, dict):
                    preserved = manifest.get("failure_attempts", [])
                    attempts = _collect_failure_attempts(
                        output_root,
                        preserved if isinstance(preserved, list) else (),
                    )
                    if root_attempt is not None:
                        attempts = _collect_failure_attempts(
                            output_root, (*attempts, root_attempt)
                        )
                    manifest.update(
                        {
                            "status": "failed",
                            "completion_state": "failed",
                            "completed_at_utc": _utc_now(),
                            "wall_time_seconds": float(perf_counter() - started),
                            "failure": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                            "failure_attempts": attempts,
                        }
                    )
                    write_root_manifest(manifest_path, manifest)
            except BaseException:
                pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("smoke", "formal"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(raw_args)
    if not args.output_dir.expanduser().is_absolute():
        raise SystemExit("--output-dir must be an absolute path")
    command = [sys.executable, "-m", "goai_al.direct_multiseed", *raw_args]
    completed = run_direct_multiseed(
        args.config,
        profile=args.profile,
        output_dir=args.output_dir,
        resume=bool(args.resume),
        device=args.device,
        command=command,
    )
    print(f"Completed GOAI v2.2 Direct multi-seed run: {completed}")


__all__ = [
    "EXPERIMENT_ID",
    "DATA_PROTOCOL",
    "MODEL_ID",
    "FORMAL_SEEDS",
    "SMOKE_SEEDS",
    "SPLIT_SEED",
    "STRATEGIES",
    "FORMAL_INITIAL_BUDGET",
    "FORMAL_ACQUISITION_BATCH_SIZE",
    "FORMAL_CHECKPOINTS",
    "FORMAL_EPOCHS",
    "FORMAL_MC_PASSES",
    "SMOKE_INITIAL_BUDGET",
    "SMOKE_ACQUISITION_BATCH_SIZE",
    "SMOKE_CHECKPOINTS",
    "SMOKE_EPOCHS",
    "SMOKE_MC_PASSES",
    "DIRECT_MODEL_DEFAULTS",
    "FROZEN_METADATA_PATH",
    "FROZEN_PROTEOME_PATH",
    "FROZEN_CACHE_PATH",
    "MISSING_RATE_THRESHOLD",
    "INTERPOLATION_FRACTION",
    "TARGET_FRACTION",
    "FeatureBundle",
    "RunnerDependencies",
    "coerce_feature_bundle",
    "validate_feature_bundle",
    "load_identity_feature_bundle",
    "load_semantic_feature_bundle",
    "deterministic_initial_ids",
    "model_seed",
    "acquisition_seed",
    "deterministic_nested_ids",
    "build_direct_model_settings",
    "frozen_config",
    "validate_config",
    "load_config",
    "canonical_json",
    "canonical_hash",
    "atomic_write_json",
    "atomic_write_json_nullable",
    "atomic_write_csv",
    "json_null_nonfinite",
    "validate_receipt",
    "validate_dataset_partitions",
    "prepare_masked_model_features",
    "curve_summary",
    "finite_metric_summary",
    "policy_superiority_decision",
    "seed_directory",
    "seed_manifest_path",
    "make_seed_manifest",
    "validate_run_identity",
    "validate_seed_manifest",
    "ensure_seed_directory",
    "artifact_inventory",
    "validate_artifact_inventory",
    "create_seed_staging",
    "finalize_seed_staging",
    "validate_complete_seed",
    "aggregate_metrics",
    "paired_policy_differences",
    "run_policy_trajectories",
    "run_seed_atomic",
    "build_run_identity",
    "rebuild_root_artifacts",
    "run_direct_multiseed",
    "main",
]


if __name__ == "__main__":
    main()
