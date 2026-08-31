from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goai_al.data import (
    CHEMICAL,
    MEDIUM,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    GroupedDataset,
)
from goai_al.direct_multiseed import (
    DATA_PROTOCOL,
    CONTROL_CONTRACT,
    EXPERIMENT_ID,
    FORMAL_ACQUISITION_BATCH_SIZE,
    FORMAL_CHECKPOINTS,
    FORMAL_EPOCHS,
    FORMAL_INITIAL_BUDGET,
    FORMAL_MC_PASSES,
    FORMAL_SEEDS,
    FROZEN_CACHE_PATH,
    FROZEN_METADATA_PATH,
    FROZEN_PROTEOME_PATH,
    INTERPOLATION_FRACTION,
    MISSING_RATE_THRESHOLD,
    MODEL_ID,
    SMOKE_ACQUISITION_BATCH_SIZE,
    SMOKE_CHECKPOINTS,
    SMOKE_EPOCHS,
    SMOKE_INITIAL_BUDGET,
    SMOKE_MC_PASSES,
    SMOKE_SEEDS,
    SPLIT_SEED,
    STRATEGIES,
    DIRECT_MODEL_DEFAULTS,
    FeatureBundle,
    PublicFeatureDatasetView,
    acquisition_seed,
    aggregate_metrics,
    atomic_write_csv,
    atomic_write_json,
    build_direct_model_settings,
    canonical_hash,
    canonical_json,
    coerce_feature_bundle,
    deterministic_initial_ids,
    ensure_seed_directory,
    load_config,
    load_identity_feature_bundle,
    load_semantic_feature_bundle,
    make_seed_manifest,
    model_seed,
    paired_policy_differences,
    seed_directory,
    validate_config,
    validate_control_contract,
    validate_receipt,
    validate_seed_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> GroupedDataset:
    ids = pd.Index(["c", "a", "b"], name="condition_id")
    metadata = pd.DataFrame(
        {
            STRAIN: ["s1", "s1", "s2"],
            CHEMICAL: ["x", "y", "x"],
            MEDIUM: ["m", "m", "n"],
            TEMPERATURE: [30, 30, 37],
            TIME: [1.0, 2.0, 1.0],
            TIME_UNIT: ["h", "h", "h"],
        },
        index=ids,
    )
    return GroupedDataset(
        metadata=metadata,
        response=pd.DataFrame([[1.0], [2.0], [3.0]], index=ids),
        proteins=("p",),
        train_ids=pd.Index(["c", "a"], name="condition_id"),
        validation_ids={},
        protein_missing_rate=pd.Series([0.0], index=["p"]),
    )


def test_frozen_protocol_constants() -> None:
    assert EXPERIMENT_ID == "goai-al-direct-semantic-multiseed-v2.2"
    assert DATA_PROTOCOL == "goai-condition-atomic-v2.1"
    assert MODEL_ID == "GOAI-AL-V22-DIRECT-SEMANTIC-01"
    assert FORMAL_SEEDS == (42, 43, 44, 45, 46)
    assert SMOKE_SEEDS == (42, 43)
    assert SPLIT_SEED == 42
    assert STRATEGIES == ("random", "coreset", "uncertainty")
    assert (FORMAL_INITIAL_BUDGET, FORMAL_ACQUISITION_BATCH_SIZE) == (128, 128)
    assert FORMAL_CHECKPOINTS == (128, 256, 512, 1024)
    assert (FORMAL_EPOCHS, FORMAL_MC_PASSES) == (80, 8)
    assert (SMOKE_INITIAL_BUDGET, SMOKE_ACQUISITION_BATCH_SIZE) == (32, 32)
    assert SMOKE_CHECKPOINTS == (32, 64, 96)
    assert (SMOKE_EPOCHS, SMOKE_MC_PASSES) == (2, 2)


def test_feature_bundle_accepts_current_mapping_and_different_widths() -> None:
    dataset = _dataset()
    mapping = {
        "row_ids": dataset.metadata.index,
        "descriptor_matrix": np.ones((3, 2)),
        "model_matrix": np.ones((3, 4)),
        "masker": None,
        "summary": {"response_used": False, "nested": [1, "x"]},
        "asset_hashes": {"asset": "abc"},
    }
    bundle = coerce_feature_bundle(mapping, dataset)
    assert isinstance(bundle, FeatureBundle)
    assert bundle.row_ids == ("c", "a", "b")
    assert bundle.descriptor_matrix.shape == (3, 2)
    assert bundle.model_matrix.shape == (3, 4)
    assert bundle.descriptor_matrix.dtype == np.float32


@pytest.mark.parametrize(
    "change, match",
    [
        ({"row_ids": ["a", "c", "b"]}, "row order"),
        ({"descriptor_matrix": np.ones(3)}, "two-dimensional"),
        ({"model_matrix": np.array([[1.0], [np.nan], [2.0]])}, "finite"),
        ({"summary": {"bad": float("nan")}}, "JSON-safe"),
    ],
)
def test_feature_bundle_rejects_misalignment_nonfinite_and_unsafe_summary(
    change: dict[str, object], match: str
) -> None:
    dataset = _dataset()
    mapping = {
        "row_ids": dataset.metadata.index,
        "descriptor_matrix": np.ones((3, 2)),
        "model_matrix": np.ones((3, 2)),
        "summary": {},
        "asset_hashes": {},
    }
    mapping.update(change)
    with pytest.raises(ValueError, match=match):
        coerce_feature_bundle(mapping, dataset)


def test_identity_loader_and_semantic_adapter_do_not_use_response() -> None:
    dataset = _dataset()
    identity = load_identity_feature_bundle(dataset)
    assert identity.row_ids == tuple(dataset.metadata.index)
    assert identity.summary["response_used"] is False
    assert identity.descriptor_matrix.shape == identity.model_matrix.shape

    semantic_mapping = {
        "row_ids": dataset.metadata.index,
        "descriptors": np.zeros((3, 2)),
        "model_features": np.ones((3, 3)),
        "summary": {"response_used": False},
        "asset_hashes": {"x": "0" * 64},
    }
    semantic = load_semantic_feature_bundle(dataset, loader=lambda _: semantic_mapping)
    assert semantic.descriptor_matrix.shape[1] == 2
    assert semantic.model_matrix.shape[1] == 3


def test_initial_ids_are_order_independent_and_seeds_ignore_strategy() -> None:
    ids = [f"id-{index}" for index in range(30)]
    first = deterministic_initial_ids(ids, 8, 42)
    second = deterministic_initial_ids(reversed(ids), 8, 42)
    assert first == second
    assert len(first) == len(set(first)) == 8
    assert deterministic_initial_ids(ids, 8, 43) != first
    assert model_seed(42, 128) == model_seed(run_seed=42, budget=128)
    assert acquisition_seed(42, 128) == acquisition_seed(run_seed=42, budget=128)
    assert model_seed(42, 128) != acquisition_seed(42, 128)


def test_direct_model_builder_is_frozen_and_direct_only() -> None:
    settings = build_direct_model_settings(epochs=80)
    assert settings.kind == "direct"
    assert settings.epochs == 80
    assert dict(DIRECT_MODEL_DEFAULTS) == {
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
    assert settings.hidden_dim == 128
    assert settings.dropout == 0.10
    assert settings.weight_decay == 0.0002
    assert settings.batch_size == 512
    assert settings.target_scale_floor == 0.05
    assert settings.device == "cuda"
    with pytest.raises(ValueError, match="frozen Direct"):
        build_direct_model_settings({"kind": "low_rank"}, epochs=80)
    with pytest.raises(ValueError, match="Unknown"):
        build_direct_model_settings({"surprise": 1}, epochs=80)


def test_yaml_matches_complete_frozen_configuration() -> None:
    config = load_config(ROOT / "configs" / "direct_multiseed.yaml")
    assert validate_config(config) == config
    assert config["data"] == {
        "metadata_path": str(FROZEN_METADATA_PATH),
        "proteome_path": str(FROZEN_PROTEOME_PATH),
        "cache_dir": "results/cache_v22",
        "missing_rate_threshold": MISSING_RATE_THRESHOLD,
        "interpolation_fraction": INTERPOLATION_FRACTION,
        "control_policy": "pooled_exact_context_water_dmso",
        "vehicle_column": None,
    }
    assert config["evaluation"] == {"target_fraction": 0.80}
    assert FROZEN_CACHE_PATH == ROOT / "results" / "cache_v22"
    assert Path(config["data"]["metadata_path"]) == FROZEN_METADATA_PATH
    assert Path(config["data"]["proteome_path"]) == FROZEN_PROTEOME_PATH
    changed = json.loads(json.dumps(config))
    changed["evaluation"]["target_fraction"] = 0.79
    with pytest.raises(ValueError, match="evaluation settings"):
        validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["profiles"]["smoke"]["mc_passes"] = 3
    with pytest.raises(ValueError, match="schedules"):
        validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["data"]["missing_rate_threshold"] = 0.79
    with pytest.raises(ValueError, match="data settings"):
        validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["model"]["hidden_dim"] = 129
    with pytest.raises(ValueError, match="frozen Direct"):
        validate_config(changed)


def test_canonical_json_hash_and_atomic_helpers(tmp_path: Path) -> None:
    left = {"b": [2, 1], "a": {"z": True}}
    right = {"a": {"z": True}, "b": [2, 1]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_hash(left) == canonical_hash(right)
    with pytest.raises(ValueError, match="JSON-safe"):
        canonical_json({"bad": float("nan")})

    json_path = atomic_write_json(tmp_path / "nested" / "value.json", left)
    assert json.loads(json_path.read_text()) == left
    csv_path = atomic_write_csv(
        tmp_path / "rows.csv", [{"seed": 42, "value": 1.0}, {"seed": 43, "value": 2.0}]
    )
    assert pd.read_csv(csv_path).to_dict("records") == [
        {"seed": 42, "value": 1.0},
        {"seed": 43, "value": 2.0},
    ]
    assert not list(tmp_path.rglob("*.tmp"))


def test_receipt_validator_is_recursive_and_json_strict() -> None:
    receipt = {"fit": {"loss": 1.2}, "events": [{"budget": 32}]}
    assert validate_receipt(receipt) == receipt
    with pytest.raises(ValueError, match="forbidden key"):
        validate_receipt({"fit": {"events": [{"predictions": [1.0]}]}})
    with pytest.raises(ValueError, match="JSON-safe"):
        validate_receipt({"fit": {"loss": float("inf")}})
    for forbidden in ("response", "targets", "model_state_dict", "oracle_impacts"):
        with pytest.raises(ValueError, match="forbidden key"):
            validate_receipt({"nested": [{forbidden: "hidden"}]})


def test_public_feature_view_has_no_response_and_feature_hashes_are_stable() -> None:
    dataset = _dataset()
    view = PublicFeatureDatasetView.from_dataset(dataset)
    assert not hasattr(view, "response")
    first = coerce_feature_bundle(
        {
            "row_ids": view.metadata.index,
            "descriptor_matrix": np.arange(6).reshape(3, 2),
            "model_matrix": np.arange(9).reshape(3, 3),
            "summary": {"response_used": False},
            "asset_hashes": {},
        },
        dataset,
    )
    second = coerce_feature_bundle(
        {
            "row_ids": view.metadata.index,
            "descriptor_matrix": np.arange(6).reshape(3, 2),
            "model_matrix": np.arange(9).reshape(3, 3),
            "summary": {"response_used": False},
            "asset_hashes": {},
        },
        dataset,
    )
    assert first.descriptor_matrix.flags.writeable is False
    assert first.model_matrix.flags.writeable is False
    assert first.descriptor_matrix_identity == second.descriptor_matrix_identity
    changed = np.arange(6).reshape(3, 2)
    changed[0, 0] = 99
    third = coerce_feature_bundle(
        {
            "row_ids": view.metadata.index,
            "descriptor_matrix": changed,
            "model_matrix": np.arange(9).reshape(3, 3),
            "summary": {"response_used": False},
            "asset_hashes": {},
        },
        dataset,
    )
    assert first.descriptor_matrix_identity != third.descriptor_matrix_identity


def test_control_contract_rejects_non_registered_summary() -> None:
    dataset = _dataset()
    object.__setattr__(
        dataset,
        "control_policy_summary",
        {
            **dict(CONTROL_CONTRACT),
            "vehicle_inference": True,
        },
    )
    with pytest.raises(ValueError, match="pooled contract"):
        validate_control_contract(dataset)


def test_seed_directory_manifest_validates_run_identity(tmp_path: Path) -> None:
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "data_protocol": DATA_PROTOCOL,
        "model_id": MODEL_ID,
        "feature_mode": "semantic",
    }
    directory = ensure_seed_directory(tmp_path, 42, identity)
    assert directory == seed_directory(tmp_path, 42)
    assert validate_seed_manifest(directory, 42, identity) == make_seed_manifest(42, identity)
    assert ensure_seed_directory(tmp_path, 42, dict(reversed(list(identity.items())))) == directory
    with pytest.raises(ValueError, match="identity"):
        ensure_seed_directory(tmp_path, 42, {**identity, "feature_mode": "identity"})


def test_aggregate_metrics_has_exact_small_sample_statistics() -> None:
    values = {46: 5.0, 42: 1.0, 45: 4.0, 43: 2.0, 44: 3.0}
    result = aggregate_metrics(values)
    assert result["raw"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result["n"] == 5
    assert result["mean"] == result["median"] == 3.0
    assert result["sample_sd"] == pytest.approx(math.sqrt(2.5))
    assert result["q25"] == 2.0
    assert result["q75"] == 4.0
    assert result["iqr"] == 2.0
    assert result["t_critical_95"] == 2.776445
    half_width = 2.776445 * math.sqrt(2.5) / math.sqrt(5)
    assert result["ci95"] == pytest.approx([3.0 - half_width, 3.0 + half_width])

    smoke = aggregate_metrics({42: 1.0, 43: 3.0})
    assert smoke["t_critical_95"] == 12.706205
    with pytest.raises(ValueError, match="exactly 2 or 5"):
        aggregate_metrics({42: 1.0})


def test_paired_policy_differences_aligns_seeds_and_directs_metric() -> None:
    policy = {42: 1.0, 43: 4.0}
    reference = {42: 2.0, 43: 1.0}
    higher = paired_policy_differences(policy, reference)
    assert higher["raw_difference"] == [-1.0, 3.0]
    assert higher["directed_difference"] == [-1.0, 3.0]
    assert higher["mean"] == 1.0
    assert (higher["wins"], higher["losses"], higher["ties"]) == (1, 1, 0)

    lower = paired_policy_differences(policy, reference, higher_is_better=False)
    assert lower["raw_difference"] == [-1.0, 3.0]
    assert lower["directed_difference"] == [1.0, -3.0]
    assert lower["mean"] == -1.0
    with pytest.raises(ValueError, match="aligned"):
        paired_policy_differences(policy, {42: 2.0, 44: 1.0})
    with pytest.raises(ValueError, match="Duplicate"):
        paired_policy_differences(
            [{"seed": 42, "value": 1.0}, {"seed": 42, "value": 2.0}],
            reference,
        )
