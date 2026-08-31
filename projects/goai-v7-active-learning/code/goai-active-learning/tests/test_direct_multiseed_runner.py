from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import goai_al.direct_multiseed as dm
from goai_al.acquisition import select_batch
from goai_al.data import (
    CHEMICAL,
    MEDIUM,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    GroupedDataset,
)
from goai_al.metrics import score_response
from goai_al.simulator import RetrospectiveOracle


ROOT = Path(__file__).resolve().parents[1]


class RecordingMasker:
    def __init__(self) -> None:
        self.support_rows: list[np.ndarray] = []

    def __call__(self, values: np.ndarray, support: np.ndarray) -> np.ndarray:
        self.support_rows.append(np.array(support, copy=True))
        return np.array(values, copy=True)


class TinyFit:
    def __init__(self, seed: int, response: np.ndarray, width: int) -> None:
        self.seed = int(seed)
        self.n_train = int(len(response))
        self.width = int(width)
        self.mean = np.nanmean(response, axis=0).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.repeat(self.mean[None, :], len(features), axis=0)

    def uncertainty(self, features: np.ndarray, passes: int) -> np.ndarray:
        assert passes >= 2
        return np.abs(features[:, -1]).astype(np.float64)

    def fit_summary(self) -> dict[str, int | float | str]:
        return {
            "kind": "direct",
            "seed": self.seed,
            "n_train": self.n_train,
            "n_features": self.width,
            "final_loss": 0.0,
        }


def _dataset() -> GroupedDataset:
    pool = [f"p{index}" for index in range(8)]
    evaluation = ["e0", "e1"]
    ids = pd.Index([*pool, *evaluation], name="condition_id")
    metadata = pd.DataFrame(
        {
            STRAIN: [f"s{index % 2}" for index in range(len(ids))],
            CHEMICAL: [f"c{index % 3}" for index in range(len(ids))],
            MEDIUM: ["m"] * len(ids),
            TEMPERATURE: [30] * len(ids),
            TIME: np.arange(1, len(ids) + 1, dtype=float),
            TIME_UNIT: ["h"] * len(ids),
        },
        index=ids,
    )
    response = pd.DataFrame(
        np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2) / 10.0,
        index=ids,
        columns=["a", "b"],
    )
    return GroupedDataset(
        metadata=metadata,
        response=response,
        proteins=("a", "b"),
        train_ids=pd.Index(pool, name="condition_id"),
        validation_ids={
            "interpolation": pd.Index(evaluation, name="condition_id")
        },
        protein_missing_rate=pd.Series([0.0, 0.0], index=["a", "b"]),
        control_policy_summary=dict(dm.CONTROL_CONTRACT),
    )


def _bundle(dataset: GroupedDataset, masker: RecordingMasker) -> dm.FeatureBundle:
    rows = len(dataset.metadata)
    descriptors = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
    model = np.column_stack(
        [
            np.ones(rows, dtype=np.float32),
            np.arange(rows, dtype=np.float32),
            np.arange(rows, dtype=np.float32)[::-1],
            np.linspace(0.0, 1.0, rows, dtype=np.float32),
        ]
    )
    return dm.coerce_feature_bundle(
        {
            "row_ids": dataset.metadata.index,
            "descriptor_matrix": descriptors,
            "model_matrix": model,
            "masker": masker,
            "summary": {"response_used": False},
            "asset_hashes": {},
        },
        dataset,
    )


def _dependencies(fit_calls: list[dict[str, object]]) -> dm.RunnerDependencies:
    def fake_fit(
        features: np.ndarray,
        response: np.ndarray,
        settings: object,
        seed: int,
    ) -> TinyFit:
        fit_calls.append(
            {
                "features": np.array(features, copy=True),
                "observed": np.array(response, copy=True),
                "seed": int(seed),
            }
        )
        return TinyFit(seed, response, features.shape[1])

    return dm.RunnerDependencies(
        load_dataset=lambda *args, **kwargs: _dataset(),
        load_semantic=lambda dataset: _bundle(dataset, RecordingMasker()),
        load_identity=lambda dataset: _bundle(dataset, RecordingMasker()),
        fit_model=fake_fit,
        select=select_batch,
        score=score_response,
        write_audits=lambda dataset, output: {},
        oracle_factory=RetrospectiveOracle,
    )


def _write_fake_audits(
    dataset: GroupedDataset, destination: str | Path
) -> dict[str, Path]:
    root = Path(destination)
    return {
        "data_audit": dm.atomic_write_json(root / "data_audit.json", {}),
        "tensor_coverage": dm.atomic_write_csv(
            root / "tensor_coverage.csv", pd.DataFrame({"value": [1]})
        ),
        "low_rank_spectrum": dm.atomic_write_csv(
            root / "low_rank_spectrum.csv", pd.DataFrame({"value": [1]})
        ),
        "control_vehicle_sensitivity": dm.atomic_write_csv(
            root / "control_vehicle_sensitivity.csv", pd.DataFrame({"value": [1]})
        ),
    }


def _configure_synthetic_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    profile_spec = {
        "name": "smoke",
        "scientific": False,
        "diagnostic_only": True,
        "seeds": (42, 43),
        "initial_budget": 2,
        "batch_size": 2,
        "checkpoints": (2, 6),
        "epochs": 2,
        "mc_passes": 2,
        "ablation_fixed_budgets": (2, 4),
    }
    real_partition_validator = dm.validate_dataset_partitions
    monkeypatch.setattr(dm, "_profile_spec", lambda config, profile: profile_spec)
    monkeypatch.setattr(
        dm,
        "validate_dataset_partitions",
        lambda dataset: real_partition_validator(
            dataset, require_registered_splits=False
        ),
    )

    def fake_run_identity(
        selected_config_path: str | Path,
        config: dict[str, object],
        selected_profile: dict[str, object],
        effective_device: str,
        bundles: dict[str, dm.FeatureBundle],
        partition_contract: dict[str, object],
        control_contract: dict[str, object],
        determinism: dict[str, object],
    ) -> dict[str, object]:
        return json.loads(
            dm.canonical_json(
                {
                    "experiment_id": dm.EXPERIMENT_ID,
                    "profile": selected_profile["name"],
                    "effective_device": effective_device,
                    "seeds": list(selected_profile["seeds"]),
                    "schedule": {
                        "initial_budget": selected_profile["initial_budget"],
                        "batch_size": selected_profile["batch_size"],
                        "checkpoints": list(selected_profile["checkpoints"]),
                        "epochs": selected_profile["epochs"],
                        "mc_passes": selected_profile["mc_passes"],
                        "ablation_fixed_budgets": list(
                            selected_profile["ablation_fixed_budgets"]
                        ),
                    },
                    "strategies": list(dm.STRATEGIES),
                    "inputs": {
                        "metadata": {"sha256": "synthetic-metadata"},
                        "proteome": {"sha256": "synthetic-proteome"},
                    },
                    "partition_contract": partition_contract,
                    "control_contract": control_contract,
                    "determinism": determinism,
                    "sources": dm._source_inventory(Path(selected_config_path)),
                }
            )
        )

    monkeypatch.setattr(dm, "build_run_identity", fake_run_identity)


def _tiny_profile() -> dict[str, object]:
    return {
        "name": "smoke",
        "initial_budget": 2,
        "batch_size": 2,
        "checkpoints": (2, 6),
        "mc_passes": 2,
    }


def test_policy_loop_has_shared_initial_ids_conserved_budgets_and_strategy_free_seeds(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    masker = RecordingMasker()
    bundle = _bundle(dataset, masker)
    fit_calls: list[dict[str, object]] = []
    metrics, acquisitions, receipts = dm.run_policy_trajectories(
        dataset,
        bundle,
        dm.build_direct_model_settings(epochs=2, device_override="cpu"),
        _tiny_profile(),
        42,
        tmp_path / "round_receipts",
        dependencies=_dependencies(fit_calls),
    )

    initial_sets = {
        strategy: tuple(
            acquisitions[
                acquisitions["strategy"].eq(strategy)
                & acquisitions["round"].eq(0)
            ]["condition_id"]
        )
        for strategy in dm.STRATEGIES
    }
    assert len(set(initial_sets.values())) == 1
    pool = set(dataset.candidate_pool_ids.astype(str))
    for strategy in dm.STRATEGIES:
        selected = acquisitions[acquisitions["strategy"].eq(strategy)]
        assert len(selected) == 6
        assert selected["condition_id"].is_unique
        assert set(selected["condition_id"]) <= pool
        assert list(selected.groupby("round")["budget_after"].first()) == [2, 4, 6]
    assert set(metrics["budget"]) == {2, 6}
    for budget in (2, 4, 6):
        seeds = {
            row["model_seed"]
            for row in receipts
            if row["role"] == "active_learning" and row["budget"] == budget
        }
        assert seeds == {dm.model_seed(42, budget)}
    for budget in (2, 4):
        values = acquisitions[acquisitions["budget_before"].eq(budget)][
            "acquisition_seed"
        ]
        assert set(values.astype(int)) == {dm.acquisition_seed(42, budget)}


def test_evaluation_ids_are_immutable_and_cannot_be_revealed() -> None:
    dataset = _dataset()
    summary = dm.validate_dataset_partitions(
        dataset, require_registered_splits=False
    )
    assert summary["evaluation_ids_revealable"] is False
    pool = tuple(dataset.candidate_pool_ids.astype(str))
    oracle = RetrospectiveOracle(
        pool,
        dataset.response.loc[list(pool)].to_numpy(),
        response_ids=pool,
    )
    with pytest.raises(ValueError, match="candidate IDs"):
        oracle.reveal(["e0"])


def test_unrevealed_hidden_label_permutation_does_not_change_next_acquisition(
    tmp_path: Path,
) -> None:
    first_dataset = _dataset()
    initial = set(dm.deterministic_initial_ids(first_dataset.candidate_pool_ids, 2, 42))
    second_dataset = _dataset()
    unrevealed = [
        row_id
        for row_id in second_dataset.candidate_pool_ids.astype(str)
        if row_id not in initial
    ]
    second_dataset.response.loc[unrevealed] = second_dataset.response.loc[
        list(reversed(unrevealed))
    ].to_numpy()

    results = []
    for index, dataset in enumerate((first_dataset, second_dataset)):
        acquisitions = dm.run_policy_trajectories(
            dataset,
            _bundle(dataset, RecordingMasker()),
            dm.build_direct_model_settings(epochs=2, device_override="cpu"),
            _tiny_profile(),
            42,
            tmp_path / f"run{index}",
            dependencies=_dependencies([]),
        )[1]
        results.append(
            acquisitions[acquisitions["round"].eq(1)]
            .sort_values(["strategy", "rank_in_batch"])[
                ["strategy", "condition_id"]
            ]
            .to_dict("records")
        )
    assert results[0] == results[1]


def test_masker_receives_only_labelled_support_and_feature_widths_may_differ(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    masker = RecordingMasker()
    bundle = _bundle(dataset, masker)
    calls: list[dict[str, object]] = []
    dm.run_policy_trajectories(
        dataset,
        bundle,
        dm.build_direct_model_settings(epochs=2, device_override="cpu"),
        _tiny_profile(),
        42,
        tmp_path,
        dependencies=_dependencies(calls),
    )
    assert bundle.descriptor_matrix.shape[1] == 2
    assert bundle.model_matrix.shape[1] == 4
    assert sorted(len(values) for values in masker.support_rows) == [2, 2, 2, 4, 4, 4, 6, 6, 6]
    assert {record["features"].shape[1] for record in calls} == {4}


def _metric_values(value: float) -> dict[str, float | int]:
    return {
        "n_conditions": 2,
        "n_proteins": 2,
        "n_observed_values": 4,
        "n_evaluable_conditions_pcc": 2,
        "n_evaluable_proteins_pcc": 2,
        "n_evaluable_proteins_r2": 2,
        "delta_rmse": 1.0 - value / 10.0,
        "delta_mae": 0.5,
        "delta_skill_zero": value,
        "pooled_delta_pcc": value,
        "condition_pcc_median": value,
        "protein_pcc_median": value,
        "protein_r2_median": value,
        "protein_r2_mean": value,
        "protein_r2_positive_fraction": 1.0,
    }


def _synthetic_identity(**extra: object) -> dict[str, object]:
    return {
        "experiment_id": dm.EXPERIMENT_ID,
        "profile": "smoke",
        "schedule": {
            "initial_budget": 32,
            "batch_size": 32,
            "checkpoints": [32, 64, 96],
            "ablation_fixed_budgets": [32, 64],
        },
        "partition_contract": {
            "candidate_pool_count": 100,
            "evaluation_counts": {"interpolation": 2},
        },
        **extra,
    }


def _write_seed_payload(staging: Path, seed: int, *, full_budget: int = 100) -> None:
    active_rows = []
    for strategy, offset in (("random", 0.0), ("coreset", 0.1), ("uncertainty", 0.2)):
        for round_index, (budget, value) in enumerate(((32, 0.0), (64, 0.2), (96, 0.4))):
            active_rows.append(
                {
                    "role": "active_learning",
                    "feature_mode": "semantic",
                    "strategy": strategy,
                    "seed": seed,
                    "round": round_index,
                    "budget": budget,
                    "model_seed": dm.model_seed(seed, budget),
                    "train_seconds": 0.01,
                    "split": "interpolation",
                    "primary_split": True,
                    **_metric_values(value + offset),
                }
            )
    full_rows = [
        {
            "role": "full_pool_reference",
            "feature_mode": "semantic",
            "strategy": "full_reference",
            "seed": seed,
            "round": np.nan,
            "budget": full_budget,
            "model_seed": dm.model_seed(seed, full_budget),
            "train_seconds": 0.01,
            "split": "interpolation",
            "primary_split": True,
            **_metric_values(0.8),
        }
    ]
    ablation_rows = []
    for budget in (32, 64, full_budget):
        for mode, value in (
            ("identity", 0.2),
            ("semantic", 0.8 if budget == full_budget else 0.3),
        ):
            ablation_rows.append(
                {
                    "role": "representation_ablation",
                    "feature_mode": mode,
                    "strategy": "representation_ablation",
                    "seed": seed,
                    "round": np.nan,
                    "budget": budget,
                    "model_seed": dm.model_seed(seed, budget),
                    "train_seconds": 0.01,
                    "split": "interpolation",
                    "primary_split": True,
                    **_metric_values(value),
                }
            )
    acquisition_rows = []
    selected_by_strategy: dict[str, list[str]] = {}
    for strategy_index, strategy in enumerate(dm.STRATEGIES):
        labelled: list[str] = []
        for round_index, budget_after in enumerate((32, 64, 96)):
            budget_before = 0 if round_index == 0 else budget_after - 32
            if round_index == 0:
                selected = [f"initial-{index:03d}" for index in range(32)]
            else:
                selected = [
                    f"{strategy}-{strategy_index}-{index:03d}"
                    for index in range(budget_before, budget_after)
                ]
            labelled.extend(selected)
            for rank, condition_id in enumerate(selected, start=1):
                acquisition_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "round": round_index,
                        "selection_type": (
                            "deterministic_initial" if round_index == 0 else strategy
                        ),
                        "rank_in_batch": rank,
                        "budget_before": budget_before,
                        "budget_after": budget_after,
                        "acquisition_seed": (
                            np.nan
                            if round_index == 0
                            else dm.acquisition_seed(seed, budget_before)
                        ),
                        "model_seed": dm.model_seed(
                            seed, 32 if round_index == 0 else budget_before
                        ),
                        "condition_id": condition_id,
                    }
                )
        selected_by_strategy[strategy] = labelled
    dm.atomic_write_csv(staging / "active_metrics.csv", pd.DataFrame(active_rows))
    dm.atomic_write_csv(staging / "full_reference_metrics.csv", pd.DataFrame(full_rows))
    dm.atomic_write_csv(staging / "ablation_metrics.csv", pd.DataFrame(ablation_rows))
    dm.atomic_write_csv(staging / "acquisitions.csv", pd.DataFrame(acquisition_rows))
    dm.atomic_write_json(
        staging / "fit_receipts.json",
        {
            "schema": "goai.direct_multiseed.fit_receipts.v1",
            "run_seed": seed,
            "receipts": [
                {
                    "role": "full_pool_reference",
                    "strategy": "full_reference",
                    "feature_mode": "semantic",
                    "run_seed": seed,
                    "round": None,
                    "budget": full_budget,
                    "model_seed": dm.model_seed(seed, full_budget),
                    "evaluated": True,
                    "support_count": full_budget,
                    "labelled_ids_sha256": dm.canonical_hash(
                        [f"full-{index}" for index in range(full_budget)]
                    ),
                    "train_seconds": 0.01,
                    "fit_reused": False,
                    "fit_summary": {"seed": dm.model_seed(seed, full_budget)},
                },
                *[
                    {
                        "role": "active_learning",
                        "strategy": strategy,
                        "feature_mode": "semantic",
                        "run_seed": seed,
                        "round": round_index,
                        "budget": budget,
                        "model_seed": dm.model_seed(seed, budget),
                        "evaluated": True,
                        "support_count": budget,
                        "labelled_ids_sha256": dm.canonical_hash(
                            selected_by_strategy[strategy][:budget]
                        ),
                        "train_seconds": 0.01,
                        "fit_reused": False,
                        "fit_summary": {"seed": dm.model_seed(seed, budget)},
                    }
                    for strategy in dm.STRATEGIES
                    for round_index, budget in enumerate((32, 64, 96))
                ],
                *[
                    {
                        "role": "representation_ablation",
                        "strategy": mode,
                        "feature_mode": mode,
                        "run_seed": seed,
                        "round": None,
                        "budget": budget,
                        "model_seed": dm.model_seed(seed, budget),
                        "evaluated": True,
                        "support_count": budget,
                        "labelled_ids_sha256": dm.canonical_hash(
                            [
                                f"{'full' if budget == full_budget else 'ablation'}-{index}"
                                for index in range(budget)
                            ]
                        ),
                        "train_seconds": 0.01,
                        "fit_reused": mode == "semantic" and budget == full_budget,
                        "fit_summary": {"seed": dm.model_seed(seed, budget)},
                    }
                    for budget in (32, 64, full_budget)
                    for mode in ("identity", "semantic")
                ],
            ],
        },
    )
    active_metrics_by_key = {
        (str(row["strategy"]), int(row["budget"])): {
            key: row[key]
            for key in ("split", "primary_split", *dm.COUNT_METRICS, *dm.SCORE_METRICS)
        }
        for row in active_rows
    }
    for strategy in dm.STRATEGIES:
        labelled: list[str] = []
        for round_index, budget_after in enumerate((32, 64, 96)):
            budget_before = 0 if round_index == 0 else budget_after - 32
            selected = selected_by_strategy[strategy][budget_before:budget_after]
            labelled.extend(selected)
            dm.atomic_write_json(
                staging / "round_receipts" / strategy / f"round_{round_index:03d}.json",
                {
                    "round_index": round_index,
                    "strategy": strategy,
                    "budget_before": budget_before,
                    "budget_after": budget_after,
                    "selected_ids": selected,
                    "global_seed": seed,
                    "acquisition_seed": (
                        None
                        if round_index == 0
                        else dm.acquisition_seed(seed, budget_before)
                    ),
                    "model_seed": dm.model_seed(seed, budget_after),
                    "checkpoint": True,
                    "labelled_ids": labelled,
                    "labelled_ids_sha256": dm.canonical_hash(labelled),
                    "model_fit_summary": {"seed": dm.model_seed(seed, budget_after)},
                    "split_metrics": [
                        active_metrics_by_key[(strategy, budget_after)]
                    ],
                    "train_seconds": 0.01,
                    "hashes": {
                        "selected_ids_sha256": dm.canonical_hash(selected),
                        "pool_before_sha256": "before",
                        "pool_after_sha256": "after",
                    },
                },
            )


def test_seed_staging_atomic_completion_resume_mismatch_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    identity = _synthetic_identity(hash="a")
    staging = dm.create_seed_staging(tmp_path, 42)
    _write_seed_payload(staging, 42)
    destination = dm.finalize_seed_staging(
        staging,
        tmp_path,
        42,
        identity,
        {"started_at_utc": "2026-01-01T00:00:00+00:00", "wall_time_seconds": 1.0},
    )
    assert destination == tmp_path / "seed_42"
    assert not staging.exists()
    assert dm.validate_complete_seed(destination, 42, identity)["status"] == "complete"
    with pytest.raises(ValueError, match="identity"):
        dm.validate_complete_seed(destination, 42, {**identity, "hash": "b"})
    with (destination / "active_metrics.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="inventory|integers"):
        dm.validate_complete_seed(destination, 42, identity)

    failed = dm.create_seed_staging(tmp_path, 43)
    with pytest.raises(ValueError, match="missing required"):
        dm.finalize_seed_staging(failed, tmp_path, 43, identity, {})
    assert not (tmp_path / "seed_43").exists()
    assert not (failed / "manifest.json").exists()


def test_aggregate_rebuild_and_paired_superiority_five_seed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _synthetic_identity()
    for seed in (42, 43):
        staging = dm.create_seed_staging(tmp_path, seed)
        _write_seed_payload(staging, seed)
        dm.finalize_seed_staging(staging, tmp_path, seed, identity, {"wall_time_seconds": 0.1})
    monkeypatch.setattr(
        dm,
        "_plot_root_artifacts",
        lambda active, ablation, root: {"status": "unavailable", "files": []},
    )
    result = dm.rebuild_root_artifacts(
        tmp_path, (42, 43), identity, profile="smoke"
    )
    assert result["tables"]["per_seed_curve_summary.csv"] == 6
    aggregate = pd.read_csv(tmp_path / "aggregate_metrics.csv")
    assert set(aggregate["raw_seed_count"]) == {2}
    analysis = json.loads((tmp_path / "analysis_summary.json").read_text())
    assert analysis["primary_decision"]["status"] == "diagnostic_only"

    qualifying = {"n": 5, "mean": 0.2, "ci95_low": 0.01, "wins": 4}
    assert dm.policy_superiority_decision(qualifying, profile="formal")[
        "beats_random"
    ] is True
    assert dm.policy_superiority_decision(
        {**qualifying, "wins": 3}, profile="formal"
    )["status"] == "retain_random"
    assert dm.policy_superiority_decision(
        {**qualifying, "n": 4}, profile="formal"
    )["status"] == "retain_random"


def test_formal_analysis_does_not_rank_multiple_qualifying_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _synthetic_identity(profile="formal")
    for seed in (42, 43, 44, 45, 46):
        staging = dm.create_seed_staging(tmp_path, seed)
        _write_seed_payload(staging, seed)
        dm.finalize_seed_staging(staging, tmp_path, seed, identity, {})
    monkeypatch.setattr(
        dm,
        "_plot_root_artifacts",
        lambda active, ablation, root: {
            "status": "unavailable",
            "files": [],
            "uncertainty_band": "sample_sd",
        },
    )
    result = dm.rebuild_root_artifacts(
        tmp_path, (42, 43, 44, 45, 46), identity, profile="formal"
    )
    decision = result["analysis"]["primary_decision"]
    assert decision["status"] == "multiple_qualified"
    assert decision["recommendation"] == "multiple_qualified"
    assert decision["qualifying_policies"] == ["coreset", "uncertainty"]


def test_irregular_budget_aulc_and_b80_null_or_reached() -> None:
    not_reached = dm.curve_summary([10, 20, 50], [0.0, 0.5, 0.7], 1.0)
    assert not_reached["normalized_aulc"] == pytest.approx(0.5125)
    assert not_reached["b80"] is None
    assert not_reached["b80_status"] == "not_reached"

    reached = dm.curve_summary([10, 20, 50], [0.0, 0.5, 0.9], 1.0)
    assert reached["b80_status"] == "reached"
    assert reached["b80"] == pytest.approx(42.5)


def test_receipts_and_acquisition_artifacts_have_no_forbidden_hidden_keys(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    _, acquisitions, fit_receipts = dm.run_policy_trajectories(
        dataset,
        _bundle(dataset, RecordingMasker()),
        dm.build_direct_model_settings(epochs=2, device_override="cpu"),
        _tiny_profile(),
        42,
        tmp_path,
        dependencies=_dependencies([]),
    )
    assert not set(acquisitions.columns) & dm.RECEIPT_FORBIDDEN_KEYS
    for receipt in fit_receipts:
        assert dm.validate_receipt(receipt) == receipt
    for path in tmp_path.rglob("round_*.json"):
        assert dm.validate_receipt(json.loads(path.read_text()))


def test_exact_grid_rejects_missing_registered_checkpoint(tmp_path: Path) -> None:
    staging = dm.create_seed_staging(tmp_path, 42)
    _write_seed_payload(staging, 42)
    active = pd.read_csv(
        staging / "active_metrics.csv", float_precision="round_trip"
    )
    active = active[
        ~(
            active["strategy"].eq("coreset")
            & active["budget"].eq(64)
            & active["split"].eq("interpolation")
        )
    ]
    dm.atomic_write_csv(staging / "active_metrics.csv", active)
    with pytest.raises(ValueError, match="exact grid mismatch"):
        dm.finalize_seed_staging(
            staging, tmp_path, 42, _synthetic_identity(), {}
        )


def test_acquisitions_reject_unregistered_strategy_and_wrong_total(
    tmp_path: Path,
) -> None:
    staging = dm.create_seed_staging(tmp_path, 42)
    _write_seed_payload(staging, 42)
    acquisitions = pd.read_csv(staging / "acquisitions.csv")
    extra = acquisitions.iloc[[0]].copy()
    extra["strategy"] = "unregistered"
    extra["condition_id"] = "unregistered-id"
    dm.atomic_write_csv(
        staging / "acquisitions.csv",
        pd.concat([acquisitions, extra], ignore_index=True),
    )
    with pytest.raises(ValueError, match="every registered strategy"):
        dm.finalize_seed_staging(
            staging, tmp_path, 42, _synthetic_identity(), {}
        )


def test_round_receipt_metrics_reconcile_values_and_normalize_nonfinite(
    tmp_path: Path,
) -> None:
    staging = dm.create_seed_staging(tmp_path, 42)
    _write_seed_payload(staging, 42)
    active = pd.read_csv(
        staging / "active_metrics.csv", float_precision="round_trip"
    )
    target = (
        active["strategy"].eq("random")
        & active["budget"].eq(32)
        & active["split"].eq("interpolation")
    )
    active.loc[target, "pooled_delta_pcc"] = np.nan
    dm.atomic_write_csv(staging / "active_metrics.csv", active)
    receipt_path = staging / "round_receipts/random/round_000.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["split_metrics"][0]["pooled_delta_pcc"] = None
    dm.atomic_write_json(receipt_path, receipt)
    destination = dm.finalize_seed_staging(
        staging, tmp_path, 42, _synthetic_identity(), {}
    )
    assert destination.is_dir()

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    mismatch = dm.create_seed_staging(mismatch_root, 42)
    _write_seed_payload(mismatch, 42)
    mismatch_path = mismatch / "round_receipts/coreset/round_001.json"
    mismatch_receipt = json.loads(mismatch_path.read_text())
    mismatch_receipt["split_metrics"][0]["delta_skill_zero"] += 0.5
    dm.atomic_write_json(mismatch_path, mismatch_receipt)
    with pytest.raises(ValueError, match="metric values do not reconcile"):
        dm.finalize_seed_staging(
            mismatch, mismatch_root, 42, _synthetic_identity(), {}
        )


def test_semantic_full_ablation_metrics_must_equal_full_reference(
    tmp_path: Path,
) -> None:
    staging = dm.create_seed_staging(tmp_path, 42)
    _write_seed_payload(staging, 42)
    ablation = pd.read_csv(staging / "ablation_metrics.csv")
    target = ablation["feature_mode"].eq("semantic") & ablation["budget"].eq(100)
    ablation.loc[target, "delta_skill_zero"] += 0.01
    dm.atomic_write_csv(staging / "ablation_metrics.csv", ablation)
    with pytest.raises(ValueError, match="reused full-reference metrics"):
        dm.finalize_seed_staging(
            staging, tmp_path, 42, _synthetic_identity(), {}
        )


def test_runtime_feature_loaders_receive_only_public_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dm.PublicFeatureDatasetView] = []

    def loader(view: dm.PublicFeatureDatasetView) -> dm.FeatureBundle:
        assert isinstance(view, dm.PublicFeatureDatasetView)
        assert not hasattr(view, "response")
        seen.append(view)
        return _bundle(view, RecordingMasker())

    dependencies = _dependencies([])
    dependencies = dm.RunnerDependencies(
        **{
            **dependencies.__dict__,
            "load_semantic": loader,
            "load_identity": loader,
        }
    )
    monkeypatch.setattr(
        dm,
        "validate_dataset_partitions",
        lambda dataset: {
            "candidate_pool_count": len(dataset.candidate_pool_ids),
            "evaluation_counts": {"interpolation": 2},
        },
    )
    config = dm.frozen_config()
    _, bundles, _, control = dm._load_runtime_inputs(config, dependencies)
    assert len(seen) == 2
    assert set(bundles) == {"identity", "semantic"}
    assert control["summary"] == dict(dm.CONTROL_CONTRACT)
    assert control["contract_sha256"] == dm.canonical_hash(dict(dm.CONTROL_CONTRACT))


def test_source_snapshot_is_exact_self_hashed_and_tamper_evident(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    config = dm.load_config(config_path)
    snapshot = dm.write_source_snapshot(tmp_path, config, config_path=config_path)
    expected = dm._source_inventory(config_path)
    assert set(dm.artifact_inventory(snapshot, exclude=("manifest.json",))) == set(expected)
    for relative, identity in expected.items():
        assert dm.sha256_file(snapshot / relative) == identity["sha256"]
    manifest = json.loads((snapshot / "manifest.json").read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_payload_sha256")
    assert recorded == dm.canonical_hash(payload)
    with (snapshot / "src/goai_al/direct_multiseed.py").open("ab") as handle:
        handle.write(b"\n# tamper\n")
    with pytest.raises(ValueError, match="inventory"):
        dm.write_source_snapshot(tmp_path, config, config_path=config_path)


def test_source_snapshot_failure_leaves_no_partial_final_and_only_cleans_own_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    config = dm.load_config(config_path)
    unrelated = tmp_path / ".source_snapshot.staging-keep"
    unrelated.mkdir()

    def fail_copy(source: str | Path, destination: str | Path) -> Path:
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(dm, "_atomic_binary_copy", fail_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        dm.write_source_snapshot(tmp_path, config, config_path=config_path)
    assert not (tmp_path / "source_snapshot").exists()
    assert unrelated.is_dir()
    assert list(tmp_path.glob(".source_snapshot.staging-*")) == [unrelated]


def test_root_manifest_payload_hash_rejects_tampering(tmp_path: Path) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    config = dm.load_config(config_path)
    dm.write_source_snapshot(tmp_path, config, config_path=config_path)
    identity = {"sources": dm._source_inventory(config_path), "key": "value"}
    manifest = {
        "schema": dm.ROOT_MANIFEST_SCHEMA,
        "status": "running",
        "run_identity": identity,
        "run_identity_sha256": dm.canonical_hash(identity),
        "artifact_inventory": {},
    }
    dm.write_root_manifest(tmp_path / "manifest.json", manifest)
    assert dm._validate_root_identity(tmp_path, identity)["status"] == "running"
    tampered = json.loads((tmp_path / "manifest.json").read_text())
    tampered["status"] = "complete"
    dm.atomic_write_json(tmp_path / "manifest.json", tampered)
    with pytest.raises(ValueError, match="payload hash"):
        dm._validate_root_identity(tmp_path, identity)


def test_complete_root_manifest_requires_mapping_artifact_inventory(
    tmp_path: Path,
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    config = dm.load_config(config_path)
    for index, manifest_extra in enumerate(({}, {"artifact_inventory": []})):
        output = tmp_path / f"case-{index}"
        output.mkdir()
        dm.write_source_snapshot(output, config, config_path=config_path)
        identity = {"sources": dm._source_inventory(config_path), "case": index}
        dm.write_root_manifest(
            output / "manifest.json",
            {
                "schema": dm.ROOT_MANIFEST_SCHEMA,
                "status": "complete",
                "run_identity": identity,
                "run_identity_sha256": dm.canonical_hash(identity),
                **manifest_extra,
            },
        )
        with pytest.raises(ValueError, match="inventory is missing or invalid"):
            dm._validate_root_identity(output, identity)


def test_representation_plot_draws_sample_sd_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    calls: list[str] = []
    original = Axes.fill_between

    def recording_fill_between(self: Axes, *args: object, **kwargs: object) -> object:
        calls.append(str(self.get_title()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", recording_fill_between)
    active_rows = []
    ablation_rows = []
    for seed, offset in ((42, 0.0), (43, 0.1)):
        for strategy in dm.STRATEGIES:
            for budget in (32, 64):
                active_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "budget": budget,
                        "split": dm.INTERPOLATION_SPLIT,
                        dm.PRIMARY_METRIC: budget / 100.0 + offset,
                    }
                )
        for mode in ("identity", "semantic"):
            for budget in (32, 64):
                ablation_rows.append(
                    {
                        "seed": seed,
                        "feature_mode": mode,
                        "budget": budget,
                        "split": dm.INTERPOLATION_SPLIT,
                        dm.PRIMARY_METRIC: budget / 100.0 + offset,
                    }
                )
    status = dm._plot_root_artifacts(
        pd.DataFrame(active_rows), pd.DataFrame(ablation_rows), tmp_path
    )
    assert status["status"] == "complete"
    assert len(calls) == len(dm.STRATEGIES) + 2
    assert (tmp_path / "representation_ablation.png").is_file()


def test_fresh_loader_failure_leaves_empty_retryable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    output = tmp_path / "loader-retry"
    _configure_synthetic_runner(monkeypatch)
    base_dependencies = _dependencies([])

    def fail_dataset_loader(*args: object, **kwargs: object) -> GroupedDataset:
        raise RuntimeError("injected dataset loader failure")

    failing_dependencies = dm.RunnerDependencies(
        **{**base_dependencies.__dict__, "load_dataset": fail_dataset_loader}
    )
    with pytest.raises(RuntimeError, match="injected dataset loader failure"):
        dm.run_direct_multiseed(
            config_path,
            profile="smoke",
            output_dir=output,
            device="cpu",
            command=("synthetic-loader-failure",),
            dependencies=failing_dependencies,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []

    working_dependencies = dm.RunnerDependencies(
        **{**base_dependencies.__dict__, "write_audits": _write_fake_audits}
    )
    assert dm.run_direct_multiseed(
        config_path,
        profile="smoke",
        output_dir=output,
        device="cpu",
        command=("synthetic-loader-retry",),
        dependencies=working_dependencies,
    ) == output
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_root_audit_failure_receipt_is_hashed_and_preserved_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    output = tmp_path / "audit-failure-resume"
    _configure_synthetic_runner(monkeypatch)
    base_dependencies = _dependencies([])

    def fail_audits(dataset: GroupedDataset, destination: str | Path) -> dict[str, Path]:
        raise RuntimeError("injected root audit failure")

    failing_dependencies = dm.RunnerDependencies(
        **{**base_dependencies.__dict__, "write_audits": fail_audits}
    )
    with pytest.raises(RuntimeError, match="injected root audit failure"):
        dm.run_direct_multiseed(
            config_path,
            profile="smoke",
            output_dir=output,
            device="cpu",
            command=("synthetic-audit-failure",),
            dependencies=failing_dependencies,
        )

    receipt_files = sorted((output / "failure_attempts").glob("root_attempt_*.json"))
    assert [path.name for path in receipt_files] == ["root_attempt_0001.json"]
    receipt = json.loads(receipt_files[0].read_text())
    assert receipt == dm.validate_receipt(receipt)
    assert receipt["schema"] == "goai.direct_multiseed.root_failure_attempt.v1"
    assert receipt["scope"] == "root"
    assert receipt["phase"] == "write_audits"
    assert receipt["type"] == "RuntimeError"
    assert receipt["message"] == "injected root audit failure"
    assert isinstance(receipt["at_utc"], str) and receipt["at_utc"]

    failed_manifest = json.loads((output / "manifest.json").read_text())
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["failure_attempts"] == [receipt]
    failed_payload = dict(failed_manifest)
    recorded_hash = failed_payload.pop("manifest_payload_sha256")
    assert recorded_hash == dm.canonical_hash(failed_payload)

    working_dependencies = dm.RunnerDependencies(
        **{**base_dependencies.__dict__, "write_audits": _write_fake_audits}
    )
    assert dm.run_direct_multiseed(
        config_path,
        profile="smoke",
        output_dir=output,
        resume=True,
        device="cpu",
        command=("synthetic-audit-resume",),
        dependencies=working_dependencies,
    ) == output
    resumed_manifest = json.loads((output / "manifest.json").read_text())
    assert resumed_manifest["status"] == "complete"
    assert resumed_manifest["failure_attempts"] == [receipt]


def test_real_runner_control_flow_fresh_resume_skip_and_root_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = ROOT / "configs" / "direct_multiseed.yaml"
    output = tmp_path / "synthetic-run"
    fit_calls: list[dict[str, object]] = []
    base_dependencies = _dependencies(fit_calls)
    dependencies = dm.RunnerDependencies(
        **{**base_dependencies.__dict__, "write_audits": _write_fake_audits}
    )
    _configure_synthetic_runner(monkeypatch)
    first = dm.run_direct_multiseed(
        config_path,
        profile="smoke",
        output_dir=output,
        device="cpu",
        command=("synthetic-run",),
        dependencies=dependencies,
    )
    assert first == output
    first_fit_count = len(fit_calls)
    assert first_fit_count > 0

    second = dm.run_direct_multiseed(
        config_path,
        profile="smoke",
        output_dir=output,
        resume=True,
        device="cpu",
        command=("synthetic-resume",),
        dependencies=dependencies,
    )
    assert second == output
    assert len(fit_calls) == first_fit_count
    resumed_manifest = json.loads((output / "manifest.json").read_text())
    assert resumed_manifest["resume_skipped_seeds"] == [42, 43]

    resumed_manifest["completion_state"] = "tampered-without-rehash"
    dm.atomic_write_json(output / "manifest.json", resumed_manifest)
    with pytest.raises(ValueError, match="payload hash"):
        dm.run_direct_multiseed(
            config_path,
            profile="smoke",
            output_dir=output,
            resume=True,
            device="cpu",
            command=("synthetic-tampered-resume",),
            dependencies=dependencies,
        )


def test_determinism_gates_fail_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(ValueError, match="PYTHONHASHSEED=0"):
        dm.configure_determinism("cpu", profile="smoke", injected_dependencies=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        dm.configure_determinism("cuda", profile="smoke", injected_dependencies=False)
    with pytest.raises(ValueError, match="only for smoke"):
        dm.configure_determinism("cpu", profile="formal", injected_dependencies=True)
    record = dm.configure_determinism(
        "cpu", profile="smoke", injected_dependencies=True
    )
    assert record["dependency_mode"] == "injected_smoke"
    assert record["torch_deterministic_algorithms"] is True


def test_cli_help_import_smoke() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "goai_al.direct_multiseed", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--profile {smoke,formal}" in completed.stdout
    assert "--resume" in completed.stdout
    assert "--device {cpu,cuda}" in completed.stdout
