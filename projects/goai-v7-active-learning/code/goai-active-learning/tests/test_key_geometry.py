from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goai_al.data import CHEMICAL, MEDIUM, STRAIN, TEMPERATURE, TIME
from goai_al.key_geometry import (
    KeyGeometrySettings,
    _classification,
    _entry_fold,
    _fit_categorical_relation_kernel,
    _fit_directed_svd_completion,
    _fit_two_way_baseline,
    _metric_values,
    _mode_group_r2,
    _score_conditions,
    _summarize_oof,
    build_panel_plan,
)


def _metadata() -> pd.DataFrame:
    rows = []
    for replicate in range(2):
        for strain in ("s0", "s1"):
            for chemical in ("d0", "d1", "d2"):
                for time in ("15", "60"):
                    for medium in ("m0", "m1"):
                        for temperature in ("30", "37"):
                            rows.append(
                                {
                                    STRAIN: strain,
                                    CHEMICAL: chemical,
                                    TIME: time,
                                    MEDIUM: medium,
                                    TEMPERATURE: temperature,
                                    "replicate": replicate,
                                }
                            )
    return pd.DataFrame(rows, index=[f"id-{index}" for index in range(len(rows))])


def test_panel_roles_are_disjoint_and_baseline_covers_metadata_levels() -> None:
    metadata = _metadata()
    plan = build_panel_plan(
        metadata,
        KeyGeometrySettings(
            target_count=8,
            donor_count=12,
            baseline_size=32,
            epochs=1,
            structure_epochs=1,
            device="cpu",
        ),
    )
    assert not (set(plan.targets) & set(plan.donors))
    assert not (set(plan.targets) & set(plan.baseline))
    assert not (set(plan.donors) & set(plan.baseline))
    for field in (STRAIN, CHEMICAL, TIME, MEDIUM, TEMPERATURE):
        assert set(metadata.loc[list(plan.baseline), field]) == set(metadata[field])


def test_score_conditions_uses_zero_normalized_mse() -> None:
    truth = np.asarray([[1.0, -1.0], [2.0, 0.0]], dtype=float)
    prediction = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    score = _score_conditions(prediction, truth)
    assert score["mse"].tolist() == pytest.approx([1.0, 0.5])
    assert score["zero_mse"].tolist() == pytest.approx([1.0, 2.0])
    assert score["normalized_loss"].tolist() == pytest.approx([1.0, 0.25])
    assert score["delta_skill_zero"].tolist() == pytest.approx([0.0, 0.75])


def test_entry_fold_is_deterministic_and_relation_stratified_input_matters() -> None:
    settings = KeyGeometrySettings(device="cpu", structure_epochs=1)
    first = _entry_fold("d", "t", "101", settings)
    second = _entry_fold("d", "t", "101", settings)
    assert first == second
    assert 0 <= first < settings.folds


def test_heldout_protocol_is_response_independent_relation_conditioned() -> None:
    prediction = pd.DataFrame(
        {
            "observed_transfer": [0.1, 0.2, 0.3, 0.4],
            "predicted_transfer": [0.1, 0.2, 0.3, 0.4],
            "two_way_prediction": [0.0, 0.0, 0.0, 0.0],
            "target_condition_id": ["t0", "t0", "t1", "t1"],
            "donor_condition_id": ["d0", "d1", "d0", "d1"],
        }
    )
    summary = _summarize_oof(
        prediction,
        label="test",
        model_kind="test",
        rank=1,
        settings=KeyGeometrySettings(device="cpu", structure_epochs=1, bootstrap_draws=5),
    )
    assert summary["heldout_protocol"] == (
        "five_fold_response_independent_relation_pattern_conditioned_heldout_entries"
    )


def test_two_way_baseline_recovers_an_additive_matrix() -> None:
    donor_index = np.repeat(np.arange(3), 4)
    target_index = np.tile(np.arange(4), 3)
    value = 1.5 + np.repeat(np.asarray([0.2, -0.4, 0.7]), 4) + np.tile(
        np.asarray([-0.3, 0.0, 0.4, 0.9]), 3
    )
    prediction = _fit_two_way_baseline(
        donor_index,
        target_index,
        value,
        np.ones(len(value), dtype=bool),
        donor_count=3,
        target_count=4,
    )
    assert prediction == pytest.approx(value)
    metric = _metric_values(value, prediction, prediction + 0.1)
    assert metric["heldout_entry_rmse"] == pytest.approx(0.0)
    assert metric["heldout_entry_r2_vs_two_way"] == pytest.approx(1.0)


def test_group_variance_r2_detects_complete_group_separation() -> None:
    assert _mode_group_r2(np.asarray([-1.0, -1.0, 1.0, 1.0]), ["a", "a", "b", "b"]) == pytest.approx(1.0)


def test_classification_prefers_nonfactorized_only_with_clear_general_gain() -> None:
    rank = pd.DataFrame(
        [
            {
                "rank": 8,
                "status": "completed",
                "heldout_entry_r2_vs_two_way": 0.15,
            }
        ]
    )
    kernel = pd.DataFrame(
        [
            {
                "model": "Factor-only",
                "heldout_entry_r2_vs_two_way": 0.03,
                "target_bootstrap_r2_ci95_low": -0.01,
            },
            {
                "model": "Factor + interaction",
                "heldout_entry_r2_vs_two_way": 0.05,
                "target_bootstrap_r2_ci95_low": -0.01,
            },
            {
                "model": "General low-rank (r=8)",
                "heldout_entry_r2_vs_two_way": 0.15,
                "target_bootstrap_r2_ci95_low": 0.04,
            },
        ]
    )
    case, recommendation, _ = _classification(rank, kernel, KeyGeometrySettings(device="cpu", structure_epochs=1))
    assert case == "Low-dimensional but non-factorized"
    assert recommendation == "General learned information Key"


def test_masked_svd_recovers_a_small_rank_one_directed_matrix() -> None:
    donor_index = np.repeat(np.arange(4), 3)
    target_index = np.tile(np.arange(3), 4)
    value = (
        0.1
        + np.repeat(np.asarray([0.2, -0.1, 0.3, -0.2]), 3)
        + np.tile(np.asarray([-0.05, 0.15, 0.0]), 4)
        + np.repeat(np.asarray([1.0, 0.5, -0.4, 0.8]), 3)
        * np.tile(np.asarray([0.2, -0.6, 0.4]), 4)
    )
    prediction = _fit_directed_svd_completion(
        rank=1,
        donor_index=donor_index,
        target_index=target_index,
        value=value,
        train_mask=np.ones(len(value), dtype=bool),
        donor_count=4,
        target_count=3,
        settings=KeyGeometrySettings(structure_epochs=200, device="cpu"),
    )
    assert np.sqrt(np.mean(np.square(prediction - value))) < 0.02


def test_categorical_factor_kernel_recovers_a_directed_factor_relation() -> None:
    donor_index = np.repeat(np.arange(4), 4)
    target_index = np.tile(np.arange(4), 4)
    donor_level = np.repeat(np.asarray([0, 0, 1, 1]), 4)
    target_level = np.tile(np.asarray([0, 1, 0, 1]), 4)
    relation = np.asarray([[0.3, -0.2], [0.1, 0.5]])
    value = relation[donor_level, target_level]
    prediction = _fit_categorical_relation_kernel(
        donor_index=donor_index,
        target_index=target_index,
        donor_codes=[donor_level],
        target_codes=[target_level],
        factor_level_counts=[2],
        value=value,
        train_mask=np.ones(len(value), dtype=bool),
        donor_count=4,
        target_count=4,
        include_interactions=False,
        settings=KeyGeometrySettings(structure_epochs=200, device="cpu"),
    )
    assert np.sqrt(np.mean(np.square(prediction - value))) < 0.05
