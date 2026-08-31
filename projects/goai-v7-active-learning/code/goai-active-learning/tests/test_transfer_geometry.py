from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goai_al.data import CHEMICAL, MEDIUM, STRAIN, TEMPERATURE, TIME
from goai_al.transfer_geometry import (
    MAIN_RELATIONS,
    ProbeSettings,
    _candidate_relation_masks,
    _classical_mds,
    _distance_r2,
    _entity_geometry_metrics,
    _build_main_plan,
    _minimum_support,
    _paired_relative_transfer,
    _relation_exposures,
    _relation_summary_rows,
    _source_neutral_geometry_baseline,
)


def _metadata() -> pd.DataFrame:
    target = {STRAIN: "s0", CHEMICAL: "d0", TIME: 15, MEDIUM: "m0", TEMPERATURE: 30}
    rows = [
        target,
        {STRAIN: "s0", CHEMICAL: "d1", TIME: 30, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s1", CHEMICAL: "d0", TIME: 30, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s1", CHEMICAL: "d1", TIME: 15, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s0", CHEMICAL: "d0", TIME: 30, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s1", CHEMICAL: "d0", TIME: 15, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s0", CHEMICAL: "d1", TIME: 15, MEDIUM: "m1", TEMPERATURE: 37},
        {STRAIN: "s1", CHEMICAL: "d1", TIME: 30, MEDIUM: "m0", TEMPERATURE: 30},
    ]
    return pd.DataFrame(rows, index=[f"id-{index}" for index in range(len(rows))])


def test_relation_masks_are_explicit_and_disjoint() -> None:
    metadata = _metadata()
    masks = _candidate_relation_masks(
        metadata, metadata.loc["id-0"], blocked_ids=frozenset({"id-0"})
    )
    expected = {
        "strain_only": "id-1",
        "chemical_only": "id-2",
        "time_only": "id-3",
        "strain_chemical": "id-4",
        "chemical_time": "id-5",
        "strain_time": "id-6",
        "random_none": "id-7",
    }
    for relation, sample_id in expected.items():
        matched = metadata.index[masks[relation]].tolist()
        assert matched == [sample_id]
    total = sum(mask.astype(int) for mask in masks.values())
    assert int(total.max()) == 1


def test_registered_support_accounts_for_equal_budget_mixtures_and_anchors() -> None:
    settings = ProbeSettings(target_count=8, baseline_size=32, donor_count=6, epochs=1)
    minimums = _minimum_support(settings)
    # Six direct donors + three mixture donors + one anchor from each single pool.
    assert minimums["strain_only"] == 10
    assert minimums["chemical_only"] == 10
    assert minimums["time_only"] == 10
    # Random donors, two M/T coverage anchors, and the rest of the baseline.
    assert minimums["random_none"] == 6 + 2 + (32 - 5)


def test_pair_summary_uses_equal_budget_pair_minus_mixture() -> None:
    rows = []
    scores = {
        "random_none": 0.01,
        "strain_only": 0.02,
        "chemical_only": 0.03,
        "time_only": 0.04,
        "strain_chemical": 0.08,
        "chemical_time": 0.07,
        "strain_time": 0.06,
        "strain_chemical_mixture": 0.05,
        "chemical_time_mixture": 0.06,
        "strain_time_mixture": 0.05,
    }
    for target_index in range(3):
        for relation in MAIN_RELATIONS:
            rows.append(
                {
                    "condition_id": f"target-{target_index}",
                    "donor_relation": relation,
                    "transfer_score": scores[relation] + target_index * 0.001,
                }
            )
    summary = pd.DataFrame(_relation_summary_rows(pd.DataFrame(rows), ProbeSettings(bootstrap_draws=25)))
    excess = summary.loc[
        summary["summary_type"].eq("interaction_excess")
        & summary["relation"].eq("strain_chemical"),
        "interaction_excess",
    ].iloc[0]
    assert excess == pytest.approx(0.03)


def test_relative_transfer_pairs_each_target_to_its_random_control() -> None:
    frame = pd.DataFrame(
        {
            "condition_id": ["a", "a", "b", "b"],
            "donor_relation": ["random_none", "strain_only", "random_none", "strain_only"],
            "transfer_score": [0.1, 0.3, -0.2, -0.1],
        }
    )
    paired = _paired_relative_transfer(frame).set_index(["condition_id", "donor_relation"])
    assert paired.loc[("a", "strain_only"), "transfer_vs_random"] == pytest.approx(0.2)
    assert paired.loc[("b", "strain_only"), "transfer_vs_random"] == pytest.approx(0.1)
    assert paired.loc[("a", "random_none"), "transfer_vs_random"] == pytest.approx(0.0)


def test_classical_mds_reconstructs_a_one_dimensional_line() -> None:
    points = np.asarray([[0.0], [1.0], [3.0], [4.0]])
    distance = np.abs(points - points.T)
    embedding, _, effective, retained = _classical_mds(distance, 1)
    reconstructed = np.sqrt(
        np.square(embedding[:, None, :] - embedding[None, :, :]).sum(axis=2)
    )
    _, r2 = _distance_r2(distance, reconstructed)
    assert effective == 1
    assert retained == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_distance_r2_does_not_turn_reversed_geometry_into_a_perfect_fit() -> None:
    reference = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    )
    reversed_distance = np.asarray(
        [[0.0, 3.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    correlation, r2 = _distance_r2(reference, reversed_distance)
    assert correlation == pytest.approx(-1.0)
    assert r2 < 0.0


def test_relation_exposures_keep_mixture_joint_terms_zero() -> None:
    mixture = _relation_exposures("strain_chemical_mixture")
    pair = _relation_exposures("strain_chemical")
    assert mixture["same_strain_fraction"] == mixture["same_chemical_fraction"] == 0.5
    assert mixture["strain_chemical_joint_fraction"] == 0.0
    assert pair["strain_chemical_joint_fraction"] == 1.0


def test_entity_geometry_metrics_hold_out_one_context_per_target_entity() -> None:
    entities = ["a", "b", "c"]
    rows = []
    for target_entity_index, target_entity in enumerate(entities):
        for context_index in range(3):
            target_id = f"target-{target_entity}-{context_index}"
            for source_index, source_entity in enumerate(entities):
                rows.append(
                    {
                        "condition_id": target_id,
                        "source_entity": source_entity,
                        "target_chemical": target_entity,
                        "transfer_score": float(source_index * 0.1 + target_entity_index * 0.02 + context_index * 0.001),
                    }
                )
    metrics, matrix = _entity_geometry_metrics(
        pd.DataFrame(rows), factor="chemical", entities=entities
    )
    assert metrics["requested_dimensions"].tolist() == [1, 2, 4]
    assert metrics["training_context_count"].iloc[0] == 6
    assert metrics["validation_context_count"].iloc[0] == 3
    assert "validation_neighbor_overlap_at_3" in metrics
    assert matrix.shape == (3, 3)


def test_zero_variance_source_profiles_are_marked_not_identifiable() -> None:
    entities = ["a", "b", "c"]
    rows = []
    for target_entity in entities:
        for context_index in range(3):
            for source_entity in entities:
                rows.append(
                    {
                        "condition_id": f"target-{target_entity}-{context_index}",
                        "source_entity": source_entity,
                        "target_chemical": target_entity,
                        "transfer_score": 0.25,
                    }
                )
    metrics, _ = _entity_geometry_metrics(
        pd.DataFrame(rows), factor="chemical", entities=entities
    )
    assert metrics["status"].eq(
        "not_identifiable_train_zero_variance_source_profile"
    ).all()
    assert metrics["validation_distance_correlation"].isna().all()


def test_source_neutral_geometry_baseline_has_equal_source_exposure() -> None:
    rows = []
    for chemical in ("d0", "d1"):
        for strain in ("s0", "s1"):
            for time in (15, 30):
                for medium in ("m0", "m1"):
                    for temperature in (30, 37):
                        rows.append(
                            {
                                STRAIN: strain,
                                CHEMICAL: chemical,
                                TIME: time,
                                MEDIUM: medium,
                                TEMPERATURE: temperature,
                            }
                        )
    metadata = pd.DataFrame(rows, index=[f"id-{index}" for index in range(len(rows))])
    target_id = "id-0"
    settings = ProbeSettings(target_count=8, baseline_size=16, donor_count=2, epochs=1)
    baseline, _ = _source_neutral_geometry_baseline(
        metadata,
        metadata.loc[target_id],
        target_id=target_id,
        factor="chemical",
        source_entities=("d0", "d1"),
        blocked_ids=frozenset({target_id}),
        donor_ids=frozenset(),
        settings=settings,
        selection_seed=42,
    )
    counts = metadata.loc[list(baseline), CHEMICAL].astype(str).value_counts()
    assert len(baseline) == 16
    assert counts.to_dict() == {"d0": 8, "d1": 8}
    assert target_id not in baseline


def test_alternate_donor_plan_keeps_the_primary_baseline_fixed() -> None:
    rows = []
    for strain in ("s0", "s1", "s2", "s3"):
        for chemical in ("d0", "d1", "d2", "d3"):
            for time in (15, 30, 60, 120):
                for medium in ("m0", "m1"):
                    for temperature in (30, 37):
                        rows.append(
                            {
                                STRAIN: strain,
                                CHEMICAL: chemical,
                                TIME: time,
                                MEDIUM: medium,
                                TEMPERATURE: temperature,
                            }
                        )
    metadata = pd.DataFrame(rows, index=[f"id-{index}" for index in range(len(rows))])
    target_id = "id-0"
    settings = ProbeSettings(target_count=8, baseline_size=16, donor_count=2, epochs=1)
    primary = _build_main_plan(
        metadata,
        target_id,
        holdout_ids=frozenset({target_id}),
        settings=settings,
        selection_seed=42,
    )
    alternate = _build_main_plan(
        metadata,
        target_id,
        holdout_ids=frozenset({target_id}),
        settings=settings,
        selection_seed=104771,
        fixed_baseline_ids=primary.baseline_ids,
    )
    assert alternate.baseline_ids == primary.baseline_ids
    for donors in alternate.donor_ids.values():
        assert set(donors).isdisjoint(primary.baseline_ids)
