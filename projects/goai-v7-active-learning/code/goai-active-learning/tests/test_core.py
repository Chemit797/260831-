from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from goai_al.acquisition import Acquisition, AcquisitionContext, select_batch
from goai_al.audit import build_data_audit
from goai_al.data import (
    CHEMICAL,
    CONDITION_ID,
    DATA_SOURCE,
    GROUP_FIELDS,
    INSTRUMENT,
    MEDIUM,
    PLATE,
    SAMPLE_ID,
    SPLIT,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    PoolFeatureEncoder,
    build_benchmark_split,
    load_grouped_dataset,
    stable_condition_id,
)
from goai_al.experiment import _deterministic_initial_ids, _model_seed
from goai_al.metrics import (
    budget_to_target,
    normalized_trapezoidal_aulc,
    score_response,
)
from goai_al.model import (
    ModelSettings,
    _mean_latent_gram_variance,
    _natural_delta_latent_gram,
    fit_response_model,
)
from goai_al.simulator import (
    BudgetSchedule,
    PoolState,
    RetrospectiveOracle,
    RoundReceipt,
)


def _condition(index: int) -> dict[str, object]:
    return {
        STRAIN: f"s{index % 2}",
        CHEMICAL: f"c{(index // 2) % 3}",
        MEDIUM: f"m{(index // 6) % 2}",
        TEMPERATURE: 30 + 7 * ((index // 12) % 2),
        TIME: 15 + 15 * (index % 3),
        TIME_UNIT: "minutes",
    }


def test_condition_id_uses_all_biological_fields_but_not_split() -> None:
    values = _condition(0)
    first = stable_condition_id(values)
    with_split = {**values, "split_final": "val_time"}
    assert stable_condition_id(with_split) == first
    changed = {**values, TIME_UNIT: "hours"}
    assert stable_condition_id(changed) != first
    assert first.startswith("condition__")


def test_benchmark_split_is_condition_atomic_and_disjoint() -> None:
    records: list[dict[str, object]] = []
    ids: list[str] = []
    for index in range(18):
        values = _condition(index)
        ids.append(stable_condition_id(values))
        records.append(
            {
                **values,
                "split_provenance": "train|val_time" if index == 0 else "train",
            }
        )
    validation = {
        STRAIN: "s9",
        CHEMICAL: "c9",
        MEDIUM: "m9",
        TEMPERATURE: 25,
        TIME: 90,
        TIME_UNIT: "minutes",
    }
    validation_id = stable_condition_id(validation)
    ids.append(validation_id)
    records.append({**validation, "split_provenance": "val_time"})
    metadata = pd.DataFrame(records, index=pd.Index(ids, name="condition_id"))

    split = build_benchmark_split(metadata, holdout_fraction=0.20, seed=42)
    pool = set(split.candidate_pool_ids)
    interpolation = set(split.interpolation_ids)
    assert pool.isdisjoint(interpolation)
    assert pool.isdisjoint(split.validation_ids["val_time"])
    assert interpolation.isdisjoint(split.validation_ids["val_time"])
    assert split.validation_ids["val_time"] == (validation_id,)
    assert stable_condition_id(_condition(0)) in split.removed_validation_overlap["val_time"]


def test_encoder_masks_unsupported_categories_but_never_continuous_time() -> None:
    pool = pd.DataFrame(
        {
            STRAIN: ["s1", "s2"],
            CHEMICAL: ["c1", "c2"],
            MEDIUM: ["m", "m"],
            TEMPERATURE: [30, 37],
            TIME: [30, 1],
            TIME_UNIT: ["minutes", "hours"],
        }
    )
    encoder = PoolFeatureEncoder().fit(pool)
    assert encoder.categorical_fields == (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE)
    assert TIME_UNIT not in encoder.categorical_column_slices
    features = encoder.transform(pool)
    masked = encoder.mask_unsupported(features, features[[0]])
    chemical_slice = encoder.categorical_column_slices[CHEMICAL]
    assert masked[1, chemical_slice].sum() == 0.0
    continuous = encoder.continuous_column_slice
    assert np.array_equal(masked[:, continuous], features[:, continuous])
    assert features[0, continuous][0] < features[1, continuous][0]


def test_encoder_rejects_unknown_time_units() -> None:
    metadata = pd.DataFrame([_condition(0)])
    metadata.loc[0, TIME_UNIT] = "fortnight"
    with pytest.raises(ValueError, match="Unknown pert_time_unit"):
        PoolFeatureEncoder().fit(metadata)


def test_grouped_train_response_discards_overlapping_validation_labels(tmp_path) -> None:
    context = {
        STRAIN: "strain-a",
        MEDIUM: "medium-a",
        TEMPERATURE: 30,
        TIME: 15,
        TIME_UNIT: "min",
        DATA_SOURCE: "source-a",
        INSTRUMENT: "instrument-a",
        PLATE: "plate-a",
    }
    metadata = pd.DataFrame(
        [
            {
                SAMPLE_ID: "control-val",
                SPLIT: "val_time",
                CHEMICAL: "Water",
                **context,
            },
            {
                SAMPLE_ID: "treat-train",
                SPLIT: "train",
                CHEMICAL: "drug-a",
                **context,
            },
            {
                SAMPLE_ID: "treat-val",
                SPLIT: "val_time",
                CHEMICAL: "drug-a",
                **context,
            },
        ]
    )
    proteome = pd.DataFrame(
        {
            SAMPLE_ID: ["control-val", "treat-train", "treat-val"],
            "protein-a": [2.0, 4.0, 16.0],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    proteome_path = tmp_path / "proteome.csv"
    metadata.to_csv(metadata_path, index=False)
    proteome.to_csv(proteome_path, index=False)

    dataset = load_grouped_dataset(
        metadata_path,
        proteome_path,
        interpolation_fraction=0.0,
    )
    condition_id = stable_condition_id({CHEMICAL: "drug-a", **context})
    grouped = dataset.metadata.loc[condition_id]
    assert dataset.response.loc[condition_id, "protein-a"] == pytest.approx(1.0)
    assert grouped["split_provenance"] == "train|val_time"
    assert grouped["replicate_count"] == 1
    assert grouped["released_replicate_count"] == 2
    assert grouped["discarded_overlap_measurement_count"] == 1
    assert grouped["measurement_context_count"] == 1
    assert grouped["source_count"] == 1
    assert grouped["instrument_count"] == 1
    assert grouped["plate_count"] == 1
    assert condition_id in dataset.removed_validation_overlap["val_time"]
    assert condition_id not in dataset.validation_ids["val_time"]

    train_control = dataset.control_provenance.loc["treat-train"]
    assert train_control["treatment_split"] == "train"
    assert json.loads(train_control["matched_control_splits"]) == ["val_time"]
    assert train_control["same_split_control_count"] == 0
    assert bool(train_control["cross_split_only"])
    audit = build_data_audit(dataset)
    assert audit["controls"][
        "cross_split_only_treatment_measurements_by_treatment_split"
    ]["train"] == 1
    assert audit["splits"]["official_train_aggregation_treatment_rows"] == 1
    assert audit["splits"]["official_train_released_treatment_rows"] == 2


def test_oracle_denies_nonpool_and_repeat_reveals_atomically() -> None:
    oracle = RetrospectiveOracle(
        ("p0", "p1"),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    first = oracle.reveal(("p0",))
    assert np.array_equal(first.labels, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="limited to candidate"):
        oracle.reveal(("evaluation-id",))
    with pytest.raises(ValueError, match="already been revealed"):
        oracle.reveal(("p1", "p0"))
    assert oracle.revealed_ids == ("p0",)


def test_acquisition_boundary_is_hidden_label_free_and_label_permutation_invariant() -> None:
    ids = ("a", "b", "c", "d")
    descriptors = np.eye(4, dtype=np.float32)
    context = AcquisitionContext(
        candidate_ids=("b", "c", "d"),
        labelled_ids=("a",),
        descriptor_ids=ids,
        descriptors=descriptors,
    )
    assert not hasattr(context, "response")
    assert not hasattr(context, "labels")
    selected = select_batch("coreset", context, 2, seed=42)

    labels = np.arange(8, dtype=np.float32).reshape(4, 2)
    permuted = np.asarray([2, 0, 3, 1])
    oracle = RetrospectiveOracle(
        ids,
        labels[permuted],
        response_ids=tuple(ids[index] for index in permuted),
    )
    assert np.array_equal(oracle.reveal(("a",)).labels, labels[[0]])
    assert select_batch("coreset", context, 2, seed=42) == selected


def test_acquisition_protocol_exposes_only_label_free_context() -> None:
    class FirstCandidate:
        def select_batch(
            self,
            context: AcquisitionContext,
            batch_size: int,
            *,
            seed: int = 0,
        ) -> tuple[str, ...]:
            del seed
            return context.candidate_ids[:batch_size]

    policy = FirstCandidate()
    assert isinstance(policy, Acquisition)
    context = AcquisitionContext(
        candidate_ids=("a", "b"),
        descriptor_ids=("a", "b"),
        descriptors=np.eye(2, dtype=np.float32),
    )
    assert policy.select_batch(context, 1, seed=42) == ("a",)


def test_budget_schedule_conserves_exact_batches_and_rejects_reselection() -> None:
    ids = tuple(f"p{index}" for index in range(8))
    state = PoolState(ids, np.eye(8, dtype=np.float32)).select(("p0", "p1"))
    schedule = BudgetSchedule(2, 2, (2, 4, 8), pool_size=8)
    advanced = schedule.advance(state, ("p2", "p3"))
    assert advanced.budget == 4
    with pytest.raises(ValueError, match="exactly one fixed batch"):
        schedule.advance(advanced, ("p4",))
    with pytest.raises(ValueError, match="reselects"):
        advanced.select(("p3",))

    formal = BudgetSchedule(128, 128, (128, 256, 512, 1024), pool_size=2670)
    assert formal.acquisition_budgets == (256, 384, 512, 640, 768, 896, 1024)
    assert formal.is_checkpoint(512)
    assert not formal.is_checkpoint(384)


def test_initial_set_and_model_seed_are_deterministic_and_strategy_independent() -> None:
    ids = tuple(f"condition-{index:03d}" for index in range(20))
    first = _deterministic_initial_ids(ids, 6, seed=42)
    assert first == _deterministic_initial_ids(tuple(reversed(ids)), 6, seed=42)
    assert len(set(first)) == 6
    assert _model_seed(42, 128) == _model_seed(42, 128)
    assert _model_seed(42, 128) != _model_seed(42, 256)


def test_score_response_matches_hand_calculations_and_excludes_constant_proteins() -> None:
    truth = np.asarray([[1.0, -1.0], [2.0, 2.0]])
    prediction = np.zeros_like(truth)
    metrics = score_response(prediction, truth)
    assert metrics["delta_rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["delta_mae"] == pytest.approx(1.5)
    assert metrics["delta_skill_zero"] == pytest.approx(0.0)
    assert metrics["protein_r2_median"] == pytest.approx((-9.0 - 1.0 / 9.0) / 2.0)

    constant_truth = np.asarray([[1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    constant_prediction = np.asarray([[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]])
    constant = score_response(constant_prediction, constant_truth)
    assert constant["n_evaluable_proteins_r2"] == 1
    assert constant["protein_r2_median"] == pytest.approx(1.0)


def test_score_response_fails_on_nonfinite_observed_prediction() -> None:
    truth = np.asarray([[1.0, np.nan]])
    with pytest.raises(ValueError, match="nonfinite"):
        score_response(np.asarray([[np.nan, 0.0]]), truth)
    allowed = score_response(np.asarray([[1.0, np.nan]]), truth)
    assert allowed["n_observed_values"] == 1


def test_aulc_uses_real_budget_spacing_and_target_reports_not_reached() -> None:
    budgets = np.asarray([1.0, 2.0, 10.0])
    values = np.asarray([0.0, 2.0, 4.0])
    assert normalized_trapezoidal_aulc(budgets, values) == pytest.approx(25.0 / 9.0)
    target = budget_to_target(budgets, values, full_reference=10.0, target_fraction=0.8)
    assert target["status"] == "not_reached"
    assert target["not_reached"] is True
    assert target["budget"] is None


def test_normalized_aulc_requires_two_strictly_increasing_budgets() -> None:
    with pytest.raises(ValueError, match="at least two strictly increasing"):
        normalized_trapezoidal_aulc(np.asarray([128.0]), np.asarray([0.5]))
    with pytest.raises(ValueError, match="strictly increasing"):
        normalized_trapezoidal_aulc(
            np.asarray([128.0, 128.0]), np.asarray([0.5, 0.6])
        )


def test_low_rank_latent_gram_uncertainty_matches_full_reconstruction() -> None:
    latent_draws = torch.tensor(
        [
            [[0.0, 1.0], [2.0, -1.0]],
            [[1.0, 3.0], [1.0, 0.0]],
            [[-2.0, 2.0], [4.0, 1.0]],
            [[3.0, -1.0], [0.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    basis = torch.tensor(
        [[1.0, 0.5, -0.25], [-0.5, 2.0, 1.5]], dtype=torch.float64
    )
    scale = torch.tensor([0.5, 2.0, 1.25], dtype=torch.float64)
    gram = _natural_delta_latent_gram(basis, scale)
    latent_score = _mean_latent_gram_variance(latent_draws, gram)
    reconstructed = torch.einsum("dbr,rp->dbp", latent_draws, basis) * scale
    full_score = reconstructed.var(dim=0, unbiased=True).mean(dim=1)
    torch.testing.assert_close(latent_score, full_score, rtol=1e-12, atol=1e-12)


def test_low_rank_basis_and_target_statistics_are_fit_local() -> None:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    revealed = np.asarray(
        [[1.0, 2.0, np.nan], [3.0, 4.0, 8.0], [8.0, 9.0, 10.0]],
        dtype=np.float32,
    )
    settings = ModelSettings(
        hidden_dim=4,
        dropout=0.0,
        learning_rate=0.001,
        weight_decay=0.0,
        epochs=1,
        batch_size=3,
        target_scale_floor=0.05,
        device="cpu",
        kind="low_rank",
        response_rank=2,
        svd_niter=1,
    )
    first = fit_response_model(features, revealed, settings, seed=42)
    second = fit_response_model(features, revealed.copy(), settings, seed=42)
    assert first.basis_hash == second.basis_hash
    assert first.response_rank == 2
    assert np.allclose(first.target_mean, np.asarray([4.0, 5.0, 9.0], dtype=np.float32))
    changed = revealed.copy()
    changed[1, 0] = 30.0
    third = fit_response_model(features, changed, settings, seed=42)
    assert third.basis_hash != first.basis_hash


def test_round_receipt_is_atomic_and_contains_no_hidden_values(tmp_path) -> None:
    ids = ("a", "b", "c")
    before = PoolState(ids, np.eye(3, dtype=np.float32)).select(("a",))
    after = before.select(("b",))
    receipt = RoundReceipt.from_transition(1, "random", before, after, ("b",))
    destination = tmp_path / "round.json"
    receipt.write_json_atomic(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["budget_before"] == 1
    assert payload["budget_after"] == 2
    assert payload["labelled_ids"] == ["a", "b"]
    assert len(payload["labelled_ids_sha256"]) == 64
    assert "labels" not in payload
    assert "response" not in payload
    assert not list(tmp_path.glob(".round.json.*.tmp"))


def test_round_receipt_validates_provenance_and_sanitizes_nested_metrics(
    tmp_path,
) -> None:
    ids = ("a", "b", "c")
    before = PoolState(ids, np.eye(3, dtype=np.float32)).select(("a",))
    after = before.select(("b",))
    receipt = RoundReceipt.from_transition(
        1,
        "random",
        before,
        after,
        ("b",),
        global_seed=42,
        acquisition_seed=123,
        model_seed=456,
        checkpoint=True,
        labelled_ids=after.selected_ids,
        model_fit_summary={"seed": 456, "final_loss": 1.0},
        split_metrics=(
            {"split": "validation", "nested": {"values": [1.0, np.nan, np.inf]}},
        ),
        train_seconds=0.0,
    )
    destination = tmp_path / "strict.json"
    receipt.write_json_atomic(destination)

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    payload = json.loads(
        destination.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
    )
    assert payload["split_metrics"][0]["nested"]["values"] == [1.0, None, None]

    with pytest.raises(ValueError, match="transition after state"):
        RoundReceipt.from_transition(
            1, "random", before, after, ("b",), labelled_ids=("a", "c")
        )
    with pytest.raises(ValueError, match="global_seed"):
        RoundReceipt.from_transition(
            1, "random", before, after, ("b",), global_seed=-1
        )
    with pytest.raises(ValueError, match="train_seconds"):
        RoundReceipt.from_transition(
            1, "random", before, after, ("b",), train_seconds=float("nan")
        )
    with pytest.raises(ValueError, match="forbidden public metadata key"):
        RoundReceipt.from_transition(
            1,
            "random",
            before,
            after,
            ("b",),
            model_fit_summary={"labels": [1.0]},
        )
