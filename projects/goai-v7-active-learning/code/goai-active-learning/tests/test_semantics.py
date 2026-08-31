from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import goai_al.semantics as semantics
from goai_al.data import (
    CHEMICAL,
    CONDITION_ID,
    MEDIUM,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    GroupedDataset,
)
from goai_al.semantics import (
    CombinedTargetFreeEncoder,
    TargetFreeSemanticEncoder,
    load_feature_bundle,
)


REAL_METADATA_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_metadata_train_val.csv"
)
SHUFFLED_CHEMICAL_PATH = (
    semantics.DEFAULT_CHEMICAL_EMBEDDINGS_PATH.with_name("chemberta_shuffled.tsv")
)
SHUFFLED_STRAIN_PATH = semantics.DEFAULT_STRAIN_SEMANTICS_PATH.with_name(
    "strain_semantics_shuffled.tsv"
)


def _metadata(
    chemicals: list[object],
    strains: list[object],
    *,
    media: list[object] | None = None,
    temperatures: list[object] | None = None,
    times: list[object] | None = None,
) -> pd.DataFrame:
    count = len(chemicals)
    assert len(strains) == count
    return pd.DataFrame(
        {
            STRAIN: strains,
            CHEMICAL: chemicals,
            MEDIUM: media if media is not None else ["YNB"] * count,
            TEMPERATURE: (
                temperatures if temperatures is not None else [30] * count
            ),
            TIME: times if times is not None else [15] * count,
            TIME_UNIT: ["minutes"] * count,
        }
    )


def test_frozen_assets_are_pinned_and_shuffled_or_wrong_hashes_are_rejected() -> None:
    encoder = TargetFreeSemanticEncoder()

    assert semantics.CHEMBERTA_MODEL_REVISION == (
        "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
    )
    assert encoder.asset_hashes == {
        "chemical_embeddings_sha256": (
            "50ba9081a7773f1438a3dfbf49054e319ce6002d44ed95d0ffd623d5f617b0a6"
        ),
        "chemical_manifest_sha256": (
            "cb64148b66d58eff94881bfc241e85725d6a6bfe5dcb41713484b5d48e608f96"
        ),
        "strain_semantics_sha256": (
            "8ee0f31d33c8dccefa90eed33cb7f4eb949470ef6dfe0681beefecf52b7d72ce"
        ),
        "strain_manifest_sha256": (
            "ba54ed7ccddd6d34663978b698ac7da39f13bd41e84937be8edbd31fbf795da2"
        ),
        "strain_identity_evidence_registry_sha256": (
            "9a3b4edf61a313d763b91ee762c4ebed2ea359c9b4fc810eaa135c30207c1add"
        ),
        "strain_identity_evidence_manifest_sha256": (
            "a2c94f7f626fd4a1b0167de6ff063b5a845649930d4a9ce21370c6810db7b1cd"
        ),
        "chemical_risk_manifest_sha256": (
            "13060a305daddd3d35f7bc70244c4c6fec793d6b13cf5645eaa0ab62f079475c"
        ),
    }
    assert encoder.asset_summary["chemical"]["model_revision"] == (
        semantics.CHEMBERTA_MODEL_REVISION
    )
    assert encoder.asset_summary["chemical"]["view"] == "real_exact"
    assert encoder.asset_summary["chemical"]["shuffled"] is False
    strain_summary = encoder.asset_summary["strain"]
    assert strain_summary["identity_warning"] == (
        "Five mappings are high-confidence public candidates, not organizer-verified "
        "identities; DHY210 is missing, never S288C"
    )
    assert strain_summary["organizer_verified"] is False
    assert strain_summary["candidate_strains"] == ["BAH", "BAI", "CEK", "CGD", "CRD"]
    assert strain_summary["unresolved_strains"] == ["DHY210"]
    assert strain_summary["identity_candidate_count"] == 5
    assert strain_summary["identity_evidence_registry_sha256"] == (
        semantics.STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256
    )
    assert strain_summary["identity_evidence_manifest_sha256"] == (
        semantics.STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256
    )

    with pytest.raises(ValueError, match="chemical real embeddings SHA256 mismatch"):
        TargetFreeSemanticEncoder(chemical_embeddings_path=SHUFFLED_CHEMICAL_PATH)
    with pytest.raises(ValueError, match="strain real semantics SHA256 mismatch"):
        TargetFreeSemanticEncoder(strain_semantics_path=SHUFFLED_STRAIN_PATH)
    with pytest.raises(ValueError, match="chemical embedding manifest SHA256 mismatch"):
        TargetFreeSemanticEncoder(
            chemical_manifest_path=semantics.DEFAULT_CHEMICAL_RISK_MANIFEST_PATH
        )


def test_fit_apis_have_no_response_argument_and_reject_response_values() -> None:
    expected_fit_parameters = ("self", "metadata")
    assert tuple(inspect.signature(TargetFreeSemanticEncoder.fit).parameters) == (
        expected_fit_parameters
    )
    assert tuple(inspect.signature(CombinedTargetFreeEncoder.fit).parameters) == (
        expected_fit_parameters
    )
    assert tuple(inspect.signature(load_feature_bundle).parameters) == (
        "dataset",
        "encoder",
    )

    metadata = _metadata(
        ["Amphotericin B", "Anisomycin"],
        ["BAH", "BAI"],
    )
    response = np.ones((len(metadata), 3), dtype=np.float32)
    with pytest.raises(TypeError):
        TargetFreeSemanticEncoder().fit(metadata, response)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CombinedTargetFreeEncoder().fit(  # type: ignore[call-arg]
            metadata, response=response
        )


def test_scalers_use_unique_entities_not_duplicated_measurement_rows() -> None:
    unique = _metadata(
        ["Amphotericin B", "Anisomycin"],
        ["BAH", "BAI"],
    )
    duplicated = pd.concat(
        [unique.iloc[[0]]] * 23 + [unique.iloc[[1]]] * 2,
        ignore_index=True,
    )

    unique_encoder = TargetFreeSemanticEncoder().fit(unique)
    duplicated_encoder = TargetFreeSemanticEncoder().fit(duplicated)

    np.testing.assert_array_equal(
        unique_encoder.chemical_mean_, duplicated_encoder.chemical_mean_
    )
    np.testing.assert_array_equal(
        unique_encoder.chemical_scale_, duplicated_encoder.chemical_scale_
    )
    np.testing.assert_array_equal(
        unique_encoder.strain_mean_, duplicated_encoder.strain_mean_
    )
    np.testing.assert_array_equal(
        unique_encoder.strain_scale_, duplicated_encoder.strain_scale_
    )
    np.testing.assert_array_equal(
        unique_encoder.transform(unique), duplicated_encoder.transform(unique)
    )
    assert duplicated_encoder.coverage_summary["measurement_frequency_weighted"] is False
    assert duplicated_encoder.coverage_summary["chemical_scaler_entity_count"] == 2
    assert duplicated_encoder.coverage_summary["strain_scaler_entity_count"] == 2
    assert duplicated_encoder.coverage_summary["response_used"] is False


def test_missing_and_unresolved_entities_are_zero_with_explicit_flags() -> None:
    encoder = TargetFreeSemanticEncoder().fit(
        _metadata(
            ["Amphotericin B", "Anisomycin"],
            ["BAH", "BAI"],
        )
    )
    query = _metadata(
        ["Quality Control", "not-a-real-chemical", "Oligomycin"],
        ["DHY210", "not-a-real-strain", "BAH"],
    )
    values = encoder.transform(query)
    blocks = encoder.block_slices

    np.testing.assert_array_equal(
        values[:2, blocks["chemical_continuous"]],
        np.zeros((2, 384), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        values[:2, blocks["strain_continuous"]],
        np.zeros((2, 53), dtype=np.float32),
    )
    # available, missing, resolved, proxy, fallback, identity-risk
    np.testing.assert_array_equal(
        values[:, blocks["chemical_flags"]],
        np.asarray(
            [
                [1, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    )
    # available, missing, resolved, proxy, fallback, identity-candidate
    np.testing.assert_array_equal(
        values[:, blocks["strain_flags"]],
        np.asarray(
            [
                [1, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 1],
            ],
            dtype=np.float32,
        ),
    )
    assert np.count_nonzero(values[2, blocks["chemical_continuous"]]) > 0


def test_candidate_strain_semantics_stay_nonzero_without_implying_verification() -> None:
    strains = ["BAH", "BAI", "CEK", "CGD", "CRD", "DHY210"]
    metadata = _metadata(["Amphotericin B"] * len(strains), strains)
    encoder = TargetFreeSemanticEncoder().fit(metadata)
    values = encoder.transform(metadata)
    blocks = encoder.block_slices

    assert encoder.output_dim == 449
    assert encoder.feature_names[-1] == "strain_flag__identity_candidate"
    for row in range(5):
        assert np.count_nonzero(values[row, blocks["strain_continuous"]]) > 0
    np.testing.assert_array_equal(
        values[5, blocks["strain_continuous"]],
        np.zeros(53, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        values[:, blocks["strain_flags"]][:, -1],
        np.asarray([1, 1, 1, 1, 1, 0], dtype=np.float32),
    )
    coverage = encoder.coverage_summary["strain"]
    assert coverage["identity_candidate_unique_entities"] == 5
    assert coverage["organizer_verified"] is False
    assert coverage["identity_warning"] == semantics.STRAIN_IDENTITY_WARNING


def test_ood_identity_can_be_zero_while_real_semantics_remain_nonzero() -> None:
    pool = _metadata(
        ["Amphotericin B", "Amphotericin B"],
        ["BAH", "BAH"],
        times=[15, 60],
    )
    encoder = CombinedTargetFreeEncoder().fit(pool)
    ood = _metadata(
        ["Anisomycin"],
        ["BAI"],
        media=["unseen-medium"],
        temperatures=[37],
        times=[30],
    )
    values = encoder.transform(ood)
    blocks = encoder.block_slices

    assert np.count_nonzero(values[0, blocks["identity_categorical"]]) == 0
    assert np.count_nonzero(values[0, blocks["chemical_continuous"]]) > 0
    assert np.count_nonzero(values[0, blocks["strain_continuous"]]) > 0
    assert np.isfinite(values).all()


def test_masking_changes_only_base_categorical_one_hot_columns() -> None:
    metadata = _metadata(
        ["Amphotericin B", "Anisomycin"],
        ["BAH", "BAI"],
        media=["medium-a", "medium-b"],
        temperatures=[30, 37],
        times=[15, 60],
    )
    encoder = CombinedTargetFreeEncoder().fit(metadata)
    values = encoder.transform(metadata)
    masked = encoder.mask_unsupported(values, values[[0]])

    for column_slice in encoder.categorical_column_slices.values():
        assert masked[1, column_slice].sum() == 0.0
    np.testing.assert_array_equal(
        masked[:, encoder.block_slices["time_continuous"]],
        values[:, encoder.block_slices["time_continuous"]],
    )
    np.testing.assert_array_equal(
        masked[:, encoder.block_slices["semantics"]],
        values[:, encoder.block_slices["semantics"]],
    )


class _ResponseBomb:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"load_feature_bundle accessed response attribute {name!r}")

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("load_feature_bundle converted response to an array")


def test_feature_bundle_is_exactly_row_aligned_and_does_not_access_response() -> None:
    row_index = pd.Index(["row-c", "row-a", "row-b"], name=CONDITION_ID)
    metadata = _metadata(
        ["Oligomycin", "Amphotericin B", "Anisomycin"],
        ["BAI", "BAH", "CEK"],
        times=[60, 15, 30],
    )
    metadata.index = row_index
    pool_ids = pd.Index(["row-b", "row-a"], name=CONDITION_ID)
    dataset = GroupedDataset(
        metadata=metadata,
        response=_ResponseBomb(),  # type: ignore[arg-type]
        proteins=(),
        train_ids=pool_ids,
        validation_ids={},
        protein_missing_rate=pd.Series(dtype=np.float64),
    )

    bundle = load_feature_bundle(dataset)

    assert {
        "row_ids",
        "descriptor_matrix",
        "model_matrix",
        "masker",
        "summary",
        "asset_hashes",
    } <= set(bundle)
    assert bundle["row_ids"] is dataset.metadata.index
    descriptors = bundle["descriptor_matrix"]
    model_features = bundle["model_matrix"]
    assert isinstance(descriptors, np.ndarray)
    assert isinstance(model_features, np.ndarray)
    assert descriptors.shape == model_features.shape
    assert descriptors.shape == (3, 456)
    assert descriptors.shape[0] == len(row_index)
    np.testing.assert_array_equal(descriptors, model_features)
    assert not np.shares_memory(descriptors, model_features)
    assert callable(bundle["masker"])
    json.dumps(bundle["summary"], allow_nan=False)
    json.dumps(bundle["asset_hashes"], allow_nan=False)
    assert bundle["summary"]["response_used"] is False
    assert bundle["summary"]["row_count"] == 3
    assert bundle["summary"]["candidate_pool_count"] == 2


def test_real_released_metadata_has_expected_frozen_asset_coverage() -> None:
    metadata = pd.read_csv(
        REAL_METADATA_PATH,
        usecols=[CHEMICAL, STRAIN],
    ).drop_duplicates(ignore_index=True)
    encoder = TargetFreeSemanticEncoder().fit(metadata)
    coverage = encoder.coverage_summary

    assert coverage["fit_scope"] == "unique_candidate_pool_entities"
    assert coverage["response_used"] is False
    assert coverage["chemical_scaler_entity_count"] == 45
    assert coverage["strain_scaler_entity_count"] == 4
    assert coverage["chemical"]["unique_entity_count"] == 46
    assert coverage["chemical"]["available_unique_entities"] == 46
    assert coverage["chemical"]["resolved_unique_entities"] == 45
    assert coverage["chemical"]["missing_unique_entities"] == 1
    assert coverage["strain"]["unique_entity_count"] == 5
    assert coverage["strain"]["available_unique_entities"] == 5
    assert coverage["strain"]["resolved_unique_entities"] == 4
    assert coverage["strain"]["missing_unique_entities"] == 1
    assert coverage["strain"]["identity_candidate_unique_entities"] == 4
    assert coverage["strain"]["organizer_verified"] is False
    assert coverage["strain"]["identity_warning"] == semantics.STRAIN_IDENTITY_WARNING
    assert coverage["strain"]["identity_evidence_registry_sha256"] == (
        semantics.STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256
    )
    assert coverage["strain"]["identity_evidence_manifest_sha256"] == (
        semantics.STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256
    )
    values = encoder.transform(metadata)
    assert values.shape == (len(metadata), encoder.output_dim)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    json.dumps(coverage, allow_nan=False)
