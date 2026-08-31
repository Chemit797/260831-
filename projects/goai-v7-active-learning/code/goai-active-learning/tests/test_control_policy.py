from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from goai_al.audit import build_data_audit, write_audit_outputs
from goai_al.data import (
    CACHE_VERSION,
    CHEMICAL,
    CONDITION_ID,
    DATA_SOURCE,
    DEFAULT_CONTROL_POLICY,
    INSTRUMENT,
    MATCH_CONTROL_FIELDS,
    MEDIUM,
    PLATE,
    POOLED_EXACT_CONTEXT_WATER_DMSO,
    PROTOCOL_VERSION,
    SAMPLE_ID,
    SPLIT,
    STRAIN,
    STRICT_EXPLICIT_VEHICLE,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    load_grouped_dataset,
    stable_condition_id,
)


VEHICLE = "vehicle"
ALTERNATE_VEHICLE = "vehicle_copy"


def _context() -> dict[str, object]:
    return {
        STRAIN: "strain-a",
        MEDIUM: "medium-a",
        TEMPERATURE: 30,
        TIME: 15,
        TIME_UNIT: "minutes",
        DATA_SOURCE: "source-a",
        INSTRUMENT: "instrument-a",
        PLATE: "plate-a",
    }


def _fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    context = _context()
    metadata = pd.DataFrame(
        [
            {
                SAMPLE_ID: "water-1",
                SPLIT: "train",
                CHEMICAL: "Water",
                VEHICLE: "Water",
                ALTERNATE_VEHICLE: "Water",
                **context,
            },
            {
                SAMPLE_ID: "water-2",
                SPLIT: "train",
                CHEMICAL: "Water",
                VEHICLE: "Water",
                ALTERNATE_VEHICLE: "Water",
                **context,
            },
            {
                SAMPLE_ID: "dmso-1",
                SPLIT: "train",
                CHEMICAL: "DMSO",
                VEHICLE: "DMSO",
                ALTERNATE_VEHICLE: "DMSO",
                **context,
            },
            {
                SAMPLE_ID: "treatment-water",
                SPLIT: "train",
                CHEMICAL: "drug-water",
                VEHICLE: "Water",
                ALTERNATE_VEHICLE: "Water",
                **context,
            },
            {
                SAMPLE_ID: "treatment-dmso",
                SPLIT: "train",
                CHEMICAL: "drug-dmso",
                VEHICLE: "DMSO",
                ALTERNATE_VEHICLE: "DMSO",
                **context,
            },
        ]
    )
    # In log2 space the controls are Water=(1, 3), DMSO=4 and both
    # treatments are 5.  The direct three-measurement pooled mean is 8/3,
    # whereas an equal average of per-type means would be 3.
    proteome = pd.DataFrame(
        {
            SAMPLE_ID: metadata[SAMPLE_ID],
            "protein-a": [2.0, 8.0, 16.0, 32.0, 32.0],
            # Exactly 80% missing in official train, so this raw column is
            # audited but excluded from the response by the strict < 0.80 rule.
            "protein-filtered": [2.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    return metadata, proteome


def _write_fixture(
    directory: Path,
    metadata: pd.DataFrame | None = None,
    proteome: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    if metadata is None or proteome is None:
        default_metadata, default_proteome = _fixture_frames()
        metadata = default_metadata if metadata is None else metadata
        proteome = default_proteome if proteome is None else proteome
    metadata_path = directory / "metadata.csv"
    proteome_path = directory / "proteome.csv"
    metadata.to_csv(metadata_path, index=False)
    proteome.to_csv(proteome_path, index=False)
    return metadata_path, proteome_path


@pytest.fixture
def explicit_control_identity_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata, proteome = _fixture_frames()
    metadata.loc[metadata[SAMPLE_ID].eq("water-1"), VEHICLE] = "wAtEr"
    metadata.loc[metadata[SAMPLE_ID].eq("water-2"), VEHICLE] = "  "
    metadata.loc[metadata[SAMPLE_ID].eq("dmso-1"), VEHICLE] = pd.NA
    return metadata, proteome


def _load(
    metadata_path: Path,
    proteome_path: Path,
    **kwargs: object,
):
    return load_grouped_dataset(
        metadata_path,
        proteome_path,
        interpolation_fraction=0.0,
        **kwargs,
    )


def _condition_id(chemical: str) -> str:
    return stable_condition_id({CHEMICAL: chemical, **_context()})


def _rewrite_cache(
    cache_path: Path,
    updates: dict[str, np.ndarray] | None = None,
    omitted: frozenset[str] = frozenset(),
) -> None:
    with np.load(cache_path, allow_pickle=False) as cached:
        arrays = {
            name: cached[name].copy()
            for name in cached.files
            if name not in omitted
        }
    arrays.update({} if updates is None else updates)
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _canonical_cache_array_digest(name: str, values: np.ndarray) -> str:
    if name in {"proteins", "missing_rate_names"}:
        payload = json.dumps(
            values.tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    else:
        payload = np.ascontiguousarray(values, dtype=np.dtype("<f8")).tobytes(
            order="C"
        )
    return hashlib.sha256(payload).hexdigest()


def _rewrite_cache_with_valid_digest(
    cache_path: Path,
    array_name: str,
    values: np.ndarray,
) -> None:
    digest_name = f"{array_name}_sha256"
    digest = _canonical_cache_array_digest(array_name, values)
    with np.load(cache_path, allow_pickle=False) as cached:
        manifest = json.loads(str(cached["cache_manifest_json"].item()))
    manifest["artifact_digests"][digest_name] = digest
    _rewrite_cache(
        cache_path,
        {
            array_name: values,
            digest_name: np.asarray(digest),
            "cache_manifest_json": np.asarray(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        },
    )


def _assert_exact_baseline_rebuild(rebuilt: object, baseline: object) -> None:
    assert not rebuilt.cache_hit
    assert rebuilt.proteins == baseline.proteins
    assert (
        rebuilt.response.to_numpy(dtype=np.float32).tobytes()
        == baseline.response.to_numpy(dtype=np.float32).tobytes()
    )
    pd.testing.assert_frame_equal(rebuilt.metadata, baseline.metadata)
    pd.testing.assert_series_equal(
        rebuilt.protein_missing_rate,
        baseline.protein_missing_rate,
    )


def test_default_pool_is_direct_measurement_weighted_and_honest(tmp_path: Path) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    dataset = _load(metadata_path, proteome_path)

    assert PROTOCOL_VERSION == "goai-condition-atomic-v2.1"
    assert CACHE_VERSION == "grouped-dataset-v4"
    assert DEFAULT_CONTROL_POLICY == POOLED_EXACT_CONTEXT_WATER_DMSO
    expected = 5.0 - np.mean([1.0, 3.0, 4.0])
    equal_type_mean = 5.0 - np.mean([np.mean([1.0, 3.0]), 4.0])
    observed = float(dataset.response.loc[_condition_id("drug-water"), "protein-a"])
    assert observed == pytest.approx(expected)
    assert observed != pytest.approx(equal_type_mean)
    assert dataset.proteins == ("protein-a",)
    assert tuple(dataset.protein_missing_rate.index) == (
        "protein-a",
        "protein-filtered",
    )
    assert set(dataset.proteins) < set(dataset.protein_missing_rate.index)
    assert dataset.protein_missing_rate["protein-filtered"] == pytest.approx(0.8)

    summary = dataset.control_policy_summary
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert summary["control_policy"] == POOLED_EXACT_CONTEXT_WATER_DMSO
    assert summary["control_aggregation"] == "direct_measurement_mean_in_log2_space"
    assert summary["control_type_means_averaged_equally"] is False
    assert summary["pooled_across_types"] is True
    assert summary["vehicle_mapping_state"] == "not_applicable_pooled_policy"
    assert summary["vehicle_inference"] is False
    assert summary["treatment_vehicle_values_validated"] is False
    assert summary["nonblank_control_vehicle_identities_validated"] is False
    assert summary["blank_control_vehicle_values_allowed"] is False
    assert summary["control_availability"] == {
        "measurement_count": 3,
        "context_count": 1,
        "both": 1,
        "dmso_only": 0,
        "water_only": 0,
        "none": 0,
    }
    assert summary["treatment_availability"] == {
        "measurement_count": 2,
        "both": 2,
        "dmso_only": 0,
        "water_only": 0,
        "none": 0,
    }

    provenance = dataset.control_provenance.loc["treatment-water"]
    assert {
        "available_control_ids",
        "available_control_types",
        "available_control_type_counts",
        "selected_control_ids",
        "selected_control_types",
        "selected_control_type_counts",
    } <= set(dataset.control_provenance.columns)
    assert provenance["available_control_count"] == 3
    assert provenance["selected_control_count"] == 3
    assert json.loads(provenance["selected_control_ids"]) == [
        "dmso-1",
        "water-1",
        "water-2",
    ]
    assert json.loads(provenance["selected_control_type_counts"]) == {
        "dmso": 1,
        "water": 2,
    }
    assert provenance["selected_control_types"] == "dmso|water"
    assert bool(provenance["pooled_across_types"])
    assert not bool(provenance["vehicle_inference"])
    assert not bool(provenance["predictor_input"])
    assert not bool(provenance["acquisition_input"])


def test_strict_policy_uses_exact_requested_water_and_dmso_controls(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    dataset = _load(
        metadata_path,
        proteome_path,
        control_policy=STRICT_EXPLICIT_VEHICLE,
        vehicle_column=VEHICLE,
    )

    assert dataset.response.loc[
        _condition_id("drug-water"), "protein-a"
    ] == pytest.approx(3.0)
    assert dataset.response.loc[
        _condition_id("drug-dmso"), "protein-a"
    ] == pytest.approx(1.0)
    summary = dataset.control_policy_summary
    assert summary["control_policy"] == STRICT_EXPLICIT_VEHICLE
    assert summary["vehicle_column"] == VEHICLE
    assert summary["vehicle_mapping_state"] == "explicit_column_validated"
    assert summary["pooled_across_types"] is False
    assert summary["vehicle_inference"] is False
    assert summary["treatment_vehicle_values_validated"] is True
    assert summary["nonblank_control_vehicle_identities_validated"] is True
    assert summary["blank_control_vehicle_values_allowed"] is True

    water = dataset.control_provenance.loc["treatment-water"]
    dmso = dataset.control_provenance.loc["treatment-dmso"]
    assert json.loads(water["available_control_type_counts"]) == {
        "dmso": 1,
        "water": 2,
    }
    assert json.loads(water["selected_control_ids"]) == ["water-1", "water-2"]
    assert water["requested_vehicle"] == "Water"
    assert water["selected_control_types"] == "water"
    assert json.loads(dmso["selected_control_ids"]) == ["dmso-1"]
    assert dmso["requested_vehicle"] == "DMSO"
    assert dmso["selected_control_types"] == "dmso"
    assert not bool(water["pooled_across_types"])
    assert not bool(dmso["pooled_across_types"])

    sensitivity = dataset.control_vehicle_sensitivity
    assert len(sensitivity) == 1
    row = sensitivity.iloc[0]
    assert row["water_control_count"] == 2
    assert row["dmso_control_count"] == 1
    assert row["delta_water_minus_delta_dmso_mean"] == pytest.approx(2.0)
    assert row["delta_water_minus_delta_dmso_mean_abs"] == pytest.approx(2.0)
    assert row["audit_role"] == "posthoc_oracle_audit"
    assert not bool(row["acquisition_input"])


def test_strict_policy_validates_control_identity_but_allows_blank_control_values(
    tmp_path: Path,
    explicit_control_identity_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    metadata, proteome = explicit_control_identity_frames
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)
    dataset = _load(
        metadata_path,
        proteome_path,
        control_policy=STRICT_EXPLICIT_VEHICLE,
        vehicle_column=VEHICLE,
    )

    assert dataset.control_policy_summary[
        "nonblank_control_vehicle_identities_validated"
    ] is True
    assert dataset.control_policy_summary["blank_control_vehicle_values_allowed"] is True


def test_strict_policy_rejects_contradictory_explicit_control_identity(
    tmp_path: Path,
    explicit_control_identity_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    metadata, proteome = explicit_control_identity_frames
    metadata.loc[metadata[SAMPLE_ID].eq("water-1"), VEHICLE] = "DMSO"
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)

    with pytest.raises(ValueError, match="control-row vehicle values must agree"):
        _load(
            metadata_path,
            proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column=VEHICLE,
        )


def test_pooled_policy_rejects_irrelevant_vehicle_column(tmp_path: Path) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="irrelevant and misleading"):
        _load(metadata_path, proteome_path, vehicle_column=VEHICLE)


@pytest.mark.parametrize(
    ("vehicle_column", "message"),
    [
        (None, "requires an explicit vehicle_column"),
        ("", "nonempty column name"),
        (CHEMICAL, "cannot infer vehicle"),
        ("pert_id", "cannot infer vehicle"),
        (DATA_SOURCE, "cannot infer vehicle"),
        (PLATE, "cannot infer vehicle"),
        ("protein_well", "cannot infer vehicle"),
    ],
)
def test_strict_policy_rejects_missing_or_inferred_vehicle_columns(
    tmp_path: Path,
    vehicle_column: str | None,
    message: str,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        _load(
            metadata_path,
            proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column=vehicle_column,
        )


def test_strict_policy_rejects_absent_invalid_and_unmatched_vehicle_values(
    tmp_path: Path,
) -> None:
    metadata, proteome = _fixture_frames()
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)
    with pytest.raises(ValueError, match="requires existing vehicle column"):
        _load(
            metadata_path,
            proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column="not_a_column",
        )

    invalid = metadata.copy()
    invalid.loc[invalid[SAMPLE_ID].eq("treatment-water"), VEHICLE] = "ethanol"
    invalid_path = tmp_path / "invalid"
    invalid_path.mkdir()
    invalid_metadata_path, invalid_proteome_path = _write_fixture(
        invalid_path, invalid, proteome
    )
    with pytest.raises(ValueError, match="values must be only Water or DMSO"):
        _load(
            invalid_metadata_path,
            invalid_proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column=VEHICLE,
        )

    keep = ~metadata[SAMPLE_ID].eq("dmso-1")
    no_dmso_metadata = metadata.loc[keep].reset_index(drop=True)
    no_dmso_proteome = proteome.loc[keep].reset_index(drop=True)
    unmatched_path = tmp_path / "unmatched"
    unmatched_path.mkdir()
    unmatched_metadata_path, unmatched_proteome_path = _write_fixture(
        unmatched_path, no_dmso_metadata, no_dmso_proteome
    )
    with pytest.raises(ValueError, match="without exact requested-type controls"):
        _load(
            unmatched_metadata_path,
            unmatched_proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column=VEHICLE,
        )


def test_strict_policy_rejects_mixed_vehicle_replicates_in_one_condition(
    tmp_path: Path,
) -> None:
    metadata, proteome = _fixture_frames()
    duplicate = metadata.loc[metadata[SAMPLE_ID].eq("treatment-water")].copy()
    duplicate.loc[:, SAMPLE_ID] = "treatment-water-dmso-replicate"
    duplicate.loc[:, VEHICLE] = "DMSO"
    duplicate.loc[:, ALTERNATE_VEHICLE] = "DMSO"
    metadata = pd.concat([metadata, duplicate], ignore_index=True)
    proteome = pd.concat(
        [
            proteome,
            pd.DataFrame(
                {
                    SAMPLE_ID: ["treatment-water-dmso-replicate"],
                    "protein-a": [64.0],
                }
            ),
        ],
        ignore_index=True,
    )
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)

    with pytest.raises(ValueError, match="mixed vehicle replicates"):
        _load(
            metadata_path,
            proteome_path,
            control_policy=STRICT_EXPLICIT_VEHICLE,
            vehicle_column=VEHICLE,
        )


def test_cache_identity_includes_policy_and_vehicle_column_and_round_trips(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    pooled = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    pooled_cached = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    strict = _load(
        metadata_path,
        proteome_path,
        cache_dir=cache_dir,
        control_policy=STRICT_EXPLICIT_VEHICLE,
        vehicle_column=VEHICLE,
    )
    strict_cached = _load(
        metadata_path,
        proteome_path,
        cache_dir=cache_dir,
        control_policy=STRICT_EXPLICIT_VEHICLE,
        vehicle_column=VEHICLE,
    )
    strict_alternate = _load(
        metadata_path,
        proteome_path,
        cache_dir=cache_dir,
        control_policy=STRICT_EXPLICIT_VEHICLE,
        vehicle_column=ALTERNATE_VEHICLE,
    )

    assert not pooled.cache_hit
    assert pooled_cached.cache_hit
    assert not strict.cache_hit
    assert strict_cached.cache_hit
    assert not strict_alternate.cache_hit
    assert len({pooled.cache_key, strict.cache_key, strict_alternate.cache_key}) == 3
    assert len(list(cache_dir.glob("grouped_*.npz"))) == 3
    assert (
        pooled_cached.response.to_numpy(dtype=np.float32).tobytes()
        == pooled.response.to_numpy(dtype=np.float32).tobytes()
    )
    assert strict_cached.control_policy_summary == strict.control_policy_summary
    pd.testing.assert_frame_equal(strict_cached.metadata, strict.metadata)
    pd.testing.assert_frame_equal(strict_cached.response, strict.response)
    pd.testing.assert_frame_equal(
        strict_cached.control_provenance,
        strict.control_provenance,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        strict_cached.control_vehicle_sensitivity,
        strict.control_vehicle_sensitivity,
        check_dtype=False,
    )

    cache_path = cache_dir / f"grouped_{strict.cache_key}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        assert {
            "cache_key",
            "cache_manifest_json",
            "source_hashes_json",
            "source_files_json",
            "missing_rate_threshold",
            "interpolation_fraction",
            "split_seed",
            "control_policy",
            "vehicle_column_json",
            "response_sha256",
            "metadata_json_sha256",
            "control_provenance_json_sha256",
            "overlap_json_sha256",
            "control_policy_summary_json_sha256",
            "control_vehicle_sensitivity_json_sha256",
            "proteins_sha256",
            "missing_rate_names_sha256",
            "missing_rate_values_sha256",
            "control_policy_summary_json",
            "control_vehicle_sensitivity_json",
        } <= set(cached.files)
        persisted_summary = json.loads(str(cached["control_policy_summary_json"].item()))
        manifest = json.loads(str(cached["cache_manifest_json"].item()))
        response_digest = hashlib.sha256(
            cached["response"].tobytes(order="C")
        ).hexdigest()
    assert persisted_summary == strict.control_policy_summary
    assert manifest["cache_key"] == strict.cache_key
    assert manifest["source_hashes"] == strict.source_hashes
    assert manifest["source_files"]["metadata"]["resolved_path"] == str(
        metadata_path.resolve()
    )
    assert manifest["source_files"]["proteome"]["resolved_path"] == str(
        proteome_path.resolve()
    )
    assert manifest["missing_rate_threshold"] == 0.8
    assert manifest["interpolation_fraction"] == 0.0
    assert manifest["split_seed"] == 42
    assert manifest["control_policy"] == STRICT_EXPLICIT_VEHICLE
    assert manifest["vehicle_column"] == VEHICLE
    assert manifest["artifact_digests"]["response_sha256"] == response_digest
    assert {
        "proteins_sha256",
        "missing_rate_names_sha256",
        "missing_rate_values_sha256",
    } <= set(manifest["artifact_digests"])


def test_v4_cache_preserves_raw_missing_rate_scope_after_filtering(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    cached = _load(metadata_path, proteome_path, cache_dir=cache_dir)

    assert cached.cache_hit
    assert cached.proteins == baseline.proteins == ("protein-a",)
    pd.testing.assert_series_equal(
        cached.protein_missing_rate,
        baseline.protein_missing_rate,
    )
    assert set(cached.proteins) < set(cached.protein_missing_rate.index)


def test_v4_cache_rejects_digest_consistent_reordered_proteins(
    tmp_path: Path,
) -> None:
    metadata, proteome = _fixture_frames()
    proteome.insert(2, "protein-b", [4.0, 16.0, 32.0, 64.0, 64.0])
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    assert baseline.proteins == ("protein-a", "protein-b")
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"
    reordered = np.asarray(baseline.proteins[::-1], dtype=str)
    _rewrite_cache_with_valid_digest(cache_path, "proteins", reordered)

    with pytest.warns(
        RuntimeWarning,
        match="ordered missing-rate filter contract",
    ):
        rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    _assert_exact_baseline_rebuild(rebuilt, baseline)
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit


def test_v4_cache_rejects_digest_consistent_filtered_protein_rate(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        missing_rate_names = cached["missing_rate_names"].tolist()
        missing_rate_values = cached["missing_rate_values"].copy()
        persisted_threshold = cached["missing_rate_threshold"].item()
    retained_index = missing_rate_names.index(baseline.proteins[0])
    missing_rate_values[retained_index] = persisted_threshold
    _rewrite_cache_with_valid_digest(
        cache_path,
        "missing_rate_values",
        missing_rate_values,
    )

    with pytest.warns(
        RuntimeWarning,
        match="ordered missing-rate filter contract",
    ):
        rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    _assert_exact_baseline_rebuild(rebuilt, baseline)
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit


@pytest.mark.parametrize(
    "retained_bytes",
    [0, 4],
    ids=["eof", "bad-zip"],
)
def test_v4_cache_rebuilds_truncated_npz(
    tmp_path: Path,
    retained_bytes: int,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"
    cache_path.write_bytes(cache_path.read_bytes()[:retained_bytes])

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable GOAI cache"):
        rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    _assert_exact_baseline_rebuild(rebuilt, baseline)
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit


@pytest.mark.parametrize(
    "array_name",
    ["proteins", "missing_rate_names", "missing_rate_values"],
)
def test_v4_cache_rejects_independently_tampered_schema_arrays(
    tmp_path: Path,
    array_name: str,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        tampered = cached[array_name].copy()
    if array_name == "proteins":
        tampered = np.asarray(["tampered-protein"], dtype=str)
    elif array_name == "missing_rate_names":
        tampered = np.asarray(
            ["tampered-name", *tampered.tolist()[1:]],
            dtype=str,
        )
    else:
        tampered[0] += np.float64(0.125)
    _rewrite_cache(cache_path, {array_name: tampered})

    with pytest.warns(RuntimeWarning, match=rf"{array_name} SHA256 mismatch"):
        rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    _assert_exact_baseline_rebuild(rebuilt, baseline)
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit


def test_v4_cache_rebuilds_when_canonical_digest_scalar_is_missing(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"

    for digest_name in (
        "proteins_sha256",
        "missing_rate_names_sha256",
        "missing_rate_values_sha256",
    ):
        _rewrite_cache(cache_path, omitted=frozenset({digest_name}))
        with pytest.warns(RuntimeWarning, match="missing v4 fields"):
            rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
        _assert_exact_baseline_rebuild(rebuilt, baseline)


@pytest.mark.parametrize(
    ("array_name", "invalid_case", "message"),
    [
        ("proteins", "duplicate", "response schema is inconsistent"),
        ("proteins", "empty", "response schema is inconsistent"),
        (
            "missing_rate_names",
            "duplicate",
            "missing-rate names must be nonempty unique strings",
        ),
        (
            "missing_rate_names",
            "empty",
            "missing-rate names must be nonempty unique strings",
        ),
        (
            "missing_rate_values",
            "negative",
            r"missing-rate values must be finite and within \[0, 1\]",
        ),
        (
            "missing_rate_values",
            "above_one",
            r"missing-rate values must be finite and within \[0, 1\]",
        ),
        (
            "missing_rate_values",
            "nan",
            r"missing-rate values must be finite and within \[0, 1\]",
        ),
        (
            "missing_rate_values",
            "inf",
            r"missing-rate values must be finite and within \[0, 1\]",
        ),
    ],
)
def test_v4_cache_rejects_semantically_invalid_names_and_missing_rates(
    tmp_path: Path,
    array_name: str,
    invalid_case: str,
    message: str,
) -> None:
    metadata, proteome = _fixture_frames()
    proteome.insert(2, "protein-b", [4.0, 16.0, 32.0, 64.0, 64.0])
    metadata_path, proteome_path = _write_fixture(tmp_path, metadata, proteome)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        invalid = cached[array_name].copy()

    if invalid_case == "duplicate":
        invalid[-1] = invalid[0]
    elif invalid_case == "empty":
        invalid = np.asarray([*invalid.tolist()[:-1], ""], dtype=str)
    elif invalid_case == "negative":
        invalid[0] = -0.01
    elif invalid_case == "above_one":
        invalid[0] = 1.01
    elif invalid_case == "nan":
        invalid[0] = np.nan
    elif invalid_case == "inf":
        invalid[0] = np.inf
    else:
        raise AssertionError(f"Unhandled invalid cache case: {invalid_case}")
    _rewrite_cache_with_valid_digest(cache_path, array_name, invalid)

    with pytest.warns(RuntimeWarning, match=message):
        rebuilt = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    _assert_exact_baseline_rebuild(rebuilt, baseline)


def test_v4_cache_rejects_tampered_response_and_wrong_manifest(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    baseline = _load(metadata_path, proteome_path, cache_dir=cache_dir)
    assert baseline.cache_key is not None
    cache_path = cache_dir / f"grouped_{baseline.cache_key}.npz"

    def rewrite_cache(updates: dict[str, np.ndarray]) -> None:
        with np.load(cache_path, allow_pickle=False) as cached:
            arrays = {name: cached[name].copy() for name in cached.files}
        arrays.update(updates)
        with cache_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    with np.load(cache_path, allow_pickle=False) as cached:
        tampered_response = cached["response"].copy()
    tampered_response[0, 0] += np.float32(1.0)
    rewrite_cache({"response": tampered_response})

    with pytest.warns(RuntimeWarning, match="response SHA256 mismatch"):
        rebuilt_response = _load(
            metadata_path,
            proteome_path,
            cache_dir=cache_dir,
        )
    assert not rebuilt_response.cache_hit
    assert (
        rebuilt_response.response.to_numpy(dtype=np.float32).tobytes()
        == baseline.response.to_numpy(dtype=np.float32).tobytes()
    )
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit

    with np.load(cache_path, allow_pickle=False) as cached:
        wrong_manifest = json.loads(str(cached["cache_manifest_json"].item()))
    wrong_manifest["cache_key"] = "0" * 64
    rewrite_cache(
        {
            "cache_manifest_json": np.asarray(
                json.dumps(wrong_manifest, sort_keys=True, separators=(",", ":"))
            )
        }
    )
    with pytest.warns(RuntimeWarning, match="manifest key differs"):
        rebuilt_manifest = _load(
            metadata_path,
            proteome_path,
            cache_dir=cache_dir,
        )
    assert not rebuilt_manifest.cache_hit
    assert (
        rebuilt_manifest.response.to_numpy(dtype=np.float32).tobytes()
        == baseline.response.to_numpy(dtype=np.float32).tobytes()
    )
    assert _load(metadata_path, proteome_path, cache_dir=cache_dir).cache_hit


def test_audit_exposes_sensitivity_as_posthoc_non_acquisition_output(
    tmp_path: Path,
) -> None:
    metadata_path, proteome_path = _write_fixture(tmp_path)
    dataset = _load(metadata_path, proteome_path)
    audit = build_data_audit(dataset)

    assert audit["audit_role"] == "posthoc_oracle_audit"
    assert audit["acquisition_input"] is False
    assert audit["control_policy_summary"] == dataset.control_policy_summary
    sensitivity_summary = audit["control_vehicle_sensitivity"]
    assert sensitivity_summary["audit_role"] == "posthoc_oracle_audit"
    assert sensitivity_summary["posthoc"] is True
    assert sensitivity_summary["acquisition_input"] is False
    assert sensitivity_summary["both_vehicle_context_count"] == 1
    assert sensitivity_summary["finite_context_count"] == 1
    assert sensitivity_summary["finite_context_protein_pair_count"] == 1
    assert sensitivity_summary["pair_weighted_global_rms"] == pytest.approx(2.0)
    assert sensitivity_summary["global_rms"] == pytest.approx(2.0)
    assert sensitivity_summary["context_rmse_quantiles"] == {
        "q0": 2.0,
        "q25": 2.0,
        "q50": 2.0,
        "q75": 2.0,
        "q100": 2.0,
    }
    assert sensitivity_summary["context_mean_absolute_difference"] == pytest.approx(
        2.0
    )
    assert sensitivity_summary["context_median_absolute_difference"] == pytest.approx(
        2.0
    )
    assert sensitivity_summary["treatment_frequency_weighted_rms"] == pytest.approx(
        2.0
    )
    assert sensitivity_summary["treatment_frequency_weighted_pair_count"] == 2
    assert sensitivity_summary["treatment_frequency_context_count"] == 1
    assert sensitivity_summary["treatment_frequency_total"] == 2
    assert sensitivity_summary["artifact"] == "control_vehicle_sensitivity.csv"

    paths = write_audit_outputs(dataset, tmp_path / "audit")
    assert set(paths) == {
        "data_audit",
        "tensor_coverage",
        "low_rank_spectrum",
        "control_vehicle_sensitivity",
    }
    written = pd.read_csv(paths["control_vehicle_sensitivity"])
    assert len(written) == 1
    assert written.loc[0, "audit_role"] == "posthoc_oracle_audit"
    assert not bool(written.loc[0, "acquisition_input"])
    assert written.loc[
        0, "delta_water_minus_delta_dmso_mean"
    ] == pytest.approx(2.0)
    assert list(written.columns[: len(MATCH_CONTROL_FIELDS)]) == list(
        MATCH_CONTROL_FIELDS
    )
    written_audit = json.loads(paths["data_audit"].read_text(encoding="utf-8"))
    assert written_audit["controls"]["vehicle_sensitivity"][
        "acquisition_input"
    ] is False
    assert CONDITION_ID not in written.columns
