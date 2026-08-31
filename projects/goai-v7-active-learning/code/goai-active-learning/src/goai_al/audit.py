"""Post-hoc oracle and tensor audits for the condition-atomic GOAI data layer.

These diagnostics may inspect the full retrospective response.  They are
explicitly post-hoc oracle audits and must never be passed to acquisition.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    CACHE_VERSION,
    CHEMICAL,
    CONDITION_ID,
    CONTROL_POLICIES,
    DEFAULT_CONTROL_POLICY,
    INTERPOLATION_SPLIT,
    MEDIUM,
    PROTOCOL_VERSION,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    GroupedDataset,
    load_grouped_dataset,
)


SPECTRUM_RANKS = (8, 16, 32, 64, 128)
_TENSOR_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, "time_level")


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
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
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _time_level(metadata: pd.DataFrame) -> pd.Series:
    return metadata[TIME].astype(str) + " " + metadata[TIME_UNIT].astype(str)


def tensor_coverage(
    metadata: pd.DataFrame,
    scopes: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Describe five-dimensional occupancy and major one-axis fibers.

    Time value and unit form one factor level, giving the registered five tensor
    axes: strain, chemical, medium, temperature, and time.
    """

    required = {STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, TIME_UNIT}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Tensor metadata is missing fields: {missing}")
    work = metadata.copy()
    work.index = work.index.astype(str)
    work["time_level"] = _time_level(work)
    if scopes is None:
        scopes = {"all_conditions": tuple(work.index)}
    rows: list[dict[str, object]] = []
    for scope_name, scope_ids in scopes.items():
        ids = work.index.intersection(pd.Index(scope_ids).astype(str), sort=False)
        frame = work.loc[ids, list(_TENSOR_FIELDS)].astype(str).drop_duplicates()
        level_counts = {field_name: int(frame[field_name].nunique()) for field_name in _TENSOR_FIELDS}
        possible = int(np.prod(list(level_counts.values()), dtype=np.int64)) if len(frame) else 0
        observed = int(len(frame))
        rows.append(
            {
                "scope": scope_name,
                "kind": "tensor",
                "fiber_axis": "",
                "condition_count": int(len(ids)),
                "observed_cells": observed,
                "possible_cells": possible,
                "occupancy": observed / possible if possible else np.nan,
                "fiber_count": np.nan,
                "complete_fibers": np.nan,
                "complete_fiber_fraction": np.nan,
                "mean_fiber_coverage": np.nan,
                "median_fiber_coverage": np.nan,
                **{f"levels_{field_name}": count for field_name, count in level_counts.items()},
            }
        )
        for axis in _TENSOR_FIELDS:
            other_fields = [field_name for field_name in _TENSOR_FIELDS if field_name != axis]
            axis_levels = level_counts[axis]
            if len(frame) == 0 or axis_levels == 0:
                coverages = np.asarray([], dtype=np.float64)
            else:
                fiber_sizes = frame.groupby(other_fields, dropna=False)[axis].nunique()
                coverages = fiber_sizes.to_numpy(dtype=np.float64) / axis_levels
            complete = int(np.count_nonzero(np.isclose(coverages, 1.0)))
            rows.append(
                {
                    "scope": scope_name,
                    "kind": "fiber",
                    "fiber_axis": axis,
                    "condition_count": int(len(ids)),
                    "observed_cells": observed,
                    "possible_cells": possible,
                    "occupancy": observed / possible if possible else np.nan,
                    "fiber_count": int(len(coverages)),
                    "complete_fibers": complete,
                    "complete_fiber_fraction": complete / len(coverages) if len(coverages) else np.nan,
                    "mean_fiber_coverage": float(coverages.mean()) if len(coverages) else np.nan,
                    "median_fiber_coverage": float(np.median(coverages)) if len(coverages) else np.nan,
                    **{f"levels_{field_name}": count for field_name, count in level_counts.items()},
                }
            )
    return pd.DataFrame(rows)


def _center_and_impute(response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("response must be two-dimensional")
    observed = np.isfinite(values)
    counts = observed.sum(axis=0)
    sums = np.where(observed, values, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(observed, values - means, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    scales = np.sqrt(variance)
    scales[~np.isfinite(scales) | (scales <= 0.0)] = 1.0
    standardized = centered / scales
    return centered, standardized


def _top_singular_values(matrix: np.ndarray, rank: int, seed: int = 42) -> tuple[np.ndarray, str]:
    rows, columns = matrix.shape
    maximum = min(rows, columns)
    if maximum == 0 or rank == 0:
        return np.asarray([], dtype=np.float64), "empty"
    rank = min(int(rank), maximum)
    if maximum <= 256:
        return np.linalg.svd(matrix, full_matrices=False, compute_uv=False)[:rank], "exact_svd"
    width = min(maximum, rank + 16)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((columns, width))
    basis, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(2):
        basis, _ = np.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
    compressed = basis.T @ matrix
    singular_values = np.linalg.svd(compressed, full_matrices=False, compute_uv=False)
    return singular_values[:rank], "randomized_svd_seed_42"


def low_rank_spectrum(
    response: pd.DataFrame | np.ndarray,
    ranks: Sequence[int] = SPECTRUM_RANKS,
) -> pd.DataFrame:
    """Compute descriptive oracle energy for centered and standardized response."""

    values = response.to_numpy(dtype=np.float64) if isinstance(response, pd.DataFrame) else np.asarray(response)
    centered, standardized = _center_and_impute(values)
    requested = tuple(int(rank) for rank in ranks)
    if any(rank <= 0 for rank in requested):
        raise ValueError("Spectrum ranks must be positive")
    maximum_requested = max(requested, default=0)
    observed_fraction = float(np.isfinite(values).mean()) if values.size else np.nan
    rows: list[dict[str, object]] = []
    for variant, matrix in (
        ("centered_response", centered),
        ("per_protein_standardized_response", standardized),
    ):
        singular_values, method = _top_singular_values(matrix, maximum_requested)
        squared = np.square(singular_values)
        total_energy = float(np.square(matrix).sum(dtype=np.float64))
        for requested_rank in requested:
            effective_rank = min(requested_rank, min(matrix.shape))
            retained = float(squared[:effective_rank].sum(dtype=np.float64))
            rows.append(
                {
                    "variant": variant,
                    "requested_rank": requested_rank,
                    "effective_rank": effective_rank,
                    "cumulative_energy": retained / total_energy if total_energy > 0.0 else 0.0,
                    "retained_energy": retained,
                    "total_energy": total_energy,
                    "condition_count": int(matrix.shape[0]),
                    "protein_count": int(matrix.shape[1]),
                    "observed_fraction": observed_fraction,
                    "method": method,
                    "audit_role": "posthoc_oracle_audit",
                    "acquisition_input": False,
                }
            )
    return pd.DataFrame(rows)


def _audit_scopes(dataset: GroupedDataset) -> dict[str, tuple[str, ...]]:
    scopes = {
        "all_conditions": tuple(dataset.metadata.index.astype(str)),
        "official_train": tuple(dataset.official_train_ids.astype(str)),
        "candidate_pool": tuple(dataset.train_ids.astype(str)),
    }
    for name, ids in dataset.validation_ids.items():
        scopes[name] = tuple(ids.astype(str))
    return scopes


def build_data_audit(dataset: GroupedDataset) -> dict[str, object]:
    """Build the JSON-serializable condition, control, split, and missingness audit."""

    provenance = dataset.control_provenance
    matched_counts = (
        provenance["matched_control_count"].to_numpy(dtype=np.int64)
        if len(provenance)
        else np.asarray([], dtype=np.int64)
    )
    control_ids: set[str] = set()
    if "matched_control_ids" in provenance:
        for encoded in provenance["matched_control_ids"]:
            control_ids.update(str(value) for value in json.loads(str(encoded)))
    replicate_counts = dataset.metadata["replicate_count"].to_numpy(dtype=np.int64)
    released_replicate_counts = dataset.metadata.get(
        "released_replicate_count", dataset.metadata["replicate_count"]
    ).to_numpy(dtype=np.int64)
    official_train_metadata = dataset.metadata.loc[dataset.official_train_ids]
    official_train_aggregation_rows = int(
        official_train_metadata["replicate_count"].sum()
    )
    official_train_released_rows = int(
        official_train_metadata.get(
            "released_replicate_count", official_train_metadata["replicate_count"]
        ).sum()
    )
    cross_split_only = (
        provenance["cross_split_only"].astype(bool)
        if "cross_split_only" in provenance
        else pd.Series(False, index=provenance.index)
    )
    treatment_split_order = ("train", *dataset.removed_validation_overlap.keys())
    observed_treatment_splits = (
        tuple(sorted(provenance["treatment_split"].astype(str).unique()))
        if "treatment_split" in provenance
        else ()
    )
    treatment_splits = tuple(dict.fromkeys((*treatment_split_order, *observed_treatment_splits)))
    cross_split_by_treatment_split = {
        split_name: (
            int(
                (
                    provenance["treatment_split"].astype(str).eq(split_name)
                    & cross_split_only
                ).sum()
            )
            if "treatment_split" in provenance
            else 0
        )
        for split_name in treatment_splits
    }
    response_values = dataset.response.to_numpy(dtype=np.float32)
    removed = {
        name: {"count": len(ids), "condition_ids": list(ids)}
        for name, ids in dataset.removed_validation_overlap.items()
    }
    overlap_records = [
        {
            CONDITION_ID: str(condition_id),
            "split_provenance": str(row["split_provenance"]),
            "split_count": int(row["split_count"]),
            "measurement_replicate_count": int(row["measurement_replicate_count"]),
            "official_train_overlap": bool(row["official_train_overlap"]),
        }
        for condition_id, row in dataset.cross_split_overlap.iterrows()
    ]
    policy_summary = dict(dataset.control_policy_summary)
    sensitivity = dataset.control_vehicle_sensitivity
    sensitivity_mean_abs_all = (
        pd.to_numeric(
            sensitivity["delta_water_minus_delta_dmso_mean_abs"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if len(sensitivity)
        and "delta_water_minus_delta_dmso_mean_abs" in sensitivity
        else np.full(len(sensitivity), np.nan, dtype=np.float64)
    )
    sensitivity_rmse_all = (
        pd.to_numeric(
            sensitivity["delta_water_minus_delta_dmso_rmse"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if len(sensitivity)
        and "delta_water_minus_delta_dmso_rmse" in sensitivity
        else np.full(len(sensitivity), np.nan, dtype=np.float64)
    )
    finite_protein_counts = (
        pd.to_numeric(
            sensitivity["finite_protein_count"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if len(sensitivity) and "finite_protein_count" in sensitivity
        else np.full(len(sensitivity), np.nan, dtype=np.float64)
    )
    sensitivity_mean_abs = sensitivity_mean_abs_all[
        np.isfinite(sensitivity_mean_abs_all)
    ]
    sensitivity_rmse = sensitivity_rmse_all[np.isfinite(sensitivity_rmse_all)]
    pair_mask = (
        np.isfinite(sensitivity_rmse_all)
        & np.isfinite(finite_protein_counts)
        & (finite_protein_counts > 0.0)
    )
    finite_pair_count = int(finite_protein_counts[pair_mask].sum())
    pair_weighted_global_rms = (
        float(
            np.sqrt(
                np.average(
                    np.square(sensitivity_rmse_all[pair_mask]),
                    weights=finite_protein_counts[pair_mask],
                )
            )
        )
        if finite_pair_count
        else None
    )
    quantile_names = ("q0", "q25", "q50", "q75", "q100")
    quantile_probabilities = (0.0, 0.25, 0.5, 0.75, 1.0)
    context_rmse_quantiles = (
        {
            name: _finite_or_none(float(value))
            for name, value in zip(
                quantile_names,
                np.quantile(sensitivity_rmse, quantile_probabilities),
            )
        }
        if len(sensitivity_rmse)
        else {name: None for name in quantile_names}
    )

    treatment_frequency_by_controls: dict[tuple[str, ...], int] = {}
    if "available_control_ids" in provenance:
        for encoded_ids in provenance["available_control_ids"]:
            try:
                available_ids = json.loads(str(encoded_ids))
            except json.JSONDecodeError:
                continue
            if isinstance(available_ids, list) and all(
                isinstance(value, str) for value in available_ids
            ):
                fingerprint = tuple(sorted(available_ids))
                treatment_frequency_by_controls[fingerprint] = (
                    treatment_frequency_by_controls.get(fingerprint, 0) + 1
                )
    treatment_frequencies = np.zeros(len(sensitivity), dtype=np.float64)
    if {
        "water_control_ids",
        "dmso_control_ids",
    } <= set(sensitivity.columns):
        for position, (_, row) in enumerate(sensitivity.iterrows()):
            try:
                water_ids = json.loads(str(row["water_control_ids"]))
                dmso_ids = json.loads(str(row["dmso_control_ids"]))
            except json.JSONDecodeError:
                continue
            if (
                isinstance(water_ids, list)
                and isinstance(dmso_ids, list)
                and all(
                    isinstance(value, str) for value in (*water_ids, *dmso_ids)
                )
            ):
                fingerprint = tuple(sorted((*water_ids, *dmso_ids)))
                treatment_frequencies[position] = treatment_frequency_by_controls.get(
                    fingerprint, 0
                )
    treatment_pair_weights = treatment_frequencies * finite_protein_counts
    treatment_pair_mask = pair_mask & (treatment_frequencies > 0.0)
    treatment_pair_weight_total = int(
        treatment_pair_weights[treatment_pair_mask].sum()
    )
    treatment_frequency_weighted_rms = (
        float(
            np.sqrt(
                np.average(
                    np.square(sensitivity_rmse_all[treatment_pair_mask]),
                    weights=treatment_pair_weights[treatment_pair_mask],
                )
            )
        )
        if treatment_pair_weight_total
        else None
    )
    treatment_context_mask = np.isfinite(sensitivity_rmse_all) & (
        treatment_frequencies > 0.0
    )
    treatment_frequency_context_rms = (
        float(
            np.sqrt(
                np.average(
                    np.square(sensitivity_rmse_all[treatment_context_mask]),
                    weights=treatment_frequencies[treatment_context_mask],
                )
            )
        )
        if bool(treatment_context_mask.any())
        else None
    )
    context_mean_absolute_difference = (
        _finite_or_none(float(sensitivity_mean_abs.mean()))
        if len(sensitivity_mean_abs)
        else None
    )
    context_median_absolute_difference = (
        _finite_or_none(float(np.median(sensitivity_mean_abs)))
        if len(sensitivity_mean_abs)
        else None
    )
    sensitivity_summary = {
        "audit_role": "posthoc_oracle_audit",
        "posthoc": True,
        "acquisition_input": False,
        "definition": (
            "delta_if_water_minus_delta_if_dmso_equals_"
            "dmso_control_mean_minus_water_control_mean"
        ),
        "both_vehicle_context_count": int(len(sensitivity)),
        "finite_context_count": int(len(sensitivity_rmse)),
        "finite_context_protein_pair_count": finite_pair_count,
        "pair_weighted_global_rms": (
            _finite_or_none(pair_weighted_global_rms)
            if pair_weighted_global_rms is not None
            else None
        ),
        "global_rms": (
            _finite_or_none(pair_weighted_global_rms)
            if pair_weighted_global_rms is not None
            else None
        ),
        "global_rms_weighting": "finite_context_protein_pairs",
        "context_rmse_quantiles": context_rmse_quantiles,
        "context_mean_absolute_difference": context_mean_absolute_difference,
        "context_median_absolute_difference": context_median_absolute_difference,
        "context_mean_abs_sensitivity_mean": context_mean_absolute_difference,
        "context_mean_abs_sensitivity_median": context_median_absolute_difference,
        "treatment_frequency_weighted_rms": (
            _finite_or_none(treatment_frequency_weighted_rms)
            if treatment_frequency_weighted_rms is not None
            else None
        ),
        "treatment_frequency_weighted_pair_count": treatment_pair_weight_total,
        "treatment_frequency_context_count": int(treatment_context_mask.sum()),
        "treatment_frequency_total": int(
            treatment_frequencies[treatment_context_mask].sum()
        ),
        "treatment_frequency_weighting": (
            "finite_context_protein_pairs_times_treatment_measurement_frequency"
        ),
        "treatment_frequency_context_rms": (
            _finite_or_none(treatment_frequency_context_rms)
            if treatment_frequency_context_rms is not None
            else None
        ),
        "artifact": "control_vehicle_sensitivity.csv",
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cache_version": CACHE_VERSION,
        "audit_role": "posthoc_oracle_audit",
        "acquisition_input": False,
        "condition_keys": [STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, TIME_UNIT],
        "condition_id_independent_of_split": True,
        "control_policy_summary": policy_summary,
        "control_vehicle_sensitivity": sensitivity_summary,
        "query_granularity": {
            "condition_count": int(len(dataset.metadata)),
            "candidate_pool_count": int(len(dataset.train_ids)),
            "measurement_replicate_count": int(replicate_counts.sum()),
            "released_treatment_measurement_count": int(len(provenance)),
            "oracle_aggregation_measurement_count": int(replicate_counts.sum()),
            "discarded_overlap_validation_measurement_count": int(
                released_replicate_counts.sum() - replicate_counts.sum()
            ),
            "replicates_per_condition_min": int(replicate_counts.min()) if len(replicate_counts) else 0,
            "replicates_per_condition_median": _finite_or_none(float(np.median(replicate_counts))) if len(replicate_counts) else None,
            "replicates_per_condition_max": int(replicate_counts.max()) if len(replicate_counts) else 0,
        },
        "controls": {
            "role": "assay_overhead",
            "included_in_query_pool": False,
            "included_in_predictor_or_acquisition": False,
            "matching": policy_summary.get(
                "selected_control_rule",
                "exact context Water/DMSO mean in log2 abundance space",
            ),
            "policy_summary": policy_summary,
            "vehicle_sensitivity": sensitivity_summary,
            "treatment_measurements_with_provenance": int(len(provenance)),
            "unique_control_measurements_used": int(len(control_ids)),
            "matched_control_links": int(matched_counts.sum()) if len(matched_counts) else 0,
            "controls_per_treatment_min": int(matched_counts.min()) if len(matched_counts) else 0,
            "controls_per_treatment_median": _finite_or_none(float(np.median(matched_counts))) if len(matched_counts) else None,
            "controls_per_treatment_max": int(matched_counts.max()) if len(matched_counts) else 0,
            "cross_split_only_treatment_measurements": int(cross_split_only.sum()),
            "cross_split_only_treatment_measurements_by_treatment_split": (
                cross_split_by_treatment_split
            ),
        },
        "splits": {
            "seed": dataset.benchmark_split.seed if dataset.benchmark_split else None,
            "interpolation_fraction": (
                dataset.benchmark_split.interpolation_fraction if dataset.benchmark_split else None
            ),
            "official_train_conditions": int(len(dataset.official_train_ids)),
            "official_train_aggregation_treatment_rows": official_train_aggregation_rows,
            "official_train_released_treatment_rows": official_train_released_rows,
            "official_train_discarded_validation_overlap_rows": int(
                official_train_released_rows - official_train_aggregation_rows
            ),
            "candidate_pool_conditions": int(len(dataset.train_ids)),
            "evaluation_conditions": {
                name: int(len(ids)) for name, ids in dataset.validation_ids.items()
            },
            "detected_cross_split_condition_count": int(len(dataset.cross_split_overlap)),
            "detected_cross_split_conditions": overlap_records,
            "removed_official_validation_overlap": removed,
        },
        "missingness": {
            "protein_schema_count": int(len(dataset.proteins)),
            "condition_protein_positions": int(response_values.size),
            "observed_condition_protein_positions": int(np.isfinite(response_values).sum()),
            "response_missing_fraction": _finite_or_none(float(np.isnan(response_values).mean())),
            "official_train_raw_protein_missing_rate_min": _finite_or_none(float(dataset.protein_missing_rate.min())),
            "official_train_raw_protein_missing_rate_median": _finite_or_none(float(dataset.protein_missing_rate.median())),
            "official_train_raw_protein_missing_rate_max": _finite_or_none(float(dataset.protein_missing_rate.max())),
        },
        "source_hashes": dict(dataset.source_hashes),
        "cache": {"key": dataset.cache_key, "hit": bool(dataset.cache_hit)},
        "artifacts": {
            "control_vehicle_sensitivity.csv": (
                "posthoc Water-DMSO exact-context sensitivity; acquisition_input=false"
            ),
            "tensor_coverage.csv": "five-dimensional occupancy and major-fiber completeness",
            "low_rank_spectrum.csv": (
                "official-train centered and per-protein-standardized response energy; "
                "posthoc oracle only"
            ),
        },
    }


def write_audit_outputs(
    dataset: GroupedDataset,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Atomically write the four registered audit artifacts."""

    destination = Path(output_dir)
    paths = {
        "data_audit": destination / "data_audit.json",
        "tensor_coverage": destination / "tensor_coverage.csv",
        "low_rank_spectrum": destination / "low_rank_spectrum.csv",
        "control_vehicle_sensitivity": destination / "control_vehicle_sensitivity.csv",
    }
    coverage = tensor_coverage(dataset.metadata, _audit_scopes(dataset))
    official_train_response = dataset.response.loc[dataset.official_train_ids]
    spectrum = low_rank_spectrum(official_train_response, SPECTRUM_RANKS)
    spectrum.insert(0, "scope", "official_train")
    _atomic_json(paths["data_audit"], build_data_audit(dataset))
    _atomic_csv(paths["tensor_coverage"], coverage)
    _atomic_csv(paths["low_rank_spectrum"], spectrum)
    sensitivity = dataset.control_vehicle_sensitivity.copy()
    if len(sensitivity) == 0 and not len(sensitivity.columns):
        sensitivity = pd.DataFrame(
            columns=["audit_role", "acquisition_input"]
        )
    _atomic_csv(paths["control_vehicle_sensitivity"], sensitivity)
    return paths


# Readable alias for callers that use the framework-spec terminology.
write_audit_artifacts = write_audit_outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--proteome", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--missing-rate-threshold", default=0.80, type=float)
    parser.add_argument("--cache-dir", default=None, type=Path)
    parser.add_argument("--interpolation-fraction", default=0.20, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--control-policy",
        choices=CONTROL_POLICIES,
        default=DEFAULT_CONTROL_POLICY,
    )
    parser.add_argument("--vehicle-column", default=None)
    args = parser.parse_args(argv)
    dataset = load_grouped_dataset(
        args.metadata,
        args.proteome,
        missing_rate_threshold=args.missing_rate_threshold,
        cache_dir=args.cache_dir,
        interpolation_fraction=args.interpolation_fraction,
        split_seed=args.seed,
        control_policy=args.control_policy,
        vehicle_column=args.vehicle_column,
    )
    paths = write_audit_outputs(dataset, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
