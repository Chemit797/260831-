"""Condition-atomic data loading for the GOAI active-learning benchmark.

Matched Water/DMSO controls are assay overhead: they are used only to build the
retrospective log2-delta oracle and never become query candidates or features.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SAMPLE_ID = "sample_ID"
SPLIT = "split_final"
CONDITION_ID = "condition_id"
STRAIN = "Strains"
CHEMICAL = "perturbation_no_concentration"
MEDIUM = "Medium"
TEMPERATURE = "Temperature"
TIME = "pert_time"
TIME_UNIT = "pert_time_unit"
DATA_SOURCE = "data_source"
INSTRUMENT = "instrument"
PLATE = "Yeast_cell_plate"

# Split is deliberately absent.  These are the complete biological query keys.
GROUP_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE, TIME, TIME_UNIT)
FEATURE_FIELDS = GROUP_FIELDS
CATEGORICAL_FEATURE_FIELDS = (STRAIN, CHEMICAL, MEDIUM, TEMPERATURE)
MATCH_CONTROL_FIELDS = (
    DATA_SOURCE,
    INSTRUMENT,
    PLATE,
    STRAIN,
    MEDIUM,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
)
CONTROL_NAMES = frozenset({"water", "dmso"})
QUALITY_CONTROL_NAME = "quality control"
POOLED_EXACT_CONTEXT_WATER_DMSO = "pooled_exact_context_water_dmso"
STRICT_EXPLICIT_VEHICLE = "strict_explicit_vehicle"
CONTROL_POLICIES = (
    POOLED_EXACT_CONTEXT_WATER_DMSO,
    STRICT_EXPLICIT_VEHICLE,
)
DEFAULT_CONTROL_POLICY = POOLED_EXACT_CONTEXT_WATER_DMSO
VALIDATION_SPLITS = (
    "val_chem_only",
    "val_strain_only",
    "val_both",
    "val_time",
)
INTERPOLATION_SPLIT = "interpolation"
PROTOCOL_VERSION = "goai-condition-atomic-v2.1"
CACHE_VERSION = "grouped-dataset-v4"
DEFAULT_INTERPOLATION_FRACTION = 0.20
DEFAULT_SPLIT_SEED = 42
_MISSING_CATEGORY = "__GOAI_MISSING__"
_FORBIDDEN_VEHICLE_COLUMNS = frozenset(
    {CHEMICAL, "pert_id", DATA_SOURCE, PLATE, "protein_well", SAMPLE_ID}
)
_SENSITIVITY_COLUMNS = (
    *MATCH_CONTROL_FIELDS,
    "water_control_count",
    "dmso_control_count",
    "water_control_ids",
    "dmso_control_ids",
    "finite_protein_count",
    "delta_water_minus_delta_dmso_mean",
    "delta_water_minus_delta_dmso_median",
    "delta_water_minus_delta_dmso_mean_abs",
    "delta_water_minus_delta_dmso_rmse",
    "delta_water_minus_delta_dmso_max_abs",
    "sensitivity_definition",
    "audit_role",
    "acquisition_input",
)


@dataclass(frozen=True)
class BenchmarkSplit:
    """Frozen condition-level candidate pool and evaluation contract."""

    candidate_pool_ids: tuple[str, ...]
    interpolation_ids: tuple[str, ...]
    validation_ids: Mapping[str, tuple[str, ...]]
    removed_validation_overlap: Mapping[str, tuple[str, ...]]
    official_train_ids: tuple[str, ...]
    seed: int = DEFAULT_SPLIT_SEED
    interpolation_fraction: float = DEFAULT_INTERPOLATION_FRACTION

    @property
    def pool_ids(self) -> tuple[str, ...]:
        """Alias used by acquisition code and older callers."""

        return self.candidate_pool_ids

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return self.candidate_pool_ids


@dataclass(frozen=True)
class GroupedDataset:
    metadata: pd.DataFrame
    response: pd.DataFrame
    proteins: tuple[str, ...]
    # Backward-compatible name.  In v2 this is the candidate pool, not all
    # official-train conditions.
    train_ids: pd.Index
    validation_ids: dict[str, pd.Index]
    protein_missing_rate: pd.Series
    benchmark_split: BenchmarkSplit | None = None
    control_provenance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_split_overlap: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    cache_key: str | None = None
    cache_hit: bool = False
    # v2.2 additions stay optional so manually constructed v2.1 fixtures and
    # downstream positional callers retain their public behavior.
    control_policy_summary: Mapping[str, object] = field(default_factory=dict)
    control_vehicle_sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def candidate_pool_ids(self) -> pd.Index:
        return self.train_ids

    @property
    def official_train_ids(self) -> pd.Index:
        if self.benchmark_split is None:
            return self.train_ids
        return pd.Index(self.benchmark_split.official_train_ids, name=CONDITION_ID)

    @property
    def removed_validation_overlap(self) -> Mapping[str, tuple[str, ...]]:
        if self.benchmark_split is None:
            return {}
        return self.benchmark_split.removed_validation_overlap


class PoolFeatureEncoder:
    """Target-free, field-balanced descriptors fitted on pool metadata only.

    Categorical fields each contribute one one-hot block, irrespective of their
    cardinality.  Time contributes one pool-normalized continuous column.  This
    keeps descriptor geometry balanced by metadata field and uses no response.
    """

    categorical_fields = CATEGORICAL_FEATURE_FIELDS
    continuous_field = TIME

    def __init__(self) -> None:
        self.categories: dict[str, list[str]] = {}
        self.categorical_column_slices: dict[str, slice] = {}
        self.time_mean_: float | None = None
        self.time_scale_: float | None = None

    @staticmethod
    def _categorical_values(metadata: pd.DataFrame, field_name: str) -> pd.Series:
        if field_name not in metadata:
            return pd.Series(_MISSING_CATEGORY, index=metadata.index, dtype="string")
        values = metadata[field_name].astype("string").fillna(_MISSING_CATEGORY)
        return values.astype(str)

    @staticmethod
    def _time_values(metadata: pd.DataFrame) -> np.ndarray:
        if TIME not in metadata:
            raise ValueError(f"Feature metadata is missing {TIME}")
        values = pd.to_numeric(metadata[TIME], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{TIME} must contain finite numeric values")
        if TIME_UNIT not in metadata:
            raise ValueError(f"Feature metadata is missing {TIME_UNIT}")
        units = metadata[TIME_UNIT].astype("string").str.strip().str.casefold()
        multipliers = units.map(
            {
                "s": 1.0 / 60.0,
                "sec": 1.0 / 60.0,
                "second": 1.0 / 60.0,
                "seconds": 1.0 / 60.0,
                "m": 1.0,
                "min": 1.0,
                "minute": 1.0,
                "minutes": 1.0,
                "h": 60.0,
                "hr": 60.0,
                "hour": 60.0,
                "hours": 60.0,
                "d": 1440.0,
                "day": 1440.0,
                "days": 1440.0,
            }
        )
        unknown = multipliers.isna()
        if unknown.any():
            unknown_units = sorted(
                {str(value) for value in units.loc[unknown].fillna("<missing>")}
            )
            raise ValueError(f"Unknown {TIME_UNIT} values: {unknown_units}")
        values = values * multipliers.to_numpy(dtype=np.float64)
        return values

    def fit(self, metadata: pd.DataFrame) -> "PoolFeatureEncoder":
        """Fit vocabulary and time normalization using candidate-pool metadata."""

        if len(metadata) == 0:
            raise ValueError("Cannot fit PoolFeatureEncoder on an empty pool")
        self.categories = {
            field_name: sorted(self._categorical_values(metadata, field_name).unique().tolist())
            for field_name in self.categorical_fields
        }
        self.categorical_column_slices = {}
        start = 0
        for field_name in self.categorical_fields:
            stop = start + len(self.categories[field_name])
            self.categorical_column_slices[field_name] = slice(start, stop)
            start = stop
        times = self._time_values(metadata)
        self.time_mean_ = float(times.mean())
        scale = float(times.std(ddof=0))
        self.time_scale_ = scale if scale > 0.0 else 1.0
        return self

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        if not self.categories or self.time_mean_ is None or self.time_scale_ is None:
            raise RuntimeError("PoolFeatureEncoder must be fit before transform")
        blocks: list[np.ndarray] = []
        for field_name in self.categorical_fields:
            categories = self.categories[field_name]
            lookup = {value: column for column, value in enumerate(categories)}
            positions = self._categorical_values(metadata, field_name).map(lookup)
            block = np.zeros((len(metadata), len(categories)), dtype=np.float32)
            valid = positions.notna().to_numpy()
            if valid.any():
                rows = np.flatnonzero(valid)
                block[rows, positions.iloc[rows].astype(int).to_numpy()] = 1.0
            blocks.append(block)
        time_block = (
            (self._time_values(metadata) - self.time_mean_) / self.time_scale_
        ).astype(np.float32)[:, None]
        blocks.append(time_block)
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def fit_transform(self, metadata: pd.DataFrame) -> np.ndarray:
        return self.fit(metadata).transform(metadata)

    def mask_unsupported(
        self,
        features: np.ndarray,
        supported_features: np.ndarray,
    ) -> np.ndarray:
        """Mask unseen categorical columns while always retaining normalized time."""

        return mask_unsupported(
            features,
            supported_features,
            self.categorical_column_slices,
        )

    @property
    def continuous_column_slice(self) -> slice:
        start = sum(len(values) for values in self.categories.values())
        return slice(start, start + 1)

    @property
    def output_dim(self) -> int:
        return sum(len(values) for values in self.categories.values()) + 1


def mask_unsupported(
    features: np.ndarray,
    supported_features: np.ndarray,
    categorical_column_slices: Mapping[str, slice],
) -> np.ndarray:
    """Zero unsupported one-hot columns and leave all noncategorical columns intact."""

    all_values = np.asarray(features)
    support = np.asarray(supported_features)
    if all_values.ndim != 2 or support.ndim != 2:
        raise ValueError("features and supported_features must be two-dimensional")
    if all_values.shape[1] != support.shape[1]:
        raise ValueError("features and supported_features must have the same width")
    result = all_values.copy()
    for column_slice in categorical_column_slices.values():
        if column_slice.step not in (None, 1):
            raise ValueError("Categorical column slices must be contiguous")
        supported = np.any(support[:, column_slice] != 0.0, axis=0)
        block = result[:, column_slice]
        block[:, ~supported] = 0.0
    return result


def _normalised_names(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.casefold()


def _control_mask(metadata: pd.DataFrame) -> pd.Series:
    return _normalised_names(metadata[CHEMICAL]).isin(CONTROL_NAMES)


def _treatment_mask(metadata: pd.DataFrame) -> pd.Series:
    names = _normalised_names(metadata[CHEMICAL])
    return ~(names.isin(CONTROL_NAMES) | names.eq(QUALITY_CONTROL_NAME))


def _keys(metadata: pd.DataFrame, fields: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(metadata.loc[:, list(fields)].astype(str))


def _key_tuples(metadata: pd.DataFrame, fields: Sequence[str]) -> list[tuple[str, ...]]:
    values = metadata.loc[:, list(fields)].astype(str).to_numpy()
    return [tuple(row) for row in values]


def stable_condition_id(values: Sequence[object] | Mapping[str, object] | pd.Series) -> str:
    """Return a stable biological condition ID that is independent of split."""

    if isinstance(values, Mapping) or isinstance(values, pd.Series):
        ordered = [values[field_name] for field_name in GROUP_FIELDS]
    else:
        ordered = list(values)
        if len(ordered) != len(GROUP_FIELDS):
            raise ValueError(f"A condition requires {len(GROUP_FIELDS)} values")
    payload = json.dumps(
        [str(value) for value in ordered],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"condition__{digest}"


def _split_values(row: pd.Series) -> tuple[str, ...]:
    if "split_provenance" in row:
        value = row["split_provenance"]
        if isinstance(value, (tuple, list, set, np.ndarray)):
            return tuple(sorted(str(item) for item in value))
        return tuple(item for item in str(value).split("|") if item)
    if SPLIT in row:
        return (str(row[SPLIT]),)
    raise ValueError(f"Metadata requires {SPLIT} or split_provenance")


def detect_cross_split_condition_overlap(metadata: pd.DataFrame) -> pd.DataFrame:
    """Report biological conditions present in more than one official split."""

    # Grouped condition metadata retains all legacy memberships here; prefer it
    # over the backward-compatible primary split column.
    if "split_provenance" in metadata:
        rows = []
        for condition_id, row in metadata.iterrows():
            splits = _split_values(row)
            if len(splits) > 1:
                rows.append(
                    {
                        CONDITION_ID: str(condition_id),
                        "split_provenance": "|".join(splits),
                        "split_count": len(splits),
                        "measurement_replicate_count": int(
                            row.get("released_replicate_count", row.get("replicate_count", 1))
                        ),
                        "official_train_overlap": "train" in splits,
                    }
                )
        return pd.DataFrame(rows).set_index(CONDITION_ID) if rows else _empty_overlap_frame()

    required = set(GROUP_FIELDS) | {SPLIT}
    if required <= set(metadata.columns):
        work = metadata.loc[:, [SPLIT, *GROUP_FIELDS]].copy()
        work[CONDITION_ID] = [
            stable_condition_id(values)
            for values in work.loc[:, GROUP_FIELDS].itertuples(index=False, name=None)
        ]
        grouped = work.groupby(CONDITION_ID, sort=True)[SPLIT].agg(
            lambda values: tuple(sorted(set(values.astype(str))))
        )
        counts = work.groupby(CONDITION_ID, sort=True).size()
        rows = [
            {
                CONDITION_ID: condition_id,
                "split_provenance": "|".join(splits),
                "split_count": len(splits),
                "measurement_replicate_count": int(counts.loc[condition_id]),
                "official_train_overlap": "train" in splits and len(splits) > 1,
            }
            for condition_id, splits in grouped.items()
            if len(splits) > 1
        ]
        return pd.DataFrame(rows).set_index(CONDITION_ID) if rows else _empty_overlap_frame()

    raise ValueError(f"Metadata requires {SPLIT} plus condition fields, or split_provenance")


def _empty_overlap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "split_provenance",
            "split_count",
            "measurement_replicate_count",
            "official_train_overlap",
        ],
        index=pd.Index([], name=CONDITION_ID),
    )


def deterministic_interpolation_split(
    metadata: pd.DataFrame,
    holdout_fraction: float = DEFAULT_INTERPOLATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    factor_fields: Sequence[str] = GROUP_FIELDS,
) -> tuple[pd.Index, pd.Index]:
    """Split conditions using metadata alone while preserving pool factor levels.

    The first returned index is the candidate pool and the second is the fixed
    interpolation holdout.  If the requested size is incompatible with retaining
    every observed factor level, the largest level-preserving holdout found is used.
    """

    if not 0.0 <= float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    missing = sorted(set(factor_fields) - set(metadata.columns))
    if missing:
        raise ValueError(f"Split metadata is missing factor fields: {missing}")
    if metadata.index.duplicated().any():
        raise ValueError("Condition metadata index must be unique")
    ids = np.asarray(sorted(metadata.index.astype(str)), dtype=str)
    n_conditions = len(ids)
    if n_conditions == 0:
        empty = pd.Index([], name=CONDITION_ID, dtype=str)
        return empty, empty.copy()
    target = int(np.floor(n_conditions * float(holdout_fraction) + 0.5))
    if holdout_fraction > 0.0 and n_conditions > 1:
        target = max(1, target)
    target = min(target, n_conditions - 1)
    if target == 0:
        return pd.Index(ids, name=CONDITION_ID), pd.Index([], name=CONDITION_ID, dtype=str)

    aligned = metadata.copy()
    aligned.index = aligned.index.astype(str)
    values = aligned.loc[ids, list(factor_fields)].astype(str).to_numpy()
    best: list[int] = []
    # Multiple deterministic attempts avoid avoidable greedy dead ends while the
    # acceptance rule guarantees preservation in every returned pool.
    rng = np.random.default_rng(int(seed))
    attempts = max(32, min(512, n_conditions * 2))
    for _ in range(attempts):
        remaining = [
            dict(pd.Series(values[:, column]).value_counts().astype(int))
            for column in range(values.shape[1])
        ]
        selected: list[int] = []
        for position in rng.permutation(n_conditions):
            if all(remaining[column][values[position, column]] > 1 for column in range(values.shape[1])):
                selected.append(int(position))
                for column in range(values.shape[1]):
                    remaining[column][values[position, column]] -= 1
                if len(selected) == target:
                    break
        if len(selected) > len(best):
            best = selected
        if len(best) == target:
            break
    holdout_set = {ids[position] for position in best}
    pool = pd.Index([item for item in ids if item not in holdout_set], name=CONDITION_ID)
    holdout = pd.Index(sorted(holdout_set), name=CONDITION_ID)
    return pool, holdout


def build_benchmark_split(
    metadata: pd.DataFrame,
    holdout_fraction: float = DEFAULT_INTERPOLATION_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> BenchmarkSplit:
    """Build the frozen condition-atomic split using metadata and no response."""

    if metadata.index.duplicated().any():
        raise ValueError("Condition metadata index must be unique")
    membership = {str(condition_id): _split_values(row) for condition_id, row in metadata.iterrows()}
    official_train = tuple(sorted(key for key, splits in membership.items() if "train" in splits))
    if not official_train:
        raise ValueError("No official train conditions were found")
    aligned = metadata.copy()
    aligned.index = aligned.index.astype(str)
    pool, interpolation = deterministic_interpolation_split(
        aligned.loc[list(official_train)],
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    validation: dict[str, tuple[str, ...]] = {INTERPOLATION_SPLIT: tuple(interpolation.astype(str))}
    removed: dict[str, tuple[str, ...]] = {}
    official_train_set = set(official_train)
    for split_name in VALIDATION_SPLITS:
        raw = {key for key, splits in membership.items() if split_name in splits}
        overlap = tuple(sorted(raw & official_train_set))
        removed[split_name] = overlap
        validation[split_name] = tuple(sorted(raw - official_train_set))
    if set(pool) & set(interpolation):
        raise AssertionError("Candidate pool and interpolation holdout overlap")
    for split_name, ids in validation.items():
        if set(pool) & set(ids):
            raise AssertionError(f"Candidate pool overlaps evaluation split {split_name}")
    return BenchmarkSplit(
        candidate_pool_ids=tuple(pool.astype(str)),
        interpolation_ids=tuple(interpolation.astype(str)),
        validation_ids=validation,
        removed_validation_overlap=removed,
        official_train_ids=official_train,
        seed=int(seed),
        interpolation_fraction=float(holdout_fraction),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_string_array_sha256(values: np.ndarray) -> str:
    """Hash a one-dimensional Unicode array independently of NumPy storage."""

    if values.ndim != 1 or values.dtype.kind != "U":
        raise ValueError("Canonical string-array digests require a 1D Unicode array")
    encoded = json.dumps(
        values.tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _canonical_numeric_array_sha256(values: np.ndarray) -> str:
    """Hash real numeric values as contiguous little-endian float64 bytes."""

    if values.ndim != 1 or values.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("Canonical numeric-array digests require a 1D real array")
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return _sha256_bytes(canonical.tobytes(order="C"))


def _source_file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = _sha256(resolved)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"Source file changed while it was being hashed: {resolved}")
    return {
        "resolved_path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest,
    }


def _validated_control_settings(
    control_policy: str,
    vehicle_column: str | None,
) -> tuple[str, str | None]:
    if control_policy not in CONTROL_POLICIES:
        raise ValueError(
            f"control_policy must be one of {CONTROL_POLICIES}, got {control_policy!r}"
        )
    if (
        control_policy == POOLED_EXACT_CONTEXT_WATER_DMSO
        and vehicle_column is not None
    ):
        raise ValueError(
            "pooled_exact_context_water_dmso rejects vehicle_column because it is "
            "irrelevant and misleading for the pooled policy"
        )
    if vehicle_column is not None:
        if not isinstance(vehicle_column, str) or not vehicle_column.strip():
            raise ValueError("vehicle_column must be a nonempty column name or None")
        vehicle_column = vehicle_column.strip()
    if control_policy == STRICT_EXPLICIT_VEHICLE:
        if vehicle_column is None:
            raise ValueError("strict_explicit_vehicle requires an explicit vehicle_column")
        if vehicle_column.casefold() in {
            name.casefold() for name in _FORBIDDEN_VEHICLE_COLUMNS
        }:
            raise ValueError(
                "strict_explicit_vehicle cannot infer vehicle from chemical, pert_id, "
                "source, plate, or well columns"
            )
    return control_policy, vehicle_column


def _policy_summary_json(summary: Mapping[str, object]) -> str:
    """Serialize and validate the policy receipt as strict, detached JSON."""

    try:
        encoded = json.dumps(
            dict(summary),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("Control policy summary must be JSON-safe") from error
    if not isinstance(decoded, dict):
        raise ValueError("Control policy summary must be a JSON object")
    return encoded


def _validated_cached_policy_summary(
    encoded: np.ndarray,
    control_policy: str,
    vehicle_column: str | None,
) -> dict[str, object]:
    try:
        summary = json.loads(str(encoded.item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Cached control policy summary is invalid JSON") from error
    required = {
        "control_policy",
        "vehicle_column",
        "vehicle_mapping_state",
        "match_control_fields",
        "control_aggregation",
        "control_type_means_averaged_equally",
        "selected_control_rule",
        "pooled_across_types",
        "vehicle_inference",
        "treatment_vehicle_values_validated",
        "nonblank_control_vehicle_identities_validated",
        "blank_control_vehicle_values_allowed",
        "control_availability",
        "treatment_availability",
    }
    if not isinstance(summary, dict) or not required <= set(summary):
        raise ValueError("Cached control policy summary is incomplete")
    if summary["control_policy"] != control_policy:
        raise ValueError("Cached control policy differs from the requested policy")
    if summary["vehicle_column"] != vehicle_column:
        raise ValueError("Cached vehicle column differs from the requested column")
    if summary["vehicle_inference"] is not False:
        raise ValueError("Cached policy must state vehicle_inference=false")
    expected_pooled = control_policy == POOLED_EXACT_CONTEXT_WATER_DMSO
    if summary["pooled_across_types"] is not expected_pooled:
        raise ValueError("Cached policy has an invalid pooled-across-types state")
    expected_mapping_state = (
        "not_applicable_pooled_policy"
        if expected_pooled
        else "explicit_column_validated"
    )
    if summary["vehicle_mapping_state"] != expected_mapping_state:
        raise ValueError("Cached policy has an invalid vehicle mapping state")
    expected_explicit_validation = not expected_pooled
    if (
        summary["treatment_vehicle_values_validated"]
        is not expected_explicit_validation
        or summary["nonblank_control_vehicle_identities_validated"]
        is not expected_explicit_validation
        or summary["blank_control_vehicle_values_allowed"]
        is not expected_explicit_validation
    ):
        raise ValueError("Cached policy has invalid explicit-vehicle validation state")
    if summary["control_aggregation"] != "direct_measurement_mean_in_log2_space":
        raise ValueError("Cached policy has an invalid log2 aggregation rule")
    if summary["control_type_means_averaged_equally"] is not False:
        raise ValueError("Cached policy cannot equally average control-type means")
    expected_selection = (
        "all_exact_context_water_dmso_measurements"
        if expected_pooled
        else "exact_context_requested_explicit_vehicle_measurements"
    )
    if summary["selected_control_rule"] != expected_selection:
        raise ValueError("Cached policy has an invalid selected-control rule")
    if list(summary["match_control_fields"]) != list(MATCH_CONTROL_FIELDS):
        raise ValueError("Cached policy has different exact control matching fields")
    for section_name, total_name in (
        ("control_availability", "context_count"),
        ("treatment_availability", "measurement_count"),
    ):
        section = summary[section_name]
        category_names = ("both", "dmso_only", "water_only", "none")
        if not isinstance(section, dict) or not {
            total_name,
            "measurement_count",
            *category_names,
        } <= set(section):
            raise ValueError(f"Cached {section_name} summary is incomplete")
        counts = [section[name] for name in category_names]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*counts, section[total_name], section["measurement_count"])
        ):
            raise ValueError(f"Cached {section_name} counts are invalid")
        if sum(counts) != section[total_name]:
            raise ValueError(f"Cached {section_name} categories do not conserve count")
    _policy_summary_json(summary)
    return summary


def _empty_sensitivity_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_SENSITIVITY_COLUMNS))


def _validated_sensitivity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_SENSITIVITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Cached control sensitivity table is missing columns: {missing}")
    result = frame.loc[:, list(_SENSITIVITY_COLUMNS)].copy()
    if len(result):
        if result.loc[:, list(MATCH_CONTROL_FIELDS)].astype(str).duplicated().any():
            raise ValueError("Cached control sensitivity contexts are not unique")
        if not result["audit_role"].astype(str).eq("posthoc_oracle_audit").all():
            raise ValueError("Cached control sensitivity has an invalid audit role")
        expected_definition = (
            "delta_if_water_minus_delta_if_dmso_equals_"
            "dmso_control_mean_minus_water_control_mean"
        )
        if not result["sensitivity_definition"].astype(str).eq(
            expected_definition
        ).all():
            raise ValueError("Cached control sensitivity has an invalid definition")
        acquisition = result["acquisition_input"]
        if not acquisition.map(lambda value: value is False or value == 0).all():
            raise ValueError("Cached control sensitivity cannot be an acquisition input")
        for control_type in ("water", "dmso"):
            count_column = f"{control_type}_control_count"
            ids_column = f"{control_type}_control_ids"
            counts = pd.to_numeric(result[count_column], errors="coerce")
            if counts.isna().any() or counts.le(0).any():
                raise ValueError("Cached control sensitivity has invalid control counts")
            for expected_count, encoded_ids in zip(counts, result[ids_column]):
                try:
                    ids = json.loads(str(encoded_ids))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Cached control sensitivity has invalid control IDs"
                    ) from error
                if (
                    not isinstance(ids, list)
                    or len(ids) != int(expected_count)
                    or len(set(ids)) != len(ids)
                    or any(not isinstance(value, str) or not value for value in ids)
                ):
                    raise ValueError(
                        "Cached control sensitivity IDs do not match control counts"
                    )
    return result


def _cache_identity(
    metadata_path: Path,
    proteome_path: Path,
    missing_rate_threshold: float,
    holdout_fraction: float,
    seed: int,
    control_policy: str = DEFAULT_CONTROL_POLICY,
    vehicle_column: str | None = None,
) -> tuple[str, dict[str, str], dict[str, object]]:
    control_policy, vehicle_column = _validated_control_settings(
        control_policy, vehicle_column
    )
    source_files = {
        "metadata": _source_file_record(metadata_path),
        "proteome": _source_file_record(proteome_path),
    }
    source_hashes = {
        "metadata_sha256": str(source_files["metadata"]["sha256"]),
        "proteome_sha256": str(source_files["proteome"]["sha256"]),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "cache_version": CACHE_VERSION,
        "source_hashes": source_hashes,
        "source_files": source_files,
        "missing_rate_threshold": float(missing_rate_threshold),
        "interpolation_fraction": float(holdout_fraction),
        "split_seed": int(seed),
        "control_policy": control_policy,
        "vehicle_column": vehicle_column,
    }
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("Cache identity inputs must be finite and JSON-safe") from error
    return hashlib.sha256(encoded).hexdigest(), source_hashes, payload


def _cache_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Cache manifest values must be finite and JSON-safe") from error


def _cached_json_object(encoded: np.ndarray, role: str) -> dict[str, object]:
    try:
        value = json.loads(str(encoded.item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Cached {role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Cached {role} must be a JSON object")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_cached_digest_scalar(
    encoded: np.ndarray,
    expected: str,
    role: str,
) -> None:
    if (
        encoded.shape != ()
        or encoded.dtype.kind != "U"
        or encoded.item() != expected
    ):
        raise ValueError(f"Cached {role} digest differs from its manifest")


def _validated_cache_manifest(
    encoded: np.ndarray,
    *,
    path: Path,
    cache_key: str,
    expected_inputs: Mapping[str, object],
    digest_names: Sequence[str],
) -> dict[str, str]:
    manifest = _cached_json_object(encoded, "cache manifest")
    required = {*expected_inputs, "cache_key", "artifact_digests"}
    if set(manifest) != required:
        raise ValueError("Cached manifest fields differ from the v4 contract")
    if manifest["cache_key"] != cache_key:
        raise ValueError("Cached manifest key differs from the requested cache key")
    if path.name != f"grouped_{cache_key}.npz":
        raise ValueError("Grouped-data cache filename differs from its cache key")

    source_hashes = manifest.get("source_hashes")
    source_files = manifest.get("source_files")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != {"metadata_sha256", "proteome_sha256"}
        or not all(_valid_sha256(value) for value in source_hashes.values())
    ):
        raise ValueError("Cached manifest source hashes are invalid")
    if not isinstance(source_files, dict) or set(source_files) != {
        "metadata",
        "proteome",
    }:
        raise ValueError("Cached manifest source file records are invalid")
    for record in source_files.values():
        if (
            not isinstance(record, dict)
            or set(record)
            != {"resolved_path", "size_bytes", "mtime_ns", "sha256"}
            or not isinstance(record["resolved_path"], str)
            or not record["resolved_path"]
            or isinstance(record["size_bytes"], bool)
            or not isinstance(record["size_bytes"], int)
            or record["size_bytes"] < 0
            or isinstance(record["mtime_ns"], bool)
            or not isinstance(record["mtime_ns"], int)
            or not _valid_sha256(record["sha256"])
        ):
            raise ValueError("Cached manifest has an invalid source file record")
    if any(
        not isinstance(manifest[name], float)
        or not np.isfinite(manifest[name])
        for name in ("missing_rate_threshold", "interpolation_fraction")
    ):
        raise ValueError("Cached manifest numeric thresholds are invalid")
    if isinstance(manifest["split_seed"], bool) or not isinstance(
        manifest["split_seed"], int
    ):
        raise ValueError("Cached manifest split seed is invalid")
    if not isinstance(manifest["control_policy"], str) or not (
        manifest["vehicle_column"] is None
        or isinstance(manifest["vehicle_column"], str)
    ):
        raise ValueError("Cached manifest control settings are invalid")
    for name, expected in expected_inputs.items():
        if manifest[name] != expected:
            raise ValueError(f"Cached manifest input {name!r} differs")

    digests = manifest["artifact_digests"]
    if (
        not isinstance(digests, dict)
        or set(digests) != set(digest_names)
        or not all(_valid_sha256(value) for value in digests.values())
    ):
        raise ValueError("Cached manifest artifact digests are invalid")
    return {name: str(value) for name, value in digests.items()}


def _frame_to_json(frame: pd.DataFrame) -> str:
    return frame.to_json(orient="table", index=True, double_precision=15)


def _frame_from_json(value: np.ndarray) -> pd.DataFrame:
    return pd.read_json(io.StringIO(str(value.item())), orient="table")


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _dataset_from_cache(
    path: Path,
    cache_key: str,
    source_hashes: Mapping[str, str],
    cache_inputs: Mapping[str, object],
    missing_rate_threshold: float,
    holdout_fraction: float,
    seed: int,
    control_policy: str = DEFAULT_CONTROL_POLICY,
    vehicle_column: str | None = None,
) -> GroupedDataset:
    control_policy, vehicle_column = _validated_control_settings(
        control_policy, vehicle_column
    )
    serialized_artifacts = {
        "metadata_json_sha256": "metadata_json",
        "control_provenance_json_sha256": "control_provenance_json",
        "overlap_json_sha256": "overlap_json",
        "control_policy_summary_json_sha256": "control_policy_summary_json",
        "control_vehicle_sensitivity_json_sha256": (
            "control_vehicle_sensitivity_json"
        ),
    }
    array_artifacts = {
        "proteins_sha256": "proteins",
        "missing_rate_names_sha256": "missing_rate_names",
        "missing_rate_values_sha256": "missing_rate_values",
    }
    digest_names = (
        "response_sha256",
        *serialized_artifacts,
        *array_artifacts,
    )
    with np.load(path, allow_pickle=False) as cached:
        required_arrays = {
            "protocol_version",
            "cache_version",
            "cache_key",
            "cache_manifest_json",
            "source_hashes_json",
            "source_files_json",
            "missing_rate_threshold",
            "interpolation_fraction",
            "split_seed",
            "control_policy",
            "vehicle_column_json",
            "metadata_json",
            "response",
            "proteins",
            "missing_rate_names",
            "missing_rate_values",
            "control_provenance_json",
            "overlap_json",
            "control_policy_summary_json",
            "control_vehicle_sensitivity_json",
            *digest_names,
        }
        missing_arrays = sorted(required_arrays - set(cached.files))
        if missing_arrays:
            raise ValueError(
                f"Grouped-data cache is missing v4 fields: {missing_arrays}"
            )
        if str(cached["protocol_version"].item()) != PROTOCOL_VERSION:
            raise ValueError("Cache protocol version differs")
        if str(cached["cache_version"].item()) != CACHE_VERSION:
            raise ValueError("Grouped-data cache version differs")
        if str(cached["cache_key"].item()) != cache_key:
            raise ValueError("Grouped-data cache key differs")
        artifact_digests = _validated_cache_manifest(
            cached["cache_manifest_json"],
            path=path,
            cache_key=cache_key,
            expected_inputs=cache_inputs,
            digest_names=digest_names,
        )
        if _cached_json_object(cached["source_hashes_json"], "source hashes") != dict(
            cache_inputs["source_hashes"]  # type: ignore[arg-type]
        ):
            raise ValueError("Cached source hashes differ")
        if _cached_json_object(cached["source_files_json"], "source files") != dict(
            cache_inputs["source_files"]  # type: ignore[arg-type]
        ):
            raise ValueError("Cached source file records differ")
        cached_missing_rate_threshold = cached["missing_rate_threshold"]
        if (
            cached_missing_rate_threshold.shape != ()
            or cached_missing_rate_threshold.dtype != np.dtype(np.float64)
            or cached_missing_rate_threshold.item()
            != float(missing_rate_threshold)
        ):
            raise ValueError("Cached missing-rate threshold differs")
        persisted_missing_rate_threshold = float(
            cached_missing_rate_threshold.item()
        )
        cached_interpolation_fraction = cached["interpolation_fraction"]
        if (
            cached_interpolation_fraction.shape != ()
            or cached_interpolation_fraction.dtype != np.dtype(np.float64)
            or cached_interpolation_fraction.item() != float(holdout_fraction)
        ):
            raise ValueError("Cached interpolation fraction differs")
        cached_split_seed = cached["split_seed"]
        if (
            cached_split_seed.shape != ()
            or cached_split_seed.dtype != np.dtype(np.int64)
            or cached_split_seed.item() != int(seed)
        ):
            raise ValueError("Cached split seed differs")
        cached_control_policy = cached["control_policy"]
        if (
            cached_control_policy.shape != ()
            or cached_control_policy.dtype.kind != "U"
            or cached_control_policy.item() != control_policy
        ):
            raise ValueError("Cached control policy differs")
        cached_vehicle_column_json = cached["vehicle_column_json"]
        if (
            cached_vehicle_column_json.shape != ()
            or cached_vehicle_column_json.dtype.kind != "U"
        ):
            raise ValueError("Cached vehicle column has an invalid representation")
        try:
            cached_vehicle_column = json.loads(
                cached_vehicle_column_json.item()
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Cached vehicle column is invalid JSON") from error
        if cached_vehicle_column != vehicle_column:
            raise ValueError("Cached vehicle column differs")

        response_values = cached["response"]
        if response_values.dtype != np.dtype(np.float32) or response_values.ndim != 2:
            raise ValueError("Cached response must be a two-dimensional float32 array")
        if (
            _sha256_bytes(response_values.tobytes(order="C"))
            != artifact_digests["response_sha256"]
        ):
            raise ValueError("Cached response SHA256 mismatch")
        _validate_cached_digest_scalar(
            cached["response_sha256"],
            artifact_digests["response_sha256"],
            "response",
        )

        serialized_values: dict[str, str] = {}
        for digest_name, array_name in serialized_artifacts.items():
            try:
                serialized = str(cached[array_name].item())
            except (AttributeError, ValueError) as error:
                raise ValueError(
                    f"Cached serialized artifact {array_name!r} is not scalar"
                ) from error
            if (
                _sha256_bytes(serialized.encode("utf-8"))
                != artifact_digests[digest_name]
            ):
                raise ValueError(f"Cached {array_name} SHA256 mismatch")
            _validate_cached_digest_scalar(
                cached[digest_name],
                artifact_digests[digest_name],
                array_name,
            )
            serialized_values[array_name] = serialized

        protein_values = cached["proteins"]
        if protein_values.ndim != 1 or protein_values.dtype.kind != "U":
            raise ValueError("Cached proteins must be a one-dimensional string array")
        missing_rate_names = cached["missing_rate_names"]
        if missing_rate_names.ndim != 1 or missing_rate_names.dtype.kind != "U":
            raise ValueError(
                "Cached missing-rate names must be a one-dimensional string array"
            )
        missing_rate_values = cached["missing_rate_values"]
        if (
            missing_rate_values.ndim != 1
            or missing_rate_values.dtype.kind not in {"i", "u", "f"}
        ):
            raise ValueError(
                "Cached missing-rate values must be a one-dimensional numeric array"
            )
        array_digests = {
            "proteins_sha256": _canonical_string_array_sha256(protein_values),
            "missing_rate_names_sha256": _canonical_string_array_sha256(
                missing_rate_names
            ),
            "missing_rate_values_sha256": _canonical_numeric_array_sha256(
                missing_rate_values
            ),
        }
        for digest_name, observed_digest in array_digests.items():
            array_name = array_artifacts[digest_name]
            if observed_digest != artifact_digests[digest_name]:
                raise ValueError(f"Cached {array_name} SHA256 mismatch")
            _validate_cached_digest_scalar(
                cached[digest_name],
                artifact_digests[digest_name],
                array_name,
            )

        metadata = _frame_from_json(np.asarray(serialized_values["metadata_json"]))
        metadata.index = pd.Index(metadata.index.astype(str), name=CONDITION_ID)
        proteins = tuple(protein_values.tolist())
        if (
            metadata.index.duplicated().any()
            or not proteins
            or any(not protein for protein in proteins)
            or len(set(proteins)) != len(proteins)
            or response_values.shape[0] != len(metadata)
            or response_values.shape[1] != len(proteins)
        ):
            raise ValueError("Cached response schema is inconsistent")
        missing_rate_name_values = tuple(missing_rate_names.tolist())
        if (
            not missing_rate_name_values
            or any(not name for name in missing_rate_name_values)
            or len(set(missing_rate_name_values)) != len(missing_rate_name_values)
        ):
            raise ValueError(
                "Cached missing-rate names must be nonempty unique strings"
            )
        if len(missing_rate_values) != len(missing_rate_name_values):
            raise ValueError(
                "Cached missing-rate values must align with missing-rate names"
            )
        missing_rate_float64 = missing_rate_values.astype(np.float64)
        if (
            not np.isfinite(missing_rate_float64).all()
            or (missing_rate_float64 < 0.0).any()
            or (missing_rate_float64 > 1.0).any()
        ):
            raise ValueError(
                "Cached missing-rate values must be finite and within [0, 1]"
            )
        expected_proteins = tuple(
            name
            for name, rate in zip(
                missing_rate_name_values,
                missing_rate_float64,
                strict=True,
            )
            if rate < persisted_missing_rate_threshold
        )
        if proteins != expected_proteins:
            raise ValueError(
                "Cached proteins violate the ordered missing-rate filter contract"
            )
        response = pd.DataFrame(
            response_values,
            index=metadata.index,
            columns=proteins,
            dtype=np.float32,
        )
        missing_rate = pd.Series(
            missing_rate_float64,
            index=missing_rate_names,
            name="protein_missing_rate",
        )
        provenance = _frame_from_json(
            np.asarray(serialized_values["control_provenance_json"])
        )
        overlap = _frame_from_json(np.asarray(serialized_values["overlap_json"]))
        policy_summary = _validated_cached_policy_summary(
            np.asarray(serialized_values["control_policy_summary_json"]),
            control_policy,
            vehicle_column,
        )
        sensitivity = _validated_sensitivity_frame(
            _frame_from_json(
                np.asarray(serialized_values["control_vehicle_sensitivity_json"])
            )
        )
        if len(overlap):
            overlap.index = pd.Index(overlap.index.astype(str), name=CONDITION_ID)
    benchmark_split = build_benchmark_split(metadata, holdout_fraction=holdout_fraction, seed=seed)
    validation_ids = {
        name: pd.Index(ids, name=CONDITION_ID)
        for name, ids in benchmark_split.validation_ids.items()
    }
    return GroupedDataset(
        metadata=metadata,
        response=response,
        proteins=proteins,
        train_ids=pd.Index(benchmark_split.candidate_pool_ids, name=CONDITION_ID),
        validation_ids=validation_ids,
        protein_missing_rate=missing_rate,
        benchmark_split=benchmark_split,
        control_provenance=provenance,
        cross_split_overlap=overlap,
        source_hashes=dict(source_hashes),
        cache_key=cache_key,
        cache_hit=True,
        control_policy_summary=policy_summary,
        control_vehicle_sensitivity=sensitivity,
    )


def load_grouped_dataset(
    metadata_path: str | Path,
    proteome_path: str | Path,
    missing_rate_threshold: float = 0.80,
    cache_dir: str | Path | None = None,
    interpolation_fraction: float = DEFAULT_INTERPOLATION_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    control_policy: str = DEFAULT_CONTROL_POLICY,
    vehicle_column: str | None = None,
) -> GroupedDataset:
    """Convert sample abundance to condition-level matched-control responses.

    The default policy retains the original direct, measurement-weighted mean of
    every exact-context Water/DMSO log2 control.  Strict policy instead requires
    a caller-named vehicle column and exact controls of that requested type.
    Conditions with train provenance aggregate only train treatment rows; their
    overlapping validation labels are retained as audit provenance but discarded
    from the oracle response.  Controls remain all-released assay overhead and
    are explicitly absent from query metadata and features.
    """

    control_policy, vehicle_column = _validated_control_settings(
        control_policy, vehicle_column
    )
    metadata_file = Path(metadata_path)
    proteome_file = Path(proteome_path)
    cache_key, source_hashes, cache_inputs = _cache_identity(
        metadata_file,
        proteome_file,
        missing_rate_threshold,
        interpolation_fraction,
        split_seed,
        control_policy,
        vehicle_column,
    )
    cache_path = None if cache_dir is None else Path(cache_dir) / f"grouped_{cache_key}.npz"
    if cache_path is not None and cache_path.is_file():
        try:
            return _dataset_from_cache(
                cache_path,
                cache_key,
                source_hashes,
                cache_inputs,
                missing_rate_threshold,
                interpolation_fraction,
                split_seed,
                control_policy,
                vehicle_column,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            zipfile.BadZipFile,
            EOFError,
        ) as error:
            warnings.warn(f"Ignoring unreadable GOAI cache {cache_path}: {error}", RuntimeWarning)

    metadata = pd.read_csv(metadata_file, low_memory=False)
    proteome = pd.read_csv(proteome_file, low_memory=False)
    required = {
        SAMPLE_ID,
        SPLIT,
        *GROUP_FIELDS,
        DATA_SOURCE,
        INSTRUMENT,
        PLATE,
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    if (
        control_policy == STRICT_EXPLICIT_VEHICLE
        and vehicle_column not in metadata.columns
    ):
        raise ValueError(
            f"strict_explicit_vehicle requires existing vehicle column {vehicle_column!r}"
        )
    if SAMPLE_ID not in proteome:
        raise ValueError("Proteome matrix is missing sample_ID")
    if metadata[SAMPLE_ID].duplicated().any() or proteome[SAMPLE_ID].duplicated().any():
        raise ValueError("sample_ID values must be unique")

    # Uniqueness was checked explicitly above; avoid pandas' deprecated
    # verify_integrity keyword while retaining the same failure behavior.
    metadata = metadata.set_index(SAMPLE_ID)
    proteome = proteome.set_index(SAMPLE_ID)
    if set(metadata.index) != set(proteome.index):
        raise ValueError("Metadata and proteome sample_ID sets differ")
    proteome = proteome.reindex(metadata.index).apply(pd.to_numeric, errors="raise").astype(np.float32)
    raw = proteome.to_numpy(copy=False)
    if not bool((np.isfinite(raw) | np.isnan(raw)).all()):
        raise ValueError("Proteome matrix contains infinite values")
    if (proteome.notna().to_numpy() & (raw <= 0)).any():
        raise ValueError("Observed proteome values must be positive before log2")

    official_train = metadata.index[metadata[SPLIT].astype(str).eq("train")]
    if len(official_train) == 0:
        raise ValueError("No official train measurement rows were found")
    missing_rate = proteome.loc[official_train].isna().mean(axis=0)
    missing_rate.name = "protein_missing_rate"
    keep = missing_rate < float(missing_rate_threshold)
    if not keep.any():
        raise ValueError("Missingness filter removed every protein")
    response_source = np.log2(proteome.loc[:, keep])

    control_ids = metadata.index[_control_mask(metadata).to_numpy()]
    if len(control_ids) == 0:
        raise ValueError("No Water/DMSO assay controls were found")
    control_keys = _keys(metadata.loc[control_ids], MATCH_CONTROL_FIELDS)
    # This is intentionally the v2.1 pooled calculation: every individual
    # exact-context measurement has equal weight in log2 space.  In particular,
    # two Water rows plus one DMSO row are a three-measurement mean, never an
    # equal average of two per-type means.
    control_means = response_source.loc[control_ids].groupby(control_keys, sort=False).mean()
    control_id_map: dict[tuple[str, ...], list[str]] = {}
    control_name_map: dict[tuple[str, ...], set[str]] = {}
    control_type_id_map: dict[tuple[str, ...], dict[str, list[str]]] = {}
    control_type_by_id: dict[str, str] = {}
    control_split_map: dict[tuple[str, ...], dict[str, str]] = {}
    control_names = _normalised_names(metadata.loc[control_ids, CHEMICAL])
    for sample_id, key, name in zip(
        control_ids.astype(str),
        _key_tuples(metadata.loc[control_ids], MATCH_CONTROL_FIELDS),
        control_names,
    ):
        control_type = str(name)
        control_id_map.setdefault(key, []).append(sample_id)
        control_name_map.setdefault(key, set()).add(control_type)
        control_type_id_map.setdefault(key, {}).setdefault(control_type, []).append(sample_id)
        control_type_by_id[sample_id] = control_type
        control_split_map.setdefault(key, {})[sample_id] = str(metadata.at[sample_id, SPLIT])

    control_means_by_type: dict[str, pd.DataFrame] = {}
    for control_type in sorted(CONTROL_NAMES):
        type_ids = control_ids[control_names.eq(control_type).to_numpy()]
        type_keys = _keys(metadata.loc[type_ids], MATCH_CONTROL_FIELDS)
        control_means_by_type[control_type] = (
            response_source.loc[type_ids].groupby(type_keys, sort=False).mean()
        )

    treatment_ids = metadata.index[_treatment_mask(metadata).to_numpy()]
    treatment_metadata = metadata.loc[treatment_ids].copy()
    condition_ids = [
        stable_condition_id(values)
        for values in treatment_metadata.loc[:, GROUP_FIELDS].itertuples(index=False, name=None)
    ]
    treatment_metadata[CONDITION_ID] = condition_ids
    treatment_keys = _keys(metadata.loc[treatment_ids], MATCH_CONTROL_FIELDS)
    treatment_key_tuples = _key_tuples(metadata.loc[treatment_ids], MATCH_CONTROL_FIELDS)
    available_control_types = [control_name_map.get(key, set()) for key in treatment_key_tuples]

    requested_vehicle = pd.Series(
        [None] * len(treatment_ids), index=treatment_ids, dtype=object
    )
    requested_vehicle_normalised = pd.Series(
        [None] * len(treatment_ids), index=treatment_ids, dtype=object
    )
    treatment_vehicle_values_validated = False
    nonblank_control_vehicle_identities_validated = False
    blank_control_vehicle_values_allowed = False
    if control_policy == STRICT_EXPLICIT_VEHICLE:
        assert vehicle_column is not None
        raw_control_vehicle = metadata.loc[control_ids, vehicle_column]
        explicit_control_vehicle = (
            raw_control_vehicle.astype("string")
            .fillna("")
            .str.strip()
            .str.casefold()
        )
        nonblank_control_vehicle = raw_control_vehicle.notna() & (
            explicit_control_vehicle.ne("")
        )
        contradictory_control_vehicle = nonblank_control_vehicle & (
            explicit_control_vehicle.ne(control_names)
        )
        if bool(contradictory_control_vehicle.any()):
            contradictory_ids = control_ids[
                contradictory_control_vehicle.to_numpy()
            ].astype(str).tolist()[:5]
            contradictions = [
                {
                    "sample_ID": sample_id,
                    "chemical": str(metadata.at[sample_id, CHEMICAL]),
                    "explicit_vehicle": str(metadata.at[sample_id, vehicle_column]),
                }
                for sample_id in contradictory_ids
            ]
            raise ValueError(
                "strict_explicit_vehicle nonblank control-row vehicle values must "
                "agree case-insensitively with the Water/DMSO chemical identity; "
                f"contradictions={contradictions}"
            )
        nonblank_control_vehicle_identities_validated = True
        blank_control_vehicle_values_allowed = True

        normalised_vehicle = _normalised_names(metadata.loc[treatment_ids, vehicle_column])
        valid_vehicle = normalised_vehicle.isin(CONTROL_NAMES)
        if not bool(valid_vehicle.all()):
            invalid_ids = treatment_ids[~valid_vehicle.to_numpy()].astype(str).tolist()[:5]
            invalid_values = metadata.loc[invalid_ids, vehicle_column].astype(str).tolist()
            raise ValueError(
                "strict_explicit_vehicle treatment values must be only Water or DMSO; "
                f"invalid={list(zip(invalid_ids, invalid_values))}"
            )
        treatment_vehicle_values_validated = True
        requested_vehicle_normalised = pd.Series(
            normalised_vehicle.to_numpy(dtype=object), index=treatment_ids, dtype=object
        )
        requested_vehicle = requested_vehicle_normalised.map(
            {"water": "Water", "dmso": "DMSO"}
        )
        vehicle_by_condition = pd.DataFrame(
            {
                CONDITION_ID: condition_ids,
                "vehicle": requested_vehicle_normalised.to_numpy(dtype=object),
            },
            index=treatment_ids,
        )
        mixed = vehicle_by_condition.groupby(CONDITION_ID, sort=True)["vehicle"].nunique()
        mixed_ids = mixed.index[mixed.gt(1)].astype(str).tolist()
        if mixed_ids:
            raise ValueError(
                "strict_explicit_vehicle rejects mixed vehicle replicates in one "
                f"biological condition: {mixed_ids[:5]}"
            )

        matched_pieces: list[pd.DataFrame] = []
        unmatched: list[str] = []
        for control_type in sorted(CONTROL_NAMES):
            type_mask = requested_vehicle_normalised.eq(control_type).to_numpy()
            type_treatment_ids = treatment_ids[type_mask]
            type_treatment_keys = _keys(
                metadata.loc[type_treatment_ids], MATCH_CONTROL_FIELDS
            )
            has_type_match = type_treatment_keys.isin(
                control_means_by_type[control_type].index
            )
            unmatched.extend(type_treatment_ids[~has_type_match].astype(str).tolist())
            selected = control_means_by_type[control_type].reindex(type_treatment_keys)
            selected.index = type_treatment_ids
            matched_pieces.append(selected)
        if unmatched:
            raise ValueError(
                "Treatment rows without exact requested-type controls: "
                f"{sorted(unmatched)[:5]}"
            )
        matched_controls = pd.concat(matched_pieces, axis=0).reindex(treatment_ids)
    else:
        matched_controls = control_means.reindex(treatment_keys)
        matched_controls.index = treatment_ids
        has_match = treatment_keys.isin(control_means.index)
        if not bool(np.all(has_match)):
            missing_examples = treatment_ids[~has_match].astype(str).tolist()[:5]
            raise ValueError(f"Treatment rows without exact controls: {missing_examples}")
    row_response = response_source.loc[treatment_ids] - matched_controls

    def availability_label(names: set[str]) -> str:
        if names == set(CONTROL_NAMES):
            return "both"
        if names == {"dmso"}:
            return "dmso_only"
        if names == {"water"}:
            return "water_only"
        return "none"

    control_availability_counts = {
        name: 0 for name in ("both", "dmso_only", "water_only", "none")
    }
    for names in control_name_map.values():
        control_availability_counts[availability_label(names)] += 1
    treatment_availability_counts = {
        name: 0 for name in ("both", "dmso_only", "water_only", "none")
    }
    for names in available_control_types:
        treatment_availability_counts[availability_label(names)] += 1

    vehicle_mapping_state = (
        "explicit_column_validated"
        if control_policy == STRICT_EXPLICIT_VEHICLE
        else "not_applicable_pooled_policy"
    )
    control_policy_summary = json.loads(
        _policy_summary_json(
            {
                "control_policy": control_policy,
                "default_policy": control_policy == DEFAULT_CONTROL_POLICY,
                "vehicle_column": vehicle_column,
                "vehicle_mapping_state": vehicle_mapping_state,
                "match_control_fields": list(MATCH_CONTROL_FIELDS),
                "control_types": ["Water", "DMSO"],
                "control_aggregation": "direct_measurement_mean_in_log2_space",
                "control_type_means_averaged_equally": False,
                "selected_control_rule": (
                    "all_exact_context_water_dmso_measurements"
                    if control_policy == POOLED_EXACT_CONTEXT_WATER_DMSO
                    else "exact_context_requested_explicit_vehicle_measurements"
                ),
                "pooled_across_types": (
                    control_policy == POOLED_EXACT_CONTEXT_WATER_DMSO
                ),
                "vehicle_inference": False,
                "treatment_vehicle_values_validated": (
                    treatment_vehicle_values_validated
                ),
                "nonblank_control_vehicle_identities_validated": (
                    nonblank_control_vehicle_identities_validated
                ),
                "blank_control_vehicle_values_allowed": (
                    blank_control_vehicle_values_allowed
                ),
                "control_availability": {
                    "measurement_count": int(len(control_ids)),
                    "context_count": int(len(control_name_map)),
                    **{key: int(value) for key, value in control_availability_counts.items()},
                },
                "treatment_availability": {
                    "measurement_count": int(len(treatment_ids)),
                    **{key: int(value) for key, value in treatment_availability_counts.items()},
                },
            }
        )
    )

    sensitivity_rows: list[dict[str, object]] = []
    for key in sorted(control_type_id_map):
        type_ids = control_type_id_map[key]
        if set(type_ids) != set(CONTROL_NAMES):
            continue
        water_mean = (
            control_means_by_type["water"].loc[[key]].iloc[0].to_numpy(dtype=np.float64)
        )
        dmso_mean = (
            control_means_by_type["dmso"].loc[[key]].iloc[0].to_numpy(dtype=np.float64)
        )
        # Delta(treatment|Water) - Delta(treatment|DMSO).
        difference = dmso_mean - water_mean
        finite = difference[np.isfinite(difference)]
        sensitivity_rows.append(
            {
                **dict(zip(MATCH_CONTROL_FIELDS, key)),
                "water_control_count": int(len(type_ids["water"])),
                "dmso_control_count": int(len(type_ids["dmso"])),
                "water_control_ids": json.dumps(sorted(type_ids["water"])),
                "dmso_control_ids": json.dumps(sorted(type_ids["dmso"])),
                "finite_protein_count": int(len(finite)),
                "delta_water_minus_delta_dmso_mean": (
                    float(finite.mean()) if len(finite) else np.nan
                ),
                "delta_water_minus_delta_dmso_median": (
                    float(np.median(finite)) if len(finite) else np.nan
                ),
                "delta_water_minus_delta_dmso_mean_abs": (
                    float(np.abs(finite).mean()) if len(finite) else np.nan
                ),
                "delta_water_minus_delta_dmso_rmse": (
                    float(np.sqrt(np.square(finite).mean())) if len(finite) else np.nan
                ),
                "delta_water_minus_delta_dmso_max_abs": (
                    float(np.abs(finite).max()) if len(finite) else np.nan
                ),
                "sensitivity_definition": (
                    "delta_if_water_minus_delta_if_dmso_equals_"
                    "dmso_control_mean_minus_water_control_mean"
                ),
                "audit_role": "posthoc_oracle_audit",
                "acquisition_input": False,
            }
        )
    control_vehicle_sensitivity = _validated_sensitivity_frame(
        pd.DataFrame(sensitivity_rows, columns=list(_SENSITIVITY_COLUMNS))
        if sensitivity_rows
        else _empty_sensitivity_frame()
    )

    condition_to_positions: dict[str, list[int]] = {}
    for position, condition_id in enumerate(condition_ids):
        condition_to_positions.setdefault(condition_id, []).append(position)

    grouped_meta_rows: list[dict[str, object]] = []
    grouped_response_rows: list[np.ndarray] = []
    ordered_condition_ids: list[str] = []
    for condition_id in sorted(condition_to_positions):
        positions = np.asarray(condition_to_positions[condition_id], dtype=np.int64)
        released_ids = treatment_metadata.index[positions]
        released_condition_frame = treatment_metadata.iloc[positions]
        first = released_condition_frame.iloc[0]
        for field_name in GROUP_FIELDS:
            if released_condition_frame[field_name].astype(str).nunique(dropna=False) != 1:
                raise AssertionError(f"Condition ID collision in {field_name}: {condition_id}")
        splits = tuple(sorted(released_condition_frame[SPLIT].astype(str).unique()))
        split_counts = (
            released_condition_frame[SPLIT].astype(str).value_counts().sort_index().to_dict()
        )
        if "train" in splits:
            primary_split = "train"
        elif len(splits) == 1:
            primary_split = splits[0]
        else:
            raise ValueError(
                "A nontrain biological condition spans multiple validation splits: "
                f"{condition_id} has {splits}"
            )
        aggregation_mask = released_condition_frame[SPLIT].astype(str).eq(primary_split)
        condition_frame = released_condition_frame.loc[aggregation_mask]
        ids = condition_frame.index
        if len(ids) == 0:
            raise AssertionError(f"No aggregation rows remain for {condition_id}")
        measurement_context_count = int(
            len(condition_frame.loc[:, MATCH_CONTROL_FIELDS].astype(str).drop_duplicates())
        )
        ordered_condition_ids.append(condition_id)
        grouped_meta_rows.append(
            {
                SPLIT: primary_split,
                **{field_name: first[field_name] for field_name in GROUP_FIELDS},
                "split_provenance": "|".join(splits),
                "split_replicate_counts": json.dumps(split_counts, sort_keys=True),
                "measurement_ids": json.dumps(ids.astype(str).tolist()),
                "replicate_count": int(len(ids)),
                "measurement_context_count": measurement_context_count,
                "source_count": int(condition_frame[DATA_SOURCE].astype(str).nunique()),
                "instrument_count": int(condition_frame[INSTRUMENT].astype(str).nunique()),
                "plate_count": int(condition_frame[PLATE].astype(str).nunique()),
                "released_measurement_ids": json.dumps(released_ids.astype(str).tolist()),
                "released_replicate_count": int(len(released_ids)),
                "discarded_overlap_measurement_count": int(len(released_ids) - len(ids)),
            }
        )
        grouped_response_rows.append(
            row_response.loc[ids].mean(axis=0, skipna=True).to_numpy(dtype=np.float32)
        )

    grouped_index = pd.Index(ordered_condition_ids, name=CONDITION_ID)
    grouped_metadata = pd.DataFrame(grouped_meta_rows, index=grouped_index)
    grouped_response = pd.DataFrame(
        np.stack(grouped_response_rows),
        index=grouped_index,
        columns=response_source.columns.astype(str),
        dtype=np.float32,
    )
    if grouped_metadata.index.duplicated().any():
        raise AssertionError("Condition IDs are not unique")

    provenance_rows = []
    for treatment_id, condition_id, key in zip(
        treatment_ids.astype(str), condition_ids, treatment_key_tuples
    ):
        available_ids = sorted(control_id_map[key])
        available_types = sorted(control_name_map[key])
        available_type_counts = {
            name: int(len(control_type_id_map[key].get(name, ())))
            for name in sorted(CONTROL_NAMES)
        }
        requested_type = requested_vehicle_normalised.at[treatment_id]
        if control_policy == STRICT_EXPLICIT_VEHICLE:
            selected_ids = sorted(control_type_id_map[key][str(requested_type)])
        else:
            selected_ids = available_ids
        selected_types = sorted({control_type_by_id[sample_id] for sample_id in selected_ids})
        selected_type_counts = {
            name: int(
                sum(
                    control_type_by_id[sample_id] == name
                    for sample_id in selected_ids
                )
            )
            for name in sorted(CONTROL_NAMES)
        }
        treatment_split = str(metadata.at[treatment_id, SPLIT])
        available_splits = sorted(
            {control_split_map[key][sample_id] for sample_id in available_ids}
        )
        selected_splits = sorted(
            {control_split_map[key][sample_id] for sample_id in selected_ids}
        )
        same_split_control_count = sum(
            control_split_map[key][sample_id] == treatment_split for sample_id in selected_ids
        )
        provenance_rows.append(
            {
                "treatment_sample_id": treatment_id,
                CONDITION_ID: condition_id,
                "treatment_split": treatment_split,
                "control_policy": control_policy,
                "vehicle_column": vehicle_column,
                "requested_vehicle": requested_vehicle.at[treatment_id],
                "vehicle_mapping_state": vehicle_mapping_state,
                "vehicle_inference": False,
                "available_control_count": len(available_ids),
                "available_control_ids": json.dumps(available_ids),
                "available_control_types": "|".join(available_types),
                "available_control_type_counts": json.dumps(
                    available_type_counts, sort_keys=True
                ),
                "available_control_splits": json.dumps(available_splits),
                "selected_control_count": len(selected_ids),
                "selected_control_ids": json.dumps(selected_ids),
                "selected_control_types": "|".join(selected_types),
                "selected_control_type_counts": json.dumps(
                    selected_type_counts, sort_keys=True
                ),
                "selected_control_splits": json.dumps(selected_splits),
                "pooled_across_types": bool(
                    control_policy == POOLED_EXACT_CONTEXT_WATER_DMSO
                    and len(selected_types) > 1
                ),
                # Backward-compatible aliases now describe the selected set.
                "matched_control_count": len(selected_ids),
                "matched_control_ids": json.dumps(selected_ids),
                "matched_control_types": "|".join(selected_types),
                "matched_control_splits": json.dumps(selected_splits),
                "same_split_control_count": int(same_split_control_count),
                "cross_split_only": bool(same_split_control_count == 0),
                "control_role": "assay_overhead",
                "predictor_input": False,
                "acquisition_input": False,
            }
        )
    control_provenance = pd.DataFrame(provenance_rows).set_index("treatment_sample_id")
    cross_split_overlap = detect_cross_split_condition_overlap(grouped_metadata)
    benchmark_split = build_benchmark_split(
        grouped_metadata,
        holdout_fraction=interpolation_fraction,
        seed=split_seed,
    )
    validation_ids = {
        name: pd.Index(ids, name=CONDITION_ID)
        for name, ids in benchmark_split.validation_ids.items()
    }
    dataset = GroupedDataset(
        metadata=grouped_metadata,
        response=grouped_response,
        proteins=tuple(grouped_response.columns),
        train_ids=pd.Index(benchmark_split.candidate_pool_ids, name=CONDITION_ID),
        validation_ids=validation_ids,
        protein_missing_rate=missing_rate,
        benchmark_split=benchmark_split,
        control_provenance=control_provenance,
        cross_split_overlap=cross_split_overlap,
        source_hashes=source_hashes,
        cache_key=cache_key,
        cache_hit=False,
        control_policy_summary=control_policy_summary,
        control_vehicle_sensitivity=control_vehicle_sensitivity,
    )

    if cache_path is not None:
        metadata_json = _frame_to_json(grouped_metadata)
        control_provenance_json = _frame_to_json(control_provenance)
        overlap_json = _frame_to_json(cross_split_overlap)
        control_policy_summary_json = _policy_summary_json(control_policy_summary)
        control_vehicle_sensitivity_json = _frame_to_json(
            control_vehicle_sensitivity
        )
        response_values = grouped_response.to_numpy(dtype=np.float32)
        protein_values = np.asarray(dataset.proteins, dtype=str)
        missing_rate_names = np.asarray(missing_rate.index.astype(str), dtype=str)
        missing_rate_values = missing_rate.to_numpy(dtype=np.float64)
        artifact_digests = {
            "response_sha256": _sha256_bytes(response_values.tobytes(order="C")),
            "metadata_json_sha256": _sha256_bytes(metadata_json.encode("utf-8")),
            "control_provenance_json_sha256": _sha256_bytes(
                control_provenance_json.encode("utf-8")
            ),
            "overlap_json_sha256": _sha256_bytes(overlap_json.encode("utf-8")),
            "control_policy_summary_json_sha256": _sha256_bytes(
                control_policy_summary_json.encode("utf-8")
            ),
            "control_vehicle_sensitivity_json_sha256": _sha256_bytes(
                control_vehicle_sensitivity_json.encode("utf-8")
            ),
            "proteins_sha256": _canonical_string_array_sha256(protein_values),
            "missing_rate_names_sha256": _canonical_string_array_sha256(
                missing_rate_names
            ),
            "missing_rate_values_sha256": _canonical_numeric_array_sha256(
                missing_rate_values
            ),
        }
        cache_manifest = {
            **cache_inputs,
            "cache_key": cache_key,
            "artifact_digests": artifact_digests,
        }
        _atomic_savez(
            cache_path,
            protocol_version=np.asarray(PROTOCOL_VERSION),
            cache_version=np.asarray(CACHE_VERSION),
            cache_key=np.asarray(cache_key),
            cache_manifest_json=np.asarray(_cache_json(cache_manifest)),
            source_hashes_json=np.asarray(
                _cache_json(cache_inputs["source_hashes"])  # type: ignore[arg-type]
            ),
            source_files_json=np.asarray(
                _cache_json(cache_inputs["source_files"])  # type: ignore[arg-type]
            ),
            missing_rate_threshold=np.asarray(
                float(missing_rate_threshold), dtype=np.float64
            ),
            interpolation_fraction=np.asarray(
                float(interpolation_fraction), dtype=np.float64
            ),
            split_seed=np.asarray(int(split_seed), dtype=np.int64),
            control_policy=np.asarray(control_policy),
            vehicle_column_json=np.asarray(
                json.dumps(vehicle_column, ensure_ascii=False, allow_nan=False)
            ),
            metadata_json=np.asarray(metadata_json),
            response=response_values,
            proteins=protein_values,
            missing_rate_names=missing_rate_names,
            missing_rate_values=missing_rate_values,
            control_provenance_json=np.asarray(control_provenance_json),
            overlap_json=np.asarray(overlap_json),
            control_policy_summary_json=np.asarray(control_policy_summary_json),
            control_vehicle_sensitivity_json=np.asarray(
                control_vehicle_sensitivity_json
            ),
            **{
                name: np.asarray(digest)
                for name, digest in artifact_digests.items()
            },
        )
    return dataset
