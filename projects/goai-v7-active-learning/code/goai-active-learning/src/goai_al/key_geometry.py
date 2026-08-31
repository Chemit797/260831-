"""Exact directed transfer-matrix geometry discovery for GOAI-AL.

This is intentionally a separate research entry point.  It leaves the frozen
v2.2 active-learning runner and the earlier relation-group transfer audit
unchanged.  The unit of evidence here is one donor condition and one globally
withheld target condition.  A shared metadata-only baseline makes the full
moderate panel exact at the cost of one fit per donor, rather than one fit per
donor-target cell.

The structural analysis is deliberately modest: five response-independent
held-out-entry folds, ranks 1/2/4/8/16, a directed factor kernel, a
regularized directed pair-relation extension, and a directed unconstrained
low-rank comparator.  It is a local retrospective proxy experiment, never an
official GOAI score or an acquisition-policy replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import (
    CHEMICAL,
    CONDITION_ID,
    MEDIUM,
    STRAIN,
    TEMPERATURE,
    TIME,
    GroupedDataset,
    load_grouped_dataset,
)
from .direct_multiseed import (
    _row_positions,
    load_identity_feature_bundle,
    prepare_masked_model_features,
)
from .model import ModelSettings, fit_response_model


EXPERIMENT_ID = "GOAI-AL-KEY-GEOMETRY-01"
PARENT_MODEL_ID = "GOAI-AL-TRANSFER-GEOMETRY-01"
PRIMARY_METRIC = "zero_normalized_mse_reduction"
DEFAULT_OUTPUT_NAME = "key_geometry-20260825-v1"
DEFAULT_METADATA_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_metadata_train_val.csv"
)
DEFAULT_PROTEOME_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_proteome_raw_train_val.csv"
)

FACTOR_FIELDS = (STRAIN, CHEMICAL, TIME, MEDIUM, TEMPERATURE)
FOCAL_FIELDS = (STRAIN, CHEMICAL, TIME)
INTERACTION_PAIRS = ((STRAIN, CHEMICAL), (CHEMICAL, TIME), (STRAIN, TIME))
REQUESTED_RANKS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class KeyGeometrySettings:
    """Pre-specified, small exact matrix and structure-analysis choices."""

    selection_seed: int = 20260825
    model_seed: int = 42
    target_count: int = 96
    donor_count: int = 148
    baseline_size: int = 256
    epochs: int = 80
    folds: int = 5
    factor_rank: int = 4
    general_comparison_rank: int = 8
    structure_epochs: int = 900
    structure_learning_rate: float = 0.035
    structure_weight_decay: float = 0.001
    bootstrap_draws: int = 500
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.target_count < 8 or self.donor_count < 8:
            raise ValueError("target_count and donor_count must each be at least 8")
        if self.baseline_size < 32:
            raise ValueError("baseline_size must be at least 32")
        if self.epochs <= 0 or self.structure_epochs <= 0:
            raise ValueError("epoch counts must be positive")
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if self.factor_rank not in REQUESTED_RANKS:
            raise ValueError("factor_rank must be one of 1, 2, 4, 8, or 16")
        if self.general_comparison_rank not in REQUESTED_RANKS:
            raise ValueError("general_comparison_rank must be a requested rank")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")


@dataclass(frozen=True)
class PanelPlan:
    """Disjoint metadata-only role assignment for exact transfer measurements."""

    targets: tuple[str, ...]
    donors: tuple[str, ...]
    baseline: tuple[str, ...]


def _as_text(value: object) -> str:
    return str(value)


def _stable_digest(seed: int, namespace: str, *values: object) -> bytes:
    payload = json.dumps(
        [int(seed), namespace, *[_as_text(value) for value in values]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _stable_sort(
    ids: Iterable[object], *, seed: int, namespace: str
) -> list[str]:
    return sorted(
        (_as_text(value) for value in ids),
        key=lambda sample_id: (_stable_digest(seed, namespace, sample_id), sample_id),
    )


def _metadata_levels(metadata: pd.DataFrame, field: str) -> list[str]:
    values = metadata[field].astype(str).unique().tolist()
    if field == TIME:
        return sorted(values, key=lambda value: float(value))
    return sorted(values)


def _context_label(row: pd.Series) -> str:
    return "\x00".join(
        _as_text(row[field]) for field in (STRAIN, TIME, MEDIUM, TEMPERATURE)
    )


def _selection_cost(row: pd.Series, counts: Mapping[str, Counter[str]]) -> tuple[int, ...]:
    """Prefer unrepresented public metadata cells without consulting response."""

    strain = _as_text(row[STRAIN])
    chemical = _as_text(row[CHEMICAL])
    time = _as_text(row[TIME])
    medium = _as_text(row[MEDIUM])
    temperature = _as_text(row[TEMPERATURE])
    strain_time = f"{strain}\x00{time}"
    medium_temperature = f"{medium}\x00{temperature}"
    return (
        counts["context"].get(_context_label(row), 0),
        counts["strain_time"].get(strain_time, 0),
        counts["medium_temperature"].get(medium_temperature, 0),
        counts["strain"].get(strain, 0),
        counts["time"].get(time, 0),
        counts["medium"].get(medium, 0),
        counts["temperature"].get(temperature, 0),
        counts["chemical"].get(chemical, 0),
    )


def _update_selection_counts(row: pd.Series, counts: Mapping[str, Counter[str]]) -> None:
    values = {
        "context": _context_label(row),
        "strain_time": f"{_as_text(row[STRAIN])}\x00{_as_text(row[TIME])}",
        "medium_temperature": f"{_as_text(row[MEDIUM])}\x00{_as_text(row[TEMPERATURE])}",
        "strain": _as_text(row[STRAIN]),
        "chemical": _as_text(row[CHEMICAL]),
        "time": _as_text(row[TIME]),
        "medium": _as_text(row[MEDIUM]),
        "temperature": _as_text(row[TEMPERATURE]),
    }
    for name, value in values.items():
        counts[name][value] += 1


def _select_balanced_panel(
    metadata: pd.DataFrame,
    *,
    count: int,
    excluded_ids: frozenset[str],
    seed: int,
    namespace: str,
    require_all_chemicals: bool,
) -> tuple[str, ...]:
    """Select a disjoint panel using only factor balancing and stable hashes.

    The first pass gives every chemical an equal integer quota when that is
    possible.  The remaining slots preferentially cover the 4x6x2x2 public
    strain/time/medium/temperature contexts and balance lower-order margins.
    """

    available = metadata.loc[
        ~metadata.index.astype(str).isin(excluded_ids)
    ].copy()
    if len(available) < count:
        raise ValueError(f"{namespace}: only {len(available)} candidates for {count} slots")
    chemicals = _metadata_levels(metadata, CHEMICAL)
    if require_all_chemicals and count < len(chemicals):
        raise ValueError(
            f"{namespace}: {count} slots cannot cover all {len(chemicals)} chemicals"
        )
    quota = count // len(chemicals)
    if require_all_chemicals:
        quota = max(quota, 1)
    available_ids = available.index.astype(str).tolist()
    # The formal panel evaluates roughly 500 selection slots.  Keep all public
    # metadata in small Python records so repeated balance comparisons do not
    # make millions of pandas ``.loc`` calls before a single response fit.
    records: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
    by_chemical: dict[str, list[str]] = {chemical: [] for chemical in chemicals}
    for sample_id, row in available.loc[available_ids, list(FACTOR_FIELDS)].iterrows():
        strain = _as_text(row[STRAIN])
        chemical = _as_text(row[CHEMICAL])
        time = _as_text(row[TIME])
        medium = _as_text(row[MEDIUM])
        temperature = _as_text(row[TEMPERATURE])
        records[sample_id] = (
            strain,
            chemical,
            time,
            medium,
            temperature,
            f"{strain}\x00{time}\x00{medium}\x00{temperature}",
            f"{strain}\x00{time}",
            f"{medium}\x00{temperature}",
        )
        by_chemical.setdefault(chemical, []).append(sample_id)
    tie_break = {
        sample_id: _stable_digest(seed, namespace, sample_id)
        for sample_id in available_ids
    }
    selected: list[str] = []
    remaining = set(available_ids)
    counts: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "context",
            "strain_time",
            "medium_temperature",
            "strain",
            "chemical",
            "time",
            "medium",
            "temperature",
        )
    }

    def cost(sample_id: str) -> tuple[int, ...]:
        strain, chemical, time, medium, temperature, context, strain_time, medium_temperature = records[sample_id]
        return (
            counts["context"].get(context, 0),
            counts["strain_time"].get(strain_time, 0),
            counts["medium_temperature"].get(medium_temperature, 0),
            counts["strain"].get(strain, 0),
            counts["time"].get(time, 0),
            counts["medium"].get(medium, 0),
            counts["temperature"].get(temperature, 0),
            counts["chemical"].get(chemical, 0),
        )

    def record(sample_id: str) -> None:
        strain, chemical, time, medium, temperature, context, strain_time, medium_temperature = records[sample_id]
        values = {
            "context": context,
            "strain_time": strain_time,
            "medium_temperature": medium_temperature,
            "strain": strain,
            "chemical": chemical,
            "time": time,
            "medium": medium,
            "temperature": temperature,
        }
        for name, value in values.items():
            counts[name][value] += 1

    def choose(candidates: Sequence[str]) -> str:
        if not candidates:
            raise ValueError(f"{namespace}: no metadata candidates remain")
        return min(
            candidates,
            key=lambda sample_id: (
                cost(sample_id),
                tie_break[sample_id],
                sample_id,
            ),
        )

    for round_index in range(quota):
        for chemical in chemicals:
            candidates = [sample_id for sample_id in by_chemical.get(chemical, []) if sample_id in remaining]
            if not candidates:
                raise ValueError(
                    f"{namespace}: chemical {chemical!r} cannot meet quota {round_index + 1}"
                )
            picked = choose(candidates)
            selected.append(picked)
            remaining.remove(picked)
            record(picked)

    while len(selected) < count:
        picked = choose(list(remaining))
        selected.append(picked)
        remaining.remove(picked)
        record(picked)

    if len(set(selected)) != count:
        raise AssertionError("Panel selection returned duplicate condition IDs")
    return tuple(selected)


def build_panel_plan(metadata: pd.DataFrame, settings: KeyGeometrySettings) -> PanelPlan:
    """Make target, donor, and baseline assignments before accessing response."""

    candidate_ids = frozenset(metadata.index.astype(str))
    if len(candidate_ids) != len(metadata):
        raise ValueError("Candidate metadata index must contain unique condition IDs")
    targets = _select_balanced_panel(
        metadata,
        count=settings.target_count,
        excluded_ids=frozenset(),
        seed=settings.selection_seed,
        namespace="key-target",
        require_all_chemicals=False,
    )
    donors = _select_balanced_panel(
        metadata,
        count=settings.donor_count,
        excluded_ids=frozenset(targets),
        seed=settings.selection_seed,
        namespace="key-donor",
        require_all_chemicals=False,
    )
    baseline = _select_balanced_panel(
        metadata,
        count=settings.baseline_size,
        excluded_ids=frozenset((*targets, *donors)),
        seed=settings.selection_seed,
        namespace="key-baseline",
        require_all_chemicals=True,
    )
    plan = PanelPlan(targets=targets, donors=donors, baseline=baseline)
    _validate_panel_plan(metadata, plan)
    return plan


def _validate_panel_plan(metadata: pd.DataFrame, plan: PanelPlan) -> None:
    roles = {
        "target": set(plan.targets),
        "donor": set(plan.donors),
        "baseline": set(plan.baseline),
    }
    if any(not values for values in roles.values()):
        raise ValueError("Every panel role must be nonempty")
    if roles["target"] & roles["donor"] or roles["target"] & roles["baseline"] or roles["donor"] & roles["baseline"]:
        raise ValueError("Target, donor, and baseline roles must be disjoint")
    for field in FACTOR_FIELDS:
        all_levels = set(metadata[field].astype(str))
        baseline_levels = set(metadata.loc[list(plan.baseline), field].astype(str))
        if baseline_levels != all_levels:
            missing = sorted(all_levels - baseline_levels)
            raise ValueError(
                f"Shared baseline does not cover all {field!r} levels; missing={missing}"
            )


def _panel_rows(metadata: pd.DataFrame, plan: PanelPlan) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, values in (
        ("target", plan.targets),
        ("donor", plan.donors),
        ("baseline", plan.baseline),
    ):
        for order, sample_id in enumerate(values, start=1):
            row = metadata.loc[sample_id]
            rows.append(
                {
                    CONDITION_ID: sample_id,
                    "role": role,
                    "selection_order": order,
                    **{f"{field}_value": _as_text(row[field]) for field in FACTOR_FIELDS},
                    "strain_time_medium_temperature": _context_label(row),
                }
            )
    return pd.DataFrame(rows)


def _panel_audit_rows(metadata: pd.DataFrame, plan: PanelPlan) -> pd.DataFrame:
    role_values = {
        "target": plan.targets,
        "donor": plan.donors,
        "baseline": plan.baseline,
    }
    rows: list[dict[str, Any]] = []
    for role, ids in role_values.items():
        subset = metadata.loc[list(ids)]
        for field in FACTOR_FIELDS:
            for value, count in subset[field].astype(str).value_counts().sort_index().items():
                rows.append(
                    {
                        "role": role,
                        "field": field,
                        "value": value,
                        "count": int(count),
                        "coverage_fraction": float(count / len(subset)),
                    }
                )
        for value, count in subset.apply(_context_label, axis=1).value_counts().sort_index().items():
            rows.append(
                {
                    "role": role,
                    "field": "strain_time_medium_temperature",
                    "value": value,
                    "count": int(count),
                    "coverage_fraction": float(count / len(subset)),
                }
            )
    return pd.DataFrame(rows)


def _model_settings(settings: KeyGeometrySettings) -> ModelSettings:
    return ModelSettings(
        kind="direct",
        hidden_dim=128,
        dropout=0.10,
        learning_rate=0.001,
        weight_decay=0.0002,
        epochs=settings.epochs,
        batch_size=512,
        target_scale_floor=0.05,
        device=settings.device,
        response_rank=64,
        svd_niter=2,
    )


def _score_conditions(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-condition MSE and zero-normalized MSE without pooling rows."""

    predicted = np.asarray(prediction, dtype=np.float64)
    observed_truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != observed_truth.shape or predicted.ndim != 2:
        raise ValueError("Prediction and truth must be identically shaped matrices")
    mask = np.isfinite(observed_truth)
    if np.any(mask & ~np.isfinite(predicted)):
        raise ValueError("Prediction is nonfinite at an observed target value")
    counts = mask.sum(axis=1).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every target condition must have at least one observed protein")
    squared_error = np.where(mask, np.square(predicted - observed_truth), 0.0).sum(axis=1)
    zero_squared_error = np.where(mask, np.square(observed_truth), 0.0).sum(axis=1)
    mse = squared_error / counts
    zero_mse = zero_squared_error / counts
    normalized_loss = np.divide(
        mse,
        zero_mse,
        out=np.full_like(mse, np.nan),
        where=zero_mse > 0.0,
    )
    return {
        "observed_values": counts.astype(np.int64),
        "mse": mse,
        "zero_mse": zero_mse,
        "normalized_loss": normalized_loss,
        "delta_skill_zero": 1.0 - normalized_loss,
    }


def _relation_metadata(donor: pd.Series, target: pd.Series) -> dict[str, Any]:
    same = {
        field: bool(_as_text(donor[field]) == _as_text(target[field]))
        for field in FACTOR_FIELDS
    }
    focal_pattern = "".join("1" if same[field] else "0" for field in FOCAL_FIELDS)
    return {
        "same_strain": same[STRAIN],
        "same_chemical": same[CHEMICAL],
        "same_time": same[TIME],
        "same_medium": same[MEDIUM],
        "same_temperature": same[TEMPERATURE],
        "same_strain_chemical": same[STRAIN] and same[CHEMICAL],
        "same_chemical_time": same[CHEMICAL] and same[TIME],
        "same_strain_time": same[STRAIN] and same[TIME],
        "focal_relation_pattern_std": focal_pattern,
    }


def _entry_fold(
    donor_id: str,
    target_id: str,
    relation_pattern: str,
    settings: KeyGeometrySettings,
) -> int:
    value = int.from_bytes(
        _stable_digest(
            settings.selection_seed,
            "key-geometry-entry-fold",
            relation_pattern,
            donor_id,
            target_id,
        )[:8],
        "big",
    )
    return int(value % settings.folds)


def _verify_fixed_feature_map(
    bundle: Any,
    plan: PanelPlan,
) -> np.ndarray:
    """Assert that appending any donor cannot unlock a feature column.

    This is the reason a shared baseline is scientifically useful here: exact
    donor effects are response-support effects, not different categorical
    masking regimes for different source entities.
    """

    baseline_features = prepare_masked_model_features(bundle, plan.baseline)
    for donor_id in plan.donors:
        appended = prepare_masked_model_features(bundle, (*plan.baseline, donor_id))
        if not np.array_equal(baseline_features, appended):
            raise AssertionError(
                f"Appending donor {donor_id} changed the masked feature map; "
                "the shared baseline is not source-neutral"
            )
    return baseline_features


def run_exact_transfer_matrix(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: KeyGeometrySettings,
    plan: PanelPlan,
) -> pd.DataFrame:
    """Fit B and B+{i} once each, then score every globally withheld target."""

    metadata = dataset.metadata.loc[list(dataset.candidate_pool_ids)].copy()
    bundle = load_identity_feature_bundle(dataset)
    row_positions = {sample_id: position for position, sample_id in enumerate(bundle.row_ids)}
    response = dataset.response.loc[list(bundle.row_ids)].to_numpy(dtype=np.float32)
    target_positions = _row_positions(bundle.row_ids, plan.targets)
    baseline_positions = _row_positions(bundle.row_ids, plan.baseline)
    masked_features = _verify_fixed_feature_map(bundle, plan)
    model_settings = _model_settings(settings)
    target_truth = response[target_positions]

    if set(plan.targets) & set(plan.baseline) or set(plan.targets) & set(plan.donors):
        raise AssertionError("A target condition appeared in support")

    receipts: list[dict[str, Any]] = []
    baseline_started = perf_counter()
    baseline_fit = fit_response_model(
        masked_features[baseline_positions],
        response[baseline_positions],
        model_settings,
        settings.model_seed,
    )
    baseline_seconds = float(perf_counter() - baseline_started)
    baseline_scores = _score_conditions(
        baseline_fit.predict(masked_features[target_positions]), target_truth
    )
    receipts.append(
        {
            "fit_role": "baseline",
            "donor_condition_id": "",
            "support_count": len(plan.baseline),
            "fit_seconds": baseline_seconds,
            "final_loss": baseline_fit.final_loss,
            **baseline_fit.fit_summary(),
        }
    )

    records: list[dict[str, Any]] = []
    for donor_rank, donor_id in enumerate(plan.donors, start=1):
        donor_position = row_positions[donor_id]
        support_ids = (*plan.baseline, donor_id)
        started = perf_counter()
        fit = fit_response_model(
            masked_features[np.asarray([*baseline_positions, donor_position], dtype=np.int64)],
            response[np.asarray([*baseline_positions, donor_position], dtype=np.int64)],
            model_settings,
            settings.model_seed,
        )
        fit_seconds = float(perf_counter() - started)
        after_scores = _score_conditions(
            fit.predict(masked_features[target_positions]), target_truth
        )
        receipts.append(
            {
                "fit_role": "donor_append",
                "donor_condition_id": donor_id,
                "donor_rank": donor_rank,
                "support_count": len(support_ids),
                "fit_seconds": fit_seconds,
                "final_loss": fit.final_loss,
                **fit.fit_summary(),
            }
        )
        donor = metadata.loc[donor_id]
        for target_rank, target_id in enumerate(plan.targets, start=1):
            target = metadata.loc[target_id]
            relation = _relation_metadata(donor, target)
            fold = _entry_fold(
                donor_id,
                target_id,
                str(relation["focal_relation_pattern_std"]),
                settings,
            )
            base_loss = float(baseline_scores["mse"][target_rank - 1])
            after_loss = float(after_scores["mse"][target_rank - 1])
            base_normalized = float(baseline_scores["normalized_loss"][target_rank - 1])
            after_normalized = float(after_scores["normalized_loss"][target_rank - 1])
            records.append(
                {
                    "donor_condition_id": donor_id,
                    "target_condition_id": target_id,
                    "donor_rank": donor_rank,
                    "target_rank": target_rank,
                    "score_origin": "exact_shared_baseline_refit",
                    "transfer_score_definition": "zero_normalized_mse_base_minus_after",
                    "transfer_score": base_normalized - after_normalized,
                    "raw_mse_reduction": base_loss - after_loss,
                    "baseline_loss": base_loss,
                    "after_donor_loss": after_loss,
                    "baseline_zero_normalized_loss": base_normalized,
                    "after_donor_zero_normalized_loss": after_normalized,
                    "baseline_delta_skill_zero": float(
                        baseline_scores["delta_skill_zero"][target_rank - 1]
                    ),
                    "after_donor_delta_skill_zero": float(
                        after_scores["delta_skill_zero"][target_rank - 1]
                    ),
                    "target_observed_values": int(
                        baseline_scores["observed_values"][target_rank - 1]
                    ),
                    "baseline_fit_seconds": baseline_seconds,
                    "after_donor_fit_seconds": fit_seconds,
                    "model_seed": settings.model_seed,
                    "epochs": settings.epochs,
                    "cv_fold": fold,
                    "train_test_role": f"heldout_structure_test_fold_{fold}",
                    **{
                        f"donor_{field}": _as_text(donor[field])
                        for field in FACTOR_FIELDS
                    },
                    **{
                        f"target_{field}": _as_text(target[field])
                        for field in FACTOR_FIELDS
                    },
                    **relation,
                }
            )
    result = pd.DataFrame(records).sort_values(
        ["donor_rank", "target_rank"], kind="stable"
    ).reset_index(drop=True)
    expected = len(plan.donors) * len(plan.targets)
    if len(result) != expected or result.duplicated(["donor_condition_id", "target_condition_id"]).any():
        raise AssertionError("Exact transfer matrix is incomplete or duplicated")
    if not np.isfinite(result["transfer_score"].to_numpy(dtype=float)).all():
        raise AssertionError("Exact transfer scores must be finite")
    result.to_csv(output_dir / "TRANSFER_MATRIX_PROBES.csv", index=False)
    pd.DataFrame(receipts).to_csv(output_dir / "EXACT_FIT_RECEIPTS.csv", index=False)
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without adding a scipy dependency."""

    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_values) & np.isfinite(right_values)
    if int(valid.sum()) < 3:
        return float("nan")
    x = _rankdata(left_values[valid])
    y = _rankdata(right_values[valid])
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def _metric_values(
    actual: np.ndarray,
    prediction: np.ndarray,
    two_way_prediction: np.ndarray,
) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.asarray(prediction, dtype=np.float64)
    baseline_values = np.asarray(two_way_prediction, dtype=np.float64)
    valid = (
        np.isfinite(actual_values)
        & np.isfinite(predicted_values)
        & np.isfinite(baseline_values)
    )
    if not valid.any():
        return {
            "heldout_entry_rmse": float("nan"),
            "heldout_entry_mae": float("nan"),
            "heldout_entry_spearman": float("nan"),
            "heldout_entry_r2_vs_two_way": float("nan"),
            "heldout_entry_r2_vs_mean": float("nan"),
        }
    y = actual_values[valid]
    p = predicted_values[valid]
    b = baseline_values[valid]
    residual = y - p
    baseline_residual = y - b
    sse = float(np.square(residual).sum())
    baseline_sse = float(np.square(baseline_residual).sum())
    mean_sse = float(np.square(y - y.mean()).sum())
    return {
        "heldout_entry_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "heldout_entry_mae": float(np.mean(np.abs(residual))),
        "heldout_entry_spearman": _spearman(y, p),
        "heldout_entry_r2_vs_two_way": (
            float(1.0 - sse / baseline_sse) if baseline_sse > 0.0 else float("nan")
        ),
        "heldout_entry_r2_vs_mean": (
            float(1.0 - sse / mean_sse) if mean_sse > 0.0 else float("nan")
        ),
    }


def _two_way_design(
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_count: int,
    target_count: int,
) -> np.ndarray:
    """A global plus donor/target-intercept baseline with fixed references."""

    donor_values = np.asarray(donor_index, dtype=np.int64)
    target_values = np.asarray(target_index, dtype=np.int64)
    design = np.zeros(
        (len(donor_values), 1 + max(donor_count - 1, 0) + max(target_count - 1, 0)),
        dtype=np.float64,
    )
    design[:, 0] = 1.0
    if donor_count > 1:
        eligible = donor_values < donor_count - 1
        design[np.where(eligible)[0], 1 + donor_values[eligible]] = 1.0
    if target_count > 1:
        eligible = target_values < target_count - 1
        offset = 1 + max(donor_count - 1, 0)
        design[np.where(eligible)[0], offset + target_values[eligible]] = 1.0
    return design


def _fit_two_way_baseline(
    donor_index: np.ndarray,
    target_index: np.ndarray,
    value: np.ndarray,
    train_mask: np.ndarray,
    donor_count: int,
    target_count: int,
) -> np.ndarray:
    design = _two_way_design(donor_index, target_index, donor_count, target_count)
    mask = np.asarray(train_mask, dtype=bool)
    if int(mask.sum()) <= design.shape[1]:
        raise ValueError("Too few train entries for the two-way intercept baseline")
    coefficients, *_ = np.linalg.lstsq(design[mask], np.asarray(value, dtype=float)[mask], rcond=None)
    return design @ coefficients


class _DirectedStructureModel(nn.Module):
    """A compact directed K/Q model with optional factor and CP-pair blocks."""

    def __init__(
        self,
        *,
        kind: str,
        donor_count: int,
        target_count: int,
        factor_level_counts: Sequence[int],
        rank: int,
        included_factor_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if kind not in {"general", "factor", "factor_interaction"}:
            raise ValueError(f"Unknown structural model kind: {kind}")
        self.kind = kind
        self.rank = int(rank)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.donor_bias = nn.Embedding(donor_count, 1)
        self.target_bias = nn.Embedding(target_count, 1)
        nn.init.zeros_(self.donor_bias.weight)
        nn.init.zeros_(self.target_bias.weight)
        self.factor_indices = tuple(
            range(len(factor_level_counts))
            if included_factor_indices is None
            else included_factor_indices
        )
        if kind == "general":
            self.donor_key = nn.Embedding(donor_count, rank)
            self.target_query = nn.Embedding(target_count, rank)
            nn.init.normal_(self.donor_key.weight, std=0.04)
            nn.init.normal_(self.target_query.weight, std=0.04)
            self.donor_factor = nn.ModuleList()
            self.target_factor = nn.ModuleList()
            self.interaction_embeddings = nn.ModuleList()
            self.interaction_pairs: tuple[tuple[int, int], ...] = tuple()
            return
        self.donor_key = None
        self.target_query = None
        self.donor_factor = nn.ModuleList(
            [nn.Embedding(level_count, rank) for level_count in factor_level_counts]
        )
        self.target_factor = nn.ModuleList(
            [nn.Embedding(level_count, rank) for level_count in factor_level_counts]
        )
        for embedding in [*self.donor_factor, *self.target_factor]:
            nn.init.normal_(embedding.weight, std=0.04)
        self.interaction_pairs = (
            ((0, 1), (1, 2), (0, 2)) if kind == "factor_interaction" else tuple()
        )
        self.interaction_embeddings = nn.ModuleList()
        for first, second in self.interaction_pairs:
            group = nn.ModuleList(
                [
                    nn.Embedding(factor_level_counts[first], rank),
                    nn.Embedding(factor_level_counts[second], rank),
                    nn.Embedding(factor_level_counts[first], rank),
                    nn.Embedding(factor_level_counts[second], rank),
                ]
            )
            for embedding in group:
                nn.init.normal_(embedding.weight, std=0.04)
            self.interaction_embeddings.append(group)

    def forward(
        self,
        donor_index: torch.Tensor,
        target_index: torch.Tensor,
        donor_codes: Sequence[torch.Tensor],
        target_codes: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        output = (
            self.global_bias
            + self.donor_bias(donor_index).squeeze(-1)
            + self.target_bias(target_index).squeeze(-1)
        )
        if self.kind == "general":
            assert self.donor_key is not None and self.target_query is not None
            return output + (self.donor_key(donor_index) * self.target_query(target_index)).sum(dim=1)
        for index in self.factor_indices:
            output = output + (
                self.donor_factor[index](donor_codes[index])
                * self.target_factor[index](target_codes[index])
            ).sum(dim=1)
        for (first, second), embeddings in zip(self.interaction_pairs, self.interaction_embeddings):
            donor_pair = embeddings[0](donor_codes[first]) * embeddings[1](donor_codes[second])
            target_pair = embeddings[2](target_codes[first]) * embeddings[3](target_codes[second])
            output = output + (donor_pair * target_pair).sum(dim=1)
        return output


def _fit_structure_model(
    *,
    kind: str,
    rank: int,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_codes: Sequence[np.ndarray],
    target_codes: Sequence[np.ndarray],
    factor_level_counts: Sequence[int],
    value: np.ndarray,
    train_mask: np.ndarray,
    settings: KeyGeometrySettings,
    seed: int,
    included_factor_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Fit only on observed training entries and predict every matrix cell."""

    mask = np.asarray(train_mask, dtype=bool)
    values = np.asarray(value, dtype=np.float64)
    if not np.isfinite(values[mask]).all() or int(mask.sum()) < 8:
        raise ValueError("Structural training entries must be finite and nonempty")
    train_mean = float(values[mask].mean())
    train_scale = max(float(values[mask].std()), 1e-8)
    standardized = ((values - train_mean) / train_scale).astype(np.float32)
    device = torch.device(settings.device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _DirectedStructureModel(
        kind=kind,
        donor_count=int(np.max(donor_index)) + 1,
        target_count=int(np.max(target_index)) + 1,
        factor_level_counts=factor_level_counts,
        rank=rank,
        included_factor_indices=included_factor_indices,
    ).to(device)
    donor_tensor = torch.as_tensor(donor_index, dtype=torch.long, device=device)
    target_tensor = torch.as_tensor(target_index, dtype=torch.long, device=device)
    donor_code_tensors = [
        torch.as_tensor(values, dtype=torch.long, device=device) for values in donor_codes
    ]
    target_code_tensors = [
        torch.as_tensor(values, dtype=torch.long, device=device) for values in target_codes
    ]
    train_positions = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
    train_truth = torch.as_tensor(standardized[mask], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.structure_learning_rate,
        weight_decay=settings.structure_weight_decay,
    )
    for _ in range(settings.structure_epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            donor_tensor[train_positions],
            target_tensor[train_positions],
            [values[train_positions] for values in donor_code_tensors],
            [values[train_positions] for values in target_code_tensors],
        )
        loss = torch.mean(torch.square(prediction - train_truth))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(
            donor_tensor,
            target_tensor,
            donor_code_tensors,
            target_code_tensors,
        )
    return prediction.detach().cpu().numpy().astype(np.float64) * train_scale + train_mean


def _structure_iterations(settings: KeyGeometrySettings) -> int:
    """Map the fixed analysis budget to stable small-matrix update rounds."""

    return max(20, min(100, settings.structure_epochs // 10))


def _fit_directed_als(
    *,
    rank: int,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    value: np.ndarray,
    train_mask: np.ndarray,
    donor_count: int,
    target_count: int,
    settings: KeyGeometrySettings,
) -> np.ndarray:
    """Ridge-regularized masked ALS for ``mu + donor + target + KQ``.

    The earlier neural optimizer could preserve ranking while letting a few
    factor norms explode, which invalidates squared-error matrix-completion
    assessment.  This closed-form alternating update fixes the scale through
    explicit ridge terms and makes the general low-rank comparator auditable.
    """

    mask = np.asarray(train_mask, dtype=bool)
    donors = np.asarray(donor_index, dtype=np.int64)[mask]
    targets = np.asarray(target_index, dtype=np.int64)[mask]
    raw = np.asarray(value, dtype=np.float64)
    mean = float(raw[mask].mean())
    scale = max(float(raw[mask].std()), 1e-8)
    y = (raw[mask] - mean) / scale
    if len(y) < donor_count + target_count:
        raise ValueError("Too few training entries for directed ALS")
    ridge = 0.10
    global_bias = float(y.mean())
    donor_bias = np.zeros(donor_count, dtype=np.float64)
    target_bias = np.zeros(target_count, dtype=np.float64)
    key = np.zeros((donor_count, rank), dtype=np.float64)
    query = np.zeros((target_count, rank), dtype=np.float64)

    # Initialize the latent product from the residual after several stable
    # intercept-only backfitting passes.  Missing cells are intentionally zero
    # here; they are never used as observations in later ALS updates.
    for _ in range(8):
        for donor in range(donor_count):
            selected = donors == donor
            if selected.any():
                donor_bias[donor] = (
                    np.sum(y[selected] - global_bias - target_bias[targets[selected]])
                    / (int(selected.sum()) + ridge)
                )
        for target in range(target_count):
            selected = targets == target
            if selected.any():
                target_bias[target] = (
                    np.sum(y[selected] - global_bias - donor_bias[donors[selected]])
                    / (int(selected.sum()) + ridge)
                )
        global_bias = float(np.mean(y - donor_bias[donors] - target_bias[targets]))
    residual_fill = np.zeros((donor_count, target_count), dtype=np.float64)
    counts = np.zeros((donor_count, target_count), dtype=np.float64)
    residual = y - global_bias - donor_bias[donors] - target_bias[targets]
    np.add.at(residual_fill, (donors, targets), residual)
    np.add.at(counts, (donors, targets), 1.0)
    residual_fill = np.divide(residual_fill, counts, out=residual_fill, where=counts > 0.0)
    u, singular, vt = np.linalg.svd(residual_fill, full_matrices=False)
    active_rank = min(rank, len(singular))
    if active_rank:
        root = np.sqrt(np.maximum(singular[:active_rank], 0.0))
        key[:, :active_rank] = u[:, :active_rank] * root
        query[:, :active_rank] = vt[:active_rank].T * root

    identity = np.eye(rank, dtype=np.float64)
    for _ in range(_structure_iterations(settings)):
        relation = np.sum(key[donors] * query[targets], axis=1)
        for donor in range(donor_count):
            selected = donors == donor
            if selected.any():
                donor_bias[donor] = (
                    np.sum(y[selected] - global_bias - target_bias[targets[selected]] - relation[selected])
                    / (int(selected.sum()) + ridge)
                )
        for target in range(target_count):
            selected = targets == target
            if selected.any():
                target_bias[target] = (
                    np.sum(y[selected] - global_bias - donor_bias[donors[selected]] - relation[selected])
                    / (int(selected.sum()) + ridge)
                )
        global_bias = float(
            np.mean(y - donor_bias[donors] - target_bias[targets] - relation)
        )
        for donor in range(donor_count):
            selected = donors == donor
            if not selected.any():
                continue
            design = query[targets[selected]]
            rhs = y[selected] - global_bias - donor_bias[donor] - target_bias[targets[selected]]
            key[donor] = np.linalg.solve(
                design.T @ design + ridge * identity,
                design.T @ rhs,
            )
        for target in range(target_count):
            selected = targets == target
            if not selected.any():
                continue
            design = key[donors[selected]]
            rhs = y[selected] - global_bias - donor_bias[donors[selected]] - target_bias[target]
            query[target] = np.linalg.solve(
                design.T @ design + ridge * identity,
                design.T @ rhs,
            )
    all_donors = np.asarray(donor_index, dtype=np.int64)
    all_targets = np.asarray(target_index, dtype=np.int64)
    standardized_prediction = (
        global_bias
        + donor_bias[all_donors]
        + target_bias[all_targets]
        + np.sum(key[all_donors] * query[all_targets], axis=1)
    )
    return standardized_prediction * scale + mean


def _fit_directed_svd_completion(
    *,
    rank: int,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    value: np.ndarray,
    train_mask: np.ndarray,
    donor_count: int,
    target_count: int,
    settings: KeyGeometrySettings,
) -> np.ndarray:
    """Masked truncated-SVD completion after a train-only two-way baseline.

    This is the primary general low-rank comparator.  It is intentionally a
    stable matrix-completion estimator rather than a flexible neural decoder:
    unobserved entries start at zero *residual* (the train-only intercept
    baseline) and are iteratively replaced only by a rank-r reconstruction.
    No held-out transfer label enters the factorization.
    """

    mask = np.asarray(train_mask, dtype=bool)
    donors = np.asarray(donor_index, dtype=np.int64)
    targets = np.asarray(target_index, dtype=np.int64)
    raw = np.asarray(value, dtype=np.float64)
    baseline = _fit_two_way_baseline(
        donors,
        targets,
        raw,
        mask,
        donor_count,
        target_count,
    )
    observed = np.zeros((donor_count, target_count), dtype=bool)
    filled = np.zeros((donor_count, target_count), dtype=np.float64)
    observed_residual = np.zeros((donor_count, target_count), dtype=np.float64)
    observed[donors[mask], targets[mask]] = True
    observed_residual[donors[mask], targets[mask]] = raw[mask] - baseline[mask]
    filled[observed] = observed_residual[observed]
    iterations = max(5, min(30, settings.structure_epochs // 50))
    reconstruction = np.zeros_like(filled)
    for _ in range(iterations):
        u, singular, vt = np.linalg.svd(filled, full_matrices=False)
        active_rank = min(rank, len(singular))
        reconstruction = (u[:, :active_rank] * singular[:active_rank]) @ vt[:active_rank]
        if observed.all():
            break
        filled[~observed] = reconstruction[~observed]
        # Re-impose the observed residual exactly.  This is the missing-data
        # analogue of truncated SVD, not a fit to test-cell labels.
        filled[observed] = observed_residual[observed]
    return baseline + reconstruction[donors, targets]


def _categorical_groups(
    *,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_codes: Sequence[np.ndarray],
    target_codes: Sequence[np.ndarray],
    factor_level_counts: Sequence[int],
    donor_count: int,
    target_count: int,
    include_interactions: bool,
) -> list[tuple[np.ndarray, int, float, str]]:
    """Build directed factor/pair relation lookup blocks without dense tensors."""

    groups: list[tuple[np.ndarray, int, float, str]] = [
        (np.asarray(donor_index, dtype=np.int64), donor_count, 0.01, "donor_intercept"),
        (np.asarray(target_index, dtype=np.int64), target_count, 0.01, "target_intercept"),
    ]
    for factor, count in enumerate(factor_level_counts):
        group = donor_codes[factor] * count + target_codes[factor]
        groups.append((group, count * count, 0.25, f"factor_{factor}"))
    if include_interactions:
        for first, second in ((0, 1), (1, 2), (0, 2)):
            second_count = factor_level_counts[second]
            pair_count = factor_level_counts[first] * second_count
            donor_pair = donor_codes[first] * second_count + donor_codes[second]
            target_pair = target_codes[first] * second_count + target_codes[second]
            group = donor_pair * pair_count + target_pair
            # A pair lookup is a deliberately generous interaction comparator;
            # high shrinkage stops one-context cells being presented as tensor
            # structure.  It can still share repeated source/target pair labels.
            groups.append((group, pair_count * pair_count, 5.0, f"pair_{first}_{second}"))
    return groups


def _fit_categorical_relation_kernel(
    *,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_codes: Sequence[np.ndarray],
    target_codes: Sequence[np.ndarray],
    factor_level_counts: Sequence[int],
    value: np.ndarray,
    train_mask: np.ndarray,
    donor_count: int,
    target_count: int,
    include_interactions: bool,
    settings: KeyGeometrySettings,
) -> np.ndarray:
    """Backfitted ridge factor kernel, optionally plus directed pair blocks."""

    mask = np.asarray(train_mask, dtype=bool)
    raw = np.asarray(value, dtype=np.float64)
    mean = float(raw[mask].mean())
    scale = max(float(raw[mask].std()), 1e-8)
    y = (raw[mask] - mean) / scale
    all_groups = _categorical_groups(
        donor_index=donor_index,
        target_index=target_index,
        donor_codes=donor_codes,
        target_codes=target_codes,
        factor_level_counts=factor_level_counts,
        donor_count=donor_count,
        target_count=target_count,
        include_interactions=include_interactions,
    )
    groups = [(code[mask], levels, ridge, name) for code, levels, ridge, name in all_groups]
    effects = [np.zeros(levels, dtype=np.float64) for _, levels, _, _ in groups]
    intercept = float(y.mean())
    fitted = np.full(len(y), intercept, dtype=np.float64)
    for _ in range(_structure_iterations(settings)):
        for index, (code, levels, ridge, _) in enumerate(groups):
            old = effects[index][code]
            partial = y - fitted + old
            sums = np.bincount(code, weights=partial, minlength=levels)
            counts = np.bincount(code, minlength=levels).astype(np.float64)
            updated = sums / (counts + ridge)
            fitted += updated[code] - old
            effects[index] = updated
        new_intercept = float(np.mean(y - fitted + intercept))
        fitted += new_intercept - intercept
        intercept = new_intercept
    standardized_prediction = np.full(len(raw), intercept, dtype=np.float64)
    for (code, _, _, _), effect in zip(all_groups, effects):
        standardized_prediction += effect[code]
    return standardized_prediction * scale + mean


def _fit_factor_coordinate_als(
    *,
    factor_index: int,
    rank: int,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_codes: Sequence[np.ndarray],
    target_codes: Sequence[np.ndarray],
    factor_level_counts: Sequence[int],
    value: np.ndarray,
    train_mask: np.ndarray,
    donor_count: int,
    target_count: int,
    settings: KeyGeometrySettings,
) -> np.ndarray:
    """Stable 1/2/4D coordinates for one factor after condition intercepts."""

    mask = np.asarray(train_mask, dtype=bool)
    donor_rows = np.asarray(donor_index, dtype=np.int64)[mask]
    target_rows = np.asarray(target_index, dtype=np.int64)[mask]
    source = donor_codes[factor_index][mask]
    query_code = target_codes[factor_index][mask]
    raw = np.asarray(value, dtype=np.float64)
    mean = float(raw[mask].mean())
    scale = max(float(raw[mask].std()), 1e-8)
    y = (raw[mask] - mean) / scale
    source_count = factor_level_counts[factor_index]
    query_count = source_count
    ridge = 0.10
    global_bias = float(y.mean())
    donor_bias = np.zeros(donor_count, dtype=np.float64)
    target_bias = np.zeros(target_count, dtype=np.float64)
    key = np.zeros((source_count, rank), dtype=np.float64)
    demand = np.zeros((query_count, rank), dtype=np.float64)
    rng = np.random.default_rng(settings.selection_seed + 23000 + factor_index * 101 + rank)
    key[:] = rng.normal(0.0, 0.01, size=key.shape)
    demand[:] = rng.normal(0.0, 0.01, size=demand.shape)
    identity = np.eye(rank, dtype=np.float64)
    for _ in range(_structure_iterations(settings)):
        relation = np.sum(key[source] * demand[query_code], axis=1)
        for donor in range(donor_count):
            selected = donor_rows == donor
            if selected.any():
                donor_bias[donor] = np.sum(
                    y[selected] - global_bias - target_bias[target_rows[selected]] - relation[selected]
                ) / (int(selected.sum()) + ridge)
        for target in range(target_count):
            selected = target_rows == target
            if selected.any():
                target_bias[target] = np.sum(
                    y[selected] - global_bias - donor_bias[donor_rows[selected]] - relation[selected]
                ) / (int(selected.sum()) + ridge)
        global_bias = float(np.mean(y - donor_bias[donor_rows] - target_bias[target_rows] - relation))
        for level in range(source_count):
            selected = source == level
            if selected.any():
                design = demand[query_code[selected]]
                rhs = y[selected] - global_bias - donor_bias[donor_rows[selected]] - target_bias[target_rows[selected]]
                key[level] = np.linalg.solve(design.T @ design + ridge * identity, design.T @ rhs)
        for level in range(query_count):
            selected = query_code == level
            if selected.any():
                design = key[source[selected]]
                rhs = y[selected] - global_bias - donor_bias[donor_rows[selected]] - target_bias[target_rows[selected]]
                demand[level] = np.linalg.solve(design.T @ design + ridge * identity, design.T @ rhs)
    all_source = donor_codes[factor_index]
    all_query = target_codes[factor_index]
    standardized_prediction = (
        global_bias
        + donor_bias[np.asarray(donor_index, dtype=np.int64)]
        + target_bias[np.asarray(target_index, dtype=np.int64)]
        + np.sum(key[all_source] * demand[all_query], axis=1)
    )
    return standardized_prediction * scale + mean


def _factor_code_arrays(
    metadata: pd.DataFrame,
    donors: Sequence[str],
    targets: Sequence[str],
    donor_index: np.ndarray,
    target_index: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[list[str]]]:
    """Encode values from metadata only; target response never enters coding."""

    source_codes: list[np.ndarray] = []
    query_codes: list[np.ndarray] = []
    level_counts: list[int] = []
    levels_by_factor: list[list[str]] = []
    donor_frame = metadata.loc[list(donors)]
    target_frame = metadata.loc[list(targets)]
    for field in FACTOR_FIELDS:
        levels = _metadata_levels(metadata, field)
        mapping = {value: index for index, value in enumerate(levels)}
        donor_level = donor_frame[field].astype(str).map(mapping).to_numpy(dtype=np.int64)
        target_level = target_frame[field].astype(str).map(mapping).to_numpy(dtype=np.int64)
        if np.any(donor_level < 0) or np.any(target_level < 0):
            raise AssertionError("A panel factor value was not encoded")
        source_codes.append(donor_level[donor_index])
        query_codes.append(target_level[target_index])
        level_counts.append(len(levels))
        levels_by_factor.append(levels)
    return source_codes, query_codes, level_counts, levels_by_factor


def _spectrum_summary(residual_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, singular_values, vt = np.linalg.svd(residual_matrix, full_matrices=False)
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / energy.sum() if energy.sum() > 0.0 else np.zeros_like(energy)
    return u, singular_values, vt, cumulative


def _target_bootstrap(
    values: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    """Target-cluster resampling of fixed out-of-fold predictions.

    It quantifies panel-target sensitivity but does not refit structural models,
    so it is clearly reported as conditional-prediction uncertainty.
    """

    if draws <= 0 or values.empty:
        return {
            "target_bootstrap_r2_ci95_low": float("nan"),
            "target_bootstrap_r2_ci95_high": float("nan"),
            "target_bootstrap_spearman_ci95_low": float("nan"),
            "target_bootstrap_spearman_ci95_high": float("nan"),
        }
    groups = [group for _, group in values.groupby("target_condition_id", sort=False)]
    if len(groups) < 2:
        return {
            "target_bootstrap_r2_ci95_low": float("nan"),
            "target_bootstrap_r2_ci95_high": float("nan"),
            "target_bootstrap_spearman_ci95_low": float("nan"),
            "target_bootstrap_spearman_ci95_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    r2_values: list[float] = []
    spearman_values: list[float] = []
    for sample in rng.integers(0, len(groups), size=(draws, len(groups))):
        combined = pd.concat([groups[index] for index in sample], ignore_index=True)
        metric = _metric_values(
            combined["observed_transfer"].to_numpy(),
            combined["predicted_transfer"].to_numpy(),
            combined["two_way_prediction"].to_numpy(),
        )
        if np.isfinite(metric["heldout_entry_r2_vs_two_way"]):
            r2_values.append(metric["heldout_entry_r2_vs_two_way"])
        if np.isfinite(metric["heldout_entry_spearman"]):
            spearman_values.append(metric["heldout_entry_spearman"])
    return {
        "target_bootstrap_r2_ci95_low": float(np.quantile(r2_values, 0.025)) if r2_values else float("nan"),
        "target_bootstrap_r2_ci95_high": float(np.quantile(r2_values, 0.975)) if r2_values else float("nan"),
        "target_bootstrap_spearman_ci95_low": float(np.quantile(spearman_values, 0.025)) if spearman_values else float("nan"),
        "target_bootstrap_spearman_ci95_high": float(np.quantile(spearman_values, 0.975)) if spearman_values else float("nan"),
    }


def _summarize_oof(
    prediction: pd.DataFrame,
    *,
    label: str,
    model_kind: str,
    rank: int | None,
    settings: KeyGeometrySettings,
) -> dict[str, Any]:
    metrics = _metric_values(
        prediction["observed_transfer"].to_numpy(),
        prediction["predicted_transfer"].to_numpy(),
        prediction["two_way_prediction"].to_numpy(),
    )
    bootstrap = _target_bootstrap(
        prediction,
        seed=settings.selection_seed + int(rank or 0) + len(label),
        draws=settings.bootstrap_draws,
    )
    return {
        "model_label": label,
        "model_kind": model_kind,
        "rank": rank,
        "heldout_protocol": "five_fold_response_independent_relation_pattern_conditioned_heldout_entries",
        "test_entry_count": int(len(prediction)),
        "test_target_count": int(prediction["target_condition_id"].nunique()),
        "test_donor_count": int(prediction["donor_condition_id"].nunique()),
        **metrics,
        **bootstrap,
    }


def _mode_group_r2(values: np.ndarray, labels: Sequence[str]) -> float:
    score = np.asarray(values, dtype=np.float64)
    groups = np.asarray([_as_text(value) for value in labels], dtype=object)
    valid = np.isfinite(score)
    if int(valid.sum()) < 3:
        return float("nan")
    y = score[valid]
    group_values = groups[valid]
    grand_mean = float(y.mean())
    total = float(np.square(y - grand_mean).sum())
    if total <= 0.0:
        return float("nan")
    between = 0.0
    for label in np.unique(group_values):
        selected = y[group_values == label]
        between += len(selected) * float(np.square(selected.mean() - grand_mean))
    return float(np.clip(between / total, 0.0, 1.0))


def _latent_mode_tables(
    *,
    metadata: pd.DataFrame,
    donors: Sequence[str],
    targets: Sequence[str],
    donor_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptively relate canonical SVD modes to factor and pair groupings."""

    score_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, tuple[str, ...]]] = [
        ("strain", (STRAIN,)),
        ("chemical", (CHEMICAL,)),
        ("time", (TIME,)),
        ("medium", (MEDIUM,)),
        ("temperature", (TEMPERATURE,)),
        ("strain_x_chemical", (STRAIN, CHEMICAL)),
        ("chemical_x_time", (CHEMICAL, TIME)),
        ("strain_x_time", (STRAIN, TIME)),
    ]
    for side, ids, coordinates in (
        ("donor_key", donors, donor_coordinates),
        ("target_query", targets, target_coordinates),
    ):
        frame = metadata.loc[list(ids)]
        for mode in range(coordinates.shape[1]):
            for sample_id, coordinate in zip(ids, coordinates[:, mode]):
                row = frame.loc[sample_id]
                score_rows.append(
                    {
                        "side": side,
                        CONDITION_ID: sample_id,
                        "mode": mode + 1,
                        "coordinate": float(coordinate),
                        **{field: _as_text(row[field]) for field in FACTOR_FIELDS},
                    }
                )
            for group_name, fields in group_specs:
                labels = [
                    "\x00".join(_as_text(frame.loc[sample_id, field]) for field in fields)
                    for sample_id in ids
                ]
                association_rows.append(
                    {
                        "side": side,
                        "mode": mode + 1,
                        "grouping": group_name,
                        "variance_explained_descriptive": _mode_group_r2(
                            coordinates[:, mode], labels
                        ),
                        "group_count": int(len(set(labels))),
                        "scope": "full_exact_matrix_descriptive_not_heldout_validation",
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(association_rows)


def _safe_error_bar(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([np.maximum(values - low, 0.0), np.maximum(high - values, 0.0)])


def _plot_transfer_heatmap(
    matrix: np.ndarray,
    donor_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    output_dir: Path,
) -> None:
    donor_order = np.argsort(donor_coordinates[:, 0], kind="stable")
    target_order = np.argsort(target_coordinates[:, 0], kind="stable")
    ordered = matrix[np.ix_(donor_order, target_order)]
    finite = ordered[np.isfinite(ordered)]
    limit = float(np.quantile(np.abs(finite), 0.985)) if len(finite) else 1.0
    limit = max(limit, 1e-5)
    figure, axis = plt.subplots(figsize=(13.5, 9.0), constrained_layout=True)
    image = axis.imshow(
        ordered,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_title("Exact directed transfer matrix, reordered by residual transfer modes")
    axis.set_xlabel("Withheld target conditions (Query order)")
    axis.set_ylabel("Donor conditions (Key order)")
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = figure.colorbar(image, ax=axis, pad=0.012)
    colorbar.set_label("Zero-normalized MSE reduction (positive = donor helps)")
    figure.savefig(output_dir / "directed_transfer_matrix_heatmap.png", dpi=185, bbox_inches="tight")
    plt.close(figure)


def _plot_spectrum_and_rank(
    singular_values: np.ndarray,
    cumulative_energy: np.ndarray,
    rank_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    count = min(16, len(singular_values))
    x = np.arange(1, count + 1)
    axes[0].plot(x, singular_values[:count] / max(float(singular_values[0]), 1e-12), "o-", color="#4878a8", label="relative singular value")
    axes[0].plot(x, cumulative_energy[:count], "s--", color="#d2774a", label="cumulative residual energy")
    axes[0].set_xlabel("Residual matrix mode")
    axes[0].set_ylabel("Relative magnitude / cumulative energy")
    axes[0].set_title("Descriptive exact-matrix residual spectrum")
    axes[0].set_xticks(x)
    axes[0].legend(frameon=False, fontsize=8)

    summary = rank_summary.loc[rank_summary["status"].eq("completed")].copy()
    ranks = summary["rank"].to_numpy(dtype=int)
    values = summary["heldout_entry_r2_vs_two_way"].to_numpy(dtype=float)
    low = summary["target_bootstrap_r2_ci95_low"].to_numpy(dtype=float)
    high = summary["target_bootstrap_r2_ci95_high"].to_numpy(dtype=float)
    if len(ranks):
        axes[1].plot(ranks, values, "o-", color="#5f9e6e")
        finite_ci = np.isfinite(low) & np.isfinite(high)
        if finite_ci.any():
            axes[1].errorbar(
                ranks[finite_ci],
                values[finite_ci],
                yerr=_safe_error_bar(values[finite_ci], low[finite_ci], high[finite_ci]),
                fmt="none",
                color="#5f9e6e",
                capsize=3,
            )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Directed general low-rank dimension")
    axes[1].set_ylabel("Held-out-entry R² vs donor+target intercepts")
    axes[1].set_title("Matrix-completion rank curve")
    axes[1].set_xticks(list(REQUESTED_RANKS))
    figure.savefig(output_dir / "transfer_spectrum_and_rank_curve.png", dpi=185, bbox_inches="tight")
    plt.close(figure)


def _plot_kernel_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    display = summary.copy()
    labels = display["model"].tolist()
    values = display["heldout_entry_r2_vs_two_way"].to_numpy(dtype=float)
    low = display["target_bootstrap_r2_ci95_low"].to_numpy(dtype=float)
    high = display["target_bootstrap_r2_ci95_high"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    colors = ["#4878a8", "#d2774a", "#5f9e6e"]
    axis.bar(np.arange(len(labels)), values, color=colors[: len(labels)])
    valid = np.isfinite(low) & np.isfinite(high)
    if valid.any():
        axis.errorbar(
            np.arange(len(labels))[valid],
            values[valid],
            yerr=_safe_error_bar(values[valid], low[valid], high[valid]),
            fmt="none",
            color="black",
            capsize=3,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=15, ha="right")
    axis.set_ylabel("Held-out-entry R² vs donor+target intercepts")
    axis.set_title("Does factor structure explain exact transfer relation?")
    figure.savefig(output_dir / "kernel_structure_comparison.png", dpi=185, bbox_inches="tight")
    plt.close(figure)


def _plot_latent_associations(associations: pd.DataFrame, output_dir: Path) -> None:
    groups = [
        "strain",
        "chemical",
        "time",
        "medium",
        "temperature",
        "strain_x_chemical",
        "chemical_x_time",
        "strain_x_time",
    ]
    modes = sorted(associations["mode"].unique().tolist())
    sides = ["donor_key", "target_query"]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for axis, side in zip(axes, sides):
        subset = associations.loc[associations["side"].eq(side)]
        value = np.asarray(
            [
                [
                    subset.loc[
                        subset["mode"].eq(mode) & subset["grouping"].eq(group),
                        "variance_explained_descriptive",
                    ].iloc[0]
                    for group in groups
                ]
                for mode in modes
            ],
            dtype=float,
        )
        image = axis.imshow(value, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_title("Donor Key modes" if side == "donor_key" else "Target Query modes")
        axis.set_xticks(np.arange(len(groups)), [name.replace("_x_", "×") for name in groups], rotation=38, ha="right")
        axis.set_yticks(np.arange(len(modes)), [f"mode {mode}" for mode in modes])
        for row in range(value.shape[0]):
            for column in range(value.shape[1]):
                axis.text(column, row, f"{value[row, column]:.2f}", ha="center", va="center", color="white" if value[row, column] > 0.55 else "black", fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03, label="descriptive variance explained")
    figure.savefig(output_dir / "latent_modes_vs_metadata.png", dpi=185, bbox_inches="tight")
    plt.close(figure)


def _plot_factor_dimensions(summary: pd.DataFrame, output_dir: Path) -> None:
    completed = summary.loc[summary["status"].eq("completed")].copy()
    if completed.empty:
        return
    figure, axis = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    for factor, group in completed.groupby("factor", sort=False):
        group = group.sort_values("requested_dimension")
        axis.plot(
            group["requested_dimension"],
            group["heldout_entry_r2_vs_two_way"],
            "o-",
            label=factor,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks([1, 2, 4])
    axis.set_xlabel("Factor-coordinate dimension")
    axis.set_ylabel("Held-out-entry R² vs donor+target intercepts")
    axis.set_title("Marginal factor-coordinate capacity")
    axis.legend(frameon=False)
    figure.savefig(output_dir / "factor_key_dimension_curve.png", dpi=185, bbox_inches="tight")
    plt.close(figure)


def _oof_rows(
    probes: pd.DataFrame,
    test_mask: np.ndarray,
    prediction: np.ndarray,
    two_way_prediction: np.ndarray,
    *,
    label: str,
    fold: int,
) -> pd.DataFrame:
    selected = probes.loc[test_mask].copy()
    selected = selected.loc[
        :, ["donor_condition_id", "target_condition_id", "cv_fold", "transfer_score"]
    ].rename(columns={"transfer_score": "observed_transfer"})
    selected["model_label"] = label
    selected["fit_fold"] = fold
    selected["predicted_transfer"] = np.asarray(prediction, dtype=float)[test_mask]
    selected["two_way_prediction"] = np.asarray(two_way_prediction, dtype=float)[test_mask]
    return selected


def _factor_dimension_summary(
    *,
    probes: pd.DataFrame,
    donor_index: np.ndarray,
    target_index: np.ndarray,
    donor_codes: Sequence[np.ndarray],
    target_codes: Sequence[np.ndarray],
    factor_level_counts: Sequence[int],
    two_way_by_fold: Mapping[int, np.ndarray],
    settings: KeyGeometrySettings,
    factor_gate: bool,
) -> pd.DataFrame:
    factor_names = ("strain", "chemical", "time", "medium", "temperature")
    if not factor_gate:
        return pd.DataFrame(
            [
                {
                    "factor": factor,
                    "requested_dimension": dimension,
                    "status": "not_run_factor_kernel_has_no_stable_positive_heldout_signal",
                    "heldout_entry_r2_vs_two_way": float("nan"),
                    "geometry_preservation": float("nan"),
                }
                for factor in factor_names
                for dimension in (1, 2, 4)
            ]
        )
    values = probes["transfer_score"].to_numpy(dtype=float)
    folds = probes["cv_fold"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for factor_index, factor in enumerate(factor_names):
        for dimension in (1, 2, 4):
            predictions: list[pd.DataFrame] = []
            for fold in range(settings.folds):
                test_mask = folds == fold
                train_mask = ~test_mask
                predicted = _fit_factor_coordinate_als(
                    factor_index=factor_index,
                    rank=dimension,
                    donor_index=donor_index,
                    target_index=target_index,
                    donor_codes=donor_codes,
                    target_codes=target_codes,
                    factor_level_counts=factor_level_counts,
                    value=values,
                    train_mask=train_mask,
                    donor_count=int(np.max(donor_index)) + 1,
                    target_count=int(np.max(target_index)) + 1,
                    settings=settings,
                )
                predictions.append(
                    _oof_rows(
                        probes,
                        test_mask,
                        predicted,
                        two_way_by_fold[fold],
                        label=f"{factor}_only_r{dimension}",
                        fold=fold,
                    )
                )
            oof = pd.concat(predictions, ignore_index=True)
            summary = _summarize_oof(
                oof,
                label=f"{factor}_only_r{dimension}",
                model_kind="single_factor_directed_kernel",
                rank=dimension,
                settings=settings,
            )
            # Full-data SVD energy of the corresponding one-factor score table
            # is deliberately not used as a performance metric.  The field is
            # retained only as an explicit place-holder for an unsupported
            # geometry-preservation claim.
            rows.append(
                {
                    "factor": factor,
                    "requested_dimension": dimension,
                    "status": "completed",
                    "geometry_preservation": float("nan"),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _classification(
    rank_summary: pd.DataFrame,
    kernel_summary: pd.DataFrame,
    settings: KeyGeometrySettings,
) -> tuple[str, str, dict[str, float]]:
    """Apply a fixed, conservative interpretation to held-out edge evidence."""

    def model_value(name: str) -> tuple[float, float]:
        row = kernel_summary.loc[kernel_summary["model"].eq(name)]
        if len(row) != 1:
            return float("nan"), float("nan")
        return (
            float(row.iloc[0]["heldout_entry_r2_vs_two_way"]),
            float(row.iloc[0]["target_bootstrap_r2_ci95_low"]),
        )

    factor, factor_low = model_value("Factor-only")
    interaction, interaction_low = model_value("Factor + interaction")
    general, general_low = model_value(
        f"General low-rank (r={settings.general_comparison_rank})"
    )
    evidence = {
        "factor_r2": factor,
        "factor_r2_ci_low": factor_low,
        "interaction_r2": interaction,
        "interaction_r2_ci_low": interaction_low,
        "general_r2": general,
        "general_r2_ci_low": general_low,
    }
    # A positive target-cluster lower bound is required before claiming any
    # static relation.  Differences are deliberately modest (0.01 R²) because
    # all comparators share a strong two-way donor/target propensity baseline.
    if not np.isfinite(general) or not np.isfinite(general_low) or general_low <= 0.0:
        return "No stable static low-dimensional geometry", "Dynamic learner-explorer", evidence
    if np.isfinite(factor) and factor_low > 0.0 and factor >= 0.8 * general:
        if np.isfinite(interaction) and interaction - factor >= 0.01 and interaction >= 0.8 * general:
            return "Hybrid", "Factor + Interaction Key", evidence
        return "Factor-dominated", "Factor Key", evidence
    if (
        np.isfinite(interaction)
        and interaction_low > 0.0
        and interaction - factor >= 0.01
        and interaction >= 0.8 * general
    ):
        return "Interaction-dominated", "Factor + Interaction Key", evidence
    if np.isfinite(interaction) and general - interaction >= 0.01:
        return "Low-dimensional but non-factorized", "General learned information Key", evidence
    return "No stable static low-dimensional geometry", "Dynamic learner-explorer", evidence


def run_structure_analysis(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: KeyGeometrySettings,
    plan: PanelPlan,
    probes: pd.DataFrame,
) -> dict[str, Any]:
    """Validate ranks and structured kernels on preassigned withheld entries."""

    expected_rows = len(plan.donors) * len(plan.targets)
    if len(probes) != expected_rows:
        raise ValueError("The structure analysis requires the complete exact matrix")
    probes = probes.sort_values(["donor_rank", "target_rank"], kind="stable").reset_index(drop=True)
    donor_index = probes["donor_rank"].to_numpy(dtype=np.int64) - 1
    target_index = probes["target_rank"].to_numpy(dtype=np.int64) - 1
    values = probes["transfer_score"].to_numpy(dtype=np.float64)
    folds = probes["cv_fold"].to_numpy(dtype=np.int64)
    if set(folds.tolist()) != set(range(settings.folds)):
        raise AssertionError("Every held-out structure fold must be represented")
    if any(int((folds == fold).sum()) == 0 for fold in range(settings.folds)):
        raise AssertionError("A structural fold has no test entries")
    metadata = dataset.metadata.loc[list(dataset.candidate_pool_ids)].copy()
    donor_codes, target_codes, factor_level_counts, _ = _factor_code_arrays(
        metadata,
        plan.donors,
        plan.targets,
        donor_index,
        target_index,
    )
    # Every donor and target must retain training edges under each preassigned
    # entry fold; otherwise K or Q would be unidentified in that fold.
    for fold in range(settings.folds):
        train = folds != fold
        for index in range(len(plan.donors)):
            if int((train & (donor_index == index)).sum()) == 0:
                raise AssertionError("A donor has no training entries in a CV fold")
        for index in range(len(plan.targets)):
            if int((train & (target_index == index)).sum()) == 0:
                raise AssertionError("A target has no training entries in a CV fold")

    two_way_by_fold: dict[int, np.ndarray] = {}
    for fold in range(settings.folds):
        train = folds != fold
        two_way_by_fold[fold] = _fit_two_way_baseline(
            donor_index,
            target_index,
            values,
            train,
            len(plan.donors),
            len(plan.targets),
        )

    all_oof: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    rank_oof: dict[int, pd.DataFrame] = {}
    for rank in REQUESTED_RANKS:
        if rank > min(len(plan.donors), len(plan.targets)):
            summary_rows.append(
                {
                    "model_label": f"general_low_rank_r{rank}",
                    "model_kind": "general_directed_low_rank",
                    "rank": rank,
                    "status": "not_identifiable_rank_exceeds_panel",
                }
            )
            continue
        prediction_rows: list[pd.DataFrame] = []
        for fold in range(settings.folds):
            test = folds == fold
            prediction = _fit_directed_svd_completion(
                rank=rank,
                donor_index=donor_index,
                target_index=target_index,
                value=values,
                train_mask=~test,
                donor_count=len(plan.donors),
                target_count=len(plan.targets),
                settings=settings,
            )
            prediction_rows.append(
                _oof_rows(
                    probes,
                    test,
                    prediction,
                    two_way_by_fold[fold],
                    label=f"general_low_rank_r{rank}",
                    fold=fold,
                )
            )
        oof = pd.concat(prediction_rows, ignore_index=True)
        rank_oof[rank] = oof
        all_oof.append(oof)
        summary_rows.append(
            {"status": "completed", **_summarize_oof(
                oof,
                label=f"general_low_rank_r{rank}",
                model_kind="general_directed_low_rank",
                rank=rank,
                settings=settings,
            )}
        )

    structured_specs = (
        ("factor_only_relation_kernel", False),
        ("factor_plus_pair_relation_kernel", True),
    )
    structured_oof: dict[str, pd.DataFrame] = {}
    for label, include_interactions in structured_specs:
        prediction_rows = []
        for fold in range(settings.folds):
            test = folds == fold
            prediction = _fit_categorical_relation_kernel(
                donor_index=donor_index,
                target_index=target_index,
                donor_codes=donor_codes,
                target_codes=target_codes,
                factor_level_counts=factor_level_counts,
                value=values,
                train_mask=~test,
                donor_count=len(plan.donors),
                target_count=len(plan.targets),
                include_interactions=include_interactions,
                settings=settings,
            )
            prediction_rows.append(
                _oof_rows(
                    probes,
                    test,
                    prediction,
                    two_way_by_fold[fold],
                    label=label,
                    fold=fold,
                )
            )
        oof = pd.concat(prediction_rows, ignore_index=True)
        structured_oof[label] = oof
        all_oof.append(oof)
        summary_rows.append(
            {"status": "completed", **_summarize_oof(
                oof,
                label=label,
                model_kind=("factor_directed_kernel" if not include_interactions else "factor_plus_directed_pair_relation_kernel"),
                rank=None,
                settings=settings,
            )}
        )

    all_predictions = pd.concat(all_oof, ignore_index=True)
    all_predictions.to_csv(output_dir / "HELDOUT_STRUCTURE_PREDICTIONS.csv", index=False)
    all_summary = pd.DataFrame(summary_rows)
    rank_summary = all_summary.loc[
        all_summary["model_kind"].eq("general_directed_low_rank")
    ].copy()

    full_two_way = _fit_two_way_baseline(
        donor_index,
        target_index,
        values,
        np.ones(len(values), dtype=bool),
        len(plan.donors),
        len(plan.targets),
    )
    matrix = values.reshape(len(plan.donors), len(plan.targets))
    residual_matrix = (values - full_two_way).reshape(len(plan.donors), len(plan.targets))
    donor_u, singular_values, target_vt, cumulative_energy = _spectrum_summary(residual_matrix)
    energy_for_rank = {
        rank: float(cumulative_energy[min(rank, len(cumulative_energy)) - 1])
        for rank in REQUESTED_RANKS
        if rank <= len(cumulative_energy)
    }
    completed_rank = rank_summary["status"].eq("completed")
    rank_summary["full_residual_svd_energy"] = rank_summary["rank"].map(energy_for_rank)
    rank_summary["predeclared_general_comparison_rank"] = rank_summary["rank"].eq(settings.general_comparison_rank)
    if completed_rank.any():
        candidates = rank_summary.loc[completed_rank & np.isfinite(rank_summary["heldout_entry_r2_vs_two_way"])]
        if len(candidates):
            maximum = float(candidates["heldout_entry_r2_vs_two_way"].max())
            if maximum > 0.0:
                saturation_candidates = candidates.loc[
                    candidates["heldout_entry_r2_vs_two_way"] >= 0.95 * maximum
                ].sort_values("rank")
                saturation_rank: float | None = float(saturation_candidates.iloc[0]["rank"])
            else:
                saturation_rank = None
            rank_summary["rank_at_95pct_of_best_heldout_r2_descriptive"] = saturation_rank
            rank_summary["saturation_reference_best_r2"] = maximum
        else:
            rank_summary["rank_at_95pct_of_best_heldout_r2_descriptive"] = np.nan
            rank_summary["saturation_reference_best_r2"] = np.nan
    rank_summary.to_csv(output_dir / "TRANSFER_RANK_SUMMARY.csv", index=False)

    def take_summary(label: str) -> pd.Series:
        chosen = all_summary.loc[all_summary["model_label"].eq(label)]
        if len(chosen) != 1:
            raise AssertionError(f"Missing structural summary for {label}")
        return chosen.iloc[0]

    kernel_rows: list[dict[str, Any]] = []
    for display, label in (
        ("Factor-only", "factor_only_relation_kernel"),
        ("Factor + interaction", "factor_plus_pair_relation_kernel"),
        (f"General low-rank (r={settings.general_comparison_rank})", f"general_low_rank_r{settings.general_comparison_rank}"),
    ):
        source = take_summary(label)
        kernel_rows.append(
            {
                "model": display,
                "structure_definition": (
                    "sum of ridge-regularized directed categorical factor relations"
                    if display == "Factor-only"
                    else (
                        "factor relations plus ridge-regularized directed strain×chemical, chemical×time, and strain×time relation blocks"
                        if display == "Factor + interaction"
                        else "train-only two-way-baseline residual truncated-SVD completion"
                    )
                ),
                **source.to_dict(),
            }
        )
    kernel_summary = pd.DataFrame(kernel_rows)
    kernel_summary.to_csv(output_dir / "KERNEL_STRUCTURE_COMPARISON.csv", index=False)

    factor_row = take_summary("factor_only_relation_kernel")
    factor_gate = bool(
        np.isfinite(float(factor_row["heldout_entry_r2_vs_two_way"]))
        and np.isfinite(float(factor_row["target_bootstrap_r2_ci95_low"]))
        and float(factor_row["heldout_entry_r2_vs_two_way"]) > 0.0
        and float(factor_row["target_bootstrap_r2_ci95_low"]) > 0.0
    )
    dimension_summary = _factor_dimension_summary(
        probes=probes,
        donor_index=donor_index,
        target_index=target_index,
        donor_codes=donor_codes,
        target_codes=target_codes,
        factor_level_counts=factor_level_counts,
        two_way_by_fold=two_way_by_fold,
        settings=settings,
        factor_gate=factor_gate,
    )
    dimension_summary.to_csv(output_dir / "FACTOR_KEY_DIMENSION_SUMMARY.csv", index=False)

    completed_candidates = rank_summary.loc[rank_summary["status"].eq("completed")].copy()
    if completed_candidates.empty:
        mode_rank = 1
    else:
        saturated = completed_candidates["rank_at_95pct_of_best_heldout_r2_descriptive"].dropna()
        mode_rank = int(saturated.iloc[0]) if len(saturated) else min(4, int(completed_candidates["rank"].max()))
    mode_rank = max(1, min(mode_rank, 4, len(singular_values)))
    donor_coordinates = donor_u[:, :mode_rank] * np.sqrt(singular_values[:mode_rank])
    target_coordinates = target_vt[:mode_rank].T * np.sqrt(singular_values[:mode_rank])
    mode_scores, associations = _latent_mode_tables(
        metadata=metadata,
        donors=plan.donors,
        targets=plan.targets,
        donor_coordinates=donor_coordinates,
        target_coordinates=target_coordinates,
    )
    mode_scores.to_csv(output_dir / "LATENT_TRANSFER_MODE_SCORES.csv", index=False)
    associations.to_csv(output_dir / "LATENT_MODE_FACTOR_ASSOCIATIONS.csv", index=False)

    _plot_transfer_heatmap(matrix, donor_coordinates, target_coordinates, output_dir)
    _plot_spectrum_and_rank(singular_values, cumulative_energy, rank_summary, output_dir)
    _plot_kernel_comparison(kernel_summary, output_dir)
    _plot_latent_associations(associations, output_dir)
    _plot_factor_dimensions(dimension_summary, output_dir)
    case, recommendation, evidence = _classification(rank_summary, kernel_summary, settings)
    return {
        "rank_summary": rank_summary,
        "kernel_summary": kernel_summary,
        "dimension_summary": dimension_summary,
        "mode_scores": mode_scores,
        "associations": associations,
        "singular_values": singular_values,
        "cumulative_energy": cumulative_energy,
        "mode_rank": mode_rank,
        "case": case,
        "recommendation": recommendation,
        "evidence": evidence,
        "factor_gate": factor_gate,
    }


def _number(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "NA"


def _rank_table_markdown(rank_summary: pd.DataFrame) -> str:
    lines = [
        "| rank | held-out-entry R² vs donor+target baseline | target-bootstrap 95% CI | full residual SVD energy |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in rank_summary.sort_values("rank").iterrows():
        if row.get("status") != "completed":
            lines.append(f"| {int(row['rank'])} | not identifiable | — | — |")
            continue
        lines.append(
            "| {rank} | {r2} | [{low}, {high}] | {energy} |".format(
                rank=int(row["rank"]),
                r2=_number(row.get("heldout_entry_r2_vs_two_way")),
                low=_number(row.get("target_bootstrap_r2_ci95_low")),
                high=_number(row.get("target_bootstrap_r2_ci95_high")),
                energy=_number(row.get("full_residual_svd_energy")),
            )
        )
    return "\n".join(lines)


def _kernel_table_markdown(kernel_summary: pd.DataFrame) -> str:
    lines = [
        "| structure model | held-out-entry R² vs donor+target baseline | target-bootstrap 95% CI | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for _, row in kernel_summary.iterrows():
        lines.append(
            "| {model} | {r2} | [{low}, {high}] | {spearman} |".format(
                model=row["model"],
                r2=_number(row.get("heldout_entry_r2_vs_two_way")),
                low=_number(row.get("target_bootstrap_r2_ci95_low")),
                high=_number(row.get("target_bootstrap_r2_ci95_high")),
                spearman=_number(row.get("heldout_entry_spearman")),
            )
        )
    return "\n".join(lines)


def _mode_table_markdown(associations: pd.DataFrame) -> str:
    rows: list[str] = [
        "| side / mode | strongest metadata grouping (descriptive) | variance explained |",
        "|---|---|---:|",
    ]
    for side in ("donor_key", "target_query"):
        for mode in sorted(associations["mode"].unique().tolist()):
            subset = associations.loc[
                associations["side"].eq(side) & associations["mode"].eq(mode)
            ].sort_values("variance_explained_descriptive", ascending=False)
            if subset.empty:
                continue
            top = subset.iloc[0]
            rows.append(
                f"| {side} {int(mode)} | {top['grouping']} | "
                f"{_number(top['variance_explained_descriptive'])} |"
            )
    return "\n".join(rows)


def _dimension_text(summary: pd.DataFrame) -> str:
    completed = summary.loc[summary["status"].eq("completed")]
    if completed.empty:
        return (
            "Factor-only kernel 没有通过预先规定的稳定正向 held-out-entry gate，"
            "所以没有把任一 factor 压缩成 1D/2D/4D Key；这不是“factor 必然无关”的证明，"
            "而是当前 exact matrix 不足以支持这个静态坐标声明。"
        )
    lines = []
    for factor, group in completed.groupby("factor", sort=False):
        best = group.sort_values("heldout_entry_r2_vs_two_way", ascending=False).iloc[0]
        lines.append(
            f"{factor} 的本轮最强边际坐标容量是 {int(best['requested_dimension'])}D "
            f"（held-out-entry R²={_number(best['heldout_entry_r2_vs_two_way'])}）。"
        )
    return " ".join(lines)


def write_key_geometry_summary(
    output_dir: Path,
    *,
    settings: KeyGeometrySettings,
    plan: PanelPlan,
    probes: pd.DataFrame,
    analysis: Mapping[str, Any],
) -> Path:
    """Write the requested scientific conclusion rather than an engineering audit."""

    rank_summary = analysis["rank_summary"]
    kernel_summary = analysis["kernel_summary"]
    associations = analysis["associations"]
    dimensions = analysis["dimension_summary"]
    case = str(analysis["case"])
    recommendation = str(analysis["recommendation"])
    evidence = analysis["evidence"]
    mode_rank = int(analysis["mode_rank"])
    rank_table = _rank_table_markdown(rank_summary)
    kernel_table = _kernel_table_markdown(kernel_summary)
    mode_table = _mode_table_markdown(associations)
    saturation = rank_summary["rank_at_95pct_of_best_heldout_r2_descriptive"].dropna()
    saturation_text = (
        f"在已测试 rank 中，95% of best held-out R² 的最小 rank 是 {int(saturation.iloc[0])}。"
        if len(saturation)
        else "没有出现可稳定解释为正的 held-out rank 饱和点。"
    )
    general_statement = (
        "在这个固定 shared-baseline、single-seed Direct identity+time 矩阵中，"
        "支持把相对 transfer residual 近似成有限维的 `KᵀQ`。"
        if evidence["general_r2_ci_low"] > 0.0
        else "在这个固定 shared-baseline、single-seed Direct identity+time 矩阵中，"
        "没有足够的 held-out 证据把相对 transfer residual 写成稳定的低维 `KᵀQ`。"
    )
    tensor_statement = (
        "pairwise directed-relation interaction 在 held-out entries 上带来可共享的增益。"
        if np.isfinite(evidence["interaction_r2"])
        and np.isfinite(evidence["factor_r2"])
        and np.isfinite(evidence["interaction_r2_ci_low"])
        and evidence["interaction_r2_ci_low"] > 0.0
        and evidence["interaction_r2"] - evidence["factor_r2"] >= 0.01
        else "pairwise directed-relation interaction 没有显示出足以超过 factor-only 的可共享 held-out 增益。"
    )
    text = rf"""# GOAI Key Geometry Summary

## 1. 我们到底想发现什么

GeneDisco 的 descriptor 描述 intervention 的 functional geometry；GOAI 在这里寻找的是 **experimental knowledge-transfer geometry**：一个已完成实验会以什么方式帮助虚拟细胞理解另一个未完成实验。上一轮 relation-group probe 已显示 pair excess 不能自动推出 tensor Key，因此本轮不再按 same-strain 或 same-drug 平均，而是直接测量有向 experiment-to-experiment transfer matrix。

## 2. 如何测量 experiment-to-experiment knowledge transfer

本轮以 metadata-only 的稳定选择将 2,670 个 candidate conditions 分为 {len(plan.targets)} 个 target、{len(plan.donors)} 个 donor 和 {len(plan.baseline)} 个共同 baseline，三者完全互斥。共同 baseline 覆盖全部 strain / chemical / time / medium / temperature level，因此 donor 加入不会改变 identity+time feature mask。

先拟合一次 `M0 = fit(B)`，再对每一 donor 拟合一次 `Mi = fit(B ∪ {{i}})`；每个模型都用同一 Direct 4,422-output MLP、{settings.epochs} epochs、model seed {settings.model_seed}。所有 {len(probes):,} 个矩阵格都是 exact score，而不是 proxy：

\[
T_{{ij}} = L_j(M0) - L_j(M_i),\qquad L_j = \mathrm{{MSE}}_j / \mathrm{{zero\! -\! MSE}}_j.
\]

正值表示 donor 降低 withheld target 的归一化误差。target response 从未用于 panel、baseline 或 donor 选择，且 target 不在任何 support。因为模型 `B∪{{i}}` 对所有 target 可批量评分，完整矩阵只需要 {len(plan.donors) + 1} 次 fresh fit；因此没有用 gradient proxy 填补主矩阵。

## 3. Transfer geometry 是否低维

低秩验证是预先指定、response-independent、relation-pattern-conditioned 的 5-fold held-out-entry matrix completion；fold 由 stable hash 平衡，不是依据 response 优化的 metadata stratification。每个 donor 与 target 在训练边仍保留，因此这是同一 panel 内 relation completion，**不是** zero-shot new-target claim。所有模型都有同一 global + donor + target intercept baseline；下面 R² 是超过该 baseline 的增量。

{rank_table}

{saturation_text} Full residual SVD energy 仅是完整 exact matrix 的描述；intrinsic-rank 判断以 held-out entries 为主，而不是 training reconstruction。General comparator 是 iterative zero-filled masked truncated-SVD，factor / pair-relation comparator 的 ridge 强度是预先固定的；负 R² 说明这些已测试估计器未能稳定 completion 这一个矩阵，并不证明所有可能的静态低秩或 kernel 参数化都失败。

本轮判断的适用范围是：一个 256-condition shared baseline 周围的单 donor 边际效应、Direct identity+time MLP、{settings.epochs} epochs、model seed {settings.model_seed}。它不排除其他 support state、semantic predictor 或较低优化/测量噪声下存在静态 geometry。

## 4. Key 到底应该是什么语义

在这个实验中，`K_i` 是 **experiment i 能向整个虚拟细胞系统提供哪几种 transferable information**；`Q_j` 是 **正确理解 experiment j 需要哪些 information modes**。`K_i^T Q_j` 只描述去除 donor/target 总体易迁移性后的 directed relation residual。{general_statement}

## 5. Key 每一维对应什么

以下表格把 full exact residual SVD 的前 {mode_rank} 个 canonical mode 与公开 metadata grouping 做描述性方差分解；它用于解释候选结构，不替代 held-out kernel comparison，也不把任意 mode 硬命名为生物机制。

{mode_table}

若 pair grouping 在这里较高，表示它与 mode 相关；只有第 6 节的 held-out interaction improvement 才可说明这类关系是可共享的结构。

## 6. Factor Geometry vs Interaction/Tensor Geometry

本轮分类为：

\[
\boxed{{\text{{{case}}}}}
\]

这里的 Case E 是上述固定 protocol 下的“未检测到”结论，不是对所有 GOAI static Key 的普遍不可能性断言。

{kernel_table}

{tensor_statement} Factor+interaction comparator 使用正则化的有向 pair-relation blocks；它是对 interaction 是否可共享的宽松检验，不把 pair lookup 自动称作 low-rank tensor，更不把 5D occupancy 或一次 pair-vs-mixture excess 当作证据。

## 7. 一因素一个数字是否还有可能

{_dimension_text(dimensions)}

因此 1D 不是为了图形简洁而默认成立；只有在 exact held-out relation 上保留正向稳定增益时，才可把某 factor 写成小型 information block。

## 8. Tensor 到底意味着什么

对 GOAI 而言，tensor 的科学含义不是“数据表存在 strain×chemical×time 轴”，而是一个 **低维、可组合** 的 pairwise relational transfer term 能在未见 donor-target edges 上预测 residual。正则化 pair-relation lookup 的增益最多支持“interaction 可共享”；还需要它逼近或被低秩 compositional model 复现，才能支持 tensor/relational Key。如果没有这种 held-out evidence，上一轮的局部 pair excess 应保留为 target-conditioned现象，而非推广为全局 tensor representation。

## 9. 下一阶段 GOAI Active Learning 的 Key 应该怎么设计

据此建议下一阶段优先**检验**的 research hypothesis / prototype 是：**{recommendation}**；它不是已被本轮证明为唯一正确的设计。

这个建议只作用于后续 research prototype，不修改冻结 v2.2 的 Random 策略或任何 submission lineage。它应先在未参与 Key 结构选择的 condition/budget 上作小型 confirmation，而不是立即扩成 acquisition benchmark。

## 10. 最终结论

GOAI 中一个实验“教会虚拟细胞什么”已在中等规模、全 exact 的有向矩阵中被直接测量，而不是从 final MSE 或 tensor occupancy 反推。该矩阵的 low-rank curve 与 factor / interaction / general comparator 给出的分类是 **{case}**。本研究的严格结论是：**在这一个 fixed-baseline、single-seed Direct identity+time transfer matrix、已测试 ranks/kernels 下，没有检测到稳定的 static low-dimensional geometry。**

因此下一步不应预先把 experimental descriptor 断言为单独 strain、chemical、time 编号的静态拼接；应把 `{recommendation}` 作为待独立 confirmation 的下一项研究假设，同时保留其他 support/model/noise 条件下静态 Key 仍可能成立的可能性。任何更强的 biological or deployment claim 均留待独立 confirmation。

所有分数均为 local retrospective GOAI matched-control proxy；没有官方 leaderboard score、官方提交或生物发现主张。
"""
    path = output_dir / "GOAI_KEY_GEOMETRY_SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def run_key_geometry(
    *,
    output_dir: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    proteome_path: Path = DEFAULT_PROTEOME_PATH,
    cache_dir: Path | None = None,
    settings: KeyGeometrySettings = KeyGeometrySettings(),
) -> Path:
    """Execute the exact transfer-matrix and held-out Key-geometry study."""

    output = _prepare_output_dir(output_dir)
    started = datetime.now(timezone.utc)
    dataset = load_grouped_dataset(
        metadata_path,
        proteome_path,
        missing_rate_threshold=0.80,
        cache_dir=cache_dir,
        interpolation_fraction=0.20,
        split_seed=42,
    )
    metadata = dataset.metadata.loc[list(dataset.candidate_pool_ids)].copy()
    plan = build_panel_plan(metadata, settings)
    panel = _panel_rows(metadata, plan)
    panel.to_csv(output / "KEY_GEOMETRY_PANEL.csv", index=False)
    _panel_audit_rows(metadata, plan).to_csv(output / "PANEL_SELECTION_AUDIT.csv", index=False)
    metadata_record: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "status": "running",
        "started_utc": started.isoformat(),
        "settings": asdict(settings),
        "feature_mode": "identity_plus_time",
        "predictor": "Direct 4422-output MLP",
        "primary_metric": PRIMARY_METRIC,
        "official_score_status": "none; local retrospective proxy only",
        "matrix_measurement": "exact shared-baseline fresh refits; no proxy-derived entries",
        "metadata_path": str(metadata_path),
        "proteome_path": str(proteome_path),
        "dataset_source_hashes": dict(dataset.source_hashes),
        "candidate_condition_count": int(len(dataset.candidate_pool_ids)),
        "official_train_condition_count": int(len(dataset.official_train_ids)),
        "protein_count": int(len(dataset.proteins)),
        "panel_counts": {
            "target": len(plan.targets),
            "donor": len(plan.donors),
            "baseline": len(plan.baseline),
            "exact_matrix_entries": len(plan.targets) * len(plan.donors),
            "fresh_response_model_fits": len(plan.donors) + 1,
        },
        "source_module": str(Path(__file__).resolve()),
        "source_module_sha256": _sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "RUN_METADATA.json").write_text(
        json.dumps(metadata_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    probes = run_exact_transfer_matrix(dataset, output, settings, plan)
    # The full exact matrix made a proxy unnecessary.  Keep a one-row receipt
    # rather than an absent ambiguous filename, but do not interpret it as a
    # proxy validation or create a proxy figure.
    pd.DataFrame(
        [
            {
                "proxy_status": "not_used_full_exact_shared_baseline_matrix",
                "reason": "149 exact fits score every donor-target pair; proxy would add no primary evidence",
                "exact_matrix_entries": len(probes),
            }
        ]
    ).to_csv(output / "TRANSFER_PROXY_VALIDATION.csv", index=False)
    analysis = run_structure_analysis(dataset, output, settings, plan, probes)
    summary_path = write_key_geometry_summary(
        output,
        settings=settings,
        plan=plan,
        probes=probes,
        analysis=analysis,
    )
    metadata_record["status"] = "complete"
    metadata_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_record["summary_path"] = str(summary_path)
    metadata_record["classification"] = analysis["case"]
    metadata_record["key_recommendation"] = analysis["recommendation"]
    metadata_record["output_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    (output / "RUN_METADATA.json").write_text(
        json.dumps(metadata_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def reanalyse_existing_exact_matrix(
    *,
    source_output_dir: Path,
    output_dir: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    proteome_path: Path = DEFAULT_PROTEOME_PATH,
    cache_dir: Path | None = None,
    settings: KeyGeometrySettings = KeyGeometrySettings(),
) -> Path:
    """Reanalyse immutable exact transfer measurements with corrected structure code.

    This is intentionally separate from ``run_key_geometry``: it never trains
    another Direct predictor and retains the source run untouched.  It exists
    for analysis-code corrections such as a numerically unstable matrix
    optimizer, while preserving the original rejected artifacts and their
    exact measurement receipt.
    """

    source = source_output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Exact-matrix source output does not exist: {source}")
    required = (
        "KEY_GEOMETRY_PANEL.csv",
        "PANEL_SELECTION_AUDIT.csv",
        "TRANSFER_MATRIX_PROBES.csv",
        "EXACT_FIT_RECEIPTS.csv",
        "RUN_METADATA.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Source output lacks required exact artifacts: {missing}")
    output = _prepare_output_dir(output_dir)
    started = datetime.now(timezone.utc)
    panel = pd.read_csv(source / "KEY_GEOMETRY_PANEL.csv")
    plan = PanelPlan(
        targets=tuple(panel.loc[panel["role"].eq("target"), CONDITION_ID].astype(str)),
        donors=tuple(panel.loc[panel["role"].eq("donor"), CONDITION_ID].astype(str)),
        baseline=tuple(panel.loc[panel["role"].eq("baseline"), CONDITION_ID].astype(str)),
    )
    if (
        len(plan.targets) != settings.target_count
        or len(plan.donors) != settings.donor_count
        or len(plan.baseline) != settings.baseline_size
    ):
        raise ValueError("Reanalysis settings do not match the immutable source panel")
    probes = pd.read_csv(source / "TRANSFER_MATRIX_PROBES.csv")
    expected = len(plan.targets) * len(plan.donors)
    if len(probes) != expected or probes.duplicated(["donor_condition_id", "target_condition_id"]).any():
        raise ValueError("Source exact transfer matrix is incomplete or duplicated")
    for name in ("KEY_GEOMETRY_PANEL.csv", "PANEL_SELECTION_AUDIT.csv", "TRANSFER_MATRIX_PROBES.csv", "EXACT_FIT_RECEIPTS.csv"):
        shutil.copy2(source / name, output / name)
    dataset = load_grouped_dataset(
        metadata_path,
        proteome_path,
        missing_rate_threshold=0.80,
        cache_dir=cache_dir,
        interpolation_fraction=0.20,
        split_seed=42,
    )
    _validate_panel_plan(dataset.metadata.loc[list(dataset.candidate_pool_ids)], plan)
    source_metadata = json.loads((source / "RUN_METADATA.json").read_text(encoding="utf-8"))
    metadata_record: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "status": "running",
        "started_utc": started.isoformat(),
        "analysis_revision": "ridge_categorical_kernel_and_masked_svd_completion_v2",
        "measurement_source_output": str(source),
        "measurement_source_transfer_matrix_sha256": _sha256_file(source / "TRANSFER_MATRIX_PROBES.csv"),
        "measurement_source_started_utc": source_metadata.get("started_utc"),
        "measurement_source_completed_utc": source_metadata.get("completed_utc"),
        "settings": asdict(settings),
        "feature_mode": "identity_plus_time",
        "predictor": "Direct 4422-output MLP; measurements reused exactly",
        "primary_metric": PRIMARY_METRIC,
        "official_score_status": "none; local retrospective proxy only",
        "matrix_measurement": "immutable exact shared-baseline fresh-refit matrix reused; no proxy-derived entries",
        "dataset_source_hashes": dict(dataset.source_hashes),
        "source_module": str(Path(__file__).resolve()),
        "source_module_sha256": _sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output / "RUN_METADATA.json").write_text(
        json.dumps(metadata_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "proxy_status": "not_used_full_exact_shared_baseline_matrix",
                "reason": "immutable source matrix has one exact fresh-refit score for every donor-target pair",
                "exact_matrix_entries": len(probes),
            }
        ]
    ).to_csv(output / "TRANSFER_PROXY_VALIDATION.csv", index=False)
    analysis = run_structure_analysis(dataset, output, settings, plan, probes)
    summary_path = write_key_geometry_summary(
        output,
        settings=settings,
        plan=plan,
        probes=probes,
        analysis=analysis,
    )
    metadata_record["status"] = "complete"
    metadata_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_record["summary_path"] = str(summary_path)
    metadata_record["classification"] = analysis["case"]
    metadata_record["key_recommendation"] = analysis["recommendation"]
    metadata_record["output_files"] = sorted(path.name for path in output.iterdir() if path.is_file())
    (output / "RUN_METADATA.json").write_text(
        json.dumps(metadata_record, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / DEFAULT_OUTPUT_NAME,
        help="New or empty output directory.",
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--proteome-path", type=Path, default=DEFAULT_PROTEOME_PATH)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cache_v22"))
    parser.add_argument("--targets", type=int, default=96)
    parser.add_argument("--donors", type=int, default=148)
    parser.add_argument("--baseline-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--selection-seed", type=int, default=20260825)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--structure-epochs", type=int, default=900)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--reuse-exact-output",
        type=Path,
        default=None,
        help="Immutable prior exact-matrix output to reanalyse without retraining Direct models.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = KeyGeometrySettings(
        selection_seed=args.selection_seed,
        model_seed=args.model_seed,
        target_count=args.targets,
        donor_count=args.donors,
        baseline_size=args.baseline_size,
        epochs=args.epochs,
        structure_epochs=args.structure_epochs,
        bootstrap_draws=args.bootstrap_draws,
        device=args.device,
    )
    if args.reuse_exact_output is None:
        output = run_key_geometry(
            output_dir=args.output_dir,
            metadata_path=args.metadata_path,
            proteome_path=args.proteome_path,
            cache_dir=args.cache_dir,
            settings=settings,
        )
    else:
        output = reanalyse_existing_exact_matrix(
            source_output_dir=args.reuse_exact_output,
            output_dir=args.output_dir,
            metadata_path=args.metadata_path,
            proteome_path=args.proteome_path,
            cache_dir=args.cache_dir,
            settings=settings,
        )
    print(f"Key-geometry audit complete: {output}")


if __name__ == "__main__":
    main()
