"""Exact, small-scale knowledge-transfer geometry probes for GOAI-AL.

This module deliberately sits outside the frozen v2.2 active-learning runner.
It reuses its condition-atomic matched-control response, identity/time feature
contract, Direct predictor, and scoring metric to ask a different question:
how much does a matched donor group improve a withheld target condition?

The default panel is intentionally modest: 64 metadata-stratified targets,
six donors per arm, one fixed optimization seed, and 80 Direct-predictor
epochs.  Every target is withheld from every other target's support within a
panel; donor selection only reads public condition metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
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
from .metrics import score_response
from .model import ModelSettings, fit_response_model


EXPERIMENT_ID = "GOAI-AL-TRANSFER-GEOMETRY-01"
PARENT_MODEL_ID = "GOAI-AL-V22-DIRECT-SEMANTIC-01"
PRIMARY_METRIC = "delta_skill_zero"
DEFAULT_OUTPUT_NAME = "transfer_geometry-20260825-v1"
DEFAULT_METADATA_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_metadata_train_val.csv"
)
DEFAULT_PROTEOME_PATH = Path(
    "/home/chenyuming/Project/go-ai/WAYB_WAYC_proteome_raw_train_val.csv"
)

FOCAL_FIELDS = (STRAIN, CHEMICAL, TIME)
CONTEXT_FIELDS = (MEDIUM, TEMPERATURE)

# The arm names make all focal-axis equality constraints explicit.  "mixture"
# arms contain equal numbers of the corresponding single-factor donors.
MAIN_RELATIONS = (
    "strain_only",
    "chemical_only",
    "time_only",
    "strain_chemical",
    "chemical_time",
    "strain_time",
    "strain_chemical_mixture",
    "chemical_time_mixture",
    "strain_time_mixture",
    "random_none",
)
PAIR_TO_MIXTURE = {
    "strain_chemical": "strain_chemical_mixture",
    "chemical_time": "chemical_time_mixture",
    "strain_time": "strain_time_mixture",
}
SINGLE_TO_FIELD = {
    "strain_only": STRAIN,
    "chemical_only": CHEMICAL,
    "time_only": TIME,
}


@dataclass(frozen=True)
class ProbeSettings:
    """Frozen, small-scale design choices for this structural audit."""

    seed: int = 42
    target_count: int = 64
    baseline_size: int = 256
    donor_count: int = 6
    epochs: int = 80
    bootstrap_draws: int = 1000
    sensitivity_target_count: int = 12
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.donor_count <= 1 or self.donor_count % 2:
            raise ValueError("donor_count must be an even integer greater than one")
        if self.baseline_size < 16:
            raise ValueError("baseline_size must be at least 16")
        if self.target_count < 8:
            raise ValueError("target_count must be at least 8")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class ProbePlan:
    """Public, condition-ID-only plan for one withheld target."""

    target_id: str
    baseline_ids: tuple[str, ...]
    donor_ids: Mapping[str, tuple[str, ...]]
    anchors: Mapping[str, str]
    selection_seed: int


def _stable_rank(seed: int, namespace: str, *values: object) -> bytes:
    payload = json.dumps(
        [int(seed), namespace, *[str(value) for value in values]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _sorted_ids(
    ids: Iterable[object], *, seed: int, namespace: str, target_id: str
) -> list[str]:
    return sorted(
        (str(value) for value in ids),
        key=lambda value: (_stable_rank(seed, namespace, target_id, value), value),
    )


def _as_text(value: object) -> str:
    return str(value)


def _candidate_relation_masks(
    metadata: pd.DataFrame,
    target: pd.Series,
    *,
    blocked_ids: frozenset[str] = frozenset(),
) -> dict[str, np.ndarray]:
    """Return mutually exclusive S/D/T donor relation masks.

    Relation definitions use exact equality after the condition loader has
    normalized time to a numeric minute value.  Medium and temperature are not
    focal axes here: their balance is enforced during selection and audited.
    """

    strain = metadata[STRAIN].astype(str).to_numpy()
    chemical = metadata[CHEMICAL].astype(str).to_numpy()
    time = metadata[TIME].astype(str).to_numpy()
    target_strain = _as_text(target[STRAIN])
    target_chemical = _as_text(target[CHEMICAL])
    target_time = _as_text(target[TIME])
    same_strain = strain == target_strain
    same_chemical = chemical == target_chemical
    same_time = time == target_time
    permitted = ~np.asarray(metadata.index.astype(str).isin(blocked_ids), dtype=bool)
    return {
        "strain_only": permitted & same_strain & ~same_chemical & ~same_time,
        "chemical_only": permitted & ~same_strain & same_chemical & ~same_time,
        "time_only": permitted & ~same_strain & ~same_chemical & same_time,
        "strain_chemical": permitted & same_strain & same_chemical & ~same_time,
        "chemical_time": permitted & ~same_strain & same_chemical & same_time,
        "strain_time": permitted & same_strain & ~same_chemical & same_time,
        "random_none": permitted & ~same_strain & ~same_chemical & ~same_time,
    }


def _context_key(metadata: pd.DataFrame, sample_id: str, target: pd.Series) -> tuple[int, int]:
    row = metadata.loc[sample_id]
    return (
        int(_as_text(row[MEDIUM]) == _as_text(target[MEDIUM])),
        int(_as_text(row[TEMPERATURE]) == _as_text(target[TEMPERATURE])),
    )


def _balanced_ids(
    metadata: pd.DataFrame,
    candidates: Sequence[str],
    count: int,
    *,
    target: pd.Series,
    target_id: str,
    seed: int,
    namespace: str,
    excluded: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Deterministically sample a group while spreading medium/temperature cells."""

    usable = [str(value) for value in candidates if str(value) not in excluded]
    if len(usable) < count:
        raise ValueError(
            f"{target_id}: only {len(usable)} public candidates for {namespace}, "
            f"need {count}"
        )
    context = metadata.loc[usable, [MEDIUM, TEMPERATURE]]
    medium_match = context[MEDIUM].astype(str).eq(_as_text(target[MEDIUM])).to_numpy()
    temperature_match = (
        context[TEMPERATURE].astype(str).eq(_as_text(target[TEMPERATURE])).to_numpy()
    )
    grouped: dict[tuple[int, int], list[str]] = {}
    for sample_id, same_medium, same_temperature in zip(
        usable, medium_match, temperature_match
    ):
        grouped.setdefault((int(same_medium), int(same_temperature)), []).append(sample_id)
    for key, values in grouped.items():
        grouped[key] = _sorted_ids(
            values,
            seed=seed,
            namespace=f"{namespace}:{key[0]}:{key[1]}",
            target_id=target_id,
        )

    # Round robin prevents one available medium/temperature cell from dominating
    # a donor group.  The priority rotates deterministically per target/arm.
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            _stable_rank(seed, f"{namespace}:cell", target_id, key),
            key,
        ),
    )
    selected: list[str] = []
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            values = grouped[key]
            if values and len(selected) < count:
                selected.append(values.pop(0))
                progressed = True
        if not progressed:
            raise AssertionError("Balanced candidate selection stalled")
    return tuple(selected)


def _coverage_anchor_candidates(
    metadata: pd.DataFrame,
    target: pd.Series,
    field_name: str,
    *,
    blocked_ids: frozenset[str],
) -> list[str]:
    """Candidates that expose target medium/temperature without S/D/T overlap."""

    masks = _candidate_relation_masks(metadata, target, blocked_ids=blocked_ids)
    base = masks["random_none"]
    match = metadata[field_name].astype(str).to_numpy() == _as_text(target[field_name])
    return metadata.index[base & match].astype(str).tolist()


def _support_counts(
    metadata: pd.DataFrame,
    target_id: str,
    *,
    blocked_ids: frozenset[str],
) -> dict[str, int]:
    target = metadata.loc[target_id]
    masks = _candidate_relation_masks(metadata, target, blocked_ids=blocked_ids)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    random_mask = masks["random_none"]
    counts["medium_anchor"] = int(
        (random_mask & metadata[MEDIUM].astype(str).eq(_as_text(target[MEDIUM])).to_numpy()).sum()
    )
    counts["temperature_anchor"] = int(
        (
            random_mask
            & metadata[TEMPERATURE].astype(str).eq(_as_text(target[TEMPERATURE])).to_numpy()
        ).sum()
    )
    return counts


def _minimum_support(settings: ProbeSettings) -> dict[str, int]:
    half = settings.donor_count // 2
    # Each single-factor pool supplies its primary arm, half of the relevant
    # mixture arm, and one baseline feature-coverage anchor.
    return {
        "strain_only": settings.donor_count + half + 1,
        "chemical_only": settings.donor_count + half + 1,
        "time_only": settings.donor_count + half + 1,
        "strain_chemical": settings.donor_count,
        "chemical_time": settings.donor_count,
        "strain_time": settings.donor_count,
        "random_none": settings.donor_count + (settings.baseline_size - 5) + 2,
        "medium_anchor": 1,
        "temperature_anchor": 1,
    }


def _eligible_target_ids(
    metadata: pd.DataFrame,
    settings: ProbeSettings,
) -> tuple[pd.DataFrame, list[str]]:
    """Availability audit based strictly on metadata and candidate IDs."""

    minimums = _minimum_support(settings)
    rows: list[dict[str, Any]] = []
    for target_id in metadata.index.astype(str):
        blocked = frozenset({target_id})
        counts = _support_counts(metadata, target_id, blocked_ids=blocked)
        rows.append(
            {
                CONDITION_ID: target_id,
                **{key: int(value) for key, value in counts.items()},
                "eligible_main": bool(
                    all(counts[name] >= needed for name, needed in minimums.items())
                ),
            }
        )
    frame = pd.DataFrame(rows).set_index(CONDITION_ID)
    return frame, frame.index[frame["eligible_main"]].astype(str).tolist()


def _selection_cost(
    row: pd.Series,
    counts: Mapping[str, Mapping[str, int]],
) -> tuple[int, int, int, int, int]:
    strain = _as_text(row[STRAIN])
    chemical = _as_text(row[CHEMICAL])
    time = _as_text(row[TIME])
    medium = _as_text(row[MEDIUM])
    temperature = _as_text(row[TEMPERATURE])
    return (
        40 * counts["strain_time"].get(f"{strain}\0{time}", 0),
        15 * counts["chemical"].get(chemical, 0),
        10 * counts["strain"].get(strain, 0),
        8 * counts["time"].get(time, 0),
        counts["context"].get(f"{medium}\0{temperature}", 0),
    )


def _select_target_panel(
    metadata: pd.DataFrame,
    eligible_ids: Sequence[str],
    settings: ProbeSettings,
) -> tuple[str, ...]:
    """Select targets using only metadata coverage, with every eligible chemical first."""

    if len(eligible_ids) < settings.target_count:
        raise ValueError(
            f"Only {len(eligible_ids)} targets support the registered probe, "
            f"need {settings.target_count}"
        )
    available = metadata.loc[list(eligible_ids)].copy()
    counts: dict[str, dict[str, int]] = {
        "strain_time": {},
        "chemical": {},
        "strain": {},
        "time": {},
        "context": {},
    }

    def choose(candidates: Sequence[str], namespace: str) -> str:
        ranked = sorted(
            (str(value) for value in candidates),
            key=lambda target_id: (
                _selection_cost(available.loc[target_id], counts),
                _stable_rank(settings.seed, namespace, target_id),
                target_id,
            ),
        )
        return ranked[0]

    def record(target_id: str) -> None:
        row = available.loc[target_id]
        values = {
            "strain_time": f"{_as_text(row[STRAIN])}\0{_as_text(row[TIME])}",
            "chemical": _as_text(row[CHEMICAL]),
            "strain": _as_text(row[STRAIN]),
            "time": _as_text(row[TIME]),
            "context": f"{_as_text(row[MEDIUM])}\0{_as_text(row[TEMPERATURE])}",
        }
        for name, value in values.items():
            counts[name][value] = counts[name].get(value, 0) + 1

    selected: list[str] = []
    # Start with chemical coverage.  Some weakly supported chemicals are
    # unavailable by design; the audit records those exclusions explicitly.
    for chemical in sorted(available[CHEMICAL].astype(str).unique()):
        if len(selected) >= settings.target_count:
            break
        candidates = [
            str(value)
            for value in available.index[available[CHEMICAL].astype(str).eq(chemical)]
            if str(value) not in selected
        ]
        if candidates:
            target_id = choose(candidates, "target-chemical-cover")
            selected.append(target_id)
            record(target_id)

    while len(selected) < settings.target_count:
        candidates = [str(value) for value in available.index if str(value) not in selected]
        target_id = choose(candidates, "target-fill")
        selected.append(target_id)
        record(target_id)
    return tuple(selected)


def _ids_for_mask(metadata: pd.DataFrame, mask: np.ndarray) -> list[str]:
    return metadata.index[mask].astype(str).tolist()


def _build_main_plan(
    metadata: pd.DataFrame,
    target_id: str,
    *,
    holdout_ids: frozenset[str],
    settings: ProbeSettings,
    selection_seed: int,
    fixed_baseline_ids: Sequence[str] | None = None,
) -> ProbePlan:
    """Build a disjoint, equal-donor-count main probe for one target.

    ``fixed_baseline_ids`` is used only for the alternate donor-draw
    sensitivity: it keeps the original baseline exactly fixed while donor
    groups are redrawn under a new metadata-only selection seed.
    """

    target = metadata.loc[target_id]
    blocked = frozenset(set(holdout_ids) | {target_id})
    masks = _candidate_relation_masks(metadata, target, blocked_ids=blocked)
    candidates = {name: _ids_for_mask(metadata, mask) for name, mask in masks.items()}
    half = settings.donor_count // 2
    chosen: dict[str, tuple[str, ...]] = {}
    fixed_baseline = tuple(str(value) for value in fixed_baseline_ids or ())
    if fixed_baseline and (target_id in fixed_baseline or set(fixed_baseline).intersection(holdout_ids)):
        raise ValueError("Fixed sensitivity baseline leaks a withheld target")
    used: frozenset[str] = frozenset(fixed_baseline)
    for relation in (
        "strain_only",
        "chemical_only",
        "time_only",
        "strain_chemical",
        "chemical_time",
        "strain_time",
        "random_none",
    ):
        chosen[relation] = _balanced_ids(
            metadata,
            candidates[relation],
            settings.donor_count,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace=f"main:{relation}",
            excluded=used,
        )
        used = frozenset(set(used) | set(chosen[relation]))

    mixture_parts = {
        "strain_chemical_mixture": ("strain_only", "chemical_only"),
        "chemical_time_mixture": ("chemical_only", "time_only"),
        "strain_time_mixture": ("strain_only", "time_only"),
    }
    for mixture, (left, right) in mixture_parts.items():
        left_ids = _balanced_ids(
            metadata,
            candidates[left],
            half,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace=f"main:{mixture}:left",
            excluded=used,
        )
        used = frozenset(set(used) | set(left_ids))
        right_ids = _balanced_ids(
            metadata,
            candidates[right],
            half,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace=f"main:{mixture}:right",
            excluded=used,
        )
        chosen[mixture] = tuple(left_ids + right_ids)
        used = frozenset(set(used) | set(right_ids))

    if fixed_baseline:
        if len(fixed_baseline) != settings.baseline_size:
            raise ValueError("Fixed sensitivity baseline has the wrong registered size")
        baseline_ids = fixed_baseline
        anchors: dict[str, str] = {}
    else:
        anchors = {}
        for name, relation in (
            ("strain_anchor", "strain_only"),
            ("chemical_anchor", "chemical_only"),
            ("time_anchor", "time_only"),
        ):
            selected = _balanced_ids(
                metadata,
                candidates[relation],
                1,
                target=target,
                target_id=target_id,
                seed=selection_seed,
                namespace=f"anchor:{name}",
                excluded=used,
            )
            anchors[name] = selected[0]
            used = frozenset(set(used) | set(selected))

        for name, field_name in (
            ("medium_anchor", MEDIUM),
            ("temperature_anchor", TEMPERATURE),
        ):
            values = _coverage_anchor_candidates(
                metadata,
                target,
                field_name,
                blocked_ids=blocked,
            )
            selected = _balanced_ids(
                metadata,
                values,
                1,
                target=target,
                target_id=target_id,
                seed=selection_seed,
                namespace=f"anchor:{name}",
                excluded=used,
            )
            anchors[name] = selected[0]
            used = frozenset(set(used) | set(selected))

        remaining = settings.baseline_size - len(anchors)
        background = _balanced_ids(
            metadata,
            candidates["random_none"],
            remaining,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace="baseline:background",
            excluded=used,
        )
        baseline_ids = tuple(anchors.values()) + tuple(background)
    if len(baseline_ids) != settings.baseline_size:
        raise AssertionError("Baseline size differs from the registered design")
    if not all(_feature_coverage(metadata, target, baseline_ids).values()):
        raise AssertionError("Baseline failed target feature coverage")
    all_ids = set(baseline_ids)
    for relation, donor_ids in chosen.items():
        if len(donor_ids) != settings.donor_count:
            raise AssertionError(f"{relation} donor count differs from the registered design")
        if all_ids.intersection(donor_ids):
            raise AssertionError("Donor IDs overlap the fixed baseline")
        all_ids.update(donor_ids)
    if target_id in all_ids or all_ids.intersection(holdout_ids):
        raise AssertionError("Target holdout leaked into probe support")
    return ProbePlan(
        target_id=target_id,
        baseline_ids=baseline_ids,
        donor_ids=chosen,
        anchors=anchors,
        selection_seed=selection_seed,
    )


def _validate_target_holdout(
    metadata: pd.DataFrame,
    target_ids: Sequence[str],
    settings: ProbeSettings,
) -> tuple[str, ...]:
    """Keep a global target holdout only when every exact plan remains feasible."""

    ranked = [str(value) for value in target_ids]
    selected = ranked[: settings.target_count]
    cursor = settings.target_count
    # Most panels pass on the first attempt.  If a rare low-support CT cell is
    # invalidated by the global holdout, replace only the failing target and
    # recheck the complete panel, avoiding quadratic plan construction.
    for _ in range(len(ranked)):
        blocked = frozenset(selected)
        failed: list[str] = []
        for target_id in selected:
            try:
                _build_main_plan(
                    metadata,
                    target_id,
                    holdout_ids=blocked,
                    settings=settings,
                    selection_seed=settings.seed,
                )
            except ValueError:
                failed.append(target_id)
        if not failed:
            return tuple(selected)
        if cursor + len(failed) > len(ranked):
            break
        replacements = ranked[cursor : cursor + len(failed)]
        cursor += len(failed)
        replacement_iter = iter(replacements)
        failed_set = set(failed)
        selected = [
            next(replacement_iter) if target_id in failed_set else target_id
            for target_id in selected
        ]
    raise RuntimeError(
        f"Could not make {settings.target_count} globally disjoint target plans"
    )


def _model_settings(settings: ProbeSettings) -> ModelSettings:
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


def _support_covariates(
    metadata: pd.DataFrame,
    target: pd.Series,
    donor_ids: Sequence[str],
) -> dict[str, float]:
    donor = metadata.loc[list(donor_ids)]
    values: dict[str, float] = {
        "donor_same_medium_fraction": float(
            donor[MEDIUM].astype(str).eq(_as_text(target[MEDIUM])).mean()
        ),
        "donor_same_temperature_fraction": float(
            donor[TEMPERATURE].astype(str).eq(_as_text(target[TEMPERATURE])).mean()
        ),
    }
    for name in ("replicate_count", "measurement_context_count", "source_count", "instrument_count", "plate_count"):
        if name in donor:
            values[f"donor_{name}_mean"] = float(
                pd.to_numeric(donor[name], errors="coerce").mean()
            )
    return values


def _feature_coverage(
    metadata: pd.DataFrame,
    target: pd.Series,
    support_ids: Sequence[str],
) -> dict[str, bool]:
    support = metadata.loc[list(support_ids)]
    return {
        "target_strain_supported": bool(
            support[STRAIN].astype(str).eq(_as_text(target[STRAIN])).any()
        ),
        "target_chemical_supported": bool(
            support[CHEMICAL].astype(str).eq(_as_text(target[CHEMICAL])).any()
        ),
        "target_medium_supported": bool(
            support[MEDIUM].astype(str).eq(_as_text(target[MEDIUM])).any()
        ),
        "target_temperature_supported": bool(
            support[TEMPERATURE].astype(str).eq(_as_text(target[TEMPERATURE])).any()
        ),
        "target_time_supported": bool(
            support[TIME].astype(str).eq(_as_text(target[TIME])).any()
        ),
    }


def _fit_and_score(
    *,
    bundle: Any,
    response: np.ndarray,
    row_positions: Mapping[str, int],
    target_id: str,
    support_ids: Sequence[str],
    target_response: np.ndarray,
    settings: ModelSettings,
    fit_seed: int,
) -> tuple[dict[str, float | int], float]:
    positions = np.asarray([row_positions[sample_id] for sample_id in support_ids], dtype=np.int64)
    target_position = row_positions[target_id]
    started = perf_counter()
    masked = prepare_masked_model_features(bundle, support_ids)
    fit = fit_response_model(
        masked[positions],
        response[positions],
        settings,
        fit_seed,
    )
    prediction = fit.predict(masked[target_position : target_position + 1])
    metrics = score_response(prediction, target_response[None, :])
    return metrics, float(perf_counter() - started)


def _relation_exposures(relation: str) -> dict[str, float]:
    values = {
        "same_strain_fraction": 0.0,
        "same_chemical_fraction": 0.0,
        "same_time_fraction": 0.0,
        "strain_chemical_joint_fraction": 0.0,
        "chemical_time_joint_fraction": 0.0,
        "strain_time_joint_fraction": 0.0,
    }
    if relation == "strain_only":
        values["same_strain_fraction"] = 1.0
    elif relation == "chemical_only":
        values["same_chemical_fraction"] = 1.0
    elif relation == "time_only":
        values["same_time_fraction"] = 1.0
    elif relation == "strain_chemical":
        values.update(
            same_strain_fraction=1.0,
            same_chemical_fraction=1.0,
            strain_chemical_joint_fraction=1.0,
        )
    elif relation == "chemical_time":
        values.update(
            same_chemical_fraction=1.0,
            same_time_fraction=1.0,
            chemical_time_joint_fraction=1.0,
        )
    elif relation == "strain_time":
        values.update(
            same_strain_fraction=1.0,
            same_time_fraction=1.0,
            strain_time_joint_fraction=1.0,
        )
    elif relation == "strain_chemical_mixture":
        values.update(same_strain_fraction=0.5, same_chemical_fraction=0.5)
    elif relation == "chemical_time_mixture":
        values.update(same_chemical_fraction=0.5, same_time_fraction=0.5)
    elif relation == "strain_time_mixture":
        values.update(same_strain_fraction=0.5, same_time_fraction=0.5)
    elif relation != "random_none":
        raise ValueError(f"Unknown relation: {relation}")
    return values


def run_main_probe(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: ProbeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Run the exact baseline-plus-donor retraining panel."""

    candidate_ids = tuple(str(value) for value in dataset.candidate_pool_ids)
    metadata = dataset.metadata.loc[list(candidate_ids)].copy()
    audit, eligible = _eligible_target_ids(metadata, settings)
    audit.to_csv(output_dir / "STRUCTURAL_AVAILABILITY_AUDIT.csv")

    # First construct a coverage-ranked panel, then enforce a global target
    # holdout with exact plan feasibility checks.
    proposed = _select_target_panel(metadata, eligible, settings)
    fallback = [target_id for target_id in eligible if target_id not in proposed]
    ranked_candidates = list(proposed) + _sorted_ids(
        fallback,
        seed=settings.seed,
        namespace="target-fallback",
        target_id="global",
    )
    target_ids = _validate_target_holdout(metadata, ranked_candidates, settings)
    target_panel = metadata.loc[list(target_ids), [STRAIN, CHEMICAL, TIME, MEDIUM, TEMPERATURE]].copy()
    target_panel.insert(0, "target_rank", np.arange(1, len(target_panel) + 1))
    target_panel.insert(1, "selection_rule", "metadata_stratified_global_holdout")
    target_panel.to_csv(output_dir / "TARGET_PANEL.csv", index_label=CONDITION_ID)

    bundle = load_identity_feature_bundle(dataset)
    row_positions = {str(sample_id): index for index, sample_id in enumerate(bundle.row_ids)}
    response = dataset.response.loc[list(bundle.row_ids)].to_numpy(dtype=np.float32)
    model_settings = _model_settings(settings)
    plans = {
        target_id: _build_main_plan(
            metadata,
            target_id,
            holdout_ids=frozenset(target_ids),
            settings=settings,
            selection_seed=settings.seed,
        )
        for target_id in target_ids
    }

    records: list[dict[str, Any]] = []
    for ordinal, target_id in enumerate(target_ids, start=1):
        plan = plans[target_id]
        target = metadata.loc[target_id]
        target_response = response[row_positions[target_id]]
        baseline_metrics, baseline_seconds = _fit_and_score(
            bundle=bundle,
            response=response,
            row_positions=row_positions,
            target_id=target_id,
            support_ids=plan.baseline_ids,
            target_response=target_response,
            settings=model_settings,
            fit_seed=settings.seed,
        )
        coverage = _feature_coverage(metadata, target, plan.baseline_ids)
        if not all(coverage.values()):
            raise AssertionError(f"Baseline failed target feature coverage for {target_id}")
        for relation in MAIN_RELATIONS:
            donors = plan.donor_ids[relation]
            support = tuple(plan.baseline_ids) + tuple(donors)
            after_metrics, after_seconds = _fit_and_score(
                bundle=bundle,
                response=response,
                row_positions=row_positions,
                target_id=target_id,
                support_ids=support,
                target_response=target_response,
                settings=model_settings,
                fit_seed=settings.seed,
            )
            if not set(support).isdisjoint(set(target_ids)):
                raise AssertionError("A global target holdout appeared in a support set")
            covariates = _support_covariates(metadata, target, donors)
            records.append(
                {
                    CONDITION_ID: target_id,
                    "target_rank": ordinal,
                    "target_strain": _as_text(target[STRAIN]),
                    "target_chemical": _as_text(target[CHEMICAL]),
                    "target_time_minutes": target[TIME],
                    "target_medium": _as_text(target[MEDIUM]),
                    "target_temperature": target[TEMPERATURE],
                    "donor_relation": relation,
                    "donor_ids": json.dumps(list(donors)),
                    "donor_count": len(donors),
                    "baseline_ids": json.dumps(list(plan.baseline_ids)),
                    "baseline_count": len(plan.baseline_ids),
                    "anchors": json.dumps(dict(plan.anchors), sort_keys=True),
                    "selection_seed": plan.selection_seed,
                    "fit_seed": settings.seed,
                    "epochs": settings.epochs,
                    "feature_mode": "identity_plus_time",
                    "baseline_delta_skill_zero": baseline_metrics[PRIMARY_METRIC],
                    "after_delta_skill_zero": after_metrics[PRIMARY_METRIC],
                    "transfer_score": float(
                        after_metrics[PRIMARY_METRIC] - baseline_metrics[PRIMARY_METRIC]
                    ),
                    "baseline_delta_rmse": baseline_metrics["delta_rmse"],
                    "after_delta_rmse": after_metrics["delta_rmse"],
                    "rmse_reduction": float(
                        baseline_metrics["delta_rmse"] - after_metrics["delta_rmse"]
                    ),
                    "baseline_condition_pcc": baseline_metrics["condition_pcc_median"],
                    "after_condition_pcc": after_metrics["condition_pcc_median"],
                    "transfer_condition_pcc": float(
                        after_metrics["condition_pcc_median"]
                        - baseline_metrics["condition_pcc_median"]
                    ),
                    "target_observed_values": baseline_metrics["n_observed_values"],
                    "baseline_train_seconds": baseline_seconds,
                    "after_train_seconds": after_seconds,
                    **coverage,
                    **_relation_exposures(relation),
                    **covariates,
                }
            )
        print(
            f"[{ordinal:02d}/{len(target_ids)}] completed transfer arms for {target_id}",
            flush=True,
        )
    results = pd.DataFrame(records)
    if len(results) != len(target_ids) * len(MAIN_RELATIONS):
        raise AssertionError("The transfer result grid is incomplete")
    results.to_csv(output_dir / "TRANSFER_PROBE_RESULTS.csv", index=False)
    return results, audit, target_ids


def _bootstrap_mean(
    values: Sequence[float], *, draws: int, seed: int
) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    if len(array) == 1:
        return (float(array[0]), 0.0, float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[positions].mean(axis=1)
    return (
        float(array.mean()),
        float(array.std(ddof=1) / math.sqrt(len(array))),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _smd(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
    return float((left.mean() - right.mean()) / pooled) if pooled > 0.0 else 0.0


def _paired_relative_transfer(results: pd.DataFrame) -> pd.DataFrame:
    required = {CONDITION_ID, "donor_relation", "transfer_score"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Transfer results are missing required columns: {missing}")
    random = results.loc[
        results["donor_relation"].eq("random_none"),
        [CONDITION_ID, "transfer_score"],
    ].rename(columns={"transfer_score": "random_transfer_score"})
    if random[CONDITION_ID].duplicated().any():
        raise ValueError("Each target must have exactly one random donor arm")
    relative = results.merge(random, on=CONDITION_ID, how="inner", validate="many_to_one")
    if len(relative) != len(results):
        raise ValueError("A target is missing its paired random donor arm")
    relative["transfer_vs_random"] = (
        relative["transfer_score"] - relative["random_transfer_score"]
    )
    return relative


def _relation_summary_rows(
    results: pd.DataFrame,
    settings: ProbeSettings,
) -> list[dict[str, Any]]:
    relative = _paired_relative_transfer(results)
    rows: list[dict[str, Any]] = []
    for index, relation in enumerate(MAIN_RELATIONS):
        frame = relative.loc[relative["donor_relation"].eq(relation)]
        transfer = frame["transfer_score"].to_numpy(dtype=float)
        relative_transfer = frame["transfer_vs_random"].to_numpy(dtype=float)
        mean, se, low, high = _bootstrap_mean(
            transfer, draws=settings.bootstrap_draws, seed=settings.seed + index
        )
        rel_mean, rel_se, rel_low, rel_high = _bootstrap_mean(
            relative_transfer,
            draws=settings.bootstrap_draws,
            seed=settings.seed + 100 + index,
        )
        rows.append(
            {
                "summary_type": "relation_transfer",
                "relation": relation,
                "factors": {
                    "strain_only": "strain",
                    "chemical_only": "chemical",
                    "time_only": "time",
                    "strain_chemical": "strain×chemical",
                    "chemical_time": "chemical×time",
                    "strain_time": "strain×time",
                    "strain_chemical_mixture": "strain/chemical mixture",
                    "chemical_time_mixture": "chemical/time mixture",
                    "strain_time_mixture": "strain/time mixture",
                    "random_none": "random control",
                }[relation],
                "n_targets": int(len(frame)),
                "mean_transfer": mean,
                "median_transfer": float(np.nanmedian(transfer)),
                "transfer_se": se,
                "transfer_ci95_low": low,
                "transfer_ci95_high": high,
                "mean_transfer_vs_random": rel_mean,
                "transfer_vs_random_se": rel_se,
                "transfer_vs_random_ci95_low": rel_low,
                "transfer_vs_random_ci95_high": rel_high,
                "interaction_excess": float("nan"),
                "interaction_excess_ci95_low": float("nan"),
                "interaction_excess_ci95_high": float("nan"),
                "interaction_excess_definition": "",
            }
        )
    wide = results.pivot(index=CONDITION_ID, columns="donor_relation", values="transfer_score")
    for index, (pair, mixture) in enumerate(PAIR_TO_MIXTURE.items()):
        values = (wide[pair] - wide[mixture]).to_numpy(dtype=float)
        mean, se, low, high = _bootstrap_mean(
            values,
            draws=settings.bootstrap_draws,
            seed=settings.seed + 300 + index,
        )
        rows.append(
            {
                "summary_type": "interaction_excess",
                "relation": pair,
                "factors": pair.replace("_", "×"),
                "n_targets": int(np.isfinite(values).sum()),
                "mean_transfer": float("nan"),
                "median_transfer": float("nan"),
                "transfer_se": float("nan"),
                "transfer_ci95_low": float("nan"),
                "transfer_ci95_high": float("nan"),
                "mean_transfer_vs_random": float("nan"),
                "transfer_vs_random_se": float("nan"),
                "transfer_vs_random_ci95_low": float("nan"),
                "transfer_vs_random_ci95_high": float("nan"),
                "interaction_excess": mean,
                "interaction_excess_se": se,
                "interaction_excess_ci95_low": low,
                "interaction_excess_ci95_high": high,
                "interaction_excess_definition": (
                    f"T({pair}) - T({mixture}); both arms contain "
                    f"{settings.donor_count} donors and share the same fixed baseline"
                ),
            }
        )
    return rows


def _covariate_balance_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    random = results.loc[results["donor_relation"].eq("random_none")].set_index(CONDITION_ID)
    covariates = [
        name
        for name in (
            "donor_same_medium_fraction",
            "donor_same_temperature_fraction",
            "donor_replicate_count_mean",
            "donor_measurement_context_count_mean",
            "donor_source_count_mean",
            "donor_instrument_count_mean",
            "donor_plate_count_mean",
        )
        if name in results
    ]
    rows: list[dict[str, Any]] = []
    for relation in MAIN_RELATIONS:
        arm = results.loc[results["donor_relation"].eq(relation)].set_index(CONDITION_ID)
        aligned = arm.index.intersection(random.index)
        for covariate in covariates:
            left = arm.loc[aligned, covariate].to_numpy(dtype=float)
            right = random.loc[aligned, covariate].to_numpy(dtype=float)
            rows.append(
                {
                    "relation": relation,
                    "covariate": covariate,
                    "n_targets": int(len(aligned)),
                    "arm_mean": float(np.nanmean(left)),
                    "random_mean": float(np.nanmean(right)),
                    "difference_from_random": float(np.nanmean(left - right)),
                    "standardized_mean_difference": _smd(left, right),
                }
            )
    return rows


def _design_matrix(frame: pd.DataFrame, *, interaction: bool) -> tuple[np.ndarray, list[str]]:
    base_names = [
        "same_strain_fraction",
        "same_chemical_fraction",
        "same_time_fraction",
        "donor_same_medium_fraction",
        "donor_same_temperature_fraction",
    ]
    names = list(base_names)
    if interaction:
        names.extend(
            [
                "strain_chemical_joint_fraction",
                "chemical_time_joint_fraction",
                "strain_time_joint_fraction",
            ]
        )
    values = frame.loc[:, names].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Explanatory design matrix contains a nonfinite covariate")
    return np.column_stack([np.ones(len(frame), dtype=np.float64), values]), ["intercept", *names]


def _loto_explanatory_score(
    frame: pd.DataFrame,
    *,
    interaction: bool,
    seed: int,
    bootstrap_draws: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Leave one target out after pairing each arm to its random control.

    The response is ``T(arm) - T(random)``.  Thus a target-specific random
    donor control is allowed to remove baseline difficulty, but no arm from the
    withheld target enters the fitted coefficient estimates.
    """

    nonrandom = frame.loc[~frame["donor_relation"].eq("random_none")].copy()
    nonrandom = nonrandom.loc[np.isfinite(nonrandom["transfer_vs_random"])]
    targets = tuple(sorted(nonrandom[CONDITION_ID].astype(str).unique()))
    prediction_rows: list[dict[str, Any]] = []
    for target_id in targets:
        train = nonrandom.loc[~nonrandom[CONDITION_ID].eq(target_id)]
        test = nonrandom.loc[nonrandom[CONDITION_ID].eq(target_id)]
        train_x, names = _design_matrix(train, interaction=interaction)
        test_x, _ = _design_matrix(test, interaction=interaction)
        beta, *_ = np.linalg.lstsq(
            train_x,
            train["transfer_vs_random"].to_numpy(dtype=np.float64),
            rcond=None,
        )
        predicted = test_x @ beta
        for row, value in zip(test.itertuples(index=False), predicted):
            prediction_rows.append(
                {
                    CONDITION_ID: target_id,
                    "donor_relation": getattr(row, "donor_relation"),
                    "observed_transfer_vs_random": float(
                        getattr(row, "transfer_vs_random")
                    ),
                    "predicted_transfer_vs_random": float(value),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    error = (
        predictions["observed_transfer_vs_random"].to_numpy()
        - predictions["predicted_transfer_vs_random"].to_numpy()
    )
    baseline = predictions["observed_transfer_vs_random"].to_numpy()
    denominator = float(np.square(baseline).sum())
    r2 = float(1.0 - np.square(error).sum() / denominator) if denominator > 0.0 else float("nan")
    by_target = predictions.assign(
        squared_error=np.square(error), baseline_square=np.square(baseline)
    ).groupby(CONDITION_ID, sort=False)[["squared_error", "baseline_square"]].sum()
    rng = np.random.default_rng(seed)
    r2_draws: list[float] = []
    if len(by_target) > 1:
        values = by_target.to_numpy(dtype=float)
        for sample in rng.integers(0, len(values), size=(bootstrap_draws, len(values))):
            selected = values[sample].sum(axis=0)
            if selected[1] > 0.0:
                r2_draws.append(float(1.0 - selected[0] / selected[1]))
    # Full-data coefficients are descriptive only; the primary performance is
    # leave-one-target-out R² above.
    full_x, names = _design_matrix(nonrandom, interaction=interaction)
    beta, *_ = np.linalg.lstsq(
        full_x,
        nonrandom["transfer_vs_random"].to_numpy(dtype=np.float64),
        rcond=None,
    )
    summary = {
        "model": "factor_plus_interaction" if interaction else "factor_only",
        "held_out_unit": "entire_target",
        "target_count": int(len(targets)),
        "observation_count": int(len(predictions)),
        "loto_r2_vs_zero_random_control": r2,
        "loto_r2_ci95_low": (
            float(np.quantile(r2_draws, 0.025)) if r2_draws else float("nan")
        ),
        "loto_r2_ci95_high": (
            float(np.quantile(r2_draws, 0.975)) if r2_draws else float("nan")
        ),
    }
    coefficient_rows = [
        {
            "model": summary["model"],
            "term": name,
            "coefficient": float(value),
            "coefficient_scope": "full_panel_descriptive",
        }
        for name, value in zip(names, beta)
    ]
    return summary, predictions, coefficient_rows


def analyze_main_probe(
    results: pd.DataFrame,
    output_dir: Path,
    settings: ProbeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write factor/interaction, balance, and transparent explanatory analyses."""

    summary = pd.DataFrame(_relation_summary_rows(results, settings))
    summary.to_csv(output_dir / "FACTOR_INTERACTION_SUMMARY.csv", index=False)
    balance = pd.DataFrame(_covariate_balance_rows(results))
    balance.to_csv(output_dir / "COVARIATE_BALANCE_SUMMARY.csv", index=False)

    paired = _paired_relative_transfer(results)
    factor, factor_predictions, factor_coefficients = _loto_explanatory_score(
        paired,
        interaction=False,
        seed=settings.seed + 700,
        bootstrap_draws=settings.bootstrap_draws,
    )
    hybrid, hybrid_predictions, hybrid_coefficients = _loto_explanatory_score(
        paired,
        interaction=True,
        seed=settings.seed + 701,
        bootstrap_draws=settings.bootstrap_draws,
    )
    model_summary = pd.DataFrame([factor, hybrid])
    model_summary["r2_gain_vs_factor_only"] = (
        model_summary["loto_r2_vs_zero_random_control"]
        - factor["loto_r2_vs_zero_random_control"]
    )
    model_summary.to_csv(output_dir / "FACTOR_EXPLANATORY_MODEL_SUMMARY.csv", index=False)
    pd.DataFrame(factor_coefficients + hybrid_coefficients).to_csv(
        output_dir / "FACTOR_EXPLANATORY_COEFFICIENTS.csv", index=False
    )
    predictions = pd.concat(
        [
            factor_predictions.assign(model=factor["model"]),
            hybrid_predictions.assign(model=hybrid["model"]),
        ],
        ignore_index=True,
    )
    predictions.to_csv(output_dir / "FACTOR_EXPLANATORY_PREDICTIONS.csv", index=False)
    _plot_main_results(summary, model_summary, output_dir)
    return summary, model_summary


def _plot_main_results(
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    relation = summary.loc[summary["summary_type"].eq("relation_transfer")].copy()
    display_order = [
        "random_none",
        "strain_only",
        "chemical_only",
        "time_only",
        "strain_chemical",
        "chemical_time",
        "strain_time",
        "strain_chemical_mixture",
        "chemical_time_mixture",
        "strain_time_mixture",
    ]
    relation["order"] = relation["relation"].map({name: index for index, name in enumerate(display_order)})
    relation = relation.sort_values("order")
    labels = [
        {
            "random_none": "Random",
            "strain_only": "S",
            "chemical_only": "D",
            "time_only": "T",
            "strain_chemical": "S×D",
            "chemical_time": "D×T",
            "strain_time": "S×T",
            "strain_chemical_mixture": "S/D mix",
            "chemical_time_mixture": "D/T mix",
            "strain_time_mixture": "S/T mix",
        }[name]
        for name in relation["relation"]
    ]
    interaction = summary.loc[summary["summary_type"].eq("interaction_excess")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    values = relation["mean_transfer"].to_numpy(dtype=float)
    low = relation["transfer_ci95_low"].to_numpy(dtype=float)
    high = relation["transfer_ci95_high"].to_numpy(dtype=float)
    error = np.vstack([values - low, high - values])
    colors = ["#8c8c8c" if name == "random_none" else "#4878a8" for name in relation["relation"]]
    axes[0].bar(np.arange(len(values)), values, yerr=error, color=colors, capsize=3)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(np.arange(len(values)), labels, rotation=38, ha="right")
    axes[0].set_ylabel("Transfer: Δ delta_skill_zero")
    axes[0].set_title("Exact donor-group transfer")
    excess = interaction["interaction_excess"].to_numpy(dtype=float)
    ilow = interaction["interaction_excess_ci95_low"].to_numpy(dtype=float)
    ihigh = interaction["interaction_excess_ci95_high"].to_numpy(dtype=float)
    ierr = np.vstack([excess - ilow, ihigh - excess])
    axes[1].bar(np.arange(len(excess)), excess, yerr=ierr, color="#d2774a", capsize=3)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(excess)), ["S×D − mix", "D×T − mix", "S×T − mix"], rotation=20, ha="right")
    axes[1].set_ylabel("Equal-budget interaction excess")
    axes[1].set_title("Pair donor versus 50/50 single-factor mixture")
    fig.savefig(output_dir / "factor_vs_interaction_transfer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    models = model_summary["model"].tolist()
    r2 = model_summary["loto_r2_vs_zero_random_control"].to_numpy(dtype=float)
    low = model_summary["loto_r2_ci95_low"].to_numpy(dtype=float)
    high = model_summary["loto_r2_ci95_high"].to_numpy(dtype=float)
    axis.bar(np.arange(len(models)), r2, yerr=np.vstack([r2 - low, high - r2]), capsize=3, color=["#5f9e6e", "#d2774a"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(models)), ["Factor-only", "Factor + pairwise"])
    axis.set_ylabel("Leave-one-target-out R²\n(relative to paired random control)")
    axis.set_title("Can factor effects explain transfer structure?")
    fig.savefig(output_dir / "factor_explanatory_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _stable_main_factors(summary: pd.DataFrame) -> tuple[str, ...]:
    mapping = {
        "strain_only": "strain",
        "chemical_only": "chemical",
        "time_only": "time",
    }
    stable: list[str] = []
    rows = summary.loc[summary["summary_type"].eq("relation_transfer")]
    for relation, factor in mapping.items():
        row = rows.loc[rows["relation"].eq(relation)]
        if len(row) != 1:
            continue
        if float(row.iloc[0]["transfer_vs_random_ci95_low"]) > 0.0:
            stable.append(factor)
    return tuple(stable)


def _entity_values(metadata: pd.DataFrame, factor: str) -> list[str]:
    field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    counts = metadata[field].astype(str).value_counts()
    if factor == "chemical":
        # A fixed metadata-only panel of well-represented chemicals is enough
        # for a light exact source-effect audit; it is never chosen by response.
        return [str(value) for value in counts.index[:12]]
    if factor == "time":
        return sorted((str(value) for value in counts.index), key=lambda value: float(value))
    return sorted(str(value) for value in counts.index)


def _entity_group_candidates(
    metadata: pd.DataFrame,
    target: pd.Series,
    *,
    factor: str,
    source_entity: str,
    blocked_ids: frozenset[str],
) -> list[str]:
    """Return source-entity donors with the other two focal axes mismatched."""

    field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    other_fields = tuple(value for value in FOCAL_FIELDS if value != field)
    mask = metadata[field].astype(str).to_numpy() == str(source_entity)
    for other in other_fields:
        mask &= metadata[other].astype(str).to_numpy() != _as_text(target[other])
    mask &= ~np.asarray(metadata.index.astype(str).isin(blocked_ids), dtype=bool)
    return metadata.index[mask].astype(str).tolist()


def _source_neutral_geometry_baseline(
    metadata: pd.DataFrame,
    target: pd.Series,
    *,
    target_id: str,
    factor: str,
    source_entities: Sequence[str],
    blocked_ids: frozenset[str],
    donor_ids: frozenset[str],
    settings: ProbeSettings,
    selection_seed: int,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Build a factor-balanced baseline for a source-effect comparison.

    Every evaluated source entity receives exactly the same number of baseline
    conditions.  This makes the marginal ``+ donor group`` comparison
    source-neutral: an on-diagonal source is not already saturated by a
    target-factor anchor, and an off-diagonal source is not given a different
    amount of prior exposure through random background.  The effective size is
    the largest multiple of the number of sources no greater than the requested
    baseline size (252 rather than 256 for six or twelve sources).
    """

    field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    sources = tuple(str(value) for value in source_entities)
    if len(set(sources)) != len(sources) or not sources:
        raise ValueError("Geometry source entities must be a nonempty unique sequence")
    per_source = settings.baseline_size // len(sources)
    if per_source < 1:
        raise ValueError("Geometry baseline cannot allocate one condition per source entity")

    source_values = metadata[field].astype(str)
    used: frozenset[str] = frozenset(set(blocked_ids) | set(donor_ids))
    selected: dict[str, list[str]] = {source: [] for source in sources}
    anchors: dict[str, str] = {}

    # Explicitly expose every target feature level without giving any evaluated
    # source entity more baseline examples than another.  Each selected anchor
    # occupies the capacity of its own source stratum and is later topped up.
    for anchor_name, coverage_field in (
        ("strain_anchor", STRAIN),
        ("chemical_anchor", CHEMICAL),
        ("time_anchor", TIME),
        ("medium_anchor", MEDIUM),
        ("temperature_anchor", TEMPERATURE),
    ):
        candidates = [
            str(sample_id)
            for sample_id in metadata.index[
                metadata[coverage_field].astype(str).eq(_as_text(target[coverage_field]))
                & source_values.isin(sources)
            ]
            if str(sample_id) not in used
            and len(selected[_as_text(metadata.at[str(sample_id), field])]) < per_source
        ]
        if not candidates:
            raise ValueError(
                f"{target_id}: no source-neutral baseline anchor for {anchor_name}"
            )
        candidates.sort(
            key=lambda sample_id: (
                len(selected[_as_text(metadata.at[sample_id, field])]),
                _stable_rank(
                    selection_seed,
                    f"geometry-baseline-anchor:{factor}:{anchor_name}",
                    target_id,
                    sample_id,
                ),
                sample_id,
            )
        )
        chosen = candidates[0]
        source = _as_text(metadata.at[chosen, field])
        selected[source].append(chosen)
        anchors[anchor_name] = chosen
        used = frozenset(set(used) | {chosen})

    for source in sources:
        remaining = per_source - len(selected[source])
        if remaining < 0:
            raise AssertionError("Geometry source baseline allocation became negative")
        candidates = [
            str(sample_id)
            for sample_id in metadata.index[source_values.eq(source)]
            if str(sample_id) not in blocked_ids
        ]
        fill = _balanced_ids(
            metadata,
            candidates,
            remaining,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace=f"geometry-baseline:{factor}:{source}",
            excluded=used,
        )
        selected[source].extend(fill)
        used = frozenset(set(used) | set(fill))

    baseline = tuple(sample_id for source in sources for sample_id in selected[source])
    expected_count = per_source * len(sources)
    if len(baseline) != expected_count or len(set(baseline)) != len(baseline):
        raise AssertionError("Geometry source-neutral baseline has an invalid allocation")
    if not all(_feature_coverage(metadata, target, baseline).values()):
        raise AssertionError("Geometry source-neutral baseline failed target feature coverage")
    return baseline, anchors


def _build_entity_plan(
    metadata: pd.DataFrame,
    target_id: str,
    *,
    factor: str,
    source_entities: Sequence[str],
    holdout_ids: frozenset[str],
    settings: ProbeSettings,
    selection_seed: int,
) -> ProbePlan:
    """Build an all-source donor plan for a source-effect geometry target."""

    target = metadata.loc[target_id]
    blocked = frozenset(set(holdout_ids) | {target_id})
    selected: dict[str, tuple[str, ...]] = {}
    used: frozenset[str] = frozenset()
    for source in source_entities:
        candidates = _entity_group_candidates(
            metadata,
            target,
            factor=factor,
            source_entity=source,
            blocked_ids=blocked,
        )
        donor_ids = _balanced_ids(
            metadata,
            candidates,
            settings.donor_count,
            target=target,
            target_id=target_id,
            seed=selection_seed,
            namespace=f"geometry:{factor}:{source}",
            excluded=used,
        )
        selected[str(source)] = donor_ids
        used = frozenset(set(used) | set(donor_ids))

    baseline, anchors = _source_neutral_geometry_baseline(
        metadata,
        target,
        target_id=target_id,
        factor=factor,
        source_entities=source_entities,
        blocked_ids=blocked,
        donor_ids=used,
        settings=settings,
        selection_seed=selection_seed,
    )
    all_support = set(baseline)
    for ids in selected.values():
        if all_support.intersection(ids):
            raise AssertionError("Entity donor group overlaps its baseline")
        all_support.update(ids)
    if target_id in all_support or all_support.intersection(holdout_ids):
        raise AssertionError("Entity geometry target leaked into support")
    return ProbePlan(
        target_id=target_id,
        baseline_ids=baseline,
        donor_ids=selected,
        anchors=anchors,
        selection_seed=selection_seed,
    )


def _geometry_candidate_order(
    metadata: pd.DataFrame,
    *,
    factor: str,
    entities: Sequence[str],
    external_holdout: frozenset[str],
    settings: ProbeSettings,
    replicates: int,
) -> list[str]:
    field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    candidate_by_entity: dict[str, list[str]] = {}
    for entity in entities:
        candidates = [
            str(value)
            for value in metadata.index[metadata[field].astype(str).eq(str(entity))]
            if str(value) not in external_holdout
        ]
        # Geometry feasibility is checked exactly below as targets enter the
        # global holdout.  Pre-limit this *metadata-only* candidate reservoir:
        # testing every condition with every source group is needless work and
        # would make a lightweight 12-chemical panel look like a pairwise sweep.
        reservoir = _sorted_ids(
            candidates,
            seed=settings.seed,
            namespace=f"geometry-reservoir:{factor}:{entity}",
            target_id="panel",
        )[: max(12, replicates * 4)]
        if len(reservoir) < replicates:
            raise RuntimeError(
                f"{factor}={entity!r} has only {len(reservoir)} candidate geometry targets; "
                f"need {replicates}"
            )
        # First order varied contexts, then use deterministic hash ties.  More
        # than ``replicates`` values stay available as global-holdout fallbacks.
        selected: list[str] = []
        context_counts: dict[str, int] = {}
        while len(selected) < len(reservoir):
            remaining = [value for value in reservoir if value not in selected]
            chosen = min(
                remaining,
                key=lambda target_id: (
                    context_counts.get(
                        f"{_as_text(metadata.at[target_id, STRAIN])}\0"
                        f"{_as_text(metadata.at[target_id, CHEMICAL])}\0"
                        f"{_as_text(metadata.at[target_id, TIME])}\0"
                        f"{_as_text(metadata.at[target_id, MEDIUM])}\0"
                        f"{_as_text(metadata.at[target_id, TEMPERATURE])}",
                        0,
                    ),
                    _stable_rank(settings.seed, f"geometry-target:{factor}:{entity}", target_id),
                    target_id,
                ),
            )
            key = (
                f"{_as_text(metadata.at[chosen, STRAIN])}\0"
                f"{_as_text(metadata.at[chosen, CHEMICAL])}\0"
                f"{_as_text(metadata.at[chosen, TIME])}\0"
                f"{_as_text(metadata.at[chosen, MEDIUM])}\0"
                f"{_as_text(metadata.at[chosen, TEMPERATURE])}"
            )
            context_counts[key] = context_counts.get(key, 0) + 1
            selected.append(chosen)
        candidate_by_entity[str(entity)] = selected
    # Interleave target entities so no entity gets a privileged position in the
    # global holdout feasibility check.
    ordered: list[str] = []
    for replica in range(max(len(values) for values in candidate_by_entity.values())):
        for entity in entities:
            candidates = candidate_by_entity[str(entity)]
            if replica < len(candidates):
                ordered.append(candidates[replica])
    return ordered


def _geometry_target_panel(
    metadata: pd.DataFrame,
    *,
    factor: str,
    entities: Sequence[str],
    external_holdout: frozenset[str],
    settings: ProbeSettings,
    replicates: int = 3,
) -> tuple[str, ...]:
    ordered = _geometry_candidate_order(
        metadata,
        factor=factor,
        entities=entities,
        external_holdout=external_holdout,
        settings=settings,
        replicates=replicates,
    )
    field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    queues: dict[str, list[str]] = {str(entity): [] for entity in entities}
    for candidate in ordered:
        queues[_as_text(metadata.at[candidate, field])].append(candidate)
    selected = {entity: values[:replicates] for entity, values in queues.items()}
    cursors = {entity: replicates for entity in queues}
    if any(len(values) < replicates for values in queues.values()):
        raise RuntimeError(f"Insufficient candidate reservoirs for {factor} geometry")
    # Validate the full candidate panel at once.  Rechecking after each target
    # would turn this compact geometry panel into an unnecessary O(n²) planning
    # pass; only failed targets are swapped with a same-entity metadata fallback.
    for _ in range(sum(len(values) for values in queues.values())):
        flat = tuple(candidate for values in selected.values() for candidate in values)
        holdout = frozenset(set(external_holdout) | set(flat))
        failed: list[str] = []
        for target_id in flat:
            try:
                _build_entity_plan(
                    metadata,
                    target_id,
                    factor=factor,
                    source_entities=entities,
                    holdout_ids=holdout,
                    settings=settings,
                    selection_seed=settings.seed,
                )
            except ValueError:
                failed.append(target_id)
        if not failed:
            return flat
        for target_id in failed:
            entity = _as_text(metadata.at[target_id, field])
            if cursors[entity] >= len(queues[entity]):
                raise RuntimeError(
                    f"Could not form a globally held-out {factor} geometry panel; "
                    f"{entity!r} exhausted its metadata candidate reservoir"
                )
            replacement = queues[entity][cursors[entity]]
            cursors[entity] += 1
            position = selected[entity].index(target_id)
            selected[entity][position] = replacement
    raise RuntimeError(f"Could not form a globally held-out {factor} geometry panel")


def _corr_distance(matrix: np.ndarray) -> np.ndarray:
    """Source-profile correlation distance for a directed source-effect matrix.

    A zero-variance source profile has no correlation geometry.  Returning a
    fabricated finite distance (for example by replacing its NaN correlation
    with zero) would turn non-identifiability into apparent structure, so the
    caller receives NaNs and records that geometry as not identifiable.
    """

    values = np.asarray(matrix, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    count = len(centered)
    if centered.shape[1] < 2:
        return np.full((count, count), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.corrcoef(centered)
    if similarity.shape != (count, count) or not np.isfinite(similarity).all():
        return np.full((count, count), np.nan, dtype=np.float64)
    similarity = np.clip((similarity + similarity.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - similarity)))


def _classical_mds(distance: np.ndarray, requested_dimensions: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    distance = np.asarray(distance, dtype=np.float64)
    count = len(distance)
    if distance.shape != (count, count):
        raise ValueError("Distance matrix must be square")
    if count < 2:
        raise ValueError("MDS needs at least two entities")
    center = np.eye(count) - np.ones((count, count), dtype=np.float64) / count
    gram = -0.5 * center @ np.square(distance) @ center
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues, 0.0, None)
    available = int(np.count_nonzero(positive > 1e-10))
    effective = min(int(requested_dimensions), max(count - 1, 0), available)
    if effective:
        embedding = eigenvectors[:, :effective] * np.sqrt(positive[:effective])
    else:
        embedding = np.zeros((count, 0), dtype=np.float64)
    retained = (
        float(positive[:effective].sum() / positive.sum())
        if positive.sum() > 0.0
        else float("nan")
    )
    return embedding, positive, effective, retained


def _pairwise_distance(embedding: np.ndarray) -> np.ndarray:
    difference = embedding[:, None, :] - embedding[None, :, :]
    return np.sqrt(np.maximum(0.0, np.square(difference).sum(axis=2)))


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[np.triu_indices(len(matrix), k=1)]


def _distance_r2(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    """Return directional distance correlation and an uncalibrated R².

    The R² is ``1 - SSE/SST`` on the supplied reference distances rather than
    correlation squared.  This preserves the fact that a reversed held-out
    geometry is a failure, not a perfect reconstruction.
    """

    left = _upper_values(reference)
    right = _upper_values(estimate)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 2 or left.std() == 0.0 or right.std() == 0.0:
        return float("nan"), float("nan")
    correlation = float(np.corrcoef(left, right)[0, 1])
    total = float(np.square(left - left.mean()).sum())
    r2 = float(1.0 - np.square(left - right).sum() / total) if total > 0.0 else float("nan")
    return correlation, r2


def _neighbor_overlap(reference: np.ndarray, estimate: np.ndarray, *, k: int = 3) -> float:
    count = len(reference)
    if count < 2:
        return float("nan")
    count_neighbors = min(k, count - 1)
    overlaps: list[float] = []
    for row in range(count):
        ref_values = np.asarray(reference[row], dtype=float).copy()
        est_values = np.asarray(estimate[row], dtype=float).copy()
        ref_values[row] = np.inf
        est_values[row] = np.inf
        ref = np.argsort(ref_values)[:count_neighbors]
        est = np.argsort(est_values)[:count_neighbors]
        overlaps.append(len(set(ref).intersection(est)) / count_neighbors)
    return float(np.mean(overlaps))


def _geometry_context_splits(
    entity_results: pd.DataFrame,
    *,
    factor: str,
    seed: int,
) -> pd.DataFrame:
    """Assign the third metadata-selected context of each entity to validation."""

    field = {
        "strain": "target_strain",
        "chemical": "target_chemical",
        "time": "target_time_minutes",
    }[factor]
    target_order = entity_results.loc[:, [CONDITION_ID, field]].drop_duplicates().copy()
    target_order = target_order.rename(columns={field: "target_entity"})
    target_order["target_entity"] = target_order["target_entity"].astype(str)
    target_order["rank"] = [
        _stable_rank(seed, f"geometry-split:{factor}", target_id)
        for target_id in target_order[CONDITION_ID].astype(str)
    ]
    target_order = target_order.sort_values(["target_entity", "rank", CONDITION_ID])
    target_order["context_split"] = target_order.groupby("target_entity", sort=False).cumcount().map(
        lambda value: "validation" if value == 2 else "training"
    )
    return target_order[[CONDITION_ID, "target_entity", "context_split"]]


def _entity_geometry_metrics(
    entity_results: pd.DataFrame,
    *,
    factor: str,
    entities: Sequence[str],
    split_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit source-effect MDS on two contexts and validate on a third context."""

    values = entity_results.copy()
    values["source_entity"] = values["source_entity"].astype(str)
    if "target_entity" not in values:
        values = values.merge(
            _geometry_context_splits(values, factor=factor, seed=split_seed),
            on=CONDITION_ID,
            how="left",
            validate="many_to_one",
        )
    else:
        values["target_entity"] = values["target_entity"].astype(str)
        if "context_split" not in values:
            values = values.merge(
                _geometry_context_splits(values, factor=factor, seed=split_seed)[
                    [CONDITION_ID, "context_split"]
                ],
                on=CONDITION_ID,
                how="left",
                validate="many_to_one",
            )
    if values["context_split"].isna().any():
        raise AssertionError("Geometry target context split was not assigned")
    target_order = values.loc[:, [CONDITION_ID, "target_entity", "context_split"]].drop_duplicates()
    # Use individual target conditions as profile columns, so the validation
    # comparison genuinely holds out biological contexts rather than matrix cells.
    pivot = values.pivot(index="source_entity", columns=CONDITION_ID, values="transfer_score")
    pivot = pivot.reindex(index=[str(value) for value in entities])
    train_ids = target_order.loc[target_order["context_split"].eq("training"), CONDITION_ID].astype(str).tolist()
    validation_ids = target_order.loc[target_order["context_split"].eq("validation"), CONDITION_ID].astype(str).tolist()
    train = pivot.loc[:, train_ids].to_numpy(dtype=float)
    validation = pivot.loc[:, validation_ids].to_numpy(dtype=float)
    train_distance = _corr_distance(train)
    validation_distance = _corr_distance(validation)
    mean_matrix = values.pivot_table(
        index="source_entity", columns="target_entity", values="transfer_score", aggfunc="mean"
    ).reindex(index=[str(value) for value in entities], columns=[str(value) for value in entities])
    rows: list[dict[str, Any]] = []
    train_identifiable = bool(np.isfinite(train_distance).all())
    validation_identifiable = bool(np.isfinite(validation_distance).all())
    for requested in (1, 2, 4):
        if not train_identifiable:
            rows.append(
                {
                    "factor": factor,
                    "entity_count": len(entities),
                    "target_context_count": len(target_order),
                    "training_context_count": len(train_ids),
                    "validation_context_count": len(validation_ids),
                    "requested_dimensions": requested,
                    "effective_dimensions": 0,
                    "status": "not_identifiable_train_zero_variance_source_profile",
                    "geometry_type": "directed_source_effect_profile_mds",
                    "train_stress": float("nan"),
                    "train_distance_correlation": float("nan"),
                    "train_distance_r2": float("nan"),
                    "validation_distance_correlation": float("nan"),
                    "validation_distance_r2": float("nan"),
                    "validation_neighbor_overlap_at_3": float("nan"),
                    "positive_eigen_energy_retained": float("nan"),
                }
            )
            continue
        embedding, _, effective, retained = _classical_mds(train_distance, requested)
        reconstructed = _pairwise_distance(embedding)
        reference_values = _upper_values(train_distance)
        reconstructed_values = _upper_values(reconstructed)
        stress = (
            float(
                math.sqrt(
                    np.square(reference_values - reconstructed_values).sum()
                    / np.square(reference_values).sum()
                )
            )
            if np.square(reference_values).sum() > 0.0
            else float("nan")
        )
        train_corr, train_r2 = _distance_r2(train_distance, reconstructed)
        validation_corr, validation_r2 = (
            _distance_r2(validation_distance, reconstructed)
            if validation_identifiable
            else (float("nan"), float("nan"))
        )
        status = "estimated"
        if requested > len(entities) - 1:
            status = "capped_at_n_minus_1"
        if effective < requested and status == "estimated":
            status = "rank_limited"
        if not validation_identifiable:
            status = "validation_not_identifiable_zero_variance_source_profile"
        rows.append(
            {
                "factor": factor,
                "entity_count": len(entities),
                "target_context_count": len(target_order),
                "training_context_count": len(train_ids),
                "validation_context_count": len(validation_ids),
                "requested_dimensions": requested,
                "effective_dimensions": effective,
                "status": status,
                "geometry_type": "directed_source_effect_profile_mds",
                "train_stress": stress,
                "train_distance_correlation": train_corr,
                "train_distance_r2": train_r2,
                "validation_distance_correlation": validation_corr,
                "validation_distance_r2": validation_r2,
                "validation_neighbor_overlap_at_3": (
                    _neighbor_overlap(validation_distance, reconstructed)
                    if validation_identifiable
                    else float("nan")
                ),
                "positive_eigen_energy_retained": retained,
            }
        )
    return pd.DataFrame(rows), mean_matrix


def _run_entity_geometry_factor(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: ProbeSettings,
    *,
    factor: str,
    external_holdout: frozenset[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_ids = tuple(str(value) for value in dataset.candidate_pool_ids)
    metadata = dataset.metadata.loc[list(candidate_ids)].copy()
    entities = _entity_values(metadata.loc[~metadata.index.astype(str).isin(external_holdout)], factor)
    target_ids = _geometry_target_panel(
        metadata,
        factor=factor,
        entities=entities,
        external_holdout=external_holdout,
        settings=settings,
        replicates=3,
    )
    geometry_holdout = frozenset(set(external_holdout) | set(target_ids))
    plans = {
        target_id: _build_entity_plan(
            metadata,
            target_id,
            factor=factor,
            source_entities=entities,
            holdout_ids=geometry_holdout,
            settings=settings,
            selection_seed=settings.seed,
        )
        for target_id in target_ids
    }
    bundle = load_identity_feature_bundle(dataset)
    row_positions = {str(sample_id): index for index, sample_id in enumerate(bundle.row_ids)}
    response = dataset.response.loc[list(bundle.row_ids)].to_numpy(dtype=np.float32)
    model_settings = _model_settings(settings)
    records: list[dict[str, Any]] = []
    factor_field = {"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]
    for target_id in target_ids:
        target = metadata.loc[target_id]
        plan = plans[target_id]
        target_response = response[row_positions[target_id]]
        baseline_metrics, baseline_seconds = _fit_and_score(
            bundle=bundle,
            response=response,
            row_positions=row_positions,
            target_id=target_id,
            support_ids=plan.baseline_ids,
            target_response=target_response,
            settings=model_settings,
            fit_seed=settings.seed,
        )
        baseline_source_counts = (
            metadata.loc[list(plan.baseline_ids), factor_field]
            .astype(str)
            .value_counts()
            .reindex([str(value) for value in entities], fill_value=0)
        )
        if baseline_source_counts.nunique() != 1:
            raise AssertionError("Geometry baseline is not source-neutral")
        coverage = _feature_coverage(metadata, target, plan.baseline_ids)
        if not all(coverage.values()):
            raise AssertionError("Geometry baseline failed target feature coverage")
        for source_entity, donors in plan.donor_ids.items():
            after_metrics, after_seconds = _fit_and_score(
                bundle=bundle,
                response=response,
                row_positions=row_positions,
                target_id=target_id,
                support_ids=tuple(plan.baseline_ids) + tuple(donors),
                target_response=target_response,
                settings=model_settings,
                fit_seed=settings.seed,
            )
            records.append(
                {
                    "factor": factor,
                    CONDITION_ID: target_id,
                    "target_strain": _as_text(target[STRAIN]),
                    "target_chemical": _as_text(target[CHEMICAL]),
                    "target_time_minutes": target[TIME],
                    "source_entity": str(source_entity),
                    "target_entity": _as_text(target[{"strain": STRAIN, "chemical": CHEMICAL, "time": TIME}[factor]]),
                    "donor_ids": json.dumps(list(donors)),
                    "donor_count": len(donors),
                    "baseline_ids": json.dumps(list(plan.baseline_ids)),
                    "baseline_count": len(plan.baseline_ids),
                    "baseline_source_entity_count": int(baseline_source_counts.loc[str(source_entity)]),
                    "baseline_source_entity_min_count": int(baseline_source_counts.min()),
                    "baseline_source_entity_max_count": int(baseline_source_counts.max()),
                    "baseline_source_entity_counts": json.dumps(
                        {str(key): int(value) for key, value in baseline_source_counts.items()},
                        sort_keys=True,
                    ),
                    "baseline_is_source_neutral": True,
                    "baseline_delta_skill_zero": baseline_metrics[PRIMARY_METRIC],
                    "after_delta_skill_zero": after_metrics[PRIMARY_METRIC],
                    "transfer_score": float(after_metrics[PRIMARY_METRIC] - baseline_metrics[PRIMARY_METRIC]),
                    "baseline_train_seconds": baseline_seconds,
                    "after_train_seconds": after_seconds,
                    **coverage,
                    **_support_covariates(metadata, target, donors),
                }
            )
        print(f"[geometry:{factor}] completed {target_id}", flush=True)
    entity_results = pd.DataFrame(records)
    context_splits = _geometry_context_splits(
        entity_results, factor=factor, seed=settings.seed
    )
    entity_results = entity_results.merge(
        context_splits[[CONDITION_ID, "context_split"]],
        on=CONDITION_ID,
        how="left",
        validate="many_to_one",
    )
    target_frame = metadata.loc[list(target_ids), [STRAIN, CHEMICAL, TIME, MEDIUM, TEMPERATURE]].copy()
    target_frame = target_frame.merge(
        context_splits.set_index(CONDITION_ID)[["context_split"]],
        left_index=True,
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    target_frame.insert(0, "factor", factor)
    target_frame.insert(1, "target_geometry_rank", np.arange(1, len(target_frame) + 1))
    target_frame.to_csv(
        output_dir / f"GEOMETRY_TARGET_PANEL_{factor}.csv", index_label=CONDITION_ID
    )
    metrics, full_matrix = _entity_geometry_metrics(
        entity_results, factor=factor, entities=entities, split_seed=settings.seed
    )
    training_matrix = (
        entity_results.loc[entity_results["context_split"].eq("training")]
        .pivot_table(
            index="source_entity",
            columns="target_entity",
            values="transfer_score",
            aggfunc="mean",
        )
        .reindex(index=[str(value) for value in entities], columns=[str(value) for value in entities])
    )
    # The default matrix is training-only so a later Key prototype cannot
    # accidentally consume held-out context labels.  The full matrix is kept as
    # an explicitly descriptive artifact for inspection only.
    training_matrix.to_csv(
        output_dir / f"ENTITY_TRANSFER_MATRIX_{factor}.csv", index_label="source_entity"
    )
    full_matrix.to_csv(
        output_dir / f"ENTITY_TRANSFER_MATRIX_{factor}_FULL_DESCRIPTIVE.csv",
        index_label="source_entity",
    )
    _plot_entity_heatmap(training_matrix, factor, output_dir)
    return entity_results, metrics, target_frame


def _plot_entity_heatmap(matrix: pd.DataFrame, factor: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    image = axis.imshow(values, aspect="auto", cmap="coolwarm")
    axis.set_xticks(np.arange(matrix.shape[1]), matrix.columns.astype(str), rotation=45, ha="right")
    axis.set_yticks(np.arange(matrix.shape[0]), matrix.index.astype(str))
    axis.set_xlabel(f"Target {factor}")
    axis.set_ylabel(f"Donor/source {factor}")
    axis.set_title(f"{factor.title()} source-effect transfer matrix (training contexts)")
    figure.colorbar(image, ax=axis, label="Δ delta_skill_zero")
    figure.savefig(output_dir / f"{factor}_source_effect_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_factor_dimension_analysis(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: ProbeSettings,
    *,
    main_summary: pd.DataFrame,
    main_target_ids: Sequence[str],
) -> pd.DataFrame:
    """Run source-effect geometry only for factors with stable main transfer."""

    stable = set(_stable_main_factors(main_summary))
    all_rows: list[pd.DataFrame] = []
    entity_outputs: list[pd.DataFrame] = []
    for factor in ("strain", "chemical", "time"):
        if factor not in stable:
            all_rows.append(
                pd.DataFrame(
                    [
                        {
                            "factor": factor,
                            "requested_dimensions": dimensions,
                            "status": "not_run_no_positive_main_transfer_signal",
                            "geometry_type": "not_estimated",
                        }
                        for dimensions in (1, 2, 4)
                    ]
                )
            )
            continue
        entity_results, metrics, _ = _run_entity_geometry_factor(
            dataset,
            output_dir,
            settings,
            factor=factor,
            external_holdout=frozenset(str(value) for value in main_target_ids),
        )
        entity_outputs.append(entity_results)
        all_rows.append(metrics)
    dimension = pd.concat(all_rows, ignore_index=True, sort=False)
    dimension.to_csv(output_dir / "FACTOR_DIMENSION_SUMMARY.csv", index=False)
    if entity_outputs:
        pd.concat(entity_outputs, ignore_index=True).to_csv(
            output_dir / "ENTITY_TRANSFER_PROBE_RESULTS.csv", index=False
        )
        _plot_dimension_curves(dimension, output_dir)
    else:
        pd.DataFrame(
            columns=["factor", CONDITION_ID, "source_entity", "target_entity", "transfer_score"]
        ).to_csv(output_dir / "ENTITY_TRANSFER_PROBE_RESULTS.csv", index=False)
    return dimension


def _plot_dimension_curves(dimension: pd.DataFrame, output_dir: Path) -> None:
    plot = dimension.loc[
        dimension["validation_distance_correlation"].map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else False
        )
    ].copy()
    if len(plot) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for factor, frame in plot.groupby("factor", sort=True):
        frame = frame.sort_values("requested_dimensions")
        axes[0].plot(
            frame["requested_dimensions"],
            frame["validation_distance_correlation"],
            marker="o",
            label=factor,
        )
        axes[1].plot(
            frame["requested_dimensions"],
            frame["train_stress"],
            marker="o",
            label=factor,
        )
    axes[0].set_xlabel("Requested dimensions")
    axes[0].set_ylabel("Held-out context distance correlation")
    axes[0].set_title("Source-effect geometry preservation")
    axes[1].set_xlabel("Requested dimensions")
    axes[1].set_ylabel("Training stress (lower is better)")
    axes[1].set_title("MDS fit")
    for axis in axes:
        axis.set_xticks([1, 2, 4])
        axis.legend(frameon=False)
    fig.savefig(output_dir / "factor_dimension_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_donor_draw_sensitivity(
    dataset: GroupedDataset,
    output_dir: Path,
    settings: ProbeSettings,
    *,
    target_ids: Sequence[str],
    primary_results: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat a small target subset with a second metadata-only donor draw.

    This is not a second model seed.  The primary baseline IDs are retained
    exactly; only the donor groups are redrawn under a second selection seed.
    It therefore asks whether relation transfer depends strongly on the one
    deterministic donor-group draw used by the main panel.
    """

    chosen_targets = tuple(target_ids[: settings.sensitivity_target_count])
    candidate_ids = tuple(str(value) for value in dataset.candidate_pool_ids)
    metadata = dataset.metadata.loc[list(candidate_ids)].copy()
    bundle = load_identity_feature_bundle(dataset)
    row_positions = {str(sample_id): index for index, sample_id in enumerate(bundle.row_ids)}
    response = dataset.response.loc[list(bundle.row_ids)].to_numpy(dtype=np.float32)
    model_settings = _model_settings(settings)
    rows: list[dict[str, Any]] = []
    required = {
        CONDITION_ID,
        "donor_relation",
        "transfer_score",
        "baseline_ids",
        "donor_ids",
        "baseline_delta_skill_zero",
    }
    missing = sorted(required - set(primary_results.columns))
    if missing:
        raise ValueError(f"Primary transfer results lack sensitivity columns: {missing}")
    primary = primary_results.loc[
        primary_results[CONDITION_ID].isin(chosen_targets),
        [
            CONDITION_ID,
            "donor_relation",
            "transfer_score",
            "baseline_ids",
            "donor_ids",
            "baseline_delta_skill_zero",
        ],
    ].copy()
    primary["donor_draw"] = "primary"
    primary["selection_seed"] = settings.seed
    rows.extend(primary.to_dict("records"))
    selection_seed = settings.seed + 104729
    holdout = frozenset(str(value) for value in target_ids)
    for target_id in chosen_targets:
        target_primary = primary.loc[primary[CONDITION_ID].eq(target_id)]
        baseline_strings = target_primary["baseline_ids"].drop_duplicates().tolist()
        baseline_scores = target_primary["baseline_delta_skill_zero"].drop_duplicates().tolist()
        if len(baseline_strings) != 1 or len(baseline_scores) != 1:
            raise AssertionError("Primary sensitivity target lacks one fixed baseline")
        fixed_baseline = tuple(str(value) for value in json.loads(baseline_strings[0]))
        baseline_score = float(baseline_scores[0])
        plan = _build_main_plan(
            metadata,
            target_id,
            holdout_ids=holdout,
            settings=settings,
            selection_seed=selection_seed,
            fixed_baseline_ids=fixed_baseline,
        )
        target_response = response[row_positions[target_id]]
        if tuple(plan.baseline_ids) != fixed_baseline:
            raise AssertionError("Alternate donor draw changed the primary baseline")
        for relation in MAIN_RELATIONS:
            after, _ = _fit_and_score(
                bundle=bundle,
                response=response,
                row_positions=row_positions,
                target_id=target_id,
                support_ids=tuple(plan.baseline_ids) + tuple(plan.donor_ids[relation]),
                target_response=target_response,
                settings=model_settings,
                fit_seed=settings.seed,
            )
            rows.append(
                {
                    CONDITION_ID: target_id,
                    "donor_relation": relation,
                    "transfer_score": float(after[PRIMARY_METRIC] - baseline_score),
                    "donor_draw": "alternate",
                    "baseline_ids": baseline_strings[0],
                    "donor_ids": json.dumps(list(plan.donor_ids[relation])),
                    "baseline_delta_skill_zero": baseline_score,
                    "selection_seed": selection_seed,
                }
            )
        print(f"[sensitivity] completed alternate donor draw for {target_id}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "DONOR_DRAW_SENSITIVITY.csv", index=False)
    paired = frame.pivot(
        index=[CONDITION_ID, "donor_relation"], columns="donor_draw", values="transfer_score"
    ).reset_index()
    summaries: list[dict[str, Any]] = []
    for relation, relation_frame in paired.groupby("donor_relation", sort=False):
        valid = relation_frame.dropna(subset=["primary", "alternate"])
        correlation = (
            float(valid["primary"].rank().corr(valid["alternate"].rank(), method="pearson"))
            if len(valid) > 1
            else float("nan")
        )
        sign_agreement = (
            float(np.mean(np.sign(valid["primary"]) == np.sign(valid["alternate"])))
            if len(valid)
            else float("nan")
        )
        summaries.append(
            {
                "donor_relation": relation,
                "n_targets": int(len(valid)),
                "spearman_transfer_correlation": correlation,
                "sign_agreement_fraction": sign_agreement,
                "mean_absolute_draw_difference": float(
                    np.mean(np.abs(valid["primary"] - valid["alternate"]))
                ) if len(valid) else float("nan"),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "DONOR_DRAW_SENSITIVITY_SUMMARY.csv", index=False)
    _plot_donor_draw_sensitivity(summary, output_dir)
    return summary


def _plot_donor_draw_sensitivity(summary: pd.DataFrame, output_dir: Path) -> None:
    """Visualize robustness to a second donor draw with the baseline fixed."""

    if len(summary) == 0:
        return
    order = {relation: index for index, relation in enumerate(MAIN_RELATIONS)}
    frame = summary.assign(order=summary["donor_relation"].map(order)).sort_values("order")
    labels = {
        "random_none": "Random",
        "strain_only": "S",
        "chemical_only": "D",
        "time_only": "T",
        "strain_chemical": "S×D",
        "chemical_time": "D×T",
        "strain_time": "S×T",
        "strain_chemical_mixture": "S/D mix",
        "chemical_time_mixture": "D/T mix",
        "strain_time_mixture": "S/T mix",
    }
    x = np.arange(len(frame))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), constrained_layout=True)
    axes[0].bar(x, frame["spearman_transfer_correlation"], color="#5f9e6e")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("Primary vs alternate donor draw")
    axes[1].bar(x, frame["sign_agreement_fraction"], color="#4878a8")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Transfer sign agreement")
    axes[1].set_title("Baseline fixed; donor groups redrawn")
    tick_labels = [labels[name] for name in frame["donor_relation"]]
    for axis in axes:
        axis.set_xticks(x, tick_labels, rotation=35, ha="right")
    figure.savefig(output_dir / "donor_draw_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _format_ci(row: pd.Series, prefix: str) -> str:
    mean = _format_number(row[prefix])
    low_column = f"{prefix}_ci95_low"
    high_column = f"{prefix}_ci95_high"
    # Relation summaries keep the explicit ``mean_`` prefix only for the
    # point estimate; their uncertainty columns are named ``transfer_*``.
    if low_column not in row.index and prefix.startswith("mean_"):
        low_column = f"{prefix.removeprefix('mean_')}_ci95_low"
        high_column = f"{prefix.removeprefix('mean_')}_ci95_high"
    low = _format_number(row[low_column])
    high = _format_number(row[high_column])
    return f"{mean}（95% target-bootstrap CI {low} 到 {high}）"


def _main_effect_sentence(summary: pd.DataFrame, relation: str, chinese_name: str) -> str:
    row = summary.loc[
        summary["summary_type"].eq("relation_transfer")
        & summary["relation"].eq(relation)
    ].iloc[0]
    direction = "有稳定的正向额外迁移" if float(row["transfer_vs_random_ci95_low"]) > 0 else "没有检出稳定的正向额外迁移"
    return (
        f"- {chinese_name}：相对等量随机 donor 的转移差为 "
        f"{_format_ci(row, 'mean_transfer_vs_random')}，因此{direction}。"
    )


def _dimension_interpretation(dimension: pd.DataFrame, factor: str) -> str:
    rows = dimension.loc[dimension["factor"].eq(factor)].copy()
    if len(rows) == 0 or "validation_distance_correlation" not in rows:
        return f"{factor}：主 probe 未提供足以启动 entity-geometry 分析的稳定正向信号。"
    valid = rows.loc[
        rows["validation_distance_correlation"].map(
            lambda value: math.isfinite(float(value)) if pd.notna(value) else False
        )
    ]
    if len(valid) == 0:
        return (
            f"{factor}：source-effect profile 在 held-out context 中没有可识别的距离变异，"
            "因此不能从本面板选择 1D/2D/4D。"
        )
    best = valid.loc[valid["validation_distance_correlation"].idxmax()]
    one_rows = valid.loc[valid["requested_dimensions"].eq(1)]
    one = one_rows.iloc[0] if len(one_rows) else None
    if float(best["validation_distance_correlation"]) <= 0.0:
        return (
            f"{factor}：最高 held-out distance correlation 仍为 "
            f"{_format_number(best['validation_distance_correlation'])}；"
            "没有方向一致的跨-context preservation，不能据此推断 intrinsic dimension。"
        )
    if (
        one is not None
        and float(best["validation_distance_correlation"])
        - float(one["validation_distance_correlation"])
        <= 0.05
    ):
        conclusion = "1D 在 held-out context preservation 上已接近本面板的最佳维数。"
    else:
        conclusion = (
            f"最佳 held-out preservation 出现在请求 {int(best['requested_dimensions'])}D；"
            "1D 不是同等保真度的压缩。"
        )
    cap = "（4D 受实体数上限截断）" if (rows["status"] == "capped_at_n_minus_1").any() else ""
    return (
        f"{factor}：{conclusion} 最佳 validation distance correlation="
        f"{_format_number(best['validation_distance_correlation'])}，"
        f"1D={_format_number(one['validation_distance_correlation']) if one is not None else 'NA'}。{cap}"
    )


def _geometry_classification(
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
) -> tuple[str, bool, bool, bool, float]:
    main = summary.loc[summary["summary_type"].eq("relation_transfer")]
    factor_positive = any(
        float(main.loc[main["relation"].eq(relation), "transfer_vs_random_ci95_low"].iloc[0]) > 0.0
        for relation in ("strain_only", "chemical_only", "time_only")
    )
    interactions = summary.loc[summary["summary_type"].eq("interaction_excess")]
    pair_excess_positive = bool(
        (interactions["interaction_excess_ci95_low"].astype(float) > 0.0).any()
    )
    factor_r2 = float(
        model_summary.loc[model_summary["model"].eq("factor_only"), "loto_r2_vs_zero_random_control"].iloc[0]
    )
    hybrid_r2 = float(
        model_summary.loc[
            model_summary["model"].eq("factor_plus_interaction"),
            "loto_r2_vs_zero_random_control",
        ].iloc[0]
    )
    gain = hybrid_r2 - factor_r2
    # A single positive pair contrast is useful evidence, but calling the
    # entire transfer geometry tensor-dominated also needs held-out evidence
    # that interaction terms improve the explanatory model.
    interaction_positive = bool(
        pair_excess_positive
        and math.isfinite(hybrid_r2)
        and hybrid_r2 > 0.0
        and gain > 0.01
    )
    if factor_positive and interaction_positive:
        label = "Hybrid Geometry"
    elif interaction_positive:
        label = "Interaction / Tensor Geometry"
    elif factor_positive:
        label = "Mainly Factorized Geometry"
    else:
        label = "Current Evidence Insufficient"
    return label, factor_positive, interaction_positive, pair_excess_positive, gain


def write_final_summary(
    output_dir: Path,
    *,
    settings: ProbeSettings,
    audit: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    dimension: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> Path:
    """Create the requested human-readable scientific conclusion in Chinese."""

    label, factor_positive, interaction_positive, pair_excess_positive, r2_gain = _geometry_classification(
        summary, model_summary
    )
    pair_rows = summary.loc[summary["summary_type"].eq("interaction_excess")]
    pair_lines = []
    for pair, name in (
        ("strain_chemical", "strain × chemical"),
        ("chemical_time", "chemical × time"),
        ("strain_time", "strain × time"),
    ):
        row = pair_rows.loc[pair_rows["relation"].eq(pair)].iloc[0]
        direction = "存在正向 excess" if float(row["interaction_excess_ci95_low"]) > 0 else "未检出稳定的正向 excess"
        pair_lines.append(
            f"- {name}：equal-budget pair − mixture 为 "
            f"{_format_ci(row, 'interaction_excess')}，{direction}。"
        )
    factor_model = model_summary.loc[model_summary["model"].eq("factor_only")].iloc[0]
    hybrid_model = model_summary.loc[model_summary["model"].eq("factor_plus_interaction")].iloc[0]
    stable_factors = _stable_main_factors(summary)
    if label == "Hybrid Geometry":
        key_advice = (
            "采用 factor key 作为主坐标，并保留只在数据支持的 pairwise interaction residual；"
            "不宜把实验空间压成纯 factor block，也不宜先上全量 tensor。"
        )
    elif label == "Interaction / Tensor Geometry":
        key_advice = (
            "Key 需要显式表示已见组合关系，例如 factorized pair residual 或 relational key；"
            "仅拼接独立 factor 坐标不足以表达本 probe 中的迁移结构。"
        )
    elif label == "Mainly Factorized Geometry":
        key_advice = (
            "Key 应优先表达各 factor 的 information/transfer semantics；"
            "pair residual 可以作为后续诊断项，而不是 acquisition 的第一设计重心。"
        )
    else:
        key_advice = (
            "当前 Direct identity+time probe 没有提供足以选择纯 scalar、low-dimensional block "
            "或 tensor 的稳定证据；下一版 Key 应保留可检验的 factor 与 interaction 插槽，避免锁死表示。"
        )
    tensor_advice = (
        "等预算 pair arm 显示稳定正向 excess，且 interaction explanatory model 在 held-out target 上有实质增益；"
        "因此值得把 pairwise relational residual 作为下一阶段 acquisition 的优先实验对象，"
        "但这并不等同于已经证明高阶 dense tensor 必要。"
        if interaction_positive
        else (
            "等预算 pair arm 已出现稳定正向 excess，但它尚未转化为 held-out target 上的 interaction-model 增益；"
            "应先做独立 residual 验证，而不是据此直接启动大规模 tensor-aware acquisition。"
            if pair_excess_positive
            else "本 panel 没有检出强而稳定的 pairwise excess；不应仅凭 tensor occupancy 启动大规模 tensor-aware acquisition。"
        )
    )
    source_count = int(len(audit))
    eligible_count = int(audit["eligible_main"].sum())
    sensitivity_note = "未运行"
    if len(sensitivity):
        median_corr = float(sensitivity["spearman_transfer_correlation"].median())
        median_sign = float(sensitivity["sign_agreement_fraction"].median())
        sensitivity_targets = int(sensitivity["n_targets"].min())
        sensitivity_note = (
            f"{sensitivity_targets}-target 固定 baseline 的第二 donor-draw 中，各 arm Spearman 相关中位数为 "
            f"{_format_number(median_corr)}，符号一致率中位数为 {_format_number(median_sign)}。"
        )
    text = rf"""# GOAI Transfer Geometry Summary

## 1. 我们想回答什么

GOAI 主动学习真正需要决定的不是哪一个 acquisition 名字更好，而是一个已做实验为何能帮助预测另一个未做实验。本轮直接测量 donor experiments 加入后，对 withheld target condition 的预测改善，并据此判断 experimental space 更接近独立 factor、低维 factor block，还是需要 interaction / relational structure。

## 2. 实验怎么做

使用 condition-atomic matched-control log2-delta response 的 candidate pool。Structural availability audit 在 {source_count} 个 candidate condition 中找到 {eligible_count} 个同时支持所有已注册 relation 的 target；主 panel 从中仅按 metadata 分层选择 {settings.target_count} 个 target，覆盖 strain、chemical、time 与 medium/temperature context，未按 response 或初步效果筛选。

每个 target 在本 panel 中对所有其他 target 保持严格 withheld。其固定 baseline 含 {settings.baseline_size} 个 condition：strain、chemical、time、medium、temperature 各一个 feature-coverage anchor，另加不匹配 strain/chemical/time 的背景条件。每个 arm 额外加入相同数量（{settings.donor_count}）的 donor，用完全相同的 Direct identity+time predictor、{settings.epochs} epochs、初始化/训练 seed {settings.seed} 重新拟合；target response 只用于事后评分。主 transfer 定义为：

\[
T = \mathrm{{delta\_skill\_zero}}_{{after}} - \mathrm{{delta\_skill\_zero}}_{{baseline}}.
\]

正值表示 donor group 降低了 target 的相对零响应平方误差。S/D/T single-factor group 要求其它两个 focal axis 都不同；pair group 要求第三轴不同。每个 pair group 还与等预算的 50/50 single-factor mixture（例如 {settings.donor_count // 2} S-only + {settings.donor_count // 2} D-only）比较，因此 interaction excess 不把 {settings.donor_count} 个 pair donor 与更多 single donor 混为一谈。

## 3. 最重要的结果

### Factor transfer

{_main_effect_sentence(summary, 'strain_only', 'strain')}
{_main_effect_sentence(summary, 'chemical_only', 'chemical')}
{_main_effect_sentence(summary, 'time_only', 'time')}

### Interaction transfer

{chr(10).join(pair_lines)}

Factor-only 模型在 leave-one-target-out、相对 paired-random control 的 transfer 上得到 R²={_format_number(factor_model['loto_r2_vs_zero_random_control'])}（95% bootstrap CI {_format_number(factor_model['loto_r2_ci95_low'])} 到 {_format_number(factor_model['loto_r2_ci95_high'])}）；加入 pairwise interaction 后为 R²={_format_number(hybrid_model['loto_r2_vs_zero_random_control'])}（CI {_format_number(hybrid_model['loto_r2_ci95_low'])} 到 {_format_number(hybrid_model['loto_r2_ci95_high'])}），增益 {_format_number(r2_gain)}。该验证一次留出一个完整 target，而不是把同一 target 的多个 arm 当作独立样本。

## 4. Factor Geometry vs Tensor Geometry

\[
\boxed{{\text{{本轮阶段性判断：{label}}}}}
\]

这是当前 Direct identity+time predictor、已注册 donor relation 与 single-seed exact retraining 下的结构判断，而不是无条件的生物学真相。{'存在可重复的单 factor transfer，同时 pair arm 仍显示额外结构。' if label == 'Hybrid Geometry' else '主判断来自等预算 pair-vs-mixture contrast、target-cluster uncertainty 与 held-out explanatory performance。'}

## 5. Key 应该是什么

{key_advice}

Key 的语义应是“该实验为其他实验带来的可迁移信息”，而不是只把原始 metadata 编号压缩一下。对本轮有稳定 signal 的 factor（{', '.join(stable_factors) if stable_factors else '无'}），Key 可由独立 factor block 表示；是否附加 pair residual 则由上面的 equal-budget excess 决定。

## 6. 一因素一数字是否成立

- {_dimension_interpretation(dimension, 'strain')}
- {_dimension_interpretation(dimension, 'chemical')}
- {_dimension_interpretation(dimension, 'time')}

这里的 embedding 是**有向 source-effect transfer profile**的 MDS：两个 donor entity 若在多个 target context 上产生相近 transfer profile，才被视为接近。1D/2D/4D 用两份 target context 拟合、第三份 held-out context 验证；它不是把原始 chemical 名称相似度误当作 information geometry。

## 7. Tensor 是否必要

{tensor_advice}

本轮只测量 S/D/T 的 pairwise relation，未系统覆盖 medium、temperature、剂量（发布 metadata 没有公开浓度）或三阶 interaction。因此“未检出 pairwise excess”只表示当前受支持关系中没有强可检出的 excess，不能证明任何高阶 tensor 永远无用。

## 8. 对 GOAI Active Learning Framework 的直接建议

下一阶段应把本轮判断变成一个**小型、单 seed** acquisition confirmation：固定 initial set、Direct predictor 与 budget，比较 Random、当前 one-hot CoreSet、以及以本轮 source-effect/factor(+residual) Key 构造的 CoreSet。不要再开新的大规模 acquisition benchmark；confirmation 必须使用未参与 Key 选择的目标条件或预算。

## 9. 最终结论

在本次精确 donor-transfer audit 中，GOAI train 的可迁移知识结构被直接测量，而非由 tensor occupancy 或最终 MSE 间接猜测。{label} 是当前数据与 Direct predictor 下最合适的阶段性表示结论。{sensitivity_note}

所有分数均为本地 retrospective GOAI-AL matched-control proxy；没有官方 leaderboard score，也没有提交或生物发现主张。
"""
    path = output_dir / "GOAI_TRANSFER_GEOMETRY_SUMMARY.md"
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


def run_transfer_geometry(
    *,
    output_dir: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    proteome_path: Path = DEFAULT_PROTEOME_PATH,
    cache_dir: Path | None = None,
    settings: ProbeSettings = ProbeSettings(),
    run_geometry: bool = True,
    run_sensitivity: bool = True,
) -> Path:
    """Execute the full requested transfer-geometry audit."""

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
    metadata = dataset.metadata.loc[list(dataset.candidate_pool_ids)]
    metadata_record = {
        "experiment_id": EXPERIMENT_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "status": "running",
        "started_utc": started.isoformat(),
        "settings": asdict(settings),
        "feature_mode": "identity_plus_time",
        "predictor": "Direct 4422-output MLP",
        "primary_metric": PRIMARY_METRIC,
        "official_score_status": "none; local retrospective proxy only",
        "metadata_path": str(metadata_path),
        "proteome_path": str(proteome_path),
        "dataset_source_hashes": dict(dataset.source_hashes),
        "candidate_condition_count": int(len(dataset.candidate_pool_ids)),
        "official_train_condition_count": int(len(dataset.official_train_ids)),
        "protein_count": int(len(dataset.proteins)),
        "candidate_factor_cardinality": {
            "strain": int(metadata[STRAIN].nunique()),
            "chemical": int(metadata[CHEMICAL].nunique()),
            "time": int(metadata[TIME].nunique()),
            "medium": int(metadata[MEDIUM].nunique()),
            "temperature": int(metadata[TEMPERATURE].nunique()),
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
    results, audit, target_ids = run_main_probe(dataset, output, settings)
    summary, model_summary = analyze_main_probe(results, output, settings)
    sensitivity = (
        run_donor_draw_sensitivity(
            dataset,
            output,
            settings,
            target_ids=target_ids,
            primary_results=results,
        )
        if run_sensitivity
        else pd.DataFrame()
    )
    dimension = (
        run_factor_dimension_analysis(
            dataset,
            output,
            settings,
            main_summary=summary,
            main_target_ids=target_ids,
        )
        if run_geometry
        else pd.DataFrame(
            [
                {"factor": factor, "requested_dimensions": dimension_count, "status": "not_run_by_cli"}
                for factor in ("strain", "chemical", "time")
                for dimension_count in (1, 2, 4)
            ]
        )
    )
    if not run_geometry:
        dimension.to_csv(output / "FACTOR_DIMENSION_SUMMARY.csv", index=False)
        pd.DataFrame(
            columns=["factor", CONDITION_ID, "source_entity", "target_entity", "transfer_score"]
        ).to_csv(output / "ENTITY_TRANSFER_PROBE_RESULTS.csv", index=False)
    summary_path = write_final_summary(
        output,
        settings=settings,
        audit=audit,
        results=results,
        summary=summary,
        model_summary=model_summary,
        dimension=dimension,
        sensitivity=sensitivity,
    )
    metadata_record["status"] = "complete"
    metadata_record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_record["summary_path"] = str(summary_path)
    metadata_record["output_files"] = sorted(
        path.name for path in output.iterdir() if path.is_file()
    )
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
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--baseline-size", type=int, default=256)
    parser.add_argument("--donors", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-donor-sensitivity", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = ProbeSettings(
        seed=args.seed,
        target_count=args.targets,
        baseline_size=args.baseline_size,
        donor_count=args.donors,
        epochs=args.epochs,
        bootstrap_draws=args.bootstrap_draws,
        device=args.device,
    )
    output = run_transfer_geometry(
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        proteome_path=args.proteome_path,
        cache_dir=args.cache_dir,
        settings=settings,
        run_geometry=not args.skip_geometry,
        run_sensitivity=not args.skip_donor_sensitivity,
    )
    print(f"Transfer-geometry audit complete: {output}")


if __name__ == "__main__":
    main()
