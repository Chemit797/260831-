"""Target-free batch acquisition policies for the GOAI condition pool.

Acquisition deliberately operates on opaque public IDs.  Hidden responses and
the retrospective oracle belong on the other side of this module's API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


VALID_STRATEGIES = ("random", "coreset", "uncertainty")


class UncertaintyPredictor(Protocol):
    """Narrow predictor surface used to prepare MC-dropout scores."""

    def uncertainty(self, features: np.ndarray, passes: int) -> np.ndarray:
        ...


@runtime_checkable
class Acquisition(Protocol):
    """Label-free acquisition interface over the public context boundary."""

    def select_batch(
        self,
        context: "AcquisitionContext",
        batch_size: int,
        *,
        seed: int = 0,
    ) -> tuple[str, ...]:
        """Return candidate IDs without access to oracle responses or labels."""

        ...


def _validated_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    ids = tuple(values)
    invalid = [value for value in ids if not isinstance(value, str) or not value]
    if invalid:
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def _readonly_matrix(values: np.ndarray, rows: int, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[0] != rows:
        raise ValueError(f"{name} row count does not match its row IDs")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(matrix, copy=True)
    result.setflags(write=False)
    return result


def _readonly_vector(values: np.ndarray, rows: int, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or len(vector) != rows:
        raise ValueError(f"{name} must have one value per row ID")
    if not np.issubdtype(vector.dtype, np.number) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite numeric values")
    if (vector < 0).any():
        raise ValueError(f"{name} cannot contain negative variances")
    result = np.array(vector, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _row_positions(requested: Sequence[str], available: Sequence[str], name: str) -> np.ndarray:
    lookup = {row_id: position for position, row_id in enumerate(available)}
    missing = [row_id for row_id in requested if row_id not in lookup]
    if missing:
        raise ValueError(f"{name} is missing row IDs: {missing[:5]}")
    return np.asarray([lookup[row_id] for row_id in requested], dtype=np.int64)


@dataclass(frozen=True, init=False)
class AcquisitionContext:
    """Immutable, target-free input to every acquisition policy.

    ``descriptor_ids`` identifies descriptor rows explicitly, so descriptor and
    predictor feature matrices may be independently ordered and have unrelated
    column counts.  Only the predictor's already-computed uncertainty is kept.
    """

    candidate_ids: tuple[str, ...]
    labelled_ids: tuple[str, ...]
    descriptor_ids: tuple[str, ...]
    _descriptors: np.ndarray = field(repr=False)
    _uncertainty: np.ndarray | None = field(repr=False)

    def __init__(
        self,
        candidate_ids: Sequence[str],
        descriptor_ids: Sequence[str],
        descriptors: np.ndarray,
        *,
        labelled_ids: Sequence[str] = (),
        uncertainty_ids: Sequence[str] | None = None,
        uncertainty: np.ndarray | None = None,
    ) -> None:
        candidates = _validated_ids(candidate_ids, "candidate_ids")
        labelled = _validated_ids(labelled_ids, "labelled_ids")
        descriptor_rows = _validated_ids(descriptor_ids, "descriptor_ids")
        overlap = set(candidates) & set(labelled)
        if overlap:
            raise ValueError(f"candidate_ids and labelled_ids overlap: {sorted(overlap)[:5]}")
        public_ids = set(candidates) | set(labelled)
        if public_ids != set(descriptor_rows):
            missing = sorted(public_ids - set(descriptor_rows))
            extra = sorted(set(descriptor_rows) - public_ids)
            raise ValueError(
                "descriptor_ids must equal candidate_ids plus labelled_ids; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        descriptor_matrix = _readonly_matrix(
            descriptors, len(descriptor_rows), "descriptors"
        )

        uncertainty_values: np.ndarray | None = None
        if uncertainty is None:
            if uncertainty_ids is not None:
                raise ValueError("uncertainty_ids requires uncertainty values")
        else:
            if uncertainty_ids is None:
                uncertainty_rows = candidates
            else:
                uncertainty_rows = _validated_ids(uncertainty_ids, "uncertainty_ids")
            if set(uncertainty_rows) != set(candidates):
                raise ValueError("uncertainty_ids must contain exactly the candidate IDs")
            raw_uncertainty = _readonly_vector(
                uncertainty, len(uncertainty_rows), "uncertainty"
            )
            positions = _row_positions(candidates, uncertainty_rows, "uncertainty")
            uncertainty_values = _readonly_vector(
                raw_uncertainty[positions], len(candidates), "uncertainty"
            )

        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "labelled_ids", labelled)
        object.__setattr__(self, "descriptor_ids", descriptor_rows)
        object.__setattr__(self, "_descriptors", descriptor_matrix)
        object.__setattr__(self, "_uncertainty", uncertainty_values)

    @property
    def descriptors(self) -> np.ndarray:
        """A read-only view in ``descriptor_ids`` row order."""

        view = self._descriptors.view()
        view.setflags(write=False)
        return view

    @property
    def uncertainty(self) -> np.ndarray | None:
        """Read-only candidate uncertainty in ``candidate_ids`` order."""

        if self._uncertainty is None:
            return None
        view = self._uncertainty.view()
        view.setflags(write=False)
        return view

    @classmethod
    def from_predictor(
        cls,
        candidate_ids: Sequence[str],
        descriptor_ids: Sequence[str],
        descriptors: np.ndarray,
        *,
        labelled_ids: Sequence[str],
        predictor: UncertaintyPredictor,
        model_feature_ids: Sequence[str],
        model_features: np.ndarray,
        mc_passes: int,
    ) -> "AcquisitionContext":
        """Prepare an uncertainty context without retaining predictor internals.

        Row-ID sets, rather than matrix shapes, establish alignment.  In
        particular, descriptor and model feature column counts need not match.
        """

        descriptor_rows = _validated_ids(descriptor_ids, "descriptor_ids")
        model_rows = _validated_ids(model_feature_ids, "model_feature_ids")
        if set(descriptor_rows) != set(model_rows):
            raise ValueError("descriptor and model feature matrices must agree on row IDs")
        model_matrix = _readonly_matrix(
            model_features, len(model_rows), "model_features"
        )
        candidates = _validated_ids(candidate_ids, "candidate_ids")
        positions = _row_positions(candidates, model_rows, "model_features")
        if mc_passes < 2:
            raise ValueError("MC-dropout uncertainty requires at least two passes")
        scores = predictor.uncertainty(model_matrix[positions], passes=int(mc_passes))
        return cls(
            candidates,
            descriptor_rows,
            descriptors,
            labelled_ids=labelled_ids,
            uncertainty_ids=candidates,
            uncertainty=scores,
        )


def _nearest_squared_distance(
    candidates: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    if len(reference) == 0:
        return np.full(len(candidates), np.inf, dtype=np.float64)
    reference = reference.astype(np.float64, copy=False)
    reference_norm = np.sum(reference * reference, axis=1)
    result = np.empty(len(candidates), dtype=np.float64)
    for start in range(0, len(candidates), chunk_size):
        block = candidates[start : start + chunk_size].astype(np.float64, copy=False)
        distance = (
            np.sum(block * block, axis=1)[:, None]
            + reference_norm[None, :]
            - 2.0 * block @ reference.T
        )
        result[start : start + len(block)] = np.maximum(distance.min(axis=1), 0.0)
    return result


def farthest_first(
    features: np.ndarray,
    candidates: np.ndarray,
    labelled: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stable farthest-first selection over validated integer row positions.

    ``rng`` remains accepted for source compatibility but ties are deliberately
    resolved by candidate order, not random jitter.
    """

    del rng
    matrix = _readonly_matrix(features, len(features), "features")
    candidate_rows = np.asarray(candidates)
    labelled_rows = np.asarray(labelled)
    for values, name in ((candidate_rows, "candidates"), (labelled_rows, "labelled")):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"{name} contains duplicate row positions")
        if ((values < 0) | (values >= len(matrix))).any():
            raise ValueError(f"{name} contains invalid row positions")
    if np.intersect1d(candidate_rows, labelled_rows).size:
        raise ValueError("candidates and labelled row positions overlap")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer")
    take = min(int(batch_size), len(candidate_rows))
    if take == 0:
        return np.empty(0, dtype=np.int64)
    candidate_rows = candidate_rows.astype(np.int64, copy=False)
    labelled_rows = labelled_rows.astype(np.int64, copy=False)
    if take == len(candidate_rows):
        return candidate_rows.copy()

    min_distance = _nearest_squared_distance(matrix[candidate_rows], matrix[labelled_rows])
    selected: list[int] = []
    available = np.ones(len(candidate_rows), dtype=bool)
    for _ in range(take):
        local = int(np.argmax(np.where(available, min_distance, -np.inf)))
        chosen = int(candidate_rows[local])
        selected.append(chosen)
        available[local] = False
        delta = matrix[candidate_rows] - matrix[chosen]
        distance = np.sum(delta * delta, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(selected, dtype=np.int64)


def select_batch(
    strategy: str,
    context: AcquisitionContext,
    batch_size: int,
    *,
    seed: int = 0,
) -> tuple[str, ...]:
    """Select public candidate IDs using one of the frozen v2 policies."""

    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    if not isinstance(context, AcquisitionContext):
        raise TypeError("acquisition requires a public AcquisitionContext")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 0:
        raise ValueError("batch_size must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    take = min(int(batch_size), len(context.candidate_ids))
    if take == 0:
        return ()

    if strategy == "random":
        order = np.random.default_rng(int(seed)).permutation(len(context.candidate_ids))
        return tuple(context.candidate_ids[position] for position in order[:take])

    descriptor_positions = _row_positions(
        (*context.candidate_ids, *context.labelled_ids),
        context.descriptor_ids,
        "descriptors",
    )
    candidate_positions = descriptor_positions[: len(context.candidate_ids)]
    labelled_positions = descriptor_positions[len(context.candidate_ids) :]
    if strategy == "coreset":
        selected = farthest_first(
            context.descriptors,
            candidate_positions,
            labelled_positions,
            take,
        )
        descriptor_id_by_position = dict(enumerate(context.descriptor_ids))
        return tuple(descriptor_id_by_position[int(position)] for position in selected)

    uncertainty = context.uncertainty
    if uncertainty is None:
        raise ValueError("uncertainty policy requires MC-dropout uncertainty scores")
    ranking = np.argsort(-uncertainty, kind="stable")
    return tuple(context.candidate_ids[position] for position in ranking[:take])
