"""Privacy-preserving primitives for retrospective active-learning simulation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


def _validated_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    ids = tuple(values)
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def _validated_public_matrix(values: np.ndarray, rows: int) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] != rows:
        raise ValueError("descriptors must be a 2D matrix with one row per descriptor ID")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("descriptors must contain only finite numeric values")
    result = np.array(matrix, copy=True)
    result.setflags(write=False)
    return result


def _id_positions(requested: Sequence[str], available: Sequence[str], name: str) -> np.ndarray:
    lookup = {row_id: position for position, row_id in enumerate(available)}
    missing = [row_id for row_id in requested if row_id not in lookup]
    if missing:
        raise ValueError(f"{name} contains invalid IDs: {missing[:5]}")
    return np.asarray([lookup[row_id] for row_id in requested], dtype=np.int64)


def _hash_json(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_id_matrix(row_ids: Sequence[str], matrix: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(_hash_json(tuple(row_ids)).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(_hash_json(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


_FORBIDDEN_PUBLIC_METADATA_KEYS = {
    "impact",
    "label",
    "labels",
    "oracle_impact",
    "oracle_response",
    "oracle_responses",
    "response",
    "responses",
}


def _json_safe_public_metadata(
    value: object,
    *,
    nonfinite_to_none: bool,
    path: str,
) -> object:
    """Return detached JSON metadata while excluding hidden-value field names."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            if key.casefold() in _FORBIDDEN_PUBLIC_METADATA_KEYS:
                raise ValueError(f"{path} contains forbidden public metadata key {key!r}")
            result[key] = _json_safe_public_metadata(
                item,
                nonfinite_to_none=nonfinite_to_none,
                path=f"{path}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe_public_metadata(
                item,
                nonfinite_to_none=nonfinite_to_none,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return _json_safe_public_metadata(
            value.item(), nonfinite_to_none=nonfinite_to_none, path=path
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if nonfinite_to_none:
            return None
        raise ValueError(f"{path} must not contain nonfinite floats")
    raise ValueError(f"{path} contains a value that is not JSON-safe")


def _validated_optional_seed(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return int(value)


@dataclass(frozen=True, init=False)
class RevealedBatch:
    """A newly revealed, ID-aligned label batch.

    Labels are copies and cannot mutate the oracle.  Their rows follow ``ids``.
    """

    ids: tuple[str, ...]
    _labels: np.ndarray = field(repr=False)

    def __init__(self, ids: Sequence[str], labels: np.ndarray) -> None:
        batch_ids = _validated_ids(ids, "revealed ids")
        values = np.asarray(labels)
        if values.ndim != 2 or values.shape[0] != len(batch_ids):
            raise ValueError("revealed labels must have one matrix row per ID")
        copied = np.array(values, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "ids", batch_ids)
        object.__setattr__(self, "_labels", copied)

    @property
    def labels(self) -> np.ndarray:
        view = self._labels.view()
        view.setflags(write=False)
        return view

    @property
    def shape(self) -> tuple[int, ...]:
        return self._labels.shape

    def __len__(self) -> int:
        return len(self.ids)


class RetrospectiveOracle:
    """Own hidden responses and reveal each eligible candidate at most once.

    Repeat reveal policy is intentionally strict: a request containing any
    previously revealed ID raises ``ValueError`` and reveals nothing.  Explicit
    ``response_ids`` make construction invariant to a joint permutation of the
    hidden response rows and their IDs.
    """

    repeat_reveal_policy = "error"

    def __init__(
        self,
        candidate_ids: Sequence[str],
        responses: np.ndarray,
        *,
        response_ids: Sequence[str] | None = None,
    ) -> None:
        candidates = _validated_ids(candidate_ids, "candidate_ids")
        hidden_ids = candidates if response_ids is None else _validated_ids(
            response_ids, "response_ids"
        )
        if set(hidden_ids) != set(candidates):
            raise ValueError("response_ids must contain exactly the candidate IDs")
        hidden = np.asarray(responses)
        if hidden.ndim != 2 or hidden.shape[0] != len(hidden_ids):
            raise ValueError("responses must be a 2D matrix with one row per response ID")
        if not np.issubdtype(hidden.dtype, np.number):
            raise ValueError("responses must be numeric")
        if np.isinf(hidden).any():
            raise ValueError("responses cannot contain infinite values")
        positions = _id_positions(candidates, hidden_ids, "response_ids")
        ordered = np.array(hidden[positions], copy=True)
        ordered.setflags(write=False)

        self.__candidate_ids = candidates
        self.__position = {row_id: index for index, row_id in enumerate(candidates)}
        self.__responses = ordered
        self.__revealed: set[str] = set()
        self.__reveal_order: list[str] = []
        self.__lock = threading.Lock()

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Public IDs eligible for reveal; no response values are exposed."""

        return self.__candidate_ids

    @property
    def revealed_ids(self) -> tuple[str, ...]:
        with self.__lock:
            return tuple(self.__reveal_order)

    @property
    def response_dimension(self) -> int:
        """Public output width without exposing response values."""

        return int(self.__responses.shape[1])

    def reveal(self, ids: Sequence[str]) -> RevealedBatch:
        """Reveal an atomic new batch of candidate labels in requested ID order."""

        requested = _validated_ids(ids, "reveal ids")
        if not requested:
            raise ValueError("reveal ids cannot be empty")
        with self.__lock:
            invalid = [row_id for row_id in requested if row_id not in self.__position]
            if invalid:
                raise ValueError(f"reveal is limited to candidate IDs: {invalid[:5]}")
            repeated = [row_id for row_id in requested if row_id in self.__revealed]
            if repeated:
                raise ValueError(f"candidate IDs have already been revealed: {repeated[:5]}")
            positions = np.asarray(
                [self.__position[row_id] for row_id in requested], dtype=np.int64
            )
            labels = np.array(self.__responses[positions], copy=True)
            self.__revealed.update(requested)
            self.__reveal_order.extend(requested)
        return RevealedBatch(requested, labels)


@dataclass(frozen=True, init=False)
class PoolState:
    """Immutable public pool state containing no response-bearing object."""

    pool_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    descriptor_ids: tuple[str, ...]
    _descriptors: np.ndarray = field(repr=False)

    def __init__(
        self,
        pool_ids: Sequence[str],
        descriptors: np.ndarray,
        *,
        descriptor_ids: Sequence[str] | None = None,
        selected_ids: Sequence[str] = (),
    ) -> None:
        public_ids = _validated_ids(pool_ids, "pool_ids")
        descriptor_rows = public_ids if descriptor_ids is None else _validated_ids(
            descriptor_ids, "descriptor_ids"
        )
        if set(descriptor_rows) != set(public_ids):
            raise ValueError("descriptor_ids must contain exactly the pool IDs")
        selected = _validated_ids(selected_ids, "selected_ids")
        invalid = [row_id for row_id in selected if row_id not in set(public_ids)]
        if invalid:
            raise ValueError(f"selected_ids contains invalid pool IDs: {invalid[:5]}")
        matrix = _validated_public_matrix(descriptors, len(descriptor_rows))
        positions = _id_positions(public_ids, descriptor_rows, "descriptor_ids")
        ordered = _validated_public_matrix(matrix[positions], len(public_ids))

        object.__setattr__(self, "pool_ids", public_ids)
        object.__setattr__(self, "selected_ids", selected)
        object.__setattr__(self, "descriptor_ids", public_ids)
        object.__setattr__(self, "_descriptors", ordered)

    @property
    def descriptors(self) -> np.ndarray:
        view = self._descriptors.view()
        view.setflags(write=False)
        return view

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        selected = set(self.selected_ids)
        return tuple(row_id for row_id in self.pool_ids if row_id not in selected)

    @property
    def budget(self) -> int:
        return len(self.selected_ids)

    @property
    def public_hash(self) -> str:
        return _hash_json(
            {
                "pool_ids": self.pool_ids,
                "selected_ids": self.selected_ids,
                "descriptors_sha256": _hash_id_matrix(
                    self.descriptor_ids, self._descriptors
                ),
            }
        )

    def select(self, ids: Sequence[str]) -> "PoolState":
        """Return the conserved next state, rejecting invalid or reselected IDs."""

        requested = _validated_ids(ids, "selected batch IDs")
        if not requested:
            raise ValueError("selected batch IDs cannot be empty")
        pool = set(self.pool_ids)
        invalid = [row_id for row_id in requested if row_id not in pool]
        if invalid:
            raise ValueError(f"selected batch contains invalid pool IDs: {invalid[:5]}")
        selected = set(self.selected_ids)
        repeated = [row_id for row_id in requested if row_id in selected]
        if repeated:
            raise ValueError(f"selected batch reselects IDs: {repeated[:5]}")
        next_ids = (*self.selected_ids, *requested)
        if len(next_ids) != self.budget + len(requested):
            raise AssertionError("budget conservation failed")
        return PoolState(
            self.pool_ids,
            self._descriptors,
            descriptor_ids=self.descriptor_ids,
            selected_ids=next_ids,
        )

    def acquisition_context(
        self,
        *,
        uncertainty_ids: Sequence[str] | None = None,
        uncertainty: np.ndarray | None = None,
    ):
        """Build the public acquisition view without granting oracle access."""

        from .acquisition import AcquisitionContext

        return AcquisitionContext(
            self.candidate_ids,
            self.descriptor_ids,
            self._descriptors,
            labelled_ids=self.selected_ids,
            uncertainty_ids=uncertainty_ids,
            uncertainty=uncertainty,
        )


@dataclass(frozen=True)
class BudgetSchedule:
    """Fixed acquisition batches with explicitly sparse evaluation checkpoints."""

    initial_budget: int
    batch_size: int
    checkpoints: tuple[int, ...]
    pool_size: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.initial_budget, "initial_budget"),
            (self.batch_size, "batch_size"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        checkpoints = tuple(self.checkpoints)
        if not checkpoints or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in checkpoints
        ):
            raise ValueError("checkpoints must contain positive integers")
        if checkpoints[0] != self.initial_budget:
            raise ValueError("the first checkpoint must equal initial_budget")
        if any(right <= left for left, right in zip(checkpoints, checkpoints[1:])):
            raise ValueError("checkpoints must be strictly increasing")
        if any(
            (checkpoint - self.initial_budget) % self.batch_size
            for checkpoint in checkpoints
        ):
            raise ValueError("every checkpoint must be reachable by fixed-size batches")
        if self.pool_size is not None:
            if (
                not isinstance(self.pool_size, int)
                or isinstance(self.pool_size, bool)
                or self.pool_size <= 0
            ):
                raise ValueError("pool_size must be a positive integer")
            if checkpoints[-1] > self.pool_size:
                raise ValueError("the final checkpoint exceeds pool_size")
        object.__setattr__(self, "checkpoints", checkpoints)

    @property
    def final_budget(self) -> int:
        return self.checkpoints[-1]

    @property
    def acquisition_budgets(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.initial_budget + self.batch_size,
                self.final_budget + 1,
                self.batch_size,
            )
        )

    def is_checkpoint(self, budget: int) -> bool:
        return budget in self.checkpoints

    def next_checkpoint(self, budget: int) -> int | None:
        if budget < 0 or budget > self.final_budget:
            raise ValueError("budget is outside the schedule")
        return next((value for value in self.checkpoints if value > budget), None)

    def advance(self, state: PoolState, selected_ids: Sequence[str]) -> PoolState:
        """Apply exactly one conserved fixed batch to a compatible pool state."""

        if self.pool_size is not None and len(state.pool_ids) != self.pool_size:
            raise ValueError("PoolState size does not match the schedule")
        if self.final_budget > len(state.pool_ids):
            raise ValueError("the final budget exceeds the PoolState size")
        if state.budget < self.initial_budget:
            raise ValueError("the initial budget must be selected before schedule advance")
        if (state.budget - self.initial_budget) % self.batch_size:
            raise ValueError("PoolState budget is not aligned to the fixed-batch schedule")
        if state.budget >= self.final_budget:
            raise ValueError("the schedule is already complete")
        requested = _validated_ids(selected_ids, "selected batch IDs")
        if len(requested) != self.batch_size:
            raise ValueError("selection must conserve exactly one fixed batch")
        if state.budget + len(requested) > self.final_budget:
            raise ValueError("selection would exceed the final budget")
        next_state = state.select(requested)
        if next_state.budget != state.budget + self.batch_size:
            raise AssertionError("budget conservation failed")
        return next_state


@dataclass(frozen=True)
class RoundReceipt:
    """Public audit receipt; its schema has no field for response labels."""

    round_index: int
    strategy: str
    budget_before: int
    budget_after: int
    selected_ids: tuple[str, ...]
    hashes: Mapping[str, str] = field(default_factory=dict)
    global_seed: int | None = None
    acquisition_seed: int | None = None
    model_seed: int | None = None
    checkpoint: bool | None = None
    labelled_ids: tuple[str, ...] | None = None
    labelled_ids_sha256: str | None = None
    model_fit_summary: Mapping[str, object] = field(default_factory=dict)
    split_metrics: tuple[Mapping[str, object], ...] = ()
    train_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.round_index, int)
            or isinstance(self.round_index, bool)
            or self.round_index < 0
        ):
            raise ValueError("round_index must be a non-negative integer")
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("strategy must be a non-empty string")
        ids = _validated_ids(self.selected_ids, "selected_ids")
        for value, name in (
            (self.budget_before, "budget_before"),
            (self.budget_after, "budget_after"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.budget_after != self.budget_before + len(ids):
            raise ValueError("receipt budgets do not conserve selected IDs")
        global_seed = _validated_optional_seed(self.global_seed, "global_seed")
        acquisition_seed = _validated_optional_seed(
            self.acquisition_seed, "acquisition_seed"
        )
        model_seed = _validated_optional_seed(self.model_seed, "model_seed")
        if self.checkpoint is not None and not isinstance(self.checkpoint, bool):
            raise ValueError("checkpoint must be a boolean or null")

        labelled_ids = None
        labelled_ids_sha256 = self.labelled_ids_sha256
        if self.labelled_ids is not None:
            labelled_ids = _validated_ids(self.labelled_ids, "labelled_ids")
            if len(labelled_ids) != self.budget_after:
                raise ValueError("labelled_ids must contain the complete budget_after set")
            if ids and labelled_ids[-len(ids) :] != ids:
                raise ValueError("labelled_ids must end with the selected transition IDs")
            computed_labelled_hash = _hash_json(labelled_ids)
            if (
                labelled_ids_sha256 is not None
                and labelled_ids_sha256 != computed_labelled_hash
            ):
                raise ValueError("labelled_ids_sha256 does not match labelled_ids")
            labelled_ids_sha256 = computed_labelled_hash
        if labelled_ids_sha256 is not None and (
            not isinstance(labelled_ids_sha256, str) or not labelled_ids_sha256
        ):
            raise ValueError("labelled_ids_sha256 must be a non-empty string or null")

        fit_summary = _json_safe_public_metadata(
            self.model_fit_summary,
            nonfinite_to_none=False,
            path="model_fit_summary",
        )
        if not isinstance(fit_summary, dict):
            raise ValueError("model_fit_summary must be a mapping")
        summary_seed = fit_summary.get("seed")
        if model_seed is not None and summary_seed is not None and summary_seed != model_seed:
            raise ValueError("model_fit_summary seed does not match model_seed")

        metric_records: list[Mapping[str, object]] = []
        if not isinstance(self.split_metrics, (list, tuple)):
            raise ValueError("split_metrics must be a sequence of mappings")
        for index, record in enumerate(self.split_metrics):
            safe_record = _json_safe_public_metadata(
                record,
                nonfinite_to_none=True,
                path=f"split_metrics[{index}]",
            )
            if not isinstance(safe_record, dict):
                raise ValueError("split_metrics must contain mappings")
            metric_records.append(MappingProxyType(safe_record))

        train_seconds = self.train_seconds
        if train_seconds is not None:
            if (
                not isinstance(train_seconds, (int, float, np.integer, np.floating))
                or isinstance(train_seconds, bool)
                or not math.isfinite(float(train_seconds))
                or train_seconds < 0
            ):
                raise ValueError("train_seconds must be finite and non-negative or null")
            train_seconds = float(train_seconds)

        hash_values = dict(self.hashes)
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in hash_values.items()
        ):
            raise ValueError("hashes must map non-empty string names to string digests")
        object.__setattr__(self, "selected_ids", ids)
        object.__setattr__(self, "hashes", MappingProxyType(hash_values))
        object.__setattr__(self, "global_seed", global_seed)
        object.__setattr__(self, "acquisition_seed", acquisition_seed)
        object.__setattr__(self, "model_seed", model_seed)
        object.__setattr__(self, "labelled_ids", labelled_ids)
        object.__setattr__(self, "labelled_ids_sha256", labelled_ids_sha256)
        object.__setattr__(self, "model_fit_summary", MappingProxyType(fit_summary))
        object.__setattr__(self, "split_metrics", tuple(metric_records))
        object.__setattr__(self, "train_seconds", train_seconds)

    @classmethod
    def from_transition(
        cls,
        round_index: int,
        strategy: str,
        before: PoolState,
        after: PoolState,
        selected_ids: Sequence[str],
        *,
        hashes: Mapping[str, str] | None = None,
        global_seed: int | None = None,
        acquisition_seed: int | None = None,
        model_seed: int | None = None,
        checkpoint: bool | None = None,
        labelled_ids: Sequence[str] | None = None,
        model_fit_summary: Mapping[str, object] | None = None,
        split_metrics: Sequence[Mapping[str, object]] = (),
        train_seconds: float | None = None,
    ) -> "RoundReceipt":
        ids = _validated_ids(selected_ids, "selected_ids")
        if before.pool_ids != after.pool_ids:
            raise ValueError("receipt states refer to different pools")
        if after.selected_ids != (*before.selected_ids, *ids):
            raise ValueError("receipt selection does not match the PoolState transition")
        complete_labelled_ids = (
            after.selected_ids
            if labelled_ids is None
            else _validated_ids(labelled_ids, "labelled_ids")
        )
        if complete_labelled_ids != after.selected_ids:
            raise ValueError("labelled_ids must equal the transition after state")
        public_hashes = {
            "pool_before_sha256": before.public_hash,
            "pool_after_sha256": after.public_hash,
            "selected_ids_sha256": _hash_json(ids),
        }
        for key, value in dict(hashes or {}).items():
            if key in public_hashes and public_hashes[key] != value:
                raise ValueError(f"cannot override computed hash {key}")
            public_hashes[key] = value
        return cls(
            round_index=round_index,
            strategy=strategy,
            budget_before=before.budget,
            budget_after=after.budget,
            selected_ids=ids,
            hashes=public_hashes,
            global_seed=global_seed,
            acquisition_seed=acquisition_seed,
            model_seed=model_seed,
            checkpoint=checkpoint,
            labelled_ids=complete_labelled_ids,
            model_fit_summary=dict(model_fit_summary or {}),
            split_metrics=tuple(split_metrics),
            train_seconds=train_seconds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "strategy": self.strategy,
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "selected_ids": list(self.selected_ids),
            "global_seed": self.global_seed,
            "acquisition_seed": self.acquisition_seed,
            "model_seed": self.model_seed,
            "checkpoint": self.checkpoint,
            "labelled_ids": (
                None if self.labelled_ids is None else list(self.labelled_ids)
            ),
            "labelled_ids_sha256": self.labelled_ids_sha256,
            "model_fit_summary": dict(self.model_fit_summary),
            "split_metrics": [dict(record) for record in self.split_metrics],
            "train_seconds": self.train_seconds,
            "hashes": dict(sorted(self.hashes.items())),
        }

    def write_json_atomic(self, path: str | Path) -> None:
        write_receipt_atomic(path, self)


def write_receipt_atomic(path: str | Path, receipt: RoundReceipt) -> None:
    """Atomically replace ``path`` with one fully flushed JSON receipt."""

    if not isinstance(receipt, RoundReceipt):
        raise TypeError("receipt must be a RoundReceipt")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                receipt.to_dict(),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
