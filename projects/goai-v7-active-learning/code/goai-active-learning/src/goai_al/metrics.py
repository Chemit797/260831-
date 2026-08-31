"""Fixed v2 metrics for matched-control response learning curves."""

from __future__ import annotations

import numpy as np


def _pearson(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = mask.astype(bool)
    if int(observed.sum()) < 2:
        return float("nan")
    x = prediction[observed].astype(np.float64)
    y = truth[observed].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0.0 else float("nan")


def _r2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    observed = mask.astype(bool)
    if int(observed.sum()) < 2:
        return float("nan")
    x = prediction[observed].astype(np.float64)
    y = truth[observed].astype(np.float64)
    denominator = float(np.square(y - y.mean()).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.square(x - y).sum() / denominator)


def _validate_response_pair(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction)
    observed_truth = np.asarray(truth)
    if predicted.ndim != 2 or observed_truth.ndim != 2:
        raise ValueError("Prediction and truth must both be two-dimensional matrices")
    if predicted.shape != observed_truth.shape:
        raise ValueError("Prediction and truth shapes differ")
    try:
        truth_mask = np.isfinite(observed_truth)
        prediction_finite = np.isfinite(predicted)
    except TypeError as exc:
        raise ValueError("Prediction and truth must be numeric") from exc
    if np.any(truth_mask & ~prediction_finite):
        raise ValueError("Prediction is nonfinite at a truth-observed position")
    return predicted, observed_truth, truth_mask


def score_response(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | int]:
    """Score natural log2-deltas using the truth-finite mask exclusively."""
    predicted, observed_truth, mask = _validate_response_pair(prediction, truth)
    n_observed = int(mask.sum())
    if n_observed:
        error = predicted[mask].astype(np.float64) - observed_truth[mask].astype(np.float64)
        squared_error = float(np.square(error).sum())
        absolute_error = float(np.abs(error).sum())
        delta_rmse = float(np.sqrt(squared_error / n_observed))
        delta_mae = float(absolute_error / n_observed)
        zero_squared_error = float(
            np.square(observed_truth[mask].astype(np.float64)).sum()
        )
        delta_skill_zero = (
            float(1.0 - squared_error / zero_squared_error)
            if zero_squared_error > 0.0
            else float("nan")
        )
    else:
        delta_rmse = float("nan")
        delta_mae = float("nan")
        delta_skill_zero = float("nan")

    condition_pcc: list[float] = []
    for row in range(observed_truth.shape[0]):
        value = _pearson(predicted[row], observed_truth[row], mask[row])
        if np.isfinite(value):
            condition_pcc.append(value)

    protein_pcc: list[float] = []
    protein_r2: list[float] = []
    for column in range(observed_truth.shape[1]):
        pcc = _pearson(predicted[:, column], observed_truth[:, column], mask[:, column])
        if np.isfinite(pcc):
            protein_pcc.append(pcc)
        r2 = _r2(predicted[:, column], observed_truth[:, column], mask[:, column])
        if np.isfinite(r2):
            protein_r2.append(r2)

    return {
        "n_conditions": int(observed_truth.shape[0]),
        "n_proteins": int(observed_truth.shape[1]),
        "n_observed_values": n_observed,
        "n_evaluable_conditions_pcc": int(len(condition_pcc)),
        "n_evaluable_proteins_pcc": int(len(protein_pcc)),
        "n_evaluable_proteins_r2": int(len(protein_r2)),
        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,
        "delta_skill_zero": delta_skill_zero,
        "pooled_delta_pcc": _pearson(predicted, observed_truth, mask),
        "condition_pcc_median": (
            float(np.median(condition_pcc)) if condition_pcc else float("nan")
        ),
        "protein_pcc_median": (
            float(np.median(protein_pcc)) if protein_pcc else float("nan")
        ),
        "protein_r2_median": (
            float(np.median(protein_r2)) if protein_r2 else float("nan")
        ),
        "protein_r2_mean": (
            float(np.mean(protein_r2)) if protein_r2 else float("nan")
        ),
        "protein_r2_positive_fraction": (
            float(np.mean(np.asarray(protein_r2) > 0.0))
            if protein_r2
            else float("nan")
        ),
    }


def _validate_curve(
    budgets: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    budget_array = np.asarray(budgets, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if budget_array.ndim != 1 or value_array.ndim != 1:
        raise ValueError("Budgets and values must be one-dimensional")
    if len(budget_array) != len(value_array):
        raise ValueError("Budgets and values must have the same length")
    if len(budget_array) == 0:
        raise ValueError("A learning curve must contain at least one point")
    if not np.isfinite(budget_array).all() or not np.isfinite(value_array).all():
        raise ValueError("Budgets and values must be finite")
    if len(budget_array) > 1 and np.any(np.diff(budget_array) <= 0.0):
        raise ValueError("Budgets must be strictly increasing")
    return budget_array, value_array


def normalized_trapezoidal_aulc(
    budgets: np.ndarray,
    values: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> float:
    """Return trapezoidal mean performance over the actual budget spacing."""
    budget_array, value_array = _validate_curve(budgets, values)
    if len(budget_array) < 2:
        raise ValueError("Normalized AULC requires at least two strictly increasing budgets")
    directed_values = value_array if higher_is_better else -value_array
    span = float(budget_array[-1] - budget_array[0])
    widths = np.diff(budget_array)
    areas = widths * (directed_values[:-1] + directed_values[1:]) * 0.5
    return float(areas.sum() / span)


def budget_to_target(
    budgets: np.ndarray,
    values: np.ndarray,
    full_reference: float,
    target_fraction: float = 0.8,
    *,
    higher_is_better: bool = True,
) -> dict[str, str | bool | float | None]:
    """Interpolate the first budget attaining a fraction of achievable gain."""
    budget_array, value_array = _validate_curve(budgets, values)
    if not np.isfinite(full_reference):
        raise ValueError("full_reference must be finite")
    if not np.isfinite(target_fraction) or not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")

    direction = 1.0 if higher_is_better else -1.0
    directed = direction * value_array
    directed_initial = float(directed[0])
    directed_full = float(direction * full_reference)
    achievable = directed_full - directed_initial
    if achievable <= 0.0:
        raise ValueError("full_reference must improve on the initial value")

    gain = (directed - directed_initial) / achievable
    envelope = np.maximum.accumulate(gain)
    tolerance = np.finfo(np.float64).eps * 16.0 * max(1.0, abs(target_fraction))
    reached = np.flatnonzero(envelope >= target_fraction - tolerance)
    base = {
        "target_fraction": float(target_fraction),
        "initial_value": float(value_array[0]),
        "full_reference": float(full_reference),
        "max_gain": float(envelope.max()),
    }
    if len(reached) == 0:
        return {
            "status": "not_reached",
            "not_reached": True,
            "budget": None,
            **base,
        }

    upper = int(reached[0])
    if upper == 0:
        target_budget = float(budget_array[0])
    else:
        lower = upper - 1
        lower_gain = float(envelope[lower])
        upper_gain = float(envelope[upper])
        fraction = (target_fraction - lower_gain) / (upper_gain - lower_gain)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        target_budget = float(
            budget_array[lower]
            + fraction * (budget_array[upper] - budget_array[lower])
        )
    return {
        "status": "reached",
        "not_reached": False,
        "budget": target_budget,
        **base,
    }
