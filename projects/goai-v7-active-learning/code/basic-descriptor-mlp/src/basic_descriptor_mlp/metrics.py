"""Mask-aware absolute and matched-control diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CHEMICAL, MATCH_FIELDS, VALIDATION_SPLITS, Dataset, is_control, is_treatment


def _finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _finite_quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _protein_r2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scores = np.full(truth.shape[1], np.nan, dtype=np.float64)
    for column in range(truth.shape[1]):
        valid = mask[:, column]
        if valid.sum() < 2:
            continue
        target = truth[valid, column]
        total = np.sum((target - target.mean()) ** 2)
        if total > 0:
            scores[column] = 1.0 - np.sum((prediction[valid, column] - target) ** 2) / total
    return scores


def _axis_pcc(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        prediction, truth, mask = prediction.T, truth.T, mask.T
    values = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    for i in range(prediction.shape[1]):
        valid = mask[:, i]
        if valid.sum() < 2:
            continue
        x, y = prediction[valid, i], truth[valid, i]
        if np.std(x) > 0 and np.std(y) > 0:
            values[i] = np.corrcoef(x, y)[0, 1]
    return values


def absolute_metrics(prediction: pd.DataFrame, truth: pd.DataFrame) -> tuple[dict[str, float | int], pd.DataFrame]:
    if not prediction.index.equals(truth.index) or not prediction.columns.equals(truth.columns):
        raise ValueError("prediction and truth indexes/columns must match")
    pred = prediction.to_numpy(dtype=np.float64)
    actual = truth.to_numpy(dtype=np.float64)
    mask = np.isfinite(actual)
    if not np.isfinite(pred).all():
        raise ValueError("prediction contains non-finite values")
    errors = pred[mask] - actual[mask]
    target = actual[mask]
    total = np.sum((target - target.mean()) ** 2)
    protein_r2 = _protein_r2(pred, actual, mask)
    protein_pcc = _axis_pcc(pred, actual, mask, axis=0)
    sample_pcc = _axis_pcc(pred, actual, mask, axis=1)
    evaluable_r2 = np.isfinite(protein_r2)
    report = {
        "n_samples": int(len(truth)),
        "n_observed_values": int(mask.sum()),
        "coverage": float(mask.mean()),
        "rmse_log2": float(np.sqrt(np.mean(errors * errors))),
        "mae_log2": float(np.mean(np.abs(errors))),
        "global_r2": float(1.0 - np.sum(errors * errors) / total) if total > 0 else float("nan"),
        "protein_r2_median": _finite_median(protein_r2),
        "protein_r2_mean": float(np.nanmean(protein_r2)),
        "protein_r2_positive_fraction": float(np.mean(protein_r2[evaluable_r2] > 0)),
        "protein_r2_q10": _finite_quantile(protein_r2, 0.10),
        "protein_r2_q25": _finite_quantile(protein_r2, 0.25),
        "protein_r2_q75": _finite_quantile(protein_r2, 0.75),
        "protein_r2_q90": _finite_quantile(protein_r2, 0.90),
        "sample_pcc_median": _finite_median(sample_pcc),
        "protein_pcc_median": _finite_median(protein_pcc),
        "n_evaluable_proteins": int(np.isfinite(protein_r2).sum()),
    }
    per_protein = pd.DataFrame({"protein_r2": protein_r2, "protein_pcc": protein_pcc}, index=truth.columns)
    return report, per_protein


def _matched_controls(metadata: pd.DataFrame, control_truth: pd.DataFrame, target_ids: pd.Index) -> pd.DataFrame:
    controls = metadata.loc[is_control(metadata)]
    if controls.empty:
        return pd.DataFrame(index=target_ids, columns=control_truth.columns, dtype=np.float32)
    controls = controls.loc[controls.index.intersection(control_truth.index)]
    if controls.empty:
        return pd.DataFrame(index=target_ids, columns=control_truth.columns, dtype=np.float32)
    keys = pd.MultiIndex.from_frame(controls.loc[:, MATCH_FIELDS].astype(str))
    means = control_truth.loc[controls.index].groupby(keys, sort=False).mean()
    target_keys = pd.MultiIndex.from_frame(metadata.loc[target_ids, MATCH_FIELDS].astype(str))
    result = means.reindex(target_keys)
    result.index = target_ids
    return result


def matched_control_metrics(
    data: Dataset,
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    split_ids: pd.Index,
) -> dict[str, float | int]:
    treatment_ids = split_ids[is_treatment(data.metadata.loc[split_ids]).to_numpy()]
    controls = _matched_controls(data.metadata, data.y_log2, treatment_ids)
    if controls.empty:
        return {"fc_n_samples": 0, "fc_n_observed_values": 0, "fc_coverage": float("nan"), "fc_rmse_log2": float("nan"), "fc_pcc": float("nan")}
    pred_delta = prediction.loc[treatment_ids] - controls
    truth_delta = truth.loc[treatment_ids] - controls
    pred_values = pred_delta.to_numpy(dtype=np.float64)
    truth_values = truth_delta.to_numpy(dtype=np.float64)
    mask = np.isfinite(pred_values) & np.isfinite(truth_values)
    if not mask.any():
        return {"fc_n_samples": 0, "fc_n_observed_values": 0, "fc_coverage": float("nan"), "fc_rmse_log2": float("nan"), "fc_pcc": float("nan")}
    error = pred_values[mask] - truth_values[mask]
    flat_pred, flat_truth = pred_values[mask], truth_values[mask]
    pcc = float(np.corrcoef(flat_pred, flat_truth)[0, 1]) if np.std(flat_pred) > 0 and np.std(flat_truth) > 0 else float("nan")
    return {
        "fc_n_samples": int(mask.any(axis=1).sum()),
        "fc_n_observed_values": int(mask.sum()),
        "fc_coverage": float(mask.mean()),
        "fc_rmse_log2": float(np.sqrt(np.mean(error * error))),
        "fc_pcc": pcc,
    }


def evaluate_splits(
    data: Dataset,
    predictions: dict[str, pd.DataFrame],
    include_treatment_metrics: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    per_protein: list[pd.DataFrame] = []
    for split in VALIDATION_SPLITS:
        ids = data.metadata.index[data.metadata["split_final"].eq(split)]
        if len(ids) == 0:
            continue
        prediction = predictions[split]
        truth = data.y_log2.loc[ids]
        absolute, protein = absolute_metrics(prediction, truth)
        fc = matched_control_metrics(data, prediction, truth, ids)
        rows.append({"split": split, "subset": "all_rows", **absolute, **fc})
        per_protein.append(protein.rename(columns={"protein_r2": f"{split}__all_rows__protein_r2", "protein_pcc": f"{split}__all_rows__protein_pcc"}))
        if include_treatment_metrics:
            treatment_ids = ids[is_treatment(data.metadata.loc[ids]).to_numpy()]
            if len(treatment_ids):
                treatment_absolute, treatment_protein = absolute_metrics(prediction.loc[treatment_ids], truth.loc[treatment_ids])
                treatment_fc = matched_control_metrics(data, prediction, truth, treatment_ids)
                rows.append({"split": split, "subset": "treatment_only", **treatment_absolute, **treatment_fc})
                per_protein.append(treatment_protein.rename(columns={"protein_r2": f"{split}__treatment_only__protein_r2", "protein_pcc": f"{split}__treatment_only__protein_pcc"}))
    return pd.DataFrame(rows), pd.concat(per_protein, axis=1)
