"""Fold-safe feature and target transforms for the fixed descriptor model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import CHEMICAL, STRAIN, Dataset


@dataclass
class FeatureState:
    categories: dict[str, list[str]]
    chemical_embeddings: pd.DataFrame
    strain_embeddings: pd.DataFrame
    normalize_descriptor_blocks: bool
    target_mean: np.ndarray
    target_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        data: Dataset,
        target_std_floor: float,
        normalize_descriptor_blocks: bool = True,
    ) -> "FeatureState":
        train_metadata = data.metadata.loc[data.train_ids]
        categories = {
            field: sorted(train_metadata[field].astype(str).unique().tolist())
            for field in ("Medium", "pert_time", "Temperature")
        }
        targets = data.y_log2.loc[data.train_ids].to_numpy(dtype=np.float32)
        observed = np.isfinite(targets)
        with np.errstate(invalid="ignore", divide="ignore"):
            target_mean = np.nanmean(targets, axis=0).astype(np.float32)
            target_scale = np.nanstd(targets, axis=0).astype(np.float32)
        target_mean = np.nan_to_num(target_mean, nan=0.0)
        target_scale = np.nan_to_num(target_scale, nan=target_std_floor)
        target_scale = np.maximum(target_scale, float(target_std_floor))
        if not observed.any(axis=0).all():
            raise ValueError("a retained protein has no finite training target")
        return cls(
            categories=categories,
            chemical_embeddings=data.chemical_embeddings,
            strain_embeddings=data.strain_embeddings,
            normalize_descriptor_blocks=normalize_descriptor_blocks,
            target_mean=target_mean,
            target_scale=target_scale,
        )

    @staticmethod
    def _one_hot(values: pd.Series, categories: list[str]) -> np.ndarray:
        result = np.zeros((len(values), len(categories)), dtype=np.float32)
        positions = values.astype(str).map({value: i for i, value in enumerate(categories)})
        valid = positions.notna().to_numpy()
        if valid.any():
            rows = np.flatnonzero(valid)
            result[rows, positions.iloc[rows].astype(int).to_numpy()] = 1.0
        return result

    @staticmethod
    def _normalize_rows(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)

    def transform_metadata(
        self,
        metadata: pd.DataFrame,
        descriptor_mode: str = "real",
        shuffle_seed: int | None = None,
    ) -> np.ndarray:
        if descriptor_mode not in {"real", "zero", "shuffle"}:
            raise ValueError(f"unknown descriptor_mode: {descriptor_mode}")
        chemical = self.chemical_embeddings.reindex(metadata[CHEMICAL].astype(str)).to_numpy(dtype=np.float32)
        strain = self.strain_embeddings.reindex(metadata[STRAIN].astype(str)).to_numpy(dtype=np.float32)
        if not np.isfinite(chemical).all() or not np.isfinite(strain).all():
            raise ValueError("descriptor lookup produced missing or non-finite values")
        if descriptor_mode == "zero":
            chemical = np.zeros_like(chemical)
            strain = np.zeros_like(strain)
        elif descriptor_mode == "shuffle":
            if shuffle_seed is None:
                raise ValueError("shuffle descriptor mode needs a seed")
            rng = np.random.default_rng(shuffle_seed)
            chemical_table = self.chemical_embeddings.to_numpy(dtype=np.float32)
            strain_table = self.strain_embeddings.to_numpy(dtype=np.float32)
            chemical_table = chemical_table[rng.permutation(len(chemical_table))]
            strain_table = strain_table[rng.permutation(len(strain_table))]
            chemical_positions = self.chemical_embeddings.index.get_indexer(metadata[CHEMICAL].astype(str))
            strain_positions = self.strain_embeddings.index.get_indexer(metadata[STRAIN].astype(str))
            chemical = chemical_table[chemical_positions]
            strain = strain_table[strain_positions]
        if self.normalize_descriptor_blocks:
            chemical = self._normalize_rows(chemical)
            strain = self._normalize_rows(strain)
        one_hot = [
            self._one_hot(metadata[field], self.categories[field])
            for field in ("Medium", "pert_time", "Temperature")
        ]
        blocks = [strain, chemical, *one_hot]
        return np.concatenate(blocks, axis=1, dtype=np.float32)

    def transform_targets(self, y_log2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = y_log2.to_numpy(dtype=np.float32)
        mask = np.isfinite(values)
        normalized = (values - self.target_mean[None, :]) / self.target_scale[None, :]
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return normalized, mask.astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale[None, :] + self.target_mean[None, :]

    def contract(self) -> dict[str, object]:
        return {
            "input_block_order": ["strain_embedding", "chemical_embedding", "Medium_onehot", "pert_time_onehot", "Temperature_onehot"],
            "input_block_dimensions": {"strain_embedding": 4096, "chemical_embedding": 512, "Medium_onehot": 2, "pert_time_onehot": 6, "Temperature_onehot": 2},
            "input_dim": 4618,
            "categories": self.categories,
            "descriptor_normalization": "per-row L2 within strain and chemical blocks" if self.normalize_descriptor_blocks else "none",
            "target_transform": "log2 then per-protein train mean/std",
            "target_std_floor": 0.10,
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
        }
