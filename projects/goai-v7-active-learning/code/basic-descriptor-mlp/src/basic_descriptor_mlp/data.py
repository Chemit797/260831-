"""Read and validate the GOAI metadata, proteome, and descriptor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_ID = "sample_ID"
SPLIT = "split_final"
STRAIN = "Strains"
CHEMICAL = "perturbation_no_concentration"
CONTROL_NAMES = frozenset({"water", "dmso"})
QUALITY_CONTROL_NAME = "quality control"
MATCH_FIELDS = (
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    STRAIN,
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)
REQUIRED_METADATA = (
    SAMPLE_ID,
    "data_source",
    STRAIN,
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
    "pert_id",
    CHEMICAL,
    "instrument",
    "Yeast_cell_plate",
    "protein_well",
    SPLIT,
    "strain_role",
    "chemical_role",
)
VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")


@dataclass
class Dataset:
    metadata: pd.DataFrame
    y_log2: pd.DataFrame
    mask: pd.DataFrame
    proteins: list[str]
    train_ids: pd.Index
    missing_rate: pd.Series
    chemical_embeddings: pd.DataFrame
    strain_embeddings: pd.DataFrame
    file_hashes: dict[str, str]


def sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = [column for column in REQUIRED_METADATA if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    if metadata[SAMPLE_ID].isna().any() or metadata[SAMPLE_ID].duplicated().any():
        raise ValueError("metadata sample_ID must be present and unique")
    if metadata[list(REQUIRED_METADATA)].isna().any().any():
        raise ValueError("metadata contains missing values in required columns")
    return metadata.set_index(SAMPLE_ID, verify_integrity=True)


def _read_proteome(path: str | Path, metadata_index: pd.Index) -> pd.DataFrame:
    proteome = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if SAMPLE_ID not in proteome.columns:
        raise ValueError("proteome is missing sample_ID")
    if proteome[SAMPLE_ID].isna().any() or proteome[SAMPLE_ID].duplicated().any():
        raise ValueError("proteome sample_ID must be present and unique")
    proteome = proteome.set_index(SAMPLE_ID, verify_integrity=True)
    if set(proteome.index) != set(metadata_index):
        raise ValueError("metadata and proteome sample_ID sets differ")
    proteome = proteome.reindex(metadata_index)
    try:
        return proteome.astype(np.float32)
    except ValueError as error:
        raise ValueError("proteome columns must be numeric or missing") from error


def _read_descriptor(path: str | Path, key: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if key not in frame.columns:
        raise ValueError(f"descriptor {path} is missing key {key}")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ValueError(f"descriptor {path} key must be present and unique")
    features = frame.drop(columns=[key])
    if features.empty:
        raise ValueError(f"descriptor {path} has no feature columns")
    values = features.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"descriptor {path} contains non-finite values")
    return pd.DataFrame(values, index=frame[key].astype(str), columns=features.columns.astype(str))


def load_dataset(config: dict[str, object]) -> Dataset:
    metadata = _read_metadata(str(config["metadata_train_val"]))
    raw = _read_proteome(str(config["proteome_train_val"]), metadata.index)
    train_ids = metadata.index[metadata[SPLIT].eq("train")]
    if train_ids.empty:
        raise ValueError("no split_final == train rows")

    values = raw.to_numpy(copy=False)
    observed = np.isfinite(values)
    if (observed & (values <= 0)).any():
        raise ValueError("observed raw protein intensities must be positive")
    missing_rate = raw.loc[train_ids].isna().mean(axis=0)
    threshold = float(config["missing_rate_threshold"])
    keep = missing_rate < threshold
    if not keep.any():
        raise ValueError("missingness filter removed all proteins")
    filtered = raw.loc[:, keep]
    y_log2 = np.log2(filtered)
    mask = y_log2.notna()

    chemical = _read_descriptor(str(config["chemical_embeddings"]), "chemical_name")
    strain = _read_descriptor(str(config["strain_embeddings"]), "strain_name")
    required_chemicals = set(metadata[CHEMICAL].astype(str))
    required_strains = set(metadata[STRAIN].astype(str))
    if not required_chemicals.issubset(chemical.index):
        raise ValueError("metadata chemical keys are not covered by descriptor table")
    if not required_strains.issubset(strain.index):
        raise ValueError("metadata strain keys are not covered by descriptor table")

    paths = {
        "metadata_train_val": str(config["metadata_train_val"]),
        "proteome_train_val": str(config["proteome_train_val"]),
        "chemical_embeddings": str(config["chemical_embeddings"]),
        "strain_embeddings": str(config["strain_embeddings"]),
    }
    return Dataset(
        metadata=metadata,
        y_log2=y_log2,
        mask=mask,
        proteins=y_log2.columns.astype(str).tolist(),
        train_ids=train_ids,
        missing_rate=missing_rate,
        chemical_embeddings=chemical,
        strain_embeddings=strain,
        file_hashes={name: sha256(path) for name, path in paths.items()},
    )


def is_control(metadata: pd.DataFrame) -> pd.Series:
    names = metadata[CHEMICAL].astype(str).str.strip().str.casefold()
    return names.isin(CONTROL_NAMES)


def is_quality_control(metadata: pd.DataFrame) -> pd.Series:
    names = metadata[CHEMICAL].astype(str).str.strip().str.casefold()
    return names.eq(QUALITY_CONTROL_NAME)


def is_treatment(metadata: pd.DataFrame) -> pd.Series:
    return ~(is_control(metadata) | is_quality_control(metadata))
