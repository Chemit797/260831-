"""Frozen target-free semantic descriptors for GOAI biological conditions.

The encoders in this module use only pinned local entity assets and condition
metadata.  They never accept or inspect response values, and they never invent
geometry from hashes, encoded names, opaque IDs, or fallback identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from .data import (
    CHEMICAL,
    STRAIN,
    GroupedDataset,
    PoolFeatureEncoder,
    mask_unsupported as mask_base_unsupported,
)


DEFAULT_CHEMICAL_EMBEDDINGS_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/chemical_embeddings/"
    "chemberta_77m_mlm/chemberta_real.tsv"
)
DEFAULT_CHEMICAL_MANIFEST_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/chemical_embeddings/"
    "chemberta_77m_mlm/manifest.json"
)
DEFAULT_STRAIN_SEMANTICS_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/entities/"
    "strain_semantics_numeric.tsv"
)
DEFAULT_STRAIN_MANIFEST_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/entities/"
    "strain_semantics_manifest.json"
)
DEFAULT_CHEMICAL_RISK_MANIFEST_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/chemical_views/manifest.json"
)

CHEMICAL_EMBEDDINGS_SHA256 = (
    "50ba9081a7773f1438a3dfbf49054e319ce6002d44ed95d0ffd623d5f617b0a6"
)
CHEMICAL_MANIFEST_SHA256 = (
    "cb64148b66d58eff94881bfc241e85725d6a6bfe5dcb41713484b5d48e608f96"
)
STRAIN_SEMANTICS_SHA256 = (
    "8ee0f31d33c8dccefa90eed33cb7f4eb949470ef6dfe0681beefecf52b7d72ce"
)
STRAIN_MANIFEST_SHA256 = (
    "ba54ed7ccddd6d34663978b698ac7da39f13bd41e84937be8edbd31fbf795da2"
)
CHEMICAL_RISK_MANIFEST_SHA256 = (
    "13060a305daddd3d35f7bc70244c4c6fec793d6b13cf5645eaa0ab62f079475c"
)
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
CHEMBERTA_MODEL_REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"
CHEMICAL_SHUFFLED_SHA256 = (
    "6384ce4b5d6115f69c7330aaa7dfb33de89decdd8dc478e58e16bba0b4d06194"
)
STRAIN_SHUFFLED_SHA256 = (
    "7f1a8d2a28f70d033a42416d59da9ac2e27022aed79b40926745f51cc0d096f7"
)
STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256 = (
    "9a3b4edf61a313d763b91ee762c4ebed2ea359c9b4fc810eaa135c30207c1add"
)
STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256 = (
    "a2c94f7f626fd4a1b0167de6ff063b5a845649930d4a9ce21370c6810db7b1cd"
)
STRAIN_IDENTITY_WARNING = (
    "Five mappings are high-confidence public candidates, not organizer-verified "
    "identities; DHY210 is missing, never S288C"
)
STRAIN_IDENTITY_CANDIDATES = ("BAH", "BAI", "CEK", "CGD", "CRD")
STRAIN_UNRESOLVED = ("DHY210",)
DEFAULT_CLIP_VALUE = 5.0

_CHEMICAL_KEY_COLUMN = "raw_name"
_STRAIN_KEY_COLUMN = "strain_code"
_STRAIN_SOURCE_FLAGS = ("resolved", "missing", "proxy")
_CHEMICAL_FLAGS = (
    "available",
    "missing",
    "resolved",
    "proxy",
    "fallback",
    "identity_risk",
)
_STRAIN_FLAGS = (
    "available",
    "missing",
    "resolved",
    "proxy",
    "fallback",
    "identity_candidate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise ValueError(f"Frozen {role} asset does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Frozen {role} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _load_json(path: Path, role: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Frozen {role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Frozen {role} must contain a JSON object")
    return value


def _json_copy(value: object) -> object:
    """Return a detached value that strict JSON can serialize."""

    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Semantic summary must be JSON-safe") from error


def _normalised_entity_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.strip().split()).casefold()
    return text or None


def _normalised_keys(values: pd.Series) -> list[str | None]:
    return [_normalised_entity_key(value) for value in values]


def _numeric_matrix(frame: pd.DataFrame, columns: list[str], role: str) -> np.ndarray:
    try:
        values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Frozen {role} columns must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"Frozen {role} contains nonfinite values")
    return values


def _validate_unique_keys(values: pd.Series, role: str) -> list[str]:
    keys = _normalised_keys(values)
    if any(key is None for key in keys):
        raise ValueError(f"Frozen {role} contains a blank entity key")
    concrete = [str(key) for key in keys]
    if len(set(concrete)) != len(concrete):
        raise ValueError(f"Frozen {role} contains duplicate normalized entity keys")
    return concrete


class TargetFreeSemanticEncoder:
    """Pinned chemical and strain semantics fitted without response values."""

    chemical_field = CHEMICAL
    strain_field = STRAIN

    def __init__(
        self,
        *,
        chemical_embeddings_path: str | Path = DEFAULT_CHEMICAL_EMBEDDINGS_PATH,
        chemical_manifest_path: str | Path = DEFAULT_CHEMICAL_MANIFEST_PATH,
        strain_semantics_path: str | Path = DEFAULT_STRAIN_SEMANTICS_PATH,
        strain_manifest_path: str | Path = DEFAULT_STRAIN_MANIFEST_PATH,
        chemical_risk_manifest_path: str | Path = DEFAULT_CHEMICAL_RISK_MANIFEST_PATH,
        clip_value: float = DEFAULT_CLIP_VALUE,
    ) -> None:
        if not np.isfinite(clip_value) or float(clip_value) <= 0.0:
            raise ValueError("clip_value must be finite and positive")
        self.chemical_embeddings_path = Path(chemical_embeddings_path)
        self.chemical_manifest_path = Path(chemical_manifest_path)
        self.strain_semantics_path = Path(strain_semantics_path)
        self.strain_manifest_path = Path(strain_manifest_path)
        self.chemical_risk_manifest_path = Path(chemical_risk_manifest_path)
        self.clip_value = float(clip_value)

        self.chemical_mean_: np.ndarray | None = None
        self.chemical_scale_: np.ndarray | None = None
        self.strain_mean_: np.ndarray | None = None
        self.strain_scale_: np.ndarray | None = None
        self._coverage_summary: dict[str, object] = {"fitted": False}

        self._load_and_validate_assets()

    def _load_and_validate_assets(self) -> None:
        asset_hashes = {
            "chemical_embeddings_sha256": _require_hash(
                self.chemical_embeddings_path,
                CHEMICAL_EMBEDDINGS_SHA256,
                "chemical real embeddings",
            ),
            "chemical_manifest_sha256": _require_hash(
                self.chemical_manifest_path,
                CHEMICAL_MANIFEST_SHA256,
                "chemical embedding manifest",
            ),
            "strain_semantics_sha256": _require_hash(
                self.strain_semantics_path,
                STRAIN_SEMANTICS_SHA256,
                "strain real semantics",
            ),
            "strain_manifest_sha256": _require_hash(
                self.strain_manifest_path,
                STRAIN_MANIFEST_SHA256,
                "strain semantic manifest",
            ),
            "chemical_risk_manifest_sha256": _require_hash(
                self.chemical_risk_manifest_path,
                CHEMICAL_RISK_MANIFEST_SHA256,
                "chemical identity-risk manifest",
            ),
        }
        chemical_manifest = _load_json(
            self.chemical_manifest_path, "chemical embedding manifest"
        )
        strain_manifest = _load_json(
            self.strain_manifest_path, "strain semantic manifest"
        )
        risk_manifest = _load_json(
            self.chemical_risk_manifest_path, "chemical identity-risk manifest"
        )

        expected_chemical_manifest = {
            "model": CHEMBERTA_MODEL,
            "model_revision": CHEMBERTA_MODEL_REVISION,
            "frozen": True,
            "pooling": "attention-mask mean pooling",
            "max_length": 256,
            "rows": 57,
            "resolved_rows": 56,
            "embedding_dim": 384,
            "real_sha256": CHEMICAL_EMBEDDINGS_SHA256,
            "shuffled_sha256": CHEMICAL_SHUFFLED_SHA256,
        }
        for key, expected in expected_chemical_manifest.items():
            if chemical_manifest.get(key) != expected:
                raise ValueError(
                    f"Chemical manifest field {key!r} does not match the frozen contract"
                )
        permutation = chemical_manifest.get("shuffle_permutation")
        if not isinstance(permutation, list) or sorted(permutation) != list(range(57)):
            raise ValueError("Chemical manifest shuffle_permutation is invalid")
        if chemical_manifest["real_sha256"] == chemical_manifest["shuffled_sha256"]:
            raise ValueError("Chemical real and shuffled artifacts must be distinct")

        chemical_columns = [
            _CHEMICAL_KEY_COLUMN,
            *[f"chemberta_{index:04d}" for index in range(384)],
        ]
        chemical = pd.read_csv(self.chemical_embeddings_path, sep="\t")
        if chemical.shape != (57, 385) or list(chemical.columns) != chemical_columns:
            raise ValueError("Chemical embedding TSV does not match the frozen exact schema")
        chemical_keys = _validate_unique_keys(
            chemical[_CHEMICAL_KEY_COLUMN], "chemical embedding TSV"
        )
        chemical_values = _numeric_matrix(
            chemical, chemical_columns[1:], "chemical embedding TSV"
        )
        zero_rows = np.all(chemical_values == 0.0, axis=1)
        zero_names = chemical.loc[zero_rows, _CHEMICAL_KEY_COLUMN].astype(str).tolist()
        if zero_names != ["Quality Control"] or int((~zero_rows).sum()) != 56:
            raise ValueError("Chemical real artifact has an invalid resolved-row pattern")

        if risk_manifest.get("schema_version") != "goai.chemical-structure-views.v3":
            raise ValueError("Chemical risk manifest schema version is invalid")
        identity_policy = str(risk_manifest.get("identity_policy", ""))
        if "exact is default" not in identity_policy:
            raise ValueError("Chemical risk manifest does not declare exact identity default")
        risky_names = risk_manifest.get("risky_raw_names")
        zero_risky_names = risk_manifest.get("zero_risky_raw_names")
        if (
            not isinstance(risky_names, list)
            or len(risky_names) != 7
            or len(set(risky_names)) != 7
            or set(risky_names) != set(zero_risky_names or ())
        ):
            raise ValueError("Chemical risk manifest risky-name lists are invalid")
        chemical_raw_names = set(chemical[_CHEMICAL_KEY_COLUMN].astype(str))
        if not set(risky_names) <= chemical_raw_names:
            raise ValueError("Chemical risk manifest names are absent from the exact asset")
        manifest_source_hash = chemical_manifest.get("source_sha256")
        risk_exact_hash = (
            risk_manifest.get("inputs", {}).get("exact_map", {}).get("sha256")
            if isinstance(risk_manifest.get("inputs"), dict)
            else None
        )
        if manifest_source_hash != risk_exact_hash:
            raise ValueError("Chemical manifests disagree on the exact-map source hash")

        if strain_manifest.get("protocol") != "goai_peter2018_strain_semantics_v1":
            raise ValueError("Strain manifest protocol is invalid")
        if strain_manifest.get("dimensions") != 32:
            raise ValueError("Strain manifest SNP semantic dimension is invalid")
        if strain_manifest.get("real_sha256") != STRAIN_SEMANTICS_SHA256:
            raise ValueError("Strain manifest real hash is invalid")
        if strain_manifest.get("shuffled_sha256") != STRAIN_SHUFFLED_SHA256:
            raise ValueError("Strain manifest shuffled hash is invalid")
        if strain_manifest["real_sha256"] == strain_manifest["shuffled_sha256"]:
            raise ValueError("Strain real and shuffled artifacts must be distinct")
        expected_identity_fields = {
            "candidate_strains": list(STRAIN_IDENTITY_CANDIDATES),
            "unresolved_strains": list(STRAIN_UNRESOLVED),
            "identity_warning": STRAIN_IDENTITY_WARNING,
            "identity_evidence_registry_sha256": (
                STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256
            ),
            "identity_evidence_manifest_sha256": (
                STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256
            ),
        }
        for key, expected in expected_identity_fields.items():
            if strain_manifest.get(key) != expected:
                raise ValueError(
                    f"Strain manifest identity field {key!r} differs from the "
                    "candidate-only contract"
                )
        identity_registry_value = strain_manifest.get("identity_evidence_registry")
        identity_manifest_value = strain_manifest.get("identity_evidence_manifest")
        if (
            not isinstance(identity_registry_value, str)
            or not identity_registry_value
            or not isinstance(identity_manifest_value, str)
            or not identity_manifest_value
        ):
            raise ValueError("Strain manifest identity evidence paths are invalid")
        identity_registry_path = Path(identity_registry_value)
        identity_manifest_path = Path(identity_manifest_value)
        if not identity_registry_path.is_absolute():
            identity_registry_path = self.strain_manifest_path.parent / identity_registry_path
        if not identity_manifest_path.is_absolute():
            identity_manifest_path = self.strain_manifest_path.parent / identity_manifest_path
        asset_hashes["strain_identity_evidence_registry_sha256"] = _require_hash(
            identity_registry_path,
            STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256,
            "strain identity evidence registry",
        )
        asset_hashes["strain_identity_evidence_manifest_sha256"] = _require_hash(
            identity_manifest_path,
            STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256,
            "strain identity evidence manifest",
        )
        strain_feature_columns = strain_manifest.get("feature_columns")
        if not isinstance(strain_feature_columns, list):
            raise ValueError("Strain manifest feature_columns must be a list")
        strain = pd.read_csv(self.strain_semantics_path, sep="\t")
        expected_strain_columns = [_STRAIN_KEY_COLUMN, *strain_feature_columns]
        if strain.shape != (6, 57) or list(strain.columns) != expected_strain_columns:
            raise ValueError("Strain semantic TSV does not match the frozen schema")
        strain_keys = _validate_unique_keys(
            strain[_STRAIN_KEY_COLUMN], "strain semantic TSV"
        )
        if strain_manifest.get("target_strains") != strain[_STRAIN_KEY_COLUMN].tolist():
            raise ValueError("Strain manifest target order differs from the real TSV")
        strain_semantic_columns = [
            name for name in strain_feature_columns if name not in _STRAIN_SOURCE_FLAGS
        ]
        if len(strain_semantic_columns) != 53:
            raise ValueError("Strain real artifact must contain 53 semantic columns")
        strain_values = _numeric_matrix(
            strain, strain_semantic_columns, "strain semantic TSV"
        )
        strain_flags = _numeric_matrix(
            strain, list(_STRAIN_SOURCE_FLAGS), "strain status flags"
        )
        if not np.isin(strain_flags, (0.0, 1.0)).all():
            raise ValueError("Strain source flags must be binary")
        if not np.equal(strain_flags[:, 0] + strain_flags[:, 1], 1.0).all():
            raise ValueError("Strain resolved and missing flags must be complementary")
        unresolved = strain_flags[:, 0] == 0.0
        if (
            strain.loc[unresolved, _STRAIN_KEY_COLUMN].astype(str).tolist() != ["DHY210"]
            or not np.all(strain_values[unresolved] == 0.0)
            or np.any(np.all(strain_values[~unresolved] == 0.0, axis=1))
        ):
            raise ValueError("Strain real artifact has an invalid resolved-row pattern")
        candidate_keys = {
            str(_normalised_entity_key(name)) for name in STRAIN_IDENTITY_CANDIDATES
        }
        if candidate_keys != {
            strain_keys[index]
            for index, is_unresolved in enumerate(unresolved)
            if not is_unresolved
        }:
            raise ValueError(
                "Strain candidate identities differ from the nonmissing semantic rows"
            )

        self._chemical_columns = tuple(chemical_columns[1:])
        self._strain_columns = tuple(strain_semantic_columns)
        self._chemical_values = {
            key: chemical_values[index].copy() for index, key in enumerate(chemical_keys)
        }
        self._chemical_resolved = {
            key: bool(not zero_rows[index]) for index, key in enumerate(chemical_keys)
        }
        self._chemical_display = {
            key: str(chemical.iloc[index][_CHEMICAL_KEY_COLUMN])
            for index, key in enumerate(chemical_keys)
        }
        self._chemical_risk_keys = {
            str(_normalised_entity_key(name)) for name in risky_names
        }
        self._strain_values = {
            key: strain_values[index].copy() for index, key in enumerate(strain_keys)
        }
        self._strain_status = {
            key: {
                name: bool(strain.iloc[index][name]) for name in _STRAIN_SOURCE_FLAGS
            }
            for index, key in enumerate(strain_keys)
        }
        self._strain_candidate_keys = candidate_keys
        self._strain_display = {
            key: str(strain.iloc[index][_STRAIN_KEY_COLUMN])
            for index, key in enumerate(strain_keys)
        }
        self._asset_hashes = asset_hashes
        self._asset_summary = {
            "frozen_local_only": True,
            "response_used": False,
            "chemical": {
                "path": str(self.chemical_embeddings_path),
                "manifest_path": str(self.chemical_manifest_path),
                "rows": 57,
                "resolved_rows": 56,
                "continuous_width": 384,
                "model": CHEMBERTA_MODEL,
                "model_revision": CHEMBERTA_MODEL_REVISION,
                "view": "real_exact",
                "shuffled": False,
            },
            "strain": {
                "path": str(self.strain_semantics_path),
                "manifest_path": str(self.strain_manifest_path),
                "rows": 6,
                "resolved_rows": 5,
                "resolved_meaning": (
                    "public_candidate_semantics_available_not_organizer_verified"
                ),
                "continuous_width": 53,
                "protocol": "goai_peter2018_strain_semantics_v1",
                "shuffled": False,
                "candidate_strains": list(STRAIN_IDENTITY_CANDIDATES),
                "identity_candidate_count": len(STRAIN_IDENTITY_CANDIDATES),
                "unresolved_strains": list(STRAIN_UNRESOLVED),
                "organizer_verified": False,
                "identity_warning": STRAIN_IDENTITY_WARNING,
                "identity_evidence_registry": identity_registry_value,
                "identity_evidence_registry_sha256": (
                    STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256
                ),
                "identity_evidence_manifest": identity_manifest_value,
                "identity_evidence_manifest_sha256": (
                    STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256
                ),
            },
            "chemical_identity_risk": {
                "manifest_path": str(self.chemical_risk_manifest_path),
                "risky_entity_count": 7,
                "semantics_zeroed_for_risk": False,
            },
        }

        feature_names = [
            *[f"chemical_semantic__{name}" for name in self._chemical_columns],
            *[f"chemical_flag__{name}" for name in _CHEMICAL_FLAGS],
            *[f"strain_semantic__{name}" for name in self._strain_columns],
            *[f"strain_flag__{name}" for name in _STRAIN_FLAGS],
        ]
        chemical_stop = len(self._chemical_columns)
        chemical_flags_stop = chemical_stop + len(_CHEMICAL_FLAGS)
        strain_stop = chemical_flags_stop + len(self._strain_columns)
        self._feature_names = tuple(feature_names)
        self._block_slices = {
            "chemical_continuous": slice(0, chemical_stop),
            "chemical_flags": slice(chemical_stop, chemical_flags_stop),
            "strain_continuous": slice(chemical_flags_stop, strain_stop),
            "strain_flags": slice(strain_stop, len(feature_names)),
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def feature_names_(self) -> tuple[str, ...]:
        """Scikit-style alias for callers that use fitted attribute names."""

        return self.feature_names

    @property
    def block_slices(self) -> dict[str, slice]:
        return dict(self._block_slices)

    @property
    def output_dim(self) -> int:
        return len(self._feature_names)

    @property
    def asset_hashes(self) -> dict[str, str]:
        return dict(self._asset_hashes)

    @property
    def asset_summary(self) -> dict[str, object]:
        return _json_copy(self._asset_summary)  # type: ignore[return-value]

    @property
    def coverage_summary(self) -> dict[str, object]:
        return _json_copy(self._coverage_summary)  # type: ignore[return-value]

    def _require_metadata_fields(self, metadata: pd.DataFrame) -> None:
        missing = sorted({CHEMICAL, STRAIN} - set(metadata.columns))
        if missing:
            raise ValueError(f"Semantic metadata is missing fields: {missing}")

    def _chemical_status(self, key: str | None) -> tuple[float, ...]:
        available = key is not None and key in self._chemical_values
        resolved = bool(available and self._chemical_resolved[str(key)])
        identity_risk = key is not None and key in self._chemical_risk_keys
        return (
            float(available),
            float(not resolved),
            float(resolved),
            0.0,
            float(not resolved),
            float(identity_risk),
        )

    def _strain_status_values(self, key: str | None) -> tuple[float, ...]:
        available = key is not None and key in self._strain_values
        source = self._strain_status.get(str(key), {}) if available else {}
        resolved = bool(source.get("resolved", False))
        missing = bool(source.get("missing", not resolved)) or not resolved
        proxy = bool(source.get("proxy", False))
        identity_candidate = bool(
            available and str(key) in self._strain_candidate_keys
        )
        return (
            float(available),
            float(missing),
            float(resolved),
            float(proxy),
            float(not resolved),
            float(identity_candidate),
        )

    @staticmethod
    def _coverage_for_keys(
        keys: list[str | None],
        status_function: Callable[[str | None], tuple[float, ...]],
        flag_names: tuple[str, ...],
    ) -> dict[str, object]:
        statuses = np.asarray([status_function(key) for key in keys], dtype=np.float64)
        unique_keys = sorted(set(keys), key=lambda value: "" if value is None else value)
        unique_statuses = np.asarray(
            [status_function(key) for key in unique_keys], dtype=np.float64
        )
        result: dict[str, object] = {
            "row_count": int(len(keys)),
            "unique_entity_count": int(len(unique_keys)),
        }
        for column, flag_name in enumerate(flag_names):
            result[f"{flag_name}_rows"] = int(statuses[:, column].sum()) if len(keys) else 0
            result[f"{flag_name}_unique_entities"] = (
                int(unique_statuses[:, column].sum()) if len(unique_keys) else 0
            )
        resolved_rows = int(result.get("resolved_rows", 0))
        result["resolved_row_fraction"] = (
            float(resolved_rows / len(keys)) if len(keys) else None
        )
        resolved_unique = int(result.get("resolved_unique_entities", 0))
        result["resolved_unique_fraction"] = (
            float(resolved_unique / len(unique_keys)) if len(unique_keys) else None
        )
        return result

    def summarize_coverage(self, metadata: pd.DataFrame) -> dict[str, object]:
        """Return JSON-safe entity coverage without fitting or reading response."""

        self._require_metadata_fields(metadata)
        chemical_keys = _normalised_keys(metadata[CHEMICAL])
        strain_keys = _normalised_keys(metadata[STRAIN])
        strain_coverage = self._coverage_for_keys(
            strain_keys, self._strain_status_values, _STRAIN_FLAGS
        )
        strain_coverage.update(
            {
                "organizer_verified": False,
                "identity_warning": STRAIN_IDENTITY_WARNING,
                "identity_evidence_registry_sha256": (
                    STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256
                ),
                "identity_evidence_manifest_sha256": (
                    STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256
                ),
            }
        )
        summary = {
            "row_count": int(len(metadata)),
            "response_used": False,
            "chemical": self._coverage_for_keys(
                chemical_keys, self._chemical_status, _CHEMICAL_FLAGS
            ),
            "strain": strain_coverage,
        }
        return _json_copy(summary)  # type: ignore[return-value]

    @staticmethod
    def _fit_scaler(rows: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            return np.zeros(width, dtype=np.float64), np.ones(width, dtype=np.float64)
        matrix = np.stack(rows).astype(np.float64, copy=False)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
        return mean, scale

    def fit(self, metadata: pd.DataFrame) -> "TargetFreeSemanticEncoder":
        """Fit scalers on unique candidate-pool entities, never measurements."""

        if len(metadata) == 0:
            raise ValueError("Cannot fit semantic encoder on an empty candidate pool")
        self._require_metadata_fields(metadata)
        chemical_keys = sorted(
            {key for key in _normalised_keys(metadata[CHEMICAL]) if key is not None}
        )
        strain_keys = sorted(
            {key for key in _normalised_keys(metadata[STRAIN]) if key is not None}
        )
        chemical_rows = [
            self._chemical_values[key]
            for key in chemical_keys
            if key in self._chemical_values and self._chemical_resolved[key]
        ]
        strain_rows = [
            self._strain_values[key]
            for key in strain_keys
            if key in self._strain_values
            and self._strain_status[key]["resolved"]
            and not self._strain_status[key]["missing"]
        ]
        self.chemical_mean_, self.chemical_scale_ = self._fit_scaler(
            chemical_rows, len(self._chemical_columns)
        )
        self.strain_mean_, self.strain_scale_ = self._fit_scaler(
            strain_rows, len(self._strain_columns)
        )
        coverage = self.summarize_coverage(metadata)
        coverage.update(
            {
                "fitted": True,
                "fit_scope": "unique_candidate_pool_entities",
                "measurement_frequency_weighted": False,
                "response_used": False,
                "chemical_scaler_entity_count": int(len(chemical_rows)),
                "strain_scaler_entity_count": int(len(strain_rows)),
                "clip_value": float(self.clip_value),
                "continuous_field_balance": "divide_each_block_by_sqrt_width",
            }
        )
        self._coverage_summary = _json_copy(coverage)  # type: ignore[assignment]
        return self

    def _require_fitted(self) -> None:
        if (
            self.chemical_mean_ is None
            or self.chemical_scale_ is None
            or self.strain_mean_ is None
            or self.strain_scale_ is None
        ):
            raise RuntimeError("TargetFreeSemanticEncoder must be fit before transform")

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self._require_metadata_fields(metadata)
        assert self.chemical_mean_ is not None
        assert self.chemical_scale_ is not None
        assert self.strain_mean_ is not None
        assert self.strain_scale_ is not None

        chemical_keys = _normalised_keys(metadata[CHEMICAL])
        strain_keys = _normalised_keys(metadata[STRAIN])
        chemical_continuous = np.zeros(
            (len(metadata), len(self._chemical_columns)), dtype=np.float64
        )
        chemical_flags = np.empty(
            (len(metadata), len(_CHEMICAL_FLAGS)), dtype=np.float64
        )
        for row, key in enumerate(chemical_keys):
            status = self._chemical_status(key)
            chemical_flags[row] = status
            if bool(status[2]):
                values = (self._chemical_values[str(key)] - self.chemical_mean_) / self.chemical_scale_
                chemical_continuous[row] = (
                    np.clip(values, -self.clip_value, self.clip_value)
                    / math.sqrt(len(self._chemical_columns))
                )

        strain_continuous = np.zeros(
            (len(metadata), len(self._strain_columns)), dtype=np.float64
        )
        strain_flags = np.empty((len(metadata), len(_STRAIN_FLAGS)), dtype=np.float64)
        for row, key in enumerate(strain_keys):
            status = self._strain_status_values(key)
            strain_flags[row] = status
            if bool(status[2]):
                values = (self._strain_values[str(key)] - self.strain_mean_) / self.strain_scale_
                strain_continuous[row] = (
                    np.clip(values, -self.clip_value, self.clip_value)
                    / math.sqrt(len(self._strain_columns))
                )

        result = np.concatenate(
            [
                chemical_continuous,
                chemical_flags,
                strain_continuous,
                strain_flags,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if result.shape != (len(metadata), self.output_dim) or not np.isfinite(result).all():
            raise AssertionError("Semantic encoder produced an invalid feature matrix")
        return result

    def fit_transform(self, metadata: pd.DataFrame) -> np.ndarray:
        return self.fit(metadata).transform(metadata)


class CombinedTargetFreeEncoder:
    """Existing identity/time descriptors followed by frozen semantics."""

    def __init__(
        self,
        base_encoder: PoolFeatureEncoder | None = None,
        semantic_encoder: TargetFreeSemanticEncoder | None = None,
    ) -> None:
        self.base_encoder = PoolFeatureEncoder() if base_encoder is None else base_encoder
        self.semantic_encoder = (
            TargetFreeSemanticEncoder() if semantic_encoder is None else semantic_encoder
        )
        self._feature_names: tuple[str, ...] = ()
        self._block_slices: dict[str, slice] = {}
        self._categorical_column_slices: dict[str, slice] = {}
        self._fitted = False

    def fit(self, metadata: pd.DataFrame) -> "CombinedTargetFreeEncoder":
        self.base_encoder.fit(metadata)
        self.semantic_encoder.fit(metadata)
        base_names: list[str] = []
        for field_name in self.base_encoder.categorical_fields:
            base_names.extend(
                f"identity__{field_name}={category}"
                for category in self.base_encoder.categories[field_name]
            )
        base_names.append("time__minutes_standardized")
        base_width = self.base_encoder.output_dim
        semantic_width = self.semantic_encoder.output_dim
        self._feature_names = (*base_names, *self.semantic_encoder.feature_names)
        self._categorical_column_slices = dict(
            self.base_encoder.categorical_column_slices
        )
        categorical_stop = self.base_encoder.continuous_column_slice.start
        self._block_slices = {
            "identity_time": slice(0, base_width),
            "identity_categorical": slice(0, categorical_stop),
            "time_continuous": self.base_encoder.continuous_column_slice,
            "semantics": slice(base_width, base_width + semantic_width),
        }
        for name, block in self.semantic_encoder.block_slices.items():
            self._block_slices[name] = slice(
                base_width + int(block.start), base_width + int(block.stop)
            )
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CombinedTargetFreeEncoder must be fit before use")

    def transform(self, metadata: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        base = self.base_encoder.transform(metadata)
        semantics = self.semantic_encoder.transform(metadata)
        result = np.concatenate([base, semantics], axis=1).astype(np.float32, copy=False)
        if result.shape[1] != self.output_dim:
            raise AssertionError("Combined encoder output width differs from its contract")
        return result

    def fit_transform(self, metadata: pd.DataFrame) -> np.ndarray:
        return self.fit(metadata).transform(metadata)

    def mask_unsupported(
        self,
        features: np.ndarray,
        supported_features: np.ndarray,
    ) -> np.ndarray:
        self._require_fitted()
        values = np.asarray(features)
        support = np.asarray(supported_features)
        if values.ndim != 2 or support.ndim != 2:
            raise ValueError("features and supported_features must be two-dimensional")
        if values.shape[1] != self.output_dim or support.shape[1] != self.output_dim:
            raise ValueError("Combined feature matrices have an unexpected width")
        return mask_base_unsupported(
            values,
            support,
            self._categorical_column_slices,
        )

    @property
    def categorical_column_slices(self) -> dict[str, slice]:
        self._require_fitted()
        return dict(self._categorical_column_slices)

    @property
    def output_dim(self) -> int:
        self._require_fitted()
        return len(self._feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return self._feature_names

    @property
    def feature_names_(self) -> tuple[str, ...]:
        return self.feature_names

    @property
    def block_slices(self) -> dict[str, slice]:
        self._require_fitted()
        return dict(self._block_slices)

    @property
    def asset_hashes(self) -> dict[str, str]:
        return self.semantic_encoder.asset_hashes

    @property
    def asset_summary(self) -> dict[str, object]:
        return self.semantic_encoder.asset_summary

    @property
    def coverage_summary(self) -> dict[str, object]:
        return self.semantic_encoder.coverage_summary


def _slice_summary(values: Mapping[str, slice]) -> dict[str, dict[str, int]]:
    return {
        name: {"start": int(column_slice.start), "stop": int(column_slice.stop)}
        for name, column_slice in values.items()
    }


def load_feature_bundle(
    dataset: GroupedDataset,
    encoder: CombinedTargetFreeEncoder | None = None,
) -> Mapping[str, object]:
    """Build ID-aligned target-free matrices for a later active-learning runner."""

    if not isinstance(dataset.metadata, pd.DataFrame):
        raise TypeError("dataset.metadata must be a pandas DataFrame")
    pool_ids = dataset.candidate_pool_ids
    missing_pool = pool_ids.difference(dataset.metadata.index)
    if len(missing_pool):
        raise ValueError(f"Candidate-pool IDs are missing from metadata: {missing_pool[:5].tolist()}")
    combined = CombinedTargetFreeEncoder() if encoder is None else encoder
    combined.fit(dataset.metadata.loc[pool_ids])
    all_features = combined.transform(dataset.metadata)
    descriptor_matrix = np.array(all_features, dtype=np.float32, copy=True)
    model_matrix = np.array(all_features, dtype=np.float32, copy=True)
    all_coverage = combined.semantic_encoder.summarize_coverage(dataset.metadata)
    summary = _json_copy(
        {
            "fit_scope": "candidate_pool_metadata_unique_entities",
            "response_used": False,
            "row_count": int(len(dataset.metadata)),
            "candidate_pool_count": int(len(pool_ids)),
            "output_dim": int(combined.output_dim),
            "feature_names": list(combined.feature_names),
            "block_slices": _slice_summary(combined.block_slices),
            "categorical_column_slices": _slice_summary(
                combined.categorical_column_slices
            ),
            "masking": (
                "base_categorical_one_hot_only; continuous_time_and_semantics_preserved"
            ),
            "assets": combined.asset_summary,
            "fit_coverage": combined.coverage_summary,
            "all_row_coverage": all_coverage,
        }
    )
    return {
        "row_ids": dataset.metadata.index,
        "descriptor_matrix": descriptor_matrix,
        "descriptor_features": descriptor_matrix,
        "descriptors": descriptor_matrix,
        "model_matrix": model_matrix,
        "model_features": model_matrix,
        "masker": combined.mask_unsupported,
        "summary": summary,
        "asset_hashes": combined.asset_hashes,
        "encoder": combined,
    }


__all__ = [
    "CHEMBERTA_MODEL_REVISION",
    "CHEMICAL_EMBEDDINGS_SHA256",
    "CHEMICAL_MANIFEST_SHA256",
    "STRAIN_SEMANTICS_SHA256",
    "STRAIN_MANIFEST_SHA256",
    "STRAIN_IDENTITY_EVIDENCE_REGISTRY_SHA256",
    "STRAIN_IDENTITY_EVIDENCE_MANIFEST_SHA256",
    "STRAIN_IDENTITY_WARNING",
    "STRAIN_IDENTITY_CANDIDATES",
    "STRAIN_UNRESOLVED",
    "CHEMICAL_RISK_MANIFEST_SHA256",
    "TargetFreeSemanticEncoder",
    "CombinedTargetFreeEncoder",
    "load_feature_bundle",
]
