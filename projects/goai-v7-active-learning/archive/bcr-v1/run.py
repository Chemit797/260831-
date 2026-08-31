#!/usr/bin/env python3
"""Run and aggregate the fixed GOAI-SEMANTIC-BCR-V1 experiment.

Commands are fold-staged so two GPUs can train independent outer folds in
parallel.  S1 and time_forward are the only scenarios whose exact, hash-locked
fold-local CalV2 caches exist; S2/S3 are intentionally emitted as N/A.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import FlatMLPSameInfo, GOAISemanticBCRV1


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path("/home/chenyuming/Project/go-ai")
METADATA_PATH = RAW_ROOT / "WAYB_WAYC_metadata_train_val.csv"
PROTEOME_PATH = RAW_ROOT / "WAYB_WAYC_proteome_raw_train_val.csv"
ASSIGNMENT_PATH = RAW_ROOT / "runs/nightly/20260812-preflight-folds/fold_assignments.csv"
PANEL_PATH = (
    ROOT
    / "runs/calibration_context_v1/protein-panel-contract-20260821/local_formal_4422_proteins.csv"
)
STRAIN_PATH = ROOT / "data/strain_embeddings.csv"
OP3_PATH = Path(
    "/home/chenyuming/Project/go-ai/data/processed/chemical_embeddings/"
    "op3_rna_strict/op3_real.tsv"
)
OP3_MANIFEST_PATH = OP3_PATH.parent / "manifest.json"
OP3_ENCODER_PATH = Path(
    "/home/chenyuming/Project/goai-rna-transfer/models/rna_pretraining/"
    "rna_real_strict_encoder.pt"
)
OP3_PRETRAIN_MANIFEST_PATH = OP3_ENCODER_PATH.with_name("manifest_strict.json")
OP3_EXTERNAL_SOURCES_PATH = Path(
    "/home/chenyuming/Project/goai-rna-transfer/configs/external_sources.json"
)
STRAIN_PROVENANCE_PATH = (
    ROOT / "experiments/basic_descriptor_mlp/provenance/descriptor_manifest.yaml"
)
MATCHED_CONTROL_CONTRACT_PATH = (
    ROOT
    / "runs/semantic_modular_pretraining_v3/smp-v3-20260823-073716/"
    "00_contract_audit/MATCHED_CONTROL_CURRENT_CONTRACT.json"
)
MATCHED_CONTROL_IMPLEMENTATION_PATH = (
    ROOT
    / "experiments/semantic_feature_engineering_v1/src/"
    "semantic_feature_engineering_v1/controls.py"
)
FORMAL_CALV2_MANIFEST = (
    ROOT
    / "experiments/biological_routing_chemical_tech_a3_v1/"
    "configs/formal_fold_semantics.json"
)
DEFAULT_RUN_DIR = ROOT / "runs/goai_semantic_bcr_v1/semantic-bcr-v1-seed42"

SEED = 42
N_PROTEINS = 4422
BATCH_SIZE = 128
CONTROL_CHEMICALS = frozenset(("Water", "DMSO"))
QC_CHEMICAL = "Quality Control"
MATCH_FIELDS = (
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)
SOURCE_CATEGORIES = ("WAYB", "WAYB_rep1", "WAYB_rep2", "WAYC")
INSTRUMENT_CATEGORIES = ("CAA", "CAB", "CAC", "O", "QE1", "QE2", "WAH")
MODELS = (
    "GOAI-SEMANTIC-BCR-V1",
    "FLAT-MLP-SAME-INFO",
    "PROTEIN-MEAN",
    "MATCHED-CONTROL-ORACLE-DIAGNOSTIC",
)
SCENARIOS = ("S1", "time_forward")

EXPECTED_HASHES = {
    METADATA_PATH: "9414f22d71e925a3b85544b49fde252613c87808d34738a84785003adb8131ef",
    PROTEOME_PATH: "a15d9a40a6ad4e8e84a4ce4ed08644fce78780d31ace5561928517c4a5fa7ccb",
    ASSIGNMENT_PATH: "1adeedb3d1566d50a2721b0fb7a246b45fc0e54fd2edc799ca3b298c4be83e81",
    PANEL_PATH: "808d8f62feb9101293ef6f7972acbacafcdce3e81a8a7c5d22e86fe104d8c7ec",
    STRAIN_PATH: "a83510062107a444e67cbfc9ff69314574cc0dc728f8dacb62529eb62987d1e7",
    OP3_PATH: "58f1ed8a5e4c76a0db9125efe7c0a1a297ab8fca90606f50938de17e663bb789",
    OP3_MANIFEST_PATH: "93cd9c5b86a6712ceffbf36ffb4861d58864638e27c6fb373237cab397b3cf7b",
    OP3_ENCODER_PATH: "c8d9091bbdd6f5d4eeae85106a9db4b773425da0544a9d73fae05ce9dbb7c996",
    OP3_PRETRAIN_MANIFEST_PATH: "ce90bb24f8c38ba83eb29a6c71fc4cb1550401a8099bcd9513fb634338f7d722",
    OP3_EXTERNAL_SOURCES_PATH: "4948962fc48b26e1793cb7076c2fcf9fe05345a57fd9bcc5ecf584364f2f6647",
    STRAIN_PROVENANCE_PATH: "b751038d9da2c5fa9edd363b1544f7abae7a8eca49fd0db323791ca0225ce5e1",
    MATCHED_CONTROL_CONTRACT_PATH: "f9cae1fd3ffb81df4eb59ccad3036da1777fdbba54409cea4013a886f30dbe90",
    MATCHED_CONTROL_IMPLEMENTATION_PATH: "28a922e990432fde92ef812f0796cfaa6a0b92267521de6d93e17e4f74d650f4",
    FORMAL_CALV2_MANIFEST: "ba4ae8a2c20129c1230586fa220d216ce279b1c1115cdca1204f2990c4eafaf7",
}


@dataclass(frozen=True)
class FeatureState:
    medium_categories: tuple[str, ...]
    temperature_mean: float
    temperature_std: float
    log_time_mean: float
    log_time_std: float


@dataclass(frozen=True)
class FeatureArrays:
    strain: np.ndarray
    medium: np.ndarray
    temperature: np.ndarray
    time: np.ndarray
    chemical: np.ndarray
    z_cal: np.ndarray
    is_treatment: np.ndarray
    flat: np.ndarray


@dataclass(frozen=True)
class FoldPrepared:
    scenario: str
    fold: int
    fit_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    proteins: tuple[str, ...]
    fit_features: FeatureArrays
    eval_features: FeatureArrays
    response_fit_positions: np.ndarray
    response_target: np.ndarray
    response_mask: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    fit_target: np.ndarray
    fit_mask: np.ndarray
    eval_truth: np.ndarray
    eval_truth_mask: np.ndarray
    oracle_prediction: np.ndarray
    oracle_has_match: np.ndarray
    control_fit_positions: np.ndarray
    cal_center: np.ndarray
    cal_decoder_scaled: np.ndarray
    feature_state: FeatureState
    calv2_checkpoint_path: str
    calv2_checkpoint_sha256: str
    response_correction_max_abs: float


@dataclass
class TorchFeatureArrays:
    strain: torch.Tensor
    medium: torch.Tensor
    temperature: torch.Tensor
    time: torch.Tensor
    chemical: torch.Tensor
    z_cal: torch.Tensor
    is_treatment: torch.Tensor
    flat: torch.Tensor

    @classmethod
    def from_numpy(cls, values: FeatureArrays, device: torch.device) -> "TorchFeatureArrays":
        return cls(
            strain=torch.as_tensor(np.ascontiguousarray(values.strain), device=device),
            medium=torch.as_tensor(np.ascontiguousarray(values.medium), device=device),
            temperature=torch.as_tensor(
                np.ascontiguousarray(values.temperature), device=device
            ),
            time=torch.as_tensor(np.ascontiguousarray(values.time), device=device),
            chemical=torch.as_tensor(np.ascontiguousarray(values.chemical), device=device),
            z_cal=torch.as_tensor(np.ascontiguousarray(values.z_cal), device=device),
            is_treatment=torch.as_tensor(
                np.ascontiguousarray(values.is_treatment), device=device
            ),
            flat=torch.as_tensor(np.ascontiguousarray(values.flat), device=device),
        )

    def subset(self, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.strain[index],
            self.medium[index],
            self.temperature[index],
            self.time[index],
            self.chemical[index],
            self.z_cal[index],
            self.is_treatment[index],
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


RUN_CODE_SHA256_AT_START = sha256_file(Path(__file__).resolve())
MODEL_CODE_SHA256_AT_START = sha256_file(Path(__file__).with_name("model.py").resolve())


def assert_code_unchanged() -> None:
    if sha256_file(Path(__file__).resolve()) != RUN_CODE_SHA256_AT_START:
        raise RuntimeError("run.py changed after this process started")
    if (
        sha256_file(Path(__file__).with_name("model.py").resolve())
        != MODEL_CODE_SHA256_AT_START
    ):
        raise RuntimeError("model.py changed after this process started")


def validate_expected_hashes() -> dict[str, str]:
    observed_hashes: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"hash mismatch: {path}")
        observed_hashes[str(path)] = observed
    return observed_hashes


def ordered_json_hash(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(map(str, values)), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe_json(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return torch.device(requested)


def canonical_time_minutes(frame: pd.DataFrame) -> np.ndarray:
    value = pd.to_numeric(frame["pert_time"], errors="raise").to_numpy(np.float64)
    unit = frame["pert_time_unit"].astype(str).str.strip().str.casefold()
    factor = unit.map(
        {
            "min": 1.0,
            "mins": 1.0,
            "minute": 1.0,
            "minutes": 1.0,
            "h": 60.0,
            "hr": 60.0,
            "hrs": 60.0,
            "hour": 60.0,
            "hours": 60.0,
            "s": 1.0 / 60.0,
            "sec": 1.0 / 60.0,
            "secs": 1.0 / 60.0,
            "second": 1.0 / 60.0,
            "seconds": 1.0 / 60.0,
        }
    )
    if factor.isna().any():
        raise ValueError(f"unsupported time units: {sorted(unit[factor.isna()].unique())}")
    result = value * factor.to_numpy(np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("canonical time must be finite and nonnegative")
    return result


def load_metadata() -> pd.DataFrame:
    frame = pd.read_csv(METADATA_PATH, low_memory=False)
    if "sample_ID" not in frame or frame["sample_ID"].isna().any():
        raise ValueError("metadata requires nonmissing sample_ID")
    frame["sample_ID"] = frame["sample_ID"].astype(str)
    if frame["sample_ID"].duplicated().any():
        raise ValueError("metadata sample_ID must be unique")
    frame = frame.set_index("sample_ID", verify_integrity=True)
    return frame


def load_proteins() -> tuple[str, ...]:
    frame = pd.read_csv(PANEL_PATH)
    proteins = tuple(frame["protein"].astype(str))
    if len(proteins) != N_PROTEINS or len(set(proteins)) != N_PROTEINS:
        raise RuntimeError("fixed protein panel must contain 4,422 unique names")
    if ordered_json_hash(proteins) != "35fe57a1276848355004232af6d9a3cfe1ee2f4ae0ab9a529797239aa04f781a":
        raise RuntimeError("fixed protein order changed")
    return proteins


def load_log2_proteome(metadata: pd.DataFrame, proteins: Sequence[str]) -> pd.DataFrame:
    # Explicit ID alignment is mandatory even though the current two source
    # files happen to share row order.
    frame = pd.read_csv(PROTEOME_PATH, usecols=["sample_ID", *proteins], low_memory=False)
    frame["sample_ID"] = frame["sample_ID"].astype(str)
    if frame["sample_ID"].isna().any() or frame["sample_ID"].duplicated().any():
        raise ValueError("proteome sample_ID must be nonmissing and unique")
    frame = frame.set_index("sample_ID", verify_integrity=True)
    if set(frame.index) != set(metadata.index):
        raise RuntimeError("metadata/proteome sample_ID sets differ")
    raw = frame.reindex(metadata.index).loc[:, list(proteins)].to_numpy(np.float32)
    observed = np.isfinite(raw)
    if np.isinf(raw).any() or np.any(raw[observed] <= 0):
        raise ValueError("observed raw proteome values must be positive finite values")
    values = np.full(raw.shape, np.nan, dtype=np.float32)
    values[observed] = np.log2(raw[observed]).astype(np.float32)
    return pd.DataFrame(values, index=metadata.index, columns=list(proteins))


def load_descriptors() -> tuple[pd.DataFrame, pd.DataFrame]:
    strain = pd.read_csv(STRAIN_PATH)
    strain["strain_name"] = strain["strain_name"].astype(str).str.strip()
    if strain["strain_name"].duplicated().any() or strain.shape[1] != 4097:
        raise RuntimeError("RAW4096 strain table schema differs")
    strain = strain.set_index("strain_name", verify_integrity=True)
    if not np.isfinite(strain.to_numpy(np.float32)).all():
        raise RuntimeError("RAW4096 contains non-finite values")

    chemical = pd.read_csv(OP3_PATH, sep="\t")
    chemical["raw_name"] = chemical["raw_name"].astype(str).str.strip()
    if chemical["raw_name"].duplicated().any() or chemical.shape[1] != 65:
        raise RuntimeError("OP3 table schema differs")
    chemical = chemical.set_index("raw_name", verify_integrity=True)
    if not np.isfinite(chemical.to_numpy(np.float32)).all():
        raise RuntimeError("OP3 contains non-finite values")
    return strain, chemical


def _calv2_python_paths() -> None:
    paths = (
        ROOT / "experiments/biological_routing_chemical_tech_a3_v1/src",
        ROOT / "experiments/biological_routing_chemical_time/src",
        ROOT / "experiments/semantic_feature_engineering_v1/src",
        ROOT / "experiments/calibration_embedding_v2/src",
    )
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_calv2_gate() -> dict[str, Any]:
    _calv2_python_paths()
    from biological_routing_chemical_tech_a3_v1.contracts import (  # type: ignore
        validate_all_calv2_caches,
    )

    gate = validate_all_calv2_caches(raise_on_failure=True)
    receipt = gate.receipt()
    expected = {
        "enabled": True,
        "folds_validated": 8,
        "folds_passed": 8,
        "provider_new_fits": 0,
        "observed_cache_surface_sha256": "8b847fd909700b678131dbf470c6032478d110aa4316e4a91c6e23597b4d345a",
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"CalV2 all-cache gate mismatch for {key}")
    return receipt


def formal_calv2_row(scenario: str, fold: int) -> dict[str, Any]:
    payload = json.loads(FORMAL_CALV2_MANIFEST.read_text(encoding="utf-8"))
    rows = payload.get(scenario)
    if not isinstance(rows, list):
        raise RuntimeError(f"formal CalV2 scenario unavailable: {scenario}")
    found = [row for row in rows if int(row.get("fold", -1)) == fold]
    if len(found) != 1:
        raise RuntimeError(f"formal CalV2 fold row unavailable: {scenario}/{fold}")
    return dict(found[0])


def load_fold_ids(scenario: str, fold: int, cal_row: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = Path(str(cal_row["calv2_cache_path"]))
    fit = tuple(pd.read_csv(root / "fit_sample_ids.csv")["sample_ID"].astype(str))
    evaluation = tuple(pd.read_csv(root / "eval_sample_ids.csv")["sample_ID"].astype(str))
    if not fit or not evaluation or set(fit) & set(evaluation):
        raise RuntimeError("fold fit/eval ID surface is invalid")
    checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("scenario") != scenario or int(contract.get("outer_fold", -1)) != fold:
        raise RuntimeError("CalV2 checkpoint fold identity differs")
    return fit, evaluation


def encode_calv2(
    frame: pd.DataFrame,
    checkpoint: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = checkpoint["model_state_dict"]
    source = np.asarray(state["source"].detach().cpu(), dtype=np.float32)
    instrument = np.asarray(state["instrument"].detach().cpu(), dtype=np.float32)
    raw_decoder = np.asarray(state["decoder.weight"].detach().cpu(), dtype=np.float32)
    decoder = np.asarray(checkpoint["canonical_decoder"], dtype=np.float32)
    protein_std = np.asarray(checkpoint["protein_std"], dtype=np.float32)
    if (
        source.shape != (4, 12)
        or instrument.shape != (7, 12)
        or raw_decoder.shape != (N_PROTEINS, 12)
        or decoder.shape != (N_PROTEINS, 12)
        or protein_std.shape != (N_PROTEINS,)
    ):
        raise RuntimeError("CalV2 checkpoint tensor interface differs")
    source_lookup = {value: index for index, value in enumerate(SOURCE_CATEGORIES)}
    instrument_lookup = {
        value: index for index, value in enumerate(INSTRUMENT_CATEGORIES)
    }
    try:
        source_code = np.asarray(
            [source_lookup[str(value)] for value in frame["data_source"]], np.int64
        )
        instrument_code = np.asarray(
            [instrument_lookup[str(value)] for value in frame["instrument"]], np.int64
        )
    except KeyError as error:
        raise RuntimeError("unsupported CalV2 source/instrument category") from error
    # Preserve the accepted provider's float32/float64 operation order.
    source_tensor = torch.from_numpy(source.copy())
    instrument_tensor = torch.from_numpy(instrument.copy())
    source_centered = source_tensor - source_tensor.mean(dim=0, keepdim=True)
    instrument_centered = instrument_tensor - instrument_tensor.mean(dim=0, keepdim=True)
    raw_z = (
        source_centered[torch.from_numpy(source_code)]
        + instrument_centered[torch.from_numpy(instrument_code)]
    ).numpy().astype(np.float64)
    _u, singular, vh = np.linalg.svd(raw_decoder.astype(np.float64), full_matrices=False)
    z_cal = ((raw_z @ vh.T) * singular[None, :]).astype(np.float32)
    decoded = np.asarray(z_cal @ decoder.T, dtype=np.float32)
    reconstructed = np.asarray(raw_z @ raw_decoder.astype(np.float64).T, dtype=np.float32)
    error = float(np.max(np.abs(decoded - reconstructed)))
    if error > 2e-4:
        raise RuntimeError(f"CalV2 canonical decode invariant failed: {error}")
    return z_cal, decoder, protein_std


def fit_feature_state(fit_metadata: pd.DataFrame) -> FeatureState:
    medium = tuple(sorted(fit_metadata["Medium"].astype(str).str.strip().unique()))
    if not medium:
        raise RuntimeError("outer fit has no medium categories")
    temperature = pd.to_numeric(fit_metadata["Temperature"], errors="raise").to_numpy(
        np.float64
    )
    log_time = np.log1p(canonical_time_minutes(fit_metadata))
    temperature_std = float(temperature.std())
    log_time_std = float(log_time.std())
    return FeatureState(
        medium_categories=medium,
        temperature_mean=float(temperature.mean()),
        temperature_std=temperature_std if temperature_std > 1e-8 else 1.0,
        log_time_mean=float(log_time.mean()),
        log_time_std=log_time_std if log_time_std > 1e-8 else 1.0,
    )


def build_feature_arrays(
    metadata: pd.DataFrame,
    *,
    state: FeatureState,
    strain_table: pd.DataFrame,
    chemical_table: pd.DataFrame,
    z_cal: np.ndarray,
    cal_center: np.ndarray,
) -> FeatureArrays:
    strains = metadata["Strains"].astype(str).str.strip()
    chemicals = metadata["perturbation_no_concentration"].astype(str).str.strip()
    missing_strains = sorted(set(strains) - set(strain_table.index))
    missing_chemicals = sorted(set(chemicals) - set(chemical_table.index))
    if missing_strains or missing_chemicals:
        raise RuntimeError(
            f"descriptor coverage failure: strains={missing_strains}, chemicals={missing_chemicals}"
        )
    strain = strain_table.loc[list(strains)].to_numpy(np.float32)
    chemical = chemical_table.loc[list(chemicals)].to_numpy(np.float32)

    medium_lookup = {value: index for index, value in enumerate(state.medium_categories)}
    medium_values = metadata["Medium"].astype(str).str.strip()
    unknown_medium = sorted(set(medium_values) - set(medium_lookup))
    if unknown_medium:
        raise RuntimeError(f"outer-eval medium unseen in outer fit: {unknown_medium}")
    medium = np.asarray([medium_lookup[value] for value in medium_values], np.int64)
    temperature_raw = pd.to_numeric(metadata["Temperature"], errors="raise").to_numpy(
        np.float64
    )
    temperature = (
        (temperature_raw - state.temperature_mean) / state.temperature_std
    ).astype(np.float32)[:, None]
    log_time = np.log1p(canonical_time_minutes(metadata))
    time_feature = ((log_time - state.log_time_mean) / state.log_time_std).astype(
        np.float32
    )[:, None]
    is_treatment = (~chemicals.isin(CONTROL_CHEMICALS | {QC_CHEMICAL})).to_numpy(
        np.float32
    )

    medium_one_hot = np.eye(len(state.medium_categories), dtype=np.float32)[medium]
    centered_z = (np.asarray(z_cal, np.float32) - cal_center[None, :]).astype(np.float32)
    flat = np.concatenate(
        (
            strain,
            medium_one_hot,
            temperature,
            time_feature,
            chemical,
            centered_z,
            is_treatment[:, None],
        ),
        axis=1,
    ).astype(np.float32)
    if not all(
        np.isfinite(value).all()
        for value in (strain, chemical, temperature, time_feature, centered_z, flat)
    ):
        raise RuntimeError("model features contain non-finite values")
    return FeatureArrays(
        strain=strain,
        medium=medium,
        temperature=temperature,
        time=time_feature,
        chemical=chemical,
        z_cal=np.asarray(z_cal, np.float32),
        is_treatment=is_treatment,
        flat=flat,
    )


def fold_target_statistics(values: np.ndarray, floor: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0).astype(np.float32)
        scale = np.nanstd(values, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0)
    scale = np.maximum(np.nan_to_num(scale, nan=floor), floor)
    if mean.shape != (N_PROTEINS,) or scale.shape != (N_PROTEINS,):
        raise RuntimeError("fold target-statistic shape differs")
    return mean, scale


def match_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in MATCH_FIELDS)


def build_control_map(
    fit_metadata: pd.DataFrame,
    fit_y: np.ndarray,
) -> dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map exact 8-key groups to log2 means, masks, and member positions."""

    chemical = fit_metadata["perturbation_no_concentration"].astype(str)
    positions = np.flatnonzero(chemical.isin(CONTROL_CHEMICALS).to_numpy())
    grouped: dict[tuple[str, ...], list[int]] = {}
    for position in positions:
        grouped.setdefault(match_key(fit_metadata.iloc[int(position)]), []).append(
            int(position)
        )
    result: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, members in grouped.items():
        member_array = np.asarray(members, np.int64)
        selected = fit_y[member_array]
        observed = np.isfinite(selected)
        count = observed.sum(axis=0)
        summed = np.where(observed, selected, 0.0).sum(axis=0, dtype=np.float64)
        mean = np.full(N_PROTEINS, np.nan, np.float32)
        keep = count > 0
        mean[keep] = (summed[keep] / count[keep]).astype(np.float32)
        result[key] = (mean, keep, member_array)
    if not result:
        raise RuntimeError("outer fit contains no matched-control groups")
    return result


def build_response_targets(
    fit_metadata: pd.DataFrame,
    fit_y: np.ndarray,
    target_std: np.ndarray,
    z_cal: np.ndarray,
    cal_decoder: np.ndarray,
    cal_protein_std: np.ndarray,
    control_map: Mapping[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    chemical = fit_metadata["perturbation_no_concentration"].astype(str)
    treatment_positions = np.flatnonzero(
        (~chemical.isin(CONTROL_CHEMICALS | {QC_CHEMICAL})).to_numpy()
    )
    kept_positions: list[int] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    max_correction = 0.0
    for position in treatment_positions:
        key = match_key(fit_metadata.iloc[int(position)])
        control = control_map.get(key)
        if control is None:
            continue
        control_mean, control_observed, members = control
        target_row = fit_y[int(position)]
        observed = np.isfinite(target_row) & control_observed
        # The frozen exact-match contract includes source+instrument, so the
        # CalV2 technical correction is mathematically zero.  Compute and audit
        # it instead of silently assuming it.
        z_control = z_cal[members].astype(np.float64).mean(axis=0).astype(np.float32)
        delta_c = np.asarray(
            (z_cal[int(position)] - z_control) @ cal_decoder.T,
            dtype=np.float32,
        ) * cal_protein_std
        if observed.any():
            max_correction = max(
                max_correction, float(np.max(np.abs(delta_c[observed])))
            )
        delta = target_row - control_mean - delta_c
        standardized = np.zeros(N_PROTEINS, np.float32)
        standardized[observed] = (delta[observed] / target_std[observed]).astype(
            np.float32
        )
        kept_positions.append(int(position))
        targets.append(standardized)
        masks.append(observed)
    if not kept_positions:
        raise RuntimeError("outer fit contains no valid matched treatment rows")
    return (
        np.asarray(kept_positions, np.int64),
        np.stack(targets).astype(np.float32),
        np.stack(masks).astype(bool),
        max_correction,
    )


def build_oracle_predictions(
    eval_metadata: pd.DataFrame,
    control_map: Mapping[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full((len(eval_metadata), N_PROTEINS), np.nan, np.float32)
    has_match = np.zeros(len(eval_metadata), bool)
    for position in range(len(eval_metadata)):
        control = control_map.get(match_key(eval_metadata.iloc[position]))
        if control is None:
            continue
        prediction[position] = control[0]
        has_match[position] = True
    return prediction, has_match


def prepare_fold(scenario: str, fold: int, *, validate_gate: bool = True) -> FoldPrepared:
    if scenario not in SCENARIOS or fold not in range(4):
        raise ValueError("only S1/time_forward folds 0..3 are legally runnable")
    if validate_gate:
        validate_calv2_gate()
    cal_row = formal_calv2_row(scenario, fold)
    cal_root = Path(str(cal_row["calv2_cache_path"]))
    checkpoint_path = cal_root / "checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != str(cal_row["calv2_checkpoint_sha256"]):
        raise RuntimeError("CalV2 checkpoint hash changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metadata = load_metadata()
    proteins = load_proteins()
    if tuple(map(str, checkpoint.get("proteins", ()))) != proteins:
        raise RuntimeError("CalV2/main protein order differs")
    fit_ids, eval_ids = load_fold_ids(scenario, fold, cal_row)
    absent = (set(fit_ids) | set(eval_ids)) - set(metadata.index)
    if absent or set(fit_ids) & set(eval_ids):
        raise RuntimeError(f"fold IDs invalid; absent examples={sorted(absent)[:5]}")
    train_ids = set(metadata.index[metadata["split_final"].astype(str).eq("train")])
    if set(fit_ids) | set(eval_ids) != train_ids:
        raise RuntimeError("fold fit+eval does not exactly partition released train IDs")
    fit_metadata = metadata.loc[list(fit_ids)].copy()
    eval_metadata = metadata.loc[list(eval_ids)].copy()
    if not eval_metadata["perturbation_no_concentration"].astype(str).map(
        lambda value: value not in CONTROL_CHEMICALS | {QC_CHEMICAL}
    ).all():
        raise RuntimeError("formal outer-eval surface must contain treatments only")

    y_log2 = load_log2_proteome(metadata, proteins)
    fit_y = y_log2.loc[list(fit_ids)].to_numpy(np.float32)
    # No training function is called before the arrays below have been split.
    # All fit statistics and labels use fit_y exclusively.
    eval_y = y_log2.loc[list(eval_ids)].to_numpy(np.float32)
    target_mean, target_std = fold_target_statistics(fit_y)
    fit_observed = np.isfinite(fit_y)
    fit_target = np.zeros_like(fit_y, np.float32)
    fit_target[fit_observed] = (
        (fit_y - target_mean[None, :]) / target_std[None, :]
    )[fit_observed]

    all_metadata = pd.concat((fit_metadata, eval_metadata), axis=0)
    z_all, cal_decoder, cal_protein_std = encode_calv2(all_metadata, checkpoint)
    z_fit = z_all[: len(fit_metadata)]
    z_eval = z_all[len(fit_metadata) :]
    # Exact current-project absolute carrier: unweighted outer-fit row mean,
    # accumulated in float64 and stored as float32.
    cal_center = z_fit.astype(np.float64).mean(axis=0).astype(np.float32)
    centered_error = float(
        np.max(np.abs((z_fit.astype(np.float64) - cal_center).mean(axis=0)))
    )
    if centered_error > 2e-5:
        raise RuntimeError(f"CalV2 absolute gauge failed: {centered_error}")
    cal_decoder_scaled = (
        cal_decoder * (cal_protein_std / target_std)[:, None]
    ).astype(np.float32)

    feature_state = fit_feature_state(fit_metadata)
    strain_table, chemical_table = load_descriptors()
    fit_features = build_feature_arrays(
        fit_metadata,
        state=feature_state,
        strain_table=strain_table,
        chemical_table=chemical_table,
        z_cal=z_fit,
        cal_center=cal_center,
    )
    eval_features = build_feature_arrays(
        eval_metadata,
        state=feature_state,
        strain_table=strain_table,
        chemical_table=chemical_table,
        z_cal=z_eval,
        cal_center=cal_center,
    )
    control_map = build_control_map(fit_metadata, fit_y)
    response_positions, response_target, response_mask, correction_max = (
        build_response_targets(
            fit_metadata,
            fit_y,
            target_std,
            z_fit,
            cal_decoder,
            cal_protein_std,
            control_map,
        )
    )
    oracle, oracle_has_match = build_oracle_predictions(eval_metadata, control_map)
    # Reuse is enforced behaviorally against the hash-locked, latest-executed
    # Water/DMSO helper selected by the attached prompt.  This prevents the
    # small local CalV2-audit extension above from drifting from its target and
    # oracle surfaces.
    _calv2_python_paths()
    from semantic_feature_engineering_v1.controls import (  # type: ignore
        exact_control_predictions,
        training_fc_targets,
    )

    reference_fc = training_fc_targets(metadata, y_log2, fit_ids, target_std)
    reference_positions = np.flatnonzero(reference_fc.mask.any(axis=1))
    if not np.array_equal(reference_positions, response_positions):
        raise RuntimeError("matched-response row surface differs from frozen helper")
    if not np.array_equal(
        reference_fc.mask[response_positions].astype(bool), response_mask
    ):
        raise RuntimeError("matched-response observation mask differs from frozen helper")
    if not np.allclose(
        reference_fc.values[response_positions], response_target, rtol=0.0, atol=1e-5
    ):
        raise RuntimeError("matched-response target differs from frozen helper")
    reference_oracle, reference_has_match = exact_control_predictions(
        metadata, y_log2, eval_ids, fit_ids
    )
    if not np.array_equal(reference_has_match.to_numpy(bool), oracle_has_match):
        raise RuntimeError("matched-control oracle row surface differs from frozen helper")
    if not np.allclose(
        reference_oracle.to_numpy(np.float32), oracle, equal_nan=True, rtol=0.0, atol=1e-5
    ):
        raise RuntimeError("matched-control oracle values differ from frozen helper")
    # Use the frozen helper's float32/pandas reduction surface byte-for-byte;
    # the local reconstruction above exists solely to audit the CalV2 delta.
    response_target = reference_fc.values[response_positions].astype(np.float32)
    response_mask = reference_fc.mask[response_positions].astype(bool)
    oracle = reference_oracle.to_numpy(np.float32)
    oracle_has_match = reference_has_match.to_numpy(bool)
    control_fit_positions = np.flatnonzero(
        fit_metadata["perturbation_no_concentration"]
        .astype(str)
        .isin(CONTROL_CHEMICALS)
        .to_numpy()
    ).astype(np.int64)
    if len(control_fit_positions) == 0:
        raise RuntimeError("Stage A has no valid control rows")
    return FoldPrepared(
        scenario=scenario,
        fold=fold,
        fit_ids=fit_ids,
        eval_ids=eval_ids,
        proteins=proteins,
        fit_features=fit_features,
        eval_features=eval_features,
        response_fit_positions=response_positions,
        response_target=response_target,
        response_mask=response_mask,
        target_mean=target_mean,
        target_std=target_std,
        fit_target=fit_target,
        fit_mask=fit_observed,
        eval_truth=eval_y,
        eval_truth_mask=np.isfinite(eval_y),
        oracle_prediction=oracle,
        oracle_has_match=oracle_has_match,
        control_fit_positions=control_fit_positions,
        cal_center=cal_center,
        cal_decoder_scaled=cal_decoder_scaled,
        feature_state=feature_state,
        calv2_checkpoint_path=str(checkpoint_path),
        calv2_checkpoint_sha256=checkpoint_hash,
        response_correction_max_abs=correction_max,
    )


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("masked loss tensors must be aligned")
    denominator = mask.sum()
    return ((prediction - target).square() * mask).sum() / denominator


def epoch_batches(
    n_items: int,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    if n_items <= 0:
        raise ValueError("batch source must be nonempty")
    permutation = torch.randperm(n_items, generator=generator, device=device)
    for start in range(0, n_items, batch_size):
        yield permutation[start : start + batch_size]


class CyclicBatches:
    def __init__(
        self,
        n_items: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        if n_items <= 0:
            raise ValueError("cyclic batch source must be nonempty")
        self.n_items = n_items
        self.generator = generator
        self.device = device
        self.permutation = torch.randperm(
            n_items, generator=generator, device=device
        )
        self.position = 0

    def next(self, batch_size: int) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        remaining = batch_size
        while remaining:
            available = self.n_items - self.position
            take = min(remaining, available)
            pieces.append(self.permutation[self.position : self.position + take])
            self.position += take
            remaining -= take
            if self.position == self.n_items:
                self.permutation = torch.randperm(
                    self.n_items, generator=self.generator, device=self.device
                )
                self.position = 0
        return torch.cat(pieces) if len(pieces) > 1 else pieces[0]


def _feature_args(features: TorchFeatureArrays, index: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return features.subset(index)


def train_bcr(
    prepared: FoldPrepared,
    *,
    device: torch.device,
) -> tuple[GOAISemanticBCRV1, pd.DataFrame]:
    set_seed(SEED)
    features = TorchFeatureArrays.from_numpy(prepared.fit_features, device)
    target = torch.as_tensor(prepared.fit_target, device=device)
    mask = torch.as_tensor(prepared.fit_mask.astype(np.float32), device=device)
    control_positions = torch.as_tensor(
        prepared.control_fit_positions, dtype=torch.long, device=device
    )
    response_positions = torch.as_tensor(
        prepared.response_fit_positions, dtype=torch.long, device=device
    )
    response_target = torch.as_tensor(prepared.response_target, device=device)
    response_mask = torch.as_tensor(
        prepared.response_mask.astype(np.float32), device=device
    )
    model = GOAISemanticBCRV1(
        n_medium=len(prepared.feature_state.medium_categories),
        chemical_dim=prepared.fit_features.chemical.shape[1],
        n_proteins=N_PROTEINS,
        cal_center=torch.from_numpy(prepared.cal_center),
        cal_decoder_scaled=torch.from_numpy(prepared.cal_decoder_scaled),
    ).to(device)
    history: list[dict[str, Any]] = []

    # Stage A: valid Water/DMSO controls only.
    optimizer = torch.optim.AdamW(
        [*model.cell.parameters(), *model.background.parameters()],
        lr=1e-3,
        weight_decay=1e-4,
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    model.train()
    for epoch in range(1, 61):
        losses: list[float] = []
        for local in epoch_batches(
            len(control_positions),
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        ):
            index = control_positions[local]
            optimizer.zero_grad(set_to_none=True)
            strain, medium, temperature, time_feature, _chem, z_cal, _flag = (
                _feature_args(features, index)
            )
            prediction = model.background_plus_calibration(
                strain, medium, temperature, time_feature, z_cal
            )
            loss = masked_mse(prediction, target[index], mask[index])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Stage A loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {"stage": "A", "epoch": epoch, "loss": float(np.mean(losses))}
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{prepared.scenario} fold {prepared.fold}] Stage A {epoch:02d}/60 "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )

    # Stage B: matched-control response labels.  Background remains frozen.
    for parameter in model.background.parameters():
        parameter.requires_grad_(False)
    for parameter in model.cell.parameters():
        parameter.requires_grad_(False)
    response_parameters = [
        *model.chemical_adapter.parameters(),
        *model.response.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        response_parameters, lr=1e-3, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    for epoch in range(1, 51):
        if epoch <= 10:
            model.cell.eval()
        elif epoch == 11:
            for parameter in model.cell.parameters():
                parameter.requires_grad_(True)
            optimizer.add_param_group(
                {
                    "params": list(model.cell.parameters()),
                    "lr": 2e-4,
                    "weight_decay": 1e-4,
                }
            )
            model.cell.train()
        else:
            model.cell.train()
        model.chemical_adapter.train()
        model.response.train()
        losses = []
        for local in epoch_batches(
            len(response_positions),
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        ):
            index = response_positions[local]
            optimizer.zero_grad(set_to_none=True)
            strain, medium, temperature, time_feature, chemical, _z, _flag = (
                _feature_args(features, index)
            )
            prediction = model.response_only(
                strain, medium, temperature, time_feature, chemical
            )
            loss = masked_mse(
                prediction, response_target[local], response_mask[local]
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Stage B loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {"stage": "B", "epoch": epoch, "loss": float(np.mean(losses))}
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{prepared.scenario} fold {prepared.fold}] Stage B {epoch:02d}/50 "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )

    # Stage C: equal-weight absolute, control, and matched-response losses.
    for parameter in model.background.parameters():
        parameter.requires_grad_(True)
    for parameter in model.cell.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.cell.parameters()), "lr": 1e-4},
            {
                "params": [
                    *model.background.parameters(),
                    *model.chemical_adapter.parameters(),
                    *model.response.parameters(),
                ],
                "lr": 5e-4,
            },
        ],
        weight_decay=1e-4,
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    model.train()
    n_fit = len(prepared.fit_ids)
    for epoch in range(1, 31):
        control_batches = CyclicBatches(
            len(control_positions), generator=generator, device=device
        )
        response_batches = CyclicBatches(
            len(response_positions), generator=generator, device=device
        )
        epoch_abs: list[float] = []
        epoch_control: list[float] = []
        epoch_response: list[float] = []
        for all_index in epoch_batches(
            n_fit,
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        ):
            current_batch_size = len(all_index)
            control_local = control_batches.next(current_batch_size)
            control_index = control_positions[control_local]
            response_local = response_batches.next(current_batch_size)
            response_index = response_positions[response_local]
            optimizer.zero_grad(set_to_none=True)

            absolute = model(*_feature_args(features, all_index))
            loss_abs = masked_mse(absolute, target[all_index], mask[all_index])

            (
                ctrl_strain,
                ctrl_medium,
                ctrl_temperature,
                ctrl_time,
                _ctrl_chem,
                ctrl_z,
                _ctrl_flag,
            ) = _feature_args(features, control_index)
            control_prediction = model.background_plus_calibration(
                ctrl_strain,
                ctrl_medium,
                ctrl_temperature,
                ctrl_time,
                ctrl_z,
            )
            loss_control = masked_mse(
                control_prediction, target[control_index], mask[control_index]
            )

            (
                resp_strain,
                resp_medium,
                resp_temperature,
                resp_time,
                resp_chemical,
                _resp_z,
                _resp_flag,
            ) = _feature_args(features, response_index)
            response_prediction = model.response_only(
                resp_strain,
                resp_medium,
                resp_temperature,
                resp_time,
                resp_chemical,
            )
            loss_response = masked_mse(
                response_prediction,
                response_target[response_local],
                response_mask[response_local],
            )
            loss = loss_abs + loss_control + loss_response
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Stage C loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            epoch_abs.append(float(loss_abs.detach().cpu()))
            epoch_control.append(float(loss_control.detach().cpu()))
            epoch_response.append(float(loss_response.detach().cpu()))
        row = {
            "stage": "C",
            "epoch": epoch,
            "loss": float(
                np.mean(epoch_abs)
                + np.mean(epoch_control)
                + np.mean(epoch_response)
            ),
            "loss_abs": float(np.mean(epoch_abs)),
            "loss_control": float(np.mean(epoch_control)),
            "loss_response": float(np.mean(epoch_response)),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[{prepared.scenario} fold {prepared.fold}] Stage C {epoch:02d}/30 "
                f"loss={row['loss']:.6f}",
                flush=True,
            )
    return model, pd.DataFrame(history)


def train_flat(
    prepared: FoldPrepared,
    *,
    device: torch.device,
) -> tuple[FlatMLPSameInfo, pd.DataFrame]:
    set_seed(SEED)
    features = torch.as_tensor(
        np.ascontiguousarray(prepared.fit_features.flat), device=device
    )
    target = torch.as_tensor(prepared.fit_target, device=device)
    mask = torch.as_tensor(prepared.fit_mask.astype(np.float32), device=device)
    model = FlatMLPSameInfo(features.shape[1], N_PROTEINS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(SEED)
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, 81):
        losses: list[float] = []
        for index in epoch_batches(
            len(features),
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[index])
            loss = masked_mse(prediction, target[index], mask[index])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite Flat MLP loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {"stage": "FLAT", "epoch": epoch, "loss": float(np.mean(losses))}
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{prepared.scenario} fold {prepared.fold}] Flat {epoch:02d}/80 "
                f"loss={np.mean(losses):.6f}",
                flush=True,
            )
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_bcr(
    model: GOAISemanticBCRV1,
    features: FeatureArrays,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    tensors = TorchFeatureArrays.from_numpy(features, device)
    surfaces: dict[str, list[np.ndarray]] = {
        "full": [],
        "zero_cal": [],
        "zero_resp": [],
    }
    for start in range(0, len(features.strain), batch_size):
        index = torch.arange(
            start, min(start + batch_size, len(features.strain)), device=device
        )
        components = model.forward_components(*_feature_args(tensors, index))
        standardized = {
            "full": components.absolute,
            "zero_cal": components.background + components.response,
            "zero_resp": components.background + components.calibration,
        }
        for name, value in standardized.items():
            physical = (
                value.detach().cpu().numpy().astype(np.float32)
                * target_std[None, :]
                + target_mean[None, :]
            ).astype(np.float32)
            surfaces[name].append(physical)
    return {name: np.concatenate(parts, axis=0) for name, parts in surfaces.items()}


@torch.no_grad()
def predict_flat(
    model: FlatMLPSameInfo,
    features: np.ndarray,
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        selected = torch.as_tensor(
            np.ascontiguousarray(features[start : start + batch_size]), device=device
        )
        standardized = model(selected).detach().cpu().numpy().astype(np.float32)
        parts.append(
            (standardized * target_std[None, :] + target_mean[None, :]).astype(
                np.float32
            )
        )
    return np.concatenate(parts, axis=0)


def per_protein_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    proteins: Sequence[str],
) -> pd.DataFrame:
    actual = np.asarray(truth, np.float64)
    predicted = np.asarray(prediction, np.float64)
    observed = np.asarray(mask, bool)
    if (
        actual.shape != predicted.shape
        or actual.shape != observed.shape
        or actual.shape[1] != len(proteins)
    ):
        raise ValueError("metric surfaces are not aligned")
    rows: list[dict[str, Any]] = []
    for protein_index, protein in enumerate(proteins):
        keep = observed[:, protein_index]
        n_observed = int(keep.sum())
        r2 = float("nan")
        pcc = float("nan")
        if n_observed >= 2:
            y = actual[keep, protein_index]
            p = predicted[keep, protein_index]
            centered_y = y - y.mean()
            tss = float(centered_y @ centered_y)
            if tss > 0:
                r2 = 1.0 - float(np.square(p - y).sum()) / tss
            centered_p = p - p.mean()
            denominator = float(
                np.sqrt((centered_y @ centered_y) * (centered_p @ centered_p))
            )
            if denominator > 0:
                pcc = float(centered_y @ centered_p) / denominator
        rows.append(
            {
                "protein_id": str(protein),
                "n_observed_eval": n_observed,
                "R2": r2,
                "PCC": pcc,
            }
        )
    return pd.DataFrame(rows)


def metric_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    proteins: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    common = np.asarray(mask, bool)
    if common.sum() == 0:
        raise ValueError("metric surface has zero common observations")
    if not np.isfinite(truth[common]).all() or not np.isfinite(prediction[common]).all():
        raise ValueError("metric common mask contains non-finite values")
    per = per_protein_metrics(truth, prediction, common, proteins)
    r2 = per["R2"].to_numpy(np.float64)
    pcc = per["PCC"].to_numpy(np.float64)
    finite_r2 = r2[np.isfinite(r2)]
    finite_pcc = pcc[np.isfinite(pcc)]
    residual = prediction[common].astype(np.float64) - truth[common].astype(np.float64)
    return (
        {
            "metric_protocol": "pooled_oof_per_protein_r2_then_median_v1",
            "n_samples": int(len(truth)),
            "n_scored_rows": int(common.any(axis=1).sum()),
            "n_proteins": int(len(proteins)),
            "n_observed_values": int(common.sum()),
            "n_evaluable_r2_proteins": int(len(finite_r2)),
            "pooled_oof_median_protein_r2": (
                float(np.median(finite_r2)) if len(finite_r2) else float("nan")
            ),
            "median_protein_pcc": (
                float(np.median(finite_pcc)) if len(finite_pcc) else float("nan")
            ),
            "rmse_log2": float(np.sqrt(np.mean(np.square(residual)))),
        },
        per,
    )


def torch_save_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def npz_save_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def fold_output_dir(run_dir: Path, scenario: str, fold: int) -> Path:
    return run_dir / "folds" / scenario / f"fold_{fold}"


def checkpoint_contract(prepared: FoldPrepared, model_name: str) -> dict[str, Any]:
    return {
        "format": "goai.semantic_bcr_v1.fold_checkpoint.v1",
        "model": model_name,
        "scenario": prepared.scenario,
        "fold": prepared.fold,
        "seed": SEED,
        "fit_sample_count": len(prepared.fit_ids),
        "eval_sample_count": len(prepared.eval_ids),
        "fit_sample_ids_sha256": ordered_json_hash(prepared.fit_ids),
        "eval_sample_ids_sha256": ordered_json_hash(prepared.eval_ids),
        "protein_count": len(prepared.proteins),
        "protein_order_sha256": ordered_json_hash(prepared.proteins),
        "calv2_checkpoint_path": prepared.calv2_checkpoint_path,
        "calv2_checkpoint_sha256": prepared.calv2_checkpoint_sha256,
        "calv2_encoder_decoder_frozen": True,
        "op3_frozen": True,
        "strain_descriptor_frozen": True,
        "strain_descriptor_status": "HISTORICAL_RAW4096_EXECUTION_ASSUMPTION_NOT_FORMALLY_VALIDATED",
        "response_contract": "TASK_LOCAL_RESOLUTION_ATTACHED_WATER_DMSO_PLUS_HASH_LOCKED_EXECUTED_8KEY_CONTRACT",
        "response_contract_path": str(MATCHED_CONTROL_CONTRACT_PATH),
        "response_contract_sha256": EXPECTED_HASHES[MATCHED_CONTROL_CONTRACT_PATH],
        "response_implementation_path": str(MATCHED_CONTROL_IMPLEMENTATION_PATH),
        "response_implementation_sha256": EXPECTED_HASHES[
            MATCHED_CONTROL_IMPLEMENTATION_PATH
        ],
        "matched_control_inference_status": "ORACLE_DIAGNOSTIC_ONLY",
        "run_code_sha256": RUN_CODE_SHA256_AT_START,
        "model_code_sha256": MODEL_CODE_SHA256_AT_START,
        "input_hashes": {str(path): digest for path, digest in EXPECTED_HASHES.items()},
        "feature_state": asdict(prepared.feature_state),
        "target_standardization": "outer-fit per-protein log2 mean/std; std floor 0.1",
        "target_mean": prepared.target_mean,
        "target_std": prepared.target_std,
        "calibration_center": prepared.cal_center,
        "response_calv2_correction_max_abs": prepared.response_correction_max_abs,
    }


def train_fold(
    scenario: str,
    fold: int,
    *,
    device_name: str,
    run_dir: Path,
) -> None:
    assert_code_unchanged()
    observed_input_hashes = validate_expected_hashes()
    output = fold_output_dir(run_dir, scenario, fold)
    completion_path = output / "completed.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("run_code_sha256") != RUN_CODE_SHA256_AT_START:
            raise RuntimeError("completed fold was produced by different run.py code")
        if completion.get("model_code_sha256") != MODEL_CODE_SHA256_AT_START:
            raise RuntimeError("completed fold was produced by different model.py code")
        required = {
            "bcr_checkpoint.pt": completion.get("bcr_checkpoint_sha256"),
            "flat_checkpoint.pt": completion.get("flat_checkpoint_sha256"),
            "oof_predictions.npz": completion.get("oof_predictions_sha256"),
            "training_history.csv": completion.get("training_history_sha256"),
            "fold_metrics.json": completion.get("fold_metrics_sha256"),
        }
        if all(
            (output / name).is_file()
            and sha256_file(output / name) == expected
            for name, expected in required.items()
        ):
            print(f"[{scenario} fold {fold}] exact completed fold reused", flush=True)
            return
        raise RuntimeError("fold completion exists but artifacts fail hash validation")
    free_gb = shutil.disk_usage(run_dir.parent if run_dir.parent.exists() else ROOT).free / 2**30
    if free_gb < 30:
        raise RuntimeError(f"training requires >=30 GiB free disk; observed {free_gb:.1f}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[{scenario} fold {fold}] preparing fold-local data", flush=True)
    prepared = prepare_fold(scenario, fold, validate_gate=True)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(
        f"[{scenario} fold {fold}] fit={len(prepared.fit_ids)} "
        f"eval={len(prepared.eval_ids)} controls={len(prepared.control_fit_positions)} "
        f"matched_response={len(prepared.response_fit_positions)} device={device}",
        flush=True,
    )

    bcr, bcr_history = train_bcr(prepared, device=device)
    assert_code_unchanged()
    bcr_predictions = predict_bcr(
        bcr,
        prepared.eval_features,
        target_mean=prepared.target_mean,
        target_std=prepared.target_std,
        device=device,
    )
    bcr = bcr.cpu()
    bcr_checkpoint = output / "bcr_checkpoint.pt"
    torch_save_atomic(
        bcr_checkpoint,
        {
            "contract": checkpoint_contract(prepared, "GOAI-SEMANTIC-BCR-V1"),
            "architecture": (
                "CellState(RAW4096,medium16,temp16,time16)->128; "
                "Background 128-512-4422; frozen CalV2 S+I 12D; "
                "OP3 64-128 adapter; Response 256-512-512-4422"
            ),
            "model_state_dict": bcr.state_dict(),
        },
    )
    del bcr
    if device.type == "cuda":
        torch.cuda.empty_cache()

    flat, flat_history = train_flat(prepared, device=device)
    assert_code_unchanged()
    flat_prediction = predict_flat(
        flat,
        prepared.eval_features.flat,
        target_mean=prepared.target_mean,
        target_std=prepared.target_std,
        device=device,
    )
    flat = flat.cpu()
    flat_checkpoint = output / "flat_checkpoint.pt"
    torch_save_atomic(
        flat_checkpoint,
        {
            "contract": checkpoint_contract(prepared, "FLAT-MLP-SAME-INFO"),
            "architecture": (
                f"LayerNorm({prepared.fit_features.flat.shape[1]})-512-512-4422"
            ),
            "input_dim": prepared.fit_features.flat.shape[1],
            "model_state_dict": flat.state_dict(),
        },
    )
    del flat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    protein_mean_prediction = np.broadcast_to(
        prepared.target_mean[None, :], prepared.eval_truth.shape
    ).copy()
    common_mask = (
        prepared.eval_truth_mask
        & np.isfinite(prepared.oracle_prediction)
        & np.isfinite(bcr_predictions["full"])
        & np.isfinite(flat_prediction)
        & np.isfinite(protein_mean_prediction)
    )
    prediction_surfaces = {
        "GOAI-SEMANTIC-BCR-V1": bcr_predictions["full"],
        "FLAT-MLP-SAME-INFO": flat_prediction,
        "PROTEIN-MEAN": protein_mean_prediction,
        "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": prepared.oracle_prediction,
        "ZERO-CAL": bcr_predictions["zero_cal"],
        "ZERO-RESP": bcr_predictions["zero_resp"],
    }
    metrics: dict[str, Any] = {}
    for name, surface in prediction_surfaces.items():
        summary, _per = metric_summary(
            prepared.eval_truth, surface, common_mask, prepared.proteins
        )
        metrics[name] = summary

    history = pd.concat(
        (
            bcr_history.assign(model="GOAI-SEMANTIC-BCR-V1"),
            flat_history.assign(model="FLAT-MLP-SAME-INFO"),
        ),
        ignore_index=True,
    )
    write_csv(output / "training_history.csv", history)
    predictions_path = output / "oof_predictions.npz"
    npz_save_atomic(
        predictions_path,
        sample_ids=np.asarray(prepared.eval_ids, dtype=np.str_),
        truth=prepared.eval_truth.astype(np.float32),
        common_mask=common_mask.astype(bool),
        bcr=bcr_predictions["full"].astype(np.float32),
        flat=flat_prediction.astype(np.float32),
        protein_mean=protein_mean_prediction.astype(np.float32),
        matched_control_oracle=prepared.oracle_prediction.astype(np.float32),
        zero_cal=bcr_predictions["zero_cal"].astype(np.float32),
        zero_resp=bcr_predictions["zero_resp"].astype(np.float32),
    )
    fold_metrics = {
        "scenario": scenario,
        "fold": fold,
        "primary_is_fold_diagnostic_only": True,
        "metrics": metrics,
        "coverage": {
            "eval_rows": len(prepared.eval_ids),
            "oracle_matched_rows": int(prepared.oracle_has_match.sum()),
            "common_observed_values": int(common_mask.sum()),
            "common_scored_rows": int(common_mask.any(axis=1).sum()),
            "response_fit_rows": len(prepared.response_fit_positions),
            "control_fit_rows": len(prepared.control_fit_positions),
        },
        "contract": checkpoint_contract(prepared, "FOLD_SHARED"),
    }
    assert_code_unchanged()
    write_json(output / "fold_metrics.json", fold_metrics)
    completion = {
        "format": "goai.semantic_bcr_v1.fold_completion.v1",
        "scenario": scenario,
        "fold": fold,
        "seed": SEED,
        "elapsed_seconds": time.time() - started,
        "bcr_checkpoint_sha256": sha256_file(bcr_checkpoint),
        "flat_checkpoint_sha256": sha256_file(flat_checkpoint),
        "oof_predictions_sha256": sha256_file(predictions_path),
        "training_history_sha256": sha256_file(output / "training_history.csv"),
        "fold_metrics_sha256": sha256_file(output / "fold_metrics.json"),
        "run_code_sha256": RUN_CODE_SHA256_AT_START,
        "model_code_sha256": MODEL_CODE_SHA256_AT_START,
        "observed_input_hashes": observed_input_hashes,
    }
    assert_code_unchanged()
    write_json(completion_path, completion)
    print(
        f"[{scenario} fold {fold}] COMPLETE elapsed={completion['elapsed_seconds']:.1f}s "
        f"BCR_fold_R2={metrics['GOAI-SEMANTIC-BCR-V1']['pooled_oof_median_protein_r2']:.6f} "
        f"Flat_fold_R2={metrics['FLAT-MLP-SAME-INFO']['pooled_oof_median_protein_r2']:.6f}",
        flush=True,
    )


def preflight() -> dict[str, Any]:
    observed_hashes = validate_expected_hashes()
    gate = validate_calv2_gate()
    metadata = load_metadata()
    proteins = load_proteins()
    strain, chemical = load_descriptors()
    used_strains = set(metadata["Strains"].astype(str).str.strip())
    used_chemicals = set(
        metadata["perturbation_no_concentration"].astype(str).str.strip()
    )
    if not used_strains.issubset(strain.index) or not used_chemicals.issubset(
        chemical.index
    ):
        raise RuntimeError("descriptor coverage is incomplete")
    assignments = pd.read_csv(ASSIGNMENT_PATH)
    scenario_counts = (
        assignments.loc[assignments["eligible"].astype(bool)]
        .groupby(["scenario", "fold"])
        .size()
        .to_dict()
    )
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < 30:
        raise RuntimeError(f"preflight requires >=30 GiB free; observed={free_gb:.1f}")
    return {
        "status": "PASS_WITH_DECLARED_LIMITS",
        "seed": SEED,
        "protein_count": len(proteins),
        "train_rows": int(metadata["split_final"].astype(str).eq("train").sum()),
        "runnable_scenarios": ["S1", "time_forward"],
        "not_runnable": {
            "S2": "N/A: no legal fold-local CalV2; fold2 rank hard stop",
            "S3": "N/A: canonical 16 folds conflicts with required 4 and no legal CalV2",
        },
        "strain_descriptor_status": (
            "HISTORICAL_RAW4096_EXECUTION_ASSUMPTION_NOT_FORMALLY_VALIDATED"
        ),
        "response_contract": "TASK_LOCAL_RESOLUTION_ATTACHED_WATER_DMSO_PLUS_HASH_LOCKED_EXECUTED_8KEY_CONTRACT",
        "matched_control_status": "ORACLE_DIAGNOSTIC_ONLY",
        "calv2_gate": gate,
        "scenario_eval_counts": {
            f"{scenario}/{fold}": int(count)
            for (scenario, fold), count in scenario_counts.items()
            if scenario in {"S1", "S2", "S3", "time_forward"}
        },
        "free_disk_gib": free_gb,
        "input_hashes": observed_hashes,
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.6f}" if math.isfinite(numeric) else "N/A"


def choose_verdict(primary: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    scenarios = ("S1", "Time")
    bcr = [
        float(primary[scenario]["GOAI-SEMANTIC-BCR-V1"]["pooled_oof_median_protein_r2"])
        for scenario in scenarios
    ]
    flat = [
        float(primary[scenario]["FLAT-MLP-SAME-INFO"]["pooled_oof_median_protein_r2"])
        for scenario in scenarios
    ]
    mean = [
        float(primary[scenario]["PROTEIN-MEAN"]["pooled_oof_median_protein_r2"])
        for scenario in scenarios
    ]
    bcr_over_flat = [candidate > baseline for candidate, baseline in zip(bcr, flat)]
    bcr_over_mean = [candidate > baseline for candidate, baseline in zip(bcr, mean)]
    if all(bcr_over_flat) and all(bcr_over_mean):
        return "DISENTANGLED_MODEL_CLEAR_WIN"
    if any(bcr_over_flat) and any(bcr_over_mean):
        return "DISENTANGLED_MODEL_PARTIAL_WIN"
    if not any(bcr_over_flat) and any(bcr_over_mean):
        return "NO_ADVANTAGE_OVER_FLAT_MLP"
    if not any(bcr_over_flat) and not any(bcr_over_mean):
        return "MODEL_UNDERPERFORMS_SIMPLE_BASELINES"
    return "INCONCLUSIVE"


def make_main_plot(
    primary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_path: Path,
) -> None:
    scenarios = ("S1", "Time")
    labels = {
        "GOAI-SEMANTIC-BCR-V1": "GOAI-SEMANTIC-BCR-V1",
        "FLAT-MLP-SAME-INFO": "Flat MLP (same info)",
        "PROTEIN-MEAN": "Protein Mean",
        "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": "Matched Control (oracle diagnostic)",
    }
    colors = {
        "GOAI-SEMANTIC-BCR-V1": "#CC79A7",
        "FLAT-MLP-SAME-INFO": "#4C78A8",
        "PROTEIN-MEAN": "#9E9E9E",
        "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": "#F2A900",
    }
    hatches = {
        "GOAI-SEMANTIC-BCR-V1": "\\\\",
        "FLAT-MLP-SAME-INFO": "xx",
        "PROTEIN-MEAN": "..",
        "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": "//",
    }
    values = {
        model: [
            float(primary[scenario][model]["pooled_oof_median_protein_r2"])
            for scenario in scenarios
        ]
        for model in MODELS
    }
    fig, axis = plt.subplots(figsize=(12.5, 7.2), dpi=180)
    x = np.arange(len(scenarios), dtype=np.float64)
    width = 0.19
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * width
    all_values: list[float] = []
    for offset, model in zip(offsets, MODELS):
        model_values = values[model]
        all_values.extend(model_values)
        bars = axis.bar(
            x + offset,
            model_values,
            width,
            label=labels[model],
            color=colors[model],
            edgecolor="#444444",
            linewidth=0.8,
            hatch=hatches[model],
            zorder=3,
        )
        for bar, value in zip(bars, model_values):
            axis.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5 if value >= 0 else -13),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                rotation=90,
            )
    minimum = min(all_values)
    maximum = max(all_values)
    span = max(maximum - min(0.0, minimum), 0.05)
    axis.set_ylim(min(0.0, minimum - 0.10 * span), maximum + 0.22 * span)
    axis.set_xticks(x, scenarios, fontsize=12)
    axis.set_ylabel("Pooled OOF median protein R²", fontsize=12)
    axis.set_title(
        "Disentangled Semantic B+C+R vs Simple Baselines",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    axis.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.015,
        "Seed 42; pooled across 4 canonical outer folds before per-protein R². "
        "S2/S3 are N/A (no legal fold-local CalV2). Oracle bar uses prediction-time-unavailable control proteomes.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fig.savefig(temporary, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    temporary.replace(output_path)


def build_report(
    primary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fold_medians: Mapping[str, Mapping[str, list[float]]],
    ablations: Mapping[str, Mapping[str, Any]],
    verdict: str,
    improvement: Mapping[str, Mapping[str, float]],
) -> str:
    def score(model: str, scenario: str) -> str:
        if scenario in {"S2", "S3"}:
            return "N/A"
        return _format_metric(primary[scenario][model]["pooled_oof_median_protein_r2"])

    bcr_scores = {
        scenario: float(
            primary[scenario]["GOAI-SEMANTIC-BCR-V1"][
                "pooled_oof_median_protein_r2"
            ]
        )
        for scenario in ("S1", "Time")
    }
    best = max(bcr_scores, key=bcr_scores.get)
    worst = min(bcr_scores, key=bcr_scores.get)
    beats_flat = all(
        bcr_scores[scenario]
        > float(
            primary[scenario]["FLAT-MLP-SAME-INFO"][
                "pooled_oof_median_protein_r2"
            ]
        )
        for scenario in bcr_scores
    )
    beats_mean = all(
        bcr_scores[scenario]
        > float(primary[scenario]["PROTEIN-MEAN"]["pooled_oof_median_protein_r2"])
        for scenario in bcr_scores
    )
    beats_oracle = all(
        bcr_scores[scenario]
        > float(
            primary[scenario]["MATCHED-CONTROL-ORACLE-DIAGNOSTIC"][
                "pooled_oof_median_protein_r2"
            ]
        )
        for scenario in bcr_scores
    )
    if verdict == "DISENTANGLED_MODEL_CLEAR_WIN":
        conclusion = (
            "在目前两个可合法运行的严格 OOF 场景中，解耦 B+C+R 一致胜过同信息 Flat MLP，"
            "值得继续，但必须先解决菌株语义资产未验证这一根本不确定性。"
        )
    elif verdict == "DISENTANGLED_MODEL_PARTIAL_WIN":
        conclusion = (
            "解耦 B+C+R 只在部分合法 OOF 场景优于同信息 Flat MLP，当前证据支持定向诊断而不支持扩大架构搜索。"
        )
    elif verdict == "NO_ADVANTAGE_OVER_FLAT_MLP":
        conclusion = (
            "解耦 B+C+R 在合法 OOF 场景没有显示出相对同信息 Flat MLP 的优势，因此不应继续围绕架构复杂化优化。"
        )
    elif verdict == "MODEL_UNDERPERFORMS_SIMPLE_BASELINES":
        conclusion = (
            "解耦 B+C+R 连简单基线也未稳定超过，当前体系不值得继续优化，除非先替换未经验证的菌株描述符。"
        )
    else:
        conclusion = (
            "由于只有两个场景可合法运行且菌株描述符未获语义验证，目前无法对解耦 B+C+R 的研究价值作确定结论。"
        )

    lines = [
        "# GOAI-SEMANTIC-BCR-V1 — One-minute result",
        "",
        "## Final model",
        "",
        "Calibration = frozen fold-local CalV2 source+instrument 12D; exact outer-fit mean gauge; fixed decoder",
        "CellState = HISTORICAL RAW4096 (frozen, complete coverage, **not formally validated**) + medium + temperature + canonical log-time",
        "Chemical semantic = frozen OP3 64D → trainable 128D adapter",
        "Response supervision = prompt-selected exact 8-key pooled Water/DMSO matched-control log2 delta; CalV2 correction audited",
        "Plate/Well = excluded from model inputs (plate used only in the frozen matching key)",
        "State×Tech interaction = none",
        "",
        "## Primary comparison",
        "",
        "| Model | S1 P-R² | S2 P-R² | S3 P-R² | Time P-R² |",
        "|---|---:|---:|---:|---:|",
        f"| GOAI-SEMANTIC-BCR-V1 | {score('GOAI-SEMANTIC-BCR-V1', 'S1')} | N/A | N/A | {score('GOAI-SEMANTIC-BCR-V1', 'Time')} |",
        f"| FLAT-MLP-SAME-INFO | {score('FLAT-MLP-SAME-INFO', 'S1')} | N/A | N/A | {score('FLAT-MLP-SAME-INFO', 'Time')} |",
        f"| PROTEIN-MEAN | {score('PROTEIN-MEAN', 'S1')} | N/A | N/A | {score('PROTEIN-MEAN', 'Time')} |",
        f"| MATCHED-CONTROL (oracle diagnostic) | {score('MATCHED-CONTROL-ORACLE-DIAGNOSTIC', 'S1')} | N/A | N/A | {score('MATCHED-CONTROL-ORACLE-DIAGNOSTIC', 'Time')} |",
        "",
        "## Main answer",
        "",
        f"Does the disentangled model beat flat MLP? = {'YES on both runnable scenarios' if beats_flat else 'NO, not on both runnable scenarios'}",
        f"Does it beat protein mean? = {'YES on both runnable scenarios' if beats_mean else 'NO, not on both runnable scenarios'}",
        f"Does it beat matched control? = {'YES' if beats_oracle else 'NO'} (mechanism diagnostic only; oracle is not deployable)",
        f"Best scenario = {best}",
        f"Worst scenario = {worst}",
        "",
        "## Branch sanity",
        "",
        f"FULL = {_format_metric(ablations['FULL']['pooled_oof_median_protein_r2'])}",
        f"ZERO-CAL = {_format_metric(ablations['ZERO-CAL']['pooled_oof_median_protein_r2'])}",
        f"ZERO-RESP = {_format_metric(ablations['ZERO-RESP']['pooled_oof_median_protein_r2'])}",
        "",
        "## Final verdict",
        "",
        verdict,
        "",
        "## One-sentence scientific conclusion",
        "",
        f"> {conclusion}",
        "",
        "---",
        "",
        "## 1. Exact architecture",
        "",
        "```text",
        "RAW4096 strain ─┐",
        "medium/temp/time ├─> CellStateEncoder(128) ─┬─> BackgroundHead ─> B",
        "                 ┘                           └─┐",
        "frozen OP3(64) ─> adapter(128) ───────────────┴─> ResponseHead ─> R",
        "source+instrument ─> frozen fold-local CalV2(12) ─> fixed decoder ─> C",
        "Y_hat = B + C + I_treatment × R   (R is structurally zero for controls)",
        "```",
        "",
        "RAW4096: `data/strain_embeddings.csv`, SHA256 `a8351006…`, exact `Strains→strain_name` mapping; provenance manifest SHA256 `b751038d…` records source/model/version/license/acquisition time as unknown, and DHY210 equals BY4741. OP3: 64D TSV SHA256 `58f1ed8a…`, manifest SHA256 `93cd9c5b…`, frozen encoder SHA256 `c8d9091b…`, exact metadata chemical→`raw_name` mapping, upstream Open Problems pseudobulk DE under CC BY 4.0; explicit acquisition time absent (artifact materialized 2026-08-15).",
        "",
        "## 2. Training contract",
        "",
        "Seed 42 only. For every legal fold, all target moments and feature statistics are outer-fit-only. Stage A trains CellState+Background on Water/DMSO controls for 60 epochs (AdamW 1e-3, wd 1e-4). Stage B trains OP3 adapter+Response for 50 epochs; CellState is frozen for epochs 1–10 then uses lr 2e-4 while response uses 1e-3; Background is frozen. Stage C uses every outer-fit row for 30 epochs with equal normalized absolute/control/response losses, CellState lr 1e-4 and heads lr 5e-4. Flat MLP uses the identical information budget for 80 epochs at 1e-3/wd 1e-4. The attached task prompt's explicit Water/DMSO choice is the task-local response authority; a later repository research track instead froze Water-only, so the two must not be conflated.",
        "",
        "Four canonical folds were run for S1 and time-forward. S2 is N/A because fold-local CalV2 is unidentifiable in fold 2 (14/17 technical contexts; rank 8<9). S3 is N/A because the canonical split has 16 rather than 4 folds and folds 8–11 have the same CalV2 hard stop. Production CalV2, zero fill, fold merging, and split regeneration were not used.",
        "",
        "## 3. Baselines",
        "",
        "PROTEIN-MEAN is the outer-fit-only per-protein log2 mean. FLAT-MLP-SAME-INFO concatenates frozen RAW4096, medium one-hot, standardized temperature/log-time, frozen OP3, centered CalV2 z, and treatment flag, then uses the fixed 512–512 MLP. MATCHED-CONTROL uses the exact outer-fit 8-key Water/DMSO log2 mean. The blind test exposes metadata but not control proteomes, so it is labeled **MATCHED-CONTROL-ORACLE-DIAGNOSTIC** and is not a deployable peer baseline.",
        "",
        "All four surfaces are scored on the same eval sample IDs, the same ordered 4,422 proteins, and one shared finite mask defined by truth plus oracle availability.",
        "",
        "## 4. Main results",
        "",
        "![Main grouped comparison](GOAI_SEMANTIC_BCR_V1_MAIN_COMPARISON.png)",
        "",
        "Primary scores pool all four OOF folds first, then compute one R² per protein and take the median. Fold medians below are diagnostics only.",
        "",
        "| Scenario | Model | Pooled P-R² | Protein PCC median | RMSE (log2) | Fold medians |",
        "|---|---|---:|---:|---:|---|",
    ]
    for scenario in ("S1", "Time"):
        for model in MODELS:
            summary = primary[scenario][model]
            diagnostic = ", ".join(
                f"{value:.4f}" for value in fold_medians[scenario][model]
            )
            lines.append(
                f"| {scenario} | {model} | {_format_metric(summary['pooled_oof_median_protein_r2'])} "
                f"| {_format_metric(summary['median_protein_pcc'])} "
                f"| {_format_metric(summary['rmse_log2'])} | {diagnostic} |"
            )
    lines.extend(
        [
            "",
            "## 5. Branch sanity",
            "",
            f"On pooled S1 OOF, FULL={_format_metric(ablations['FULL']['pooled_oof_median_protein_r2'])}, ZERO-CAL={_format_metric(ablations['ZERO-CAL']['pooled_oof_median_protein_r2'])}, and ZERO-RESP={_format_metric(ablations['ZERO-RESP']['pooled_oof_median_protein_r2'])}. These are inference-only ablations of the saved fold models; no retraining occurred.",
            "",
            "## 6. Scientific interpretation",
            "",
            f"Across runnable scenarios, BCR−Flat pooled P-R² is {improvement['S1']['median_score_delta']:+.6f} on S1 and {improvement['Time']['median_score_delta']:+.6f} on Time. The fraction of evaluable proteins whose R² improves is {improvement['S1']['fraction_proteins_improved']:.3f} and {improvement['Time']['fraction_proteins_improved']:.3f}, respectively. Calibration contribution is measured by FULL−ZERO-CAL={float(ablations['FULL']['pooled_oof_median_protein_r2']) - float(ablations['ZERO-CAL']['pooled_oof_median_protein_r2']):+.6f}; response contribution is FULL−ZERO-RESP={float(ablations['FULL']['pooled_oof_median_protein_r2']) - float(ablations['ZERO-RESP']['pooled_oof_median_protein_r2']):+.6f}.",
            "",
            "The interpretation is bounded: half the requested scenarios are legally unavailable, and RAW4096 is a coverage-complete historical fallback that failed prior real-vs-zero/shuffle validation. A negative result therefore tests this exact fixed assembly, not every possible disentangled model.",
            "",
            "**SINGLE BEST NEXT ACTION:** obtain or experimentally validate a correctly mapped strain descriptor (especially DHY210) before any further B+C+R architecture optimization.",
            "",
            "EXPERIMENT_COMPLETE",
            "STOP",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate(run_dir: Path) -> None:
    assert_code_unchanged()
    observed_input_hashes = validate_expected_hashes()
    run_dir.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins()
    primary: dict[str, dict[str, dict[str, Any]]] = {}
    fold_medians: dict[str, dict[str, list[float]]] = {}
    improvement: dict[str, dict[str, float]] = {}
    result_rows: list[dict[str, Any]] = []
    for internal_scenario, display_scenario in (("S1", "S1"), ("time_forward", "Time")):
        fold_arrays: list[dict[str, np.ndarray]] = []
        fold_metric_payloads: list[dict[str, Any]] = []
        for fold in range(4):
            directory = fold_output_dir(run_dir, internal_scenario, fold)
            completion_path = directory / "completed.json"
            if not completion_path.is_file():
                raise RuntimeError(f"fold incomplete: {internal_scenario}/{fold}")
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if completion.get("run_code_sha256") != RUN_CODE_SHA256_AT_START:
                raise RuntimeError(f"run.py code hash differs for {internal_scenario}/{fold}")
            if completion.get("model_code_sha256") != MODEL_CODE_SHA256_AT_START:
                raise RuntimeError(f"model.py code hash differs for {internal_scenario}/{fold}")
            if completion.get("observed_input_hashes") != observed_input_hashes:
                raise RuntimeError(f"input hashes differ for {internal_scenario}/{fold}")
            predictions_path = directory / "oof_predictions.npz"
            if sha256_file(predictions_path) != completion["oof_predictions_sha256"]:
                raise RuntimeError(f"fold OOF hash mismatch: {internal_scenario}/{fold}")
            if sha256_file(directory / "fold_metrics.json") != completion["fold_metrics_sha256"]:
                raise RuntimeError(f"fold metrics hash mismatch: {internal_scenario}/{fold}")
            if sha256_file(directory / "bcr_checkpoint.pt") != completion["bcr_checkpoint_sha256"]:
                raise RuntimeError(f"BCR checkpoint hash mismatch: {internal_scenario}/{fold}")
            if sha256_file(directory / "flat_checkpoint.pt") != completion["flat_checkpoint_sha256"]:
                raise RuntimeError(f"Flat checkpoint hash mismatch: {internal_scenario}/{fold}")
            with np.load(predictions_path, allow_pickle=False) as loaded:
                fold_arrays.append({name: loaded[name].copy() for name in loaded.files})
            fold_metric_payloads.append(
                json.loads((directory / "fold_metrics.json").read_text(encoding="utf-8"))
            )
        sample_ids = np.concatenate([row["sample_ids"] for row in fold_arrays])
        if len(sample_ids) != len(set(map(str, sample_ids))):
            raise RuntimeError(f"pooled OOF IDs are not unique for {internal_scenario}")
        expected_count = 5078 if internal_scenario == "S1" else 891
        if len(sample_ids) != expected_count:
            raise RuntimeError(
                f"pooled OOF count differs for {internal_scenario}: {len(sample_ids)}"
            )
        pooled = {
            name: np.concatenate([row[name] for row in fold_arrays], axis=0)
            for name in (
                "truth",
                "common_mask",
                "bcr",
                "flat",
                "protein_mean",
                "matched_control_oracle",
                "zero_cal",
                "zero_resp",
            )
        }
        model_surface = {
            "GOAI-SEMANTIC-BCR-V1": pooled["bcr"],
            "FLAT-MLP-SAME-INFO": pooled["flat"],
            "PROTEIN-MEAN": pooled["protein_mean"],
            "MATCHED-CONTROL-ORACLE-DIAGNOSTIC": pooled[
                "matched_control_oracle"
            ],
        }
        primary[display_scenario] = {}
        fold_medians[display_scenario] = {}
        per_tables: list[pd.DataFrame] = []
        for model, surface in model_surface.items():
            summary, per = metric_summary(
                pooled["truth"], surface, pooled["common_mask"], proteins
            )
            primary[display_scenario][model] = summary
            per.insert(0, "model", model)
            per_tables.append(per)
            diagnostics = [
                float(payload["metrics"][model]["pooled_oof_median_protein_r2"])
                for payload in fold_metric_payloads
            ]
            fold_medians[display_scenario][model] = diagnostics
            result_rows.append(
                {
                    "kind": "primary",
                    "scenario": display_scenario,
                    "model": model,
                    "status": "OK",
                    **summary,
                    "fold_medians_diagnostic": ";".join(
                        f"{value:.8f}" for value in diagnostics
                    ),
                    "reason": "",
                }
            )
        per_frame = pd.concat(per_tables, ignore_index=True)
        write_csv(run_dir / "oof" / f"{internal_scenario}_per_protein_metrics.csv", per_frame)
        npz_save_atomic(
            run_dir / "oof" / f"{internal_scenario}_oof_predictions.npz",
            sample_ids=sample_ids.astype(np.str_),
            **pooled,
        )
        bcr_per = per_tables[0][["protein_id", "R2"]].rename(
            columns={"R2": "bcr_r2"}
        )
        flat_per = per_tables[1][["protein_id", "R2"]].rename(
            columns={"R2": "flat_r2"}
        )
        merged = bcr_per.merge(flat_per, on="protein_id", validate="one_to_one")
        finite = np.isfinite(merged[["bcr_r2", "flat_r2"]].to_numpy(np.float64)).all(
            axis=1
        )
        delta = (
            merged.loc[finite, "bcr_r2"].to_numpy(np.float64)
            - merged.loc[finite, "flat_r2"].to_numpy(np.float64)
        )
        improvement[display_scenario] = {
            "median_score_delta": float(
                primary[display_scenario]["GOAI-SEMANTIC-BCR-V1"][
                    "pooled_oof_median_protein_r2"
                ]
                - primary[display_scenario]["FLAT-MLP-SAME-INFO"][
                    "pooled_oof_median_protein_r2"
                ]
            ),
            "median_paired_protein_delta_r2": float(np.median(delta)),
            "fraction_proteins_improved": float(np.mean(delta > 0)),
            "n_evaluable_proteins": int(len(delta)),
        }

    for scenario, reason in (
        ("S2", "N/A: no legal fold-local CalV2; canonical fold 2 is rank-deficient"),
        (
            "S3",
            "N/A: canonical split has 16 not 4 folds and CalV2 folds 8-11 are rank-deficient",
        ),
    ):
        for model in MODELS:
            result_rows.append(
                {
                    "kind": "primary",
                    "scenario": scenario,
                    "model": model,
                    "status": "N/A",
                    "metric_protocol": "pooled_oof_per_protein_r2_then_median_v1",
                    "n_samples": np.nan,
                    "n_scored_rows": np.nan,
                    "n_proteins": N_PROTEINS,
                    "n_observed_values": np.nan,
                    "n_evaluable_r2_proteins": np.nan,
                    "pooled_oof_median_protein_r2": np.nan,
                    "median_protein_pcc": np.nan,
                    "rmse_log2": np.nan,
                    "fold_medians_diagnostic": "",
                    "reason": reason,
                }
            )

    # The only permitted inference ablations are pooled S1 FULL/ZERO-CAL/ZERO-RESP.
    s1_oof = np.load(run_dir / "oof" / "S1_oof_predictions.npz", allow_pickle=False)
    ablations: dict[str, dict[str, Any]] = {}
    for name, array_name in (
        ("FULL", "bcr"),
        ("ZERO-CAL", "zero_cal"),
        ("ZERO-RESP", "zero_resp"),
    ):
        summary, _per = metric_summary(
            s1_oof["truth"], s1_oof[array_name], s1_oof["common_mask"], proteins
        )
        ablations[name] = summary
        result_rows.append(
            {
                "kind": "inference_ablation",
                "scenario": "S1",
                "model": name,
                "status": "OK",
                **summary,
                "fold_medians_diagnostic": "",
                "reason": "zero-retrain inference ablation",
            }
        )
    s1_oof.close()

    results = pd.DataFrame(result_rows)
    preferred_columns = [
        "kind",
        "scenario",
        "model",
        "status",
        "pooled_oof_median_protein_r2",
        "median_protein_pcc",
        "rmse_log2",
        "n_samples",
        "n_scored_rows",
        "n_proteins",
        "n_observed_values",
        "n_evaluable_r2_proteins",
        "metric_protocol",
        "fold_medians_diagnostic",
        "reason",
    ]
    results = results.reindex(columns=preferred_columns)
    results_path = run_dir / "GOAI_SEMANTIC_BCR_V1_RESULTS.csv"
    write_csv(results_path, results)
    verdict = choose_verdict(primary)
    chart_path = run_dir / "GOAI_SEMANTIC_BCR_V1_MAIN_COMPARISON.png"
    make_main_plot(primary, chart_path)
    report = build_report(
        primary, fold_medians, ablations, verdict, improvement
    )
    report_path = run_dir / "GOAI_SEMANTIC_BCR_V1_REPORT.md"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    write_json(
        run_dir / "completed.json",
        {
            "format": "goai.semantic_bcr_v1.experiment_completion.v1",
            "status": "EXPERIMENT_COMPLETE",
            "seed": SEED,
            "runnable_scenarios": ["S1", "time_forward"],
            "not_run_scenarios": ["S2", "S3"],
            "verdict": verdict,
            "results_sha256": sha256_file(results_path),
            "report_sha256": sha256_file(report_path),
            "figure_sha256": sha256_file(chart_path),
            "matched_control_status": "ORACLE_DIAGNOSTIC_ONLY",
            "strain_descriptor_status": (
                "HISTORICAL_RAW4096_EXECUTION_ASSUMPTION_NOT_FORMALLY_VALIDATED"
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "EXPERIMENT_COMPLETE",
                "verdict": verdict,
                "results": str(results_path),
                "report": str(report_path),
                "figure": str(chart_path),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("preflight", "train-fold", "aggregate")
    )
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--fold", type=int, choices=range(4))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        print(json.dumps(safe_json(preflight()), ensure_ascii=False, indent=2))
    elif args.command == "train-fold":
        if args.scenario is None or args.fold is None:
            raise SystemExit("train-fold requires --scenario and --fold")
        train_fold(
            args.scenario,
            args.fold,
            device_name=args.device,
            run_dir=args.run_dir.resolve(),
        )
    elif args.command == "aggregate":
        aggregate(args.run_dir.resolve())
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
