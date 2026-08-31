"""Predictors for matched-control response in natural log2-delta space."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ModelKind = Literal["direct", "low_rank"]


class DropoutMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@runtime_checkable
class FittedPredictor(Protocol):
    """A fitted response predictor available to evaluation and acquisition."""

    def predict(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        ...

    def uncertainty(
        self,
        features: np.ndarray,
        passes: int,
        batch_size: int = 512,
    ) -> np.ndarray:
        ...

    def fit_summary(self) -> dict[str, str | int | float | None]:
        ...


@runtime_checkable
class Predictor(Protocol):
    """A predictor whose fit is isolated to the response supplied to that call."""

    def fit(
        self,
        features: np.ndarray,
        response: np.ndarray,
        seed: int,
    ) -> FittedPredictor:
        ...


def _natural_delta_latent_gram(
    basis: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Return the protein-mean natural-delta Gram matrix in latent space."""

    if basis.ndim != 2 or target_scale.ndim != 1:
        raise ValueError("basis must be two-dimensional and target_scale one-dimensional")
    if basis.shape[1] != target_scale.shape[0] or target_scale.numel() == 0:
        raise ValueError("basis and target_scale protein dimensions must agree and be nonempty")
    scaled_basis = basis * target_scale.unsqueeze(0)
    return scaled_basis @ scaled_basis.transpose(0, 1) / target_scale.numel()


def _mean_latent_gram_variance(
    latent_draws: torch.Tensor,
    latent_gram: torch.Tensor,
) -> torch.Tensor:
    """Mean per-protein sample variance without reconstructing protein draws."""

    if latent_draws.ndim != 3:
        raise ValueError("latent_draws must have draw, row, and latent dimensions")
    if latent_draws.shape[0] < 2:
        raise ValueError("At least two latent draws are required")
    if latent_gram.shape != (latent_draws.shape[2], latent_draws.shape[2]):
        raise ValueError("latent_gram dimensions do not match latent draws")
    centered = latent_draws - latent_draws.mean(dim=0, keepdim=True)
    summed_quadratic = torch.einsum(
        "dbr,rs,dbs->b", centered, latent_gram, centered
    )
    return summed_quadratic / (latent_draws.shape[0] - 1)


@dataclass(frozen=True)
class ModelSettings:
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    target_scale_floor: float
    device: str
    kind: ModelKind = "low_rank"
    response_rank: int = 64
    svd_niter: int = 2


def _validate_feature_matrix(
    features: np.ndarray,
    *,
    expected_columns: int | None = None,
    allow_empty_rows: bool = False,
) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError("Features must be a two-dimensional matrix")
    if values.shape[1] == 0:
        raise ValueError("Features must contain at least one column")
    if not allow_empty_rows and values.shape[0] == 0:
        raise ValueError("Features must contain at least one row")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(
            f"Feature width is {values.shape[1]}, expected {expected_columns}"
        )
    try:
        finite = np.isfinite(values)
    except TypeError as exc:
        raise ValueError("Features must be numeric and finite") from exc
    if not finite.all():
        raise ValueError("Features must contain only finite values")
    converted = np.asarray(values, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("Features must be finite when represented as float32")
    return converted


def _validate_settings(settings: ModelSettings) -> None:
    if settings.kind not in ("direct", "low_rank"):
        raise ValueError("Model kind must be 'direct' or 'low_rank'")
    if (
        isinstance(settings.hidden_dim, bool)
        or not isinstance(settings.hidden_dim, (int, np.integer))
        or settings.hidden_dim <= 0
    ):
        raise ValueError("hidden_dim must be a positive integer")
    if not np.isfinite(settings.dropout) or not 0.0 <= settings.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not np.isfinite(settings.learning_rate) or settings.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(settings.weight_decay) or settings.weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if (
        isinstance(settings.epochs, bool)
        or not isinstance(settings.epochs, (int, np.integer))
        or settings.epochs <= 0
    ):
        raise ValueError("epochs must be a positive integer")
    if (
        isinstance(settings.batch_size, bool)
        or not isinstance(settings.batch_size, (int, np.integer))
        or settings.batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(settings.target_scale_floor) or settings.target_scale_floor <= 0.0:
        raise ValueError("target_scale_floor must be finite and positive")
    if (
        isinstance(settings.response_rank, bool)
        or not isinstance(settings.response_rank, (int, np.integer))
        or settings.response_rank <= 0
    ):
        raise ValueError("response_rank must be a positive integer")
    if (
        isinstance(settings.svd_niter, bool)
        or not isinstance(settings.svd_niter, (int, np.integer))
        or settings.svd_niter < 0
    ):
        raise ValueError("svd_niter must be a nonnegative integer")


@dataclass
class ResponseFit:
    """Compatibility fit object implementing the v2 fitted-predictor contract."""

    model: DropoutMLP
    target_mean: np.ndarray
    target_scale: np.ndarray
    device: torch.device
    final_loss: float
    kind: ModelKind = "direct"
    basis: np.ndarray | None = None
    basis_hash: str | None = None
    explained_energy: float | None = None
    seed: int = 0
    input_dim: int = 0
    n_train: int = 0
    n_observed_values: int = 0
    requested_response_rank: int | None = None

    @property
    def response_rank(self) -> int:
        return 0 if self.basis is None else int(self.basis.shape[0])

    @property
    def summary(self) -> dict[str, str | int | float | None]:
        """A JSON-safe fit receipt."""
        return self.fit_summary()

    def fit_summary(self) -> dict[str, str | int | float | None]:
        """Return metadata without arrays, tensors, devices, NaN, or infinity."""
        final_loss = float(self.final_loss) if np.isfinite(self.final_loss) else None
        explained = (
            float(self.explained_energy)
            if self.explained_energy is not None and np.isfinite(self.explained_energy)
            else None
        )
        return {
            "kind": str(self.kind),
            "seed": int(self.seed),
            "n_train": int(self.n_train),
            "n_features": int(self._input_dim()),
            "n_proteins": int(self.target_mean.shape[0]),
            "n_observed_values": int(self.n_observed_values),
            "requested_response_rank": (
                None
                if self.requested_response_rank is None
                else int(self.requested_response_rank)
            ),
            "response_rank": int(self.response_rank),
            "basis_hash": self.basis_hash,
            "explained_energy": explained,
            "final_loss": final_loss,
        }

    def _input_dim(self) -> int:
        if self.input_dim > 0:
            return int(self.input_dim)
        first_layer = self.model.network[0]
        if not isinstance(first_layer, nn.Linear):
            raise TypeError("DropoutMLP does not start with a linear layer")
        return int(first_layer.in_features)

    def _basis_tensor(self) -> torch.Tensor | None:
        if self.basis is None:
            return None
        return torch.from_numpy(self.basis).to(self.device)

    def _reconstruct(
        self,
        model_output: torch.Tensor,
        basis: torch.Tensor | None,
    ) -> torch.Tensor:
        standardised = model_output if basis is None else model_output @ basis
        mean = torch.as_tensor(self.target_mean, device=self.device)
        scale = torch.as_tensor(self.target_scale, device=self.device)
        return standardised * scale + mean

    def predict(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        values = _validate_feature_matrix(
            features,
            expected_columns=self._input_dim(),
            allow_empty_rows=True,
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(values) == 0:
            return np.empty((0, self.target_mean.shape[0]), dtype=np.float32)
        self.model.eval()
        basis = self._basis_tensor()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                inputs = torch.from_numpy(values[start : start + batch_size]).to(self.device)
                reconstructed = self._reconstruct(self.model(inputs), basis)
                outputs.append(reconstructed.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def uncertainty(
        self,
        features: np.ndarray,
        passes: int,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Mean MC-dropout variance across reconstructed natural-delta proteins."""
        values = _validate_feature_matrix(
            features,
            expected_columns=self._input_dim(),
            allow_empty_rows=True,
        )
        if isinstance(passes, bool) or not isinstance(passes, (int, np.integer)) or passes < 2:
            raise ValueError("MC-dropout uncertainty requires at least two integer passes")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scores = np.empty(len(values), dtype=np.float64)
        if len(values) == 0:
            return scores

        cuda_devices: list[int] = []
        if self.device.type == "cuda":
            cuda_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        basis = self._basis_tensor()
        latent_gram = None
        if basis is not None:
            target_scale = torch.as_tensor(self.target_scale, device=self.device)
            latent_gram = _natural_delta_latent_gram(basis, target_scale)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            self.model.train()
            try:
                with torch.no_grad():
                    for start in range(0, len(values), batch_size):
                        block = torch.from_numpy(values[start : start + batch_size]).to(self.device)
                        if latent_gram is not None:
                            latent_draws = torch.stack(
                                [self.model(block) for _ in range(passes)], dim=0
                            )
                            variance_score = _mean_latent_gram_variance(
                                latent_draws, latent_gram
                            )
                            scores[start : start + len(block)] = (
                                variance_score.cpu().numpy()
                            )
                            continue
                        running_mean: torch.Tensor | None = None
                        running_m2: torch.Tensor | None = None
                        for draw in range(1, passes + 1):
                            reconstructed = self._reconstruct(self.model(block), basis)
                            if running_mean is None:
                                running_mean = reconstructed.clone()
                                running_m2 = torch.zeros_like(reconstructed)
                            else:
                                delta = reconstructed - running_mean
                                running_mean = running_mean + delta / draw
                                running_m2 = running_m2 + delta * (
                                    reconstructed - running_mean
                                )
                        assert running_m2 is not None
                        variance = running_m2 / (passes - 1)
                        scores[start : start + len(block)] = (
                            variance.mean(dim=1).cpu().numpy()
                        )
            finally:
                self.model.eval()
        return scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _fit_local_targets(
    response: np.ndarray,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(response)
    counts = observed.sum(axis=0)
    finite_response = np.where(observed, response, 0.0).astype(np.float64)
    target_mean64 = np.divide(
        finite_response.sum(axis=0),
        counts,
        out=np.zeros(response.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    centered = np.where(observed, finite_response - target_mean64, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.zeros(response.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    target_scale64 = np.maximum(np.sqrt(variance), scale_floor)
    standardised = np.where(
        observed,
        centered / target_scale64,
        0.0,
    ).astype(np.float32)
    natural_targets = finite_response.astype(np.float32)
    return (
        target_mean64.astype(np.float32),
        target_scale64.astype(np.float32),
        standardised,
        natural_targets,
        observed.astype(np.float32),
    )


def _response_basis(
    standardised: np.ndarray,
    requested_rank: int,
    niter: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    rank = min(requested_rank, standardised.shape[0], standardised.shape[1])
    matrix = torch.from_numpy(standardised).to(device)
    _, singular_values, right_vectors = torch.pca_lowrank(
        matrix,
        q=rank,
        center=False,
        niter=niter,
    )
    basis = right_vectors.transpose(0, 1).contiguous().cpu().numpy().astype(
        np.float32, copy=False
    )
    total_energy = float(matrix.square().sum().cpu())
    if total_energy == 0.0:
        explained_energy = 0.0
    else:
        retained_energy = float(singular_values.square().sum().cpu())
        explained_energy = float(np.clip(retained_energy / total_energy, 0.0, 1.0))
    return basis, explained_energy


def fit_response_model(
    features: np.ndarray,
    response: np.ndarray,
    settings: ModelSettings,
    seed: int,
) -> ResponseFit:
    """Fit a fresh direct or low-rank model using only the supplied response."""
    _validate_settings(settings)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    feature_values = _validate_feature_matrix(features)
    response_values = np.asarray(response)
    if response_values.ndim != 2:
        raise ValueError("Response must be a two-dimensional matrix")
    if response_values.shape[0] != feature_values.shape[0]:
        raise ValueError("Features and response must contain the same number of rows")
    if response_values.shape[1] == 0:
        raise ValueError("Response must contain at least one protein column")
    try:
        observed = np.isfinite(response_values)
    except TypeError as exc:
        raise ValueError("Response must be numeric") from exc
    if not observed.any():
        raise ValueError("Response must contain at least one finite value")
    response_values = np.asarray(response_values, dtype=np.float32)
    if np.any(observed & ~np.isfinite(response_values)):
        raise ValueError("Finite response values must be representable as float32")

    set_seed(seed)
    device = resolve_device(settings.device)
    target_mean, target_scale, standardised, natural_targets, masks = _fit_local_targets(
        response_values,
        settings.target_scale_floor,
    )

    basis: np.ndarray | None = None
    basis_hash: str | None = None
    explained_energy: float | None = None
    output_dim = response_values.shape[1]
    if settings.kind == "low_rank":
        basis, explained_energy = _response_basis(
            standardised,
            settings.response_rank,
            settings.svd_niter,
            device,
        )
        basis_hash = hashlib.sha256(
            np.ascontiguousarray(basis).tobytes()
        ).hexdigest()
        output_dim = basis.shape[0]

    dataset = TensorDataset(
        torch.from_numpy(feature_values),
        torch.from_numpy(natural_targets),
        torch.from_numpy(masks),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(settings.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    model = DropoutMLP(
        feature_values.shape[1],
        output_dim,
        settings.hidden_dim,
        settings.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    target_mean_tensor = torch.from_numpy(target_mean).to(device)
    target_scale_tensor = torch.from_numpy(target_scale).to(device)
    basis_tensor = None if basis is None else torch.from_numpy(basis).to(device)

    final_loss = float("nan")
    for _ in range(settings.epochs):
        model.train()
        weighted_loss = 0.0
        observed_count = 0.0
        for batch_features, batch_targets, batch_masks in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_masks = batch_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(batch_features)
            standardised_prediction = (
                model_output if basis_tensor is None else model_output @ basis_tensor
            )
            prediction = (
                standardised_prediction * target_scale_tensor + target_mean_tensor
            )
            count = batch_masks.sum().clamp_min(1.0)
            loss = ((prediction - batch_targets).square() * batch_masks).sum() / count
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * float(count.detach().cpu())
            observed_count += float(count.detach().cpu())
        final_loss = weighted_loss / max(observed_count, 1.0)
    model.eval()
    return ResponseFit(
        model=model,
        target_mean=target_mean,
        target_scale=target_scale,
        device=device,
        final_loss=final_loss,
        kind=settings.kind,
        basis=basis,
        basis_hash=basis_hash,
        explained_energy=explained_energy,
        seed=int(seed),
        input_dim=feature_values.shape[1],
        n_train=feature_values.shape[0],
        n_observed_values=int(observed.sum()),
        requested_response_rank=(
            settings.response_rank if settings.kind == "low_rank" else None
        ),
    )
