"""Fixed GOAI-SEMANTIC-BCR-V1 and same-information flat baseline.

The module deliberately contains only the two architectures frozen by the
experiment prompt.  Frozen descriptors and frozen CalV2 tensors enter as
inputs/buffers; no encoder from those upstream artifacts is trainable here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BCRComponents:
    absolute: torch.Tensor
    background: torch.Tensor
    calibration: torch.Tensor
    response: torch.Tensor


class CellStateEncoder(nn.Module):
    """Biology-only encoder fixed by the user-provided experiment contract."""

    def __init__(self, n_medium: int, strain_dim: int = 4096) -> None:
        super().__init__()
        if n_medium <= 0 or strain_dim <= 0:
            raise ValueError("CellState dimensions must be positive")
        self.strain = nn.Sequential(
            nn.LayerNorm(strain_dim),
            nn.Linear(strain_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        # The final slot is a fixed all-zero unknown category.  Formal folds
        # are required to have no unseen medium; this slot is fail-safe only.
        self.medium = nn.Embedding(n_medium + 1, 16, padding_idx=n_medium)
        self.temperature = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.time = nn.Sequential(nn.Linear(1, 16), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(128 + 16 + 16 + 16, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )

    def forward(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        pieces = (
            self.strain(strain),
            self.medium(medium),
            self.temperature(temperature),
            self.time(time),
        )
        return self.fusion(torch.cat(pieces, dim=1))


class FrozenCalV2(nn.Module):
    """Frozen canonical 12D CalV2 decoder in main-target standardized units."""

    def __init__(self, center: torch.Tensor, decoder_scaled: torch.Tensor) -> None:
        super().__init__()
        if center.shape != (12,):
            raise ValueError("CalV2 center must have shape [12]")
        if decoder_scaled.ndim != 2 or decoder_scaled.shape[1] != 12:
            raise ValueError("CalV2 decoder must have shape [protein, 12]")
        if not torch.isfinite(center).all() or not torch.isfinite(decoder_scaled).all():
            raise ValueError("CalV2 buffers must be finite")
        self.register_buffer("center", center.detach().clone().float())
        self.register_buffer("decoder_scaled", decoder_scaled.detach().clone().float())

    def forward(self, z_cal: torch.Tensor) -> torch.Tensor:
        return (z_cal - self.center) @ self.decoder_scaled.T


class GOAISemanticBCRV1(nn.Module):
    """Strictly disentangled ``B(h_cell) + C(z_cal) + I_treatment R``."""

    def __init__(
        self,
        *,
        n_medium: int,
        chemical_dim: int,
        n_proteins: int,
        cal_center: torch.Tensor,
        cal_decoder_scaled: torch.Tensor,
        strain_dim: int = 4096,
    ) -> None:
        super().__init__()
        if chemical_dim <= 0 or n_proteins <= 0:
            raise ValueError("chemical/protein dimensions must be positive")
        if cal_decoder_scaled.shape != (n_proteins, 12):
            raise ValueError("CalV2 decoder/protein interface differs")
        self.cell = CellStateEncoder(n_medium=n_medium, strain_dim=strain_dim)
        self.background = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_proteins),
        )
        self.chemical_adapter = nn.Sequential(
            nn.LayerNorm(chemical_dim),
            nn.Linear(chemical_dim, 128),
            nn.GELU(),
        )
        self.response = nn.Sequential(
            nn.Linear(128 + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_proteins),
        )
        self.calibration = FrozenCalV2(cal_center, cal_decoder_scaled)

    def encode_cell(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.cell(strain, medium, temperature, time)

    def response_only(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
        chemical: torch.Tensor,
    ) -> torch.Tensor:
        h_cell = self.encode_cell(strain, medium, temperature, time)
        h_chemical = self.chemical_adapter(chemical)
        return self.response(torch.cat((h_cell, h_chemical), dim=1))

    def background_plus_calibration(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
        z_cal: torch.Tensor,
    ) -> torch.Tensor:
        h_cell = self.encode_cell(strain, medium, temperature, time)
        return self.background(h_cell) + self.calibration(z_cal)

    def forward_components(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
        chemical: torch.Tensor,
        z_cal: torch.Tensor,
        is_treatment: torch.Tensor,
    ) -> BCRComponents:
        h_cell = self.encode_cell(strain, medium, temperature, time)
        background = self.background(h_cell)
        calibration = self.calibration(z_cal)
        h_chemical = self.chemical_adapter(chemical)
        raw_response = self.response(torch.cat((h_cell, h_chemical), dim=1))
        # This multiplication is the structural control invariant: response is
        # exactly zero for every control, independent of learned parameters.
        response = raw_response * is_treatment.reshape(-1, 1)
        absolute = background + calibration + response
        return BCRComponents(
            absolute=absolute,
            background=background,
            calibration=calibration,
            response=response,
        )

    def forward(
        self,
        strain: torch.Tensor,
        medium: torch.Tensor,
        temperature: torch.Tensor,
        time: torch.Tensor,
        chemical: torch.Tensor,
        z_cal: torch.Tensor,
        is_treatment: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_components(
            strain, medium, temperature, time, chemical, z_cal, is_treatment
        ).absolute


class FlatMLPSameInfo(nn.Module):
    """Fixed concat MLP with the same information budget as B+C+R."""

    def __init__(self, input_dim: int, n_proteins: int) -> None:
        super().__init__()
        if input_dim <= 0 or n_proteins <= 0:
            raise ValueError("flat MLP dimensions must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_proteins),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
