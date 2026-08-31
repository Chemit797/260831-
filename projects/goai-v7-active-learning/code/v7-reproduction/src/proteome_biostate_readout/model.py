"""Proteome-only BioState-Readout modules adapted from WAY-FM v7_final.

The adaptation deliberately keeps one chemical route because the released GOAI
WAYB/WAYC table has no KO rows.  The condition blocks and proteome target panel
are supplied by the existing fold-safe basic_descriptor_mlp data contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm1d(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def grad_reverse(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(inputs, scale)


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float, norm: str) -> None:
        super().__init__()
        middle = max(64, hidden_dim // 2)
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, middle),
            make_norm(norm, middle),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(middle, middle),
            make_norm(norm, middle),
            nn.GELU(),
        )
        self.mu = nn.Linear(middle, latent_dim)
        self.logvar = nn.Linear(middle, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(inputs)
        return self.mu(hidden), torch.clamp(self.logvar(hidden), min=-8.0, max=4.0)


class RoutedChemicalConditionEncoder(nn.Module):
    """Block-aware condition encoder with the v7 chemical perturbation route."""

    def __init__(
        self,
        strain_dim: int,
        chemical_dim: int,
        context_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        self.strain_dim = int(strain_dim)
        self.chemical_dim = int(chemical_dim)
        self.context_dim = int(context_dim)
        expert_dim = max(64, hidden_dim // 2)

        def expert(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                make_norm(norm, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, expert_dim),
                make_norm(norm, expert_dim),
                nn.GELU(),
            )

        self.strain_encoder = expert(self.strain_dim)
        self.chemical_expert = expert(self.chemical_dim)
        self.context_encoder = expert(self.context_dim)
        self.fusion = nn.Sequential(
            nn.Linear(expert_dim * 3, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strain_end = self.strain_dim
        chemical_end = strain_end + self.chemical_dim
        strain = condition[:, :strain_end]
        chemical = condition[:, strain_end:chemical_end]
        context = condition[:, chemical_end:chemical_end + self.context_dim]
        hidden = self.fusion(
            torch.cat(
                [
                    self.strain_encoder(strain),
                    self.chemical_expert(chemical),
                    self.context_encoder(context),
                ],
                dim=1,
            )
        )
        return self.mu(hidden), torch.clamp(self.logvar(hidden), min=-8.0, max=4.0)


class ModalityProjector(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float, norm: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class ProteomePosteriorEncoder(nn.Module):
    def __init__(self, n_proteins: int, hidden_dim: int, latent_dim: int, dropout: float, norm: str) -> None:
        super().__init__()
        self.encoder = GaussianEncoder(n_proteins * 2, hidden_dim, latent_dim, dropout, norm)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(torch.cat([values, mask.float()], dim=1))


class ProteomeObserver(nn.Module):
    """Canonical proteome state plus train-fitted instrument/plate calibration."""

    def __init__(
        self,
        latent_dim: int,
        n_instruments: int,
        n_plates: int,
        side_dim: int,
        hidden_dim: int,
        n_proteins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Index 0 is a neutral fold-safe unknown category.
        self.abundance_instrument = nn.Embedding(n_instruments, side_dim, padding_idx=0)
        self.mask_instrument = nn.Embedding(n_instruments, side_dim, padding_idx=0)
        self.abundance_plate = nn.Embedding(n_plates, side_dim, padding_idx=0)
        self.mask_plate = nn.Embedding(n_plates, side_dim, padding_idx=0)
        input_dim = latent_dim + side_dim * 2

        def body() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.abundance_body = body()
        self.mask_body = body()
        self.abundance_head = nn.Linear(hidden_dim, n_proteins)
        self.mask_head = nn.Linear(hidden_dim, n_proteins)

    def forward(
        self,
        latent: torch.Tensor,
        instrument_idx: torch.Tensor,
        plate_idx: torch.Tensor,
        detach_mask_latent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        abundance_side = torch.cat(
            [self.abundance_instrument(instrument_idx), self.abundance_plate(plate_idx)], dim=1
        )
        mask_side = torch.cat([self.mask_instrument(instrument_idx), self.mask_plate(plate_idx)], dim=1)
        abundance = self.abundance_head(self.abundance_body(torch.cat([latent, abundance_side], dim=1)))
        mask_latent = latent.detach() if detach_mask_latent else latent
        mask_logits = self.mask_head(self.mask_body(torch.cat([mask_latent, mask_side], dim=1)))
        return abundance, mask_logits


class ProteomeBioStateReadout(nn.Module):
    def __init__(
        self,
        strain_dim: int,
        chemical_dim: int,
        context_dim: int,
        n_instruments: int,
        n_plates: int,
        n_proteins: int,
        hidden_dim: int = 2048,
        posterior_hidden_dim: int = 2048,
        decoder_hidden_dim: int = 2048,
        latent_dim: int = 512,
        side_embedding_dim: int = 32,
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        self.bio_encoder = RoutedChemicalConditionEncoder(
            strain_dim, chemical_dim, context_dim, hidden_dim, latent_dim, encoder_dropout, norm
        )
        self.proteome_projector = ModalityProjector(latent_dim, hidden_dim, encoder_dropout, norm)
        self.proteome_posterior_encoder = ProteomePosteriorEncoder(
            n_proteins, posterior_hidden_dim, latent_dim, encoder_dropout, norm
        )
        self.proteome_observer = ProteomeObserver(
            latent_dim,
            n_instruments,
            n_plates,
            side_embedding_dim,
            decoder_hidden_dim,
            n_proteins,
            decoder_dropout,
        )
        adversary_hidden = max(64, latent_dim // 2)
        self.instrument_adversary = nn.Sequential(
            nn.Linear(latent_dim, adversary_hidden),
            nn.GELU(),
            nn.Dropout(encoder_dropout),
            nn.Linear(adversary_hidden, n_instruments),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def predict_from_condition(
        self,
        condition: torch.Tensor,
        instrument_idx: torch.Tensor,
        plate_idx: torch.Tensor,
        detach_mask_latent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_bio_mu, _ = self.bio_encoder(condition)
        z_pro_prior = self.proteome_projector(z_bio_mu)
        return self.proteome_observer(z_pro_prior, instrument_idx, plate_idx, detach_mask_latent)

    def forward(
        self,
        condition: torch.Tensor,
        instrument_idx: torch.Tensor,
        plate_idx: torch.Tensor,
        target_values: torch.Tensor,
        target_mask: torch.Tensor,
        sample_posterior: bool,
        detach_mask_latent: bool,
        adversary_grl_scale: float,
    ) -> dict[str, torch.Tensor]:
        z_bio_mu, z_bio_logvar = self.bio_encoder(condition)
        z_pro_prior = self.proteome_projector(z_bio_mu)
        posterior_mu, posterior_logvar = self.proteome_posterior_encoder(target_values, target_mask)
        posterior_sample = self.reparameterize(posterior_mu, posterior_logvar, sample_posterior)
        pred_q, mask_logits_q = self.proteome_observer(
            posterior_sample, instrument_idx, plate_idx, detach_mask_latent
        )
        pred_c, mask_logits_c = self.proteome_observer(
            z_pro_prior, instrument_idx, plate_idx, detach_mask_latent
        )
        adv_logits = self.instrument_adversary(grad_reverse(posterior_mu, adversary_grl_scale))
        return {
            "pred_q": pred_q,
            "pred_c": pred_c,
            "mask_logits_q": mask_logits_q,
            "mask_logits_c": mask_logits_c,
            "z_bio_mu": z_bio_mu,
            "z_bio_logvar": z_bio_logvar,
            "z_pro_prior": z_pro_prior,
            "posterior_mu": posterior_mu,
            "posterior_logvar": posterior_logvar,
            "adv_logits": adv_logits,
        }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.float()
    return (((prediction - target) ** 2) * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def diagonal_kl_q_to_unit_variance_prior(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
) -> torch.Tensor:
    q_var = torch.exp(q_logvar)
    kl = 0.5 * (-q_logvar + q_var + (q_mu - prior_mu) ** 2 - 1.0)
    return kl.sum(dim=1).mean()


def supervised_symmetric_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    group_idx: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    left = nn.functional.normalize(left, dim=1)
    right = nn.functional.normalize(right, dim=1)
    logits = left @ right.T / float(temperature)
    labels = group_idx.view(-1, 1)
    positives = labels.eq(labels.T).float()
    row = -(torch.log_softmax(logits, dim=1) * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1.0)
    column = -(torch.log_softmax(logits.T, dim=1) * positives.T).sum(dim=1) / positives.T.sum(dim=1).clamp_min(1.0)
    return 0.5 * (row.mean() + column.mean())


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        init_lr: float,
        target_lr: float,
        total_steps: int,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.init_lr = float(init_lr)
        self.target_lr = float(target_lr)
        self.total_steps = max(1, int(total_steps))
        self.min_lr = float(min_lr)
        self.current_step = 0
        self._set_lr(self.init_lr if self.warmup_steps else self.target_lr)

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> None:
        self.current_step += 1
        if self.warmup_steps and self.current_step <= self.warmup_steps:
            learning_rate = self.init_lr + (
                (self.target_lr - self.init_lr) * self.current_step / self.warmup_steps
            )
        else:
            denominator = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (self.current_step - self.warmup_steps) / denominator))
            learning_rate = self.min_lr + 0.5 * (self.target_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        self._set_lr(learning_rate)

    def get_last_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

