"""Train the fold-safe proteome-only BioState-Readout reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BASIC_SRC = PROJECT_ROOT / "experiments" / "basic_descriptor_mlp" / "src"
if str(BASIC_SRC) not in sys.path:
    sys.path.insert(0, str(BASIC_SRC))

from basic_descriptor_mlp.data import VALIDATION_SPLITS, Dataset, load_dataset  # noqa: E402
from basic_descriptor_mlp.features import FeatureState  # noqa: E402
from basic_descriptor_mlp.metrics import evaluate_splits  # noqa: E402

from proteome_biostate_readout.model import (  # noqa: E402
    ProteomeBioStateReadout,
    WarmupCosineScheduler,
    diagonal_kl_q_to_unit_variance_prior,
    masked_mse,
    supervised_symmetric_nce,
)


GROUP_FIELDS = ("Strains", "Medium", "Temperature", "pert_time", "perturbation_no_concentration")


@dataclass
class SideState:
    instrument_vocab: list[str]
    plate_vocab: list[str]

    @classmethod
    def fit(cls, metadata: pd.DataFrame, train_ids: pd.Index) -> "SideState":
        train = metadata.loc[train_ids]
        return cls(
            instrument_vocab=["<UNK>"] + sorted(train["instrument"].astype(str).unique().tolist()),
            plate_vocab=["<UNK>"] + sorted(train["Yeast_cell_plate"].astype(str).unique().tolist()),
        )

    @staticmethod
    def _encode(values: pd.Series, vocab: list[str]) -> np.ndarray:
        mapping = {value: index for index, value in enumerate(vocab)}
        return values.astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)

    def transform(self, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._encode(metadata["instrument"], self.instrument_vocab),
            self._encode(metadata["Yeast_cell_plate"], self.plate_vocab),
        )

    def contract(self, metadata: pd.DataFrame, train_ids: pd.Index) -> dict[str, object]:
        nontrain = metadata.index.difference(train_ids)
        instruments, plates = self.transform(metadata.loc[nontrain])
        return {
            "fit_scope": "split_final == train only",
            "unknown_policy": "index 0, fixed neutral padding embedding",
            "instrument_vocab": self.instrument_vocab,
            "plate_vocab": self.plate_vocab,
            "nontrain_unknown_instrument_rows": int((instruments == 0).sum()),
            "nontrain_unknown_plate_rows": int((plates == 0).sum()),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(seed: int) -> str:
    return f"biostate-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def biological_groups(metadata: pd.DataFrame, train_ids: pd.Index) -> np.ndarray:
    frame = metadata.loc[train_ids, GROUP_FIELDS].astype(str)
    keys = pd.MultiIndex.from_frame(frame)
    codes, _ = pd.factorize(keys, sort=True)
    return codes.astype(np.int64)


def compute_mask_pos_weight(mask: np.ndarray, enabled: bool) -> np.ndarray | None:
    if not enabled:
        return None
    positive = mask.sum(axis=0, dtype=np.float64)
    negative = mask.shape[0] - positive
    return np.clip(negative / np.maximum(positive, 1.0), 0.1, 10.0).astype(np.float32)


def model_kwargs(config: dict[str, object], data: Dataset, side: SideState) -> dict[str, object]:
    settings = config["model"]
    return {
        "strain_dim": 4096,
        "chemical_dim": 512,
        "context_dim": 10,
        "n_instruments": len(side.instrument_vocab),
        "n_plates": len(side.plate_vocab),
        "n_proteins": len(data.proteins),
        "hidden_dim": int(settings["hidden_dim"]),
        "posterior_hidden_dim": int(settings["posterior_hidden_dim"]),
        "decoder_hidden_dim": int(settings["decoder_hidden_dim"]),
        "latent_dim": int(settings["latent_dim"]),
        "side_embedding_dim": int(settings["side_embedding_dim"]),
        "encoder_dropout": float(settings["encoder_dropout"]),
        "decoder_dropout": float(settings["decoder_dropout"]),
        "norm": str(settings["norm"]),
    }


def batch_loss(
    model: ProteomeBioStateReadout,
    batch: tuple[torch.Tensor, ...],
    settings: dict[str, object],
    pos_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    condition, targets, masks, instrument, plate, groups = batch
    outputs = model(
        condition,
        instrument,
        plate,
        targets,
        masks,
        sample_posterior=bool(settings["sample_posterior"]),
        detach_mask_latent=bool(settings["detach_mask_latent"]),
        adversary_grl_scale=float(settings["adversary_grl_scale"]),
    )
    recon_q = masked_mse(outputs["pred_q"], targets, masks)
    predict_c = masked_mse(outputs["pred_c"], targets, masks)
    kl = diagonal_kl_q_to_unit_variance_prior(
        outputs["posterior_mu"], outputs["posterior_logvar"], outputs["z_pro_prior"]
    )
    nce = supervised_symmetric_nce(
        outputs["z_pro_prior"], outputs["posterior_mu"], groups, float(settings["temperature"])
    )
    bio_nce = supervised_symmetric_nce(
        outputs["z_bio_mu"], outputs["posterior_mu"], groups, float(settings["temperature"])
    )
    mask_q = nn.functional.binary_cross_entropy_with_logits(
        outputs["mask_logits_q"], masks.float(), pos_weight=pos_weight
    )
    mask_c = nn.functional.binary_cross_entropy_with_logits(
        outputs["mask_logits_c"], masks.float(), pos_weight=pos_weight
    )
    adversary = nn.functional.cross_entropy(outputs["adv_logits"], instrument)
    loss = (
        float(settings["lambda_recon_q"]) * recon_q
        + float(settings["lambda_predict_c"]) * predict_c
        + float(settings["lambda_mask"])
        * (float(settings["lambda_recon_q"]) * mask_q + float(settings["lambda_predict_c"]) * mask_c)
        + float(settings["lambda_kl"]) * kl
        + float(settings["lambda_nce"]) * nce
        + float(settings["lambda_bio_posterior_nce"]) * bio_nce
        + float(settings["lambda_adv_instrument"]) * adversary
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "recon_q": float(recon_q.detach().cpu()),
        "predict_c": float(predict_c.detach().cpu()),
        "mask_q": float(mask_q.detach().cpu()),
        "mask_c": float(mask_c.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "nce": float(nce.detach().cpu()),
        "bio_nce": float(bio_nce.detach().cpu()),
        "adversary": float(adversary.detach().cpu()),
    }


@torch.no_grad()
def predict(
    model: ProteomeBioStateReadout,
    conditions: np.ndarray,
    instruments: np.ndarray,
    plates: np.ndarray,
    device: torch.device,
    batch_size: int,
    detach_mask_latent: bool,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(conditions), batch_size):
        stop = start + batch_size
        condition = torch.from_numpy(conditions[start:stop]).to(device)
        instrument = torch.from_numpy(instruments[start:stop]).to(device)
        plate = torch.from_numpy(plates[start:stop]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values, _ = model.predict_from_condition(condition, instrument, plate, detach_mask_latent)
        output.append(values.float().cpu().numpy())
    return np.concatenate(output, axis=0)


def train(
    config_path: str | Path,
    device_name: str,
    seed: int,
    run_dir: str | Path | None = None,
    epochs_override: int | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    data = load_dataset(config["data"])
    feature_state = FeatureState.fit(
        data,
        float(config["data"]["target_std_floor"]),
        bool(config["features"]["normalize_descriptor_blocks"]),
    )
    side_state = SideState.fit(data.metadata, data.train_ids)
    settings = dict(config["training"])
    if epochs_override is not None:
        settings["epochs"] = int(epochs_override)
    device = resolve_device(device_name)
    set_seed(seed)

    output = Path(run_dir) if run_dir else Path(config["runtime"]["runs_dir"]) / make_run_id(seed)
    output.mkdir(parents=True, exist_ok=False)
    resolved_config = json.loads(json.dumps(config))
    resolved_config["training"] = settings
    resolved_config["runtime"].update(
        {
            "device": str(device),
            "seed": seed,
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "torch_version": torch.__version__,
        }
    )
    write_json(output / "config.json", resolved_config)
    write_json(output / "file_hashes.json", data.file_hashes)
    write_json(output / "feature_contract.json", feature_state.contract())
    write_json(output / "side_context_contract.json", side_state.contract(data.metadata, data.train_ids))
    write_json(
        output / "experiment_contract.json",
        {
            "hypothesis": "The v7 proteome prior/posterior readout improves released-validation proteome prediction over the fixed Descriptor MLP.",
            "single_primary_change": "DescriptorMLP replaced by proteome-only BioState-Readout v7 architecture and loss recipe.",
            "data_boundary": "same basic_descriptor_mlp train rows and released validation; validation targets unused in fitting or selection",
            "features": "same normalized strain/chemical descriptors and train-fitted medium/time/temperature one-hot blocks",
            "stage": "proteome_pretrain adaptation; chemical route only; no metabolome, growth, KO, or paired alignment",
            "training_budget": {"epochs": int(settings["epochs"]), "batch_size": int(settings["batch_size"])},
            "seed": seed,
            "primary_metric": "per-protein R2 median, reported separately by released validation scenario",
            "auxiliary_metrics": ["RMSE", "MAE", "Global R2", "Sample PCC median", "Protein PCC median"],
        },
    )

    train_metadata = data.metadata.loc[data.train_ids]
    x_train = feature_state.transform_metadata(train_metadata, descriptor_mode="real")
    if x_train.shape[1] != 4618:
        raise ValueError(f"unexpected condition width: {x_train.shape[1]}")
    y_train, mask_train = feature_state.transform_targets(data.y_log2.loc[data.train_ids])
    instrument_train, plate_train = side_state.transform(train_metadata)
    groups_train = biological_groups(data.metadata, data.train_ids)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(mask_train),
        torch.from_numpy(instrument_train),
        torch.from_numpy(plate_train),
        torch.from_numpy(groups_train),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    kwargs = model_kwargs(config, data, side_state)
    model = ProteomeBioStateReadout(**kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["warmup_initial_learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        int(settings["warmup_steps"]),
        float(settings["warmup_initial_learning_rate"]),
        float(settings["learning_rate"]),
        int(settings["epochs"]) * len(loader),
        float(settings["min_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(settings["amp"]) and device.type == "cuda")
    pos_weight_array = compute_mask_pos_weight(mask_train, bool(settings["mask_pos_weight"]))
    pos_weight = torch.from_numpy(pos_weight_array).to(device) if pos_weight_array is not None else None

    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        epoch_logs: list[dict[str, float]] = []
        for cpu_batch in loader:
            batch = tuple(value.to(device, non_blocking=True) for value in cpu_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=bool(settings["amp"]) and device.type == "cuda",
            ):
                loss, logs = batch_loss(model, batch, settings, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_logs.append(logs)
        row: dict[str, float | int] = {"epoch": epoch, "learning_rate": scheduler.get_last_lr()}
        for key in epoch_logs[0]:
            row[key] = float(np.mean([entry[key] for entry in epoch_logs]))
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(settings["epochs"]):
            print(
                f"seed={seed} epoch={epoch:03d}/{int(settings['epochs'])} "
                f"loss={row['loss']:.6f} predict_c={row['predict_c']:.6f} "
                f"recon_q={row['recon_q']:.6f} lr={row['learning_rate']:.3e}",
                flush=True,
            )

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": kwargs,
        "target_mean": feature_state.target_mean,
        "target_scale": feature_state.target_scale,
        "feature_categories": feature_state.categories,
        "side_state": side_state,
        "proteins": data.proteins,
        "seed": seed,
        "selected_epoch": int(settings["epochs"]),
    }
    torch.save(checkpoint, output / "checkpoint.pt")

    predictions: dict[str, pd.DataFrame] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    for split in VALIDATION_SPLITS:
        ids = data.metadata.index[data.metadata["split_final"].eq(split)]
        metadata = data.metadata.loc[ids]
        conditions = feature_state.transform_metadata(metadata, descriptor_mode="real")
        instruments, plates = side_state.transform(metadata)
        standardized = predict(
            model,
            conditions,
            instruments,
            plates,
            device,
            int(settings["batch_size"]),
            bool(settings["detach_mask_latent"]),
        )
        values = feature_state.inverse_targets(standardized).astype(np.float32)
        predictions[split] = pd.DataFrame(values, index=ids, columns=data.proteins)
        prediction_arrays[split] = values

    metrics, per_protein = evaluate_splits(data, predictions)
    metrics.to_csv(output / "metrics.csv", index=False)
    per_protein.to_csv(output / "per_protein_metrics.csv", index_label="protein")
    np.savez_compressed(output / "validation_predictions.npz", **prediction_arrays)
    write_json(
        output / "summary.json",
        {
            "model": "proteome_biostate_readout_v7_reproduction",
            "seed": seed,
            "device": str(device),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "input_dim": 4618,
            "output_dim": len(data.proteins),
            "selected_epoch": int(settings["epochs"]),
            "selection": "fixed budget; released validation not used for early stopping",
            "checkpoint_sha256": file_sha256(output / "checkpoint.pt"),
        },
    )
    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote run: {output.resolve()}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--epochs-override", type=int, default=None)
    args = parser.parse_args()
    train(args.config, args.device, args.seed, args.run_dir, args.epochs_override)


if __name__ == "__main__":
    main()

