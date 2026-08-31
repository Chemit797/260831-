"""Train the fixed descriptor MLP and write one auditable run directory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic_descriptor_mlp.data import VALIDATION_SPLITS, Dataset, load_dataset
from basic_descriptor_mlp.features import FeatureState
from basic_descriptor_mlp.metrics import evaluate_splits
from basic_descriptor_mlp.model import DescriptorMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return ((prediction - target).square() * mask).sum() / denominator


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _copy_config(config: dict[str, object], variant: str, device: str, seed: int) -> dict[str, object]:
    payload = json.loads(json.dumps(config))
    payload.setdefault("runtime", {})["variant"] = variant
    payload["runtime"]["device"] = device
    payload["runtime"]["shuffle_descriptor_seed"] = seed + 100003
    payload["runtime"]["run_started_utc"] = datetime.now(timezone.utc).isoformat()
    payload["runtime"]["python"] = sys.version
    payload["runtime"]["torch_version"] = torch.__version__
    payload["model"]["seed"] = seed
    return payload


def _device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _run_id(variant: str, seed: int) -> str:
    return f"{variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _predict(model: DescriptorMLP, inputs: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, model.decoder[-1].out_features), dtype=np.float32)


def _mean_predictions(data: Dataset) -> dict[str, pd.DataFrame]:
    mean = data.y_log2.loc[data.train_ids].mean(axis=0).fillna(0.0)
    result: dict[str, pd.DataFrame] = {}
    for split in VALIDATION_SPLITS:
        split_ids = data.metadata.index[data.metadata["split_final"].eq(split)]
        values = np.repeat(mean.to_numpy(dtype=np.float32)[None, :], len(split_ids), axis=0)
        result[split] = pd.DataFrame(values, index=split_ids, columns=data.proteins)
    return result


def train(config_path: str | Path, variant: str, device_name: str, seed: int, run_dir: str | Path | None = None) -> Path:
    if variant not in {"mean", "real", "zero", "shuffle"}:
        raise ValueError("variant must be mean, real, zero, or shuffle")
    config = _load_config(config_path)
    data_config = config["data"]
    model_config = config["model"]
    feature_config = config["features"]
    data = load_dataset(data_config)
    output = Path(run_dir) if run_dir else Path(config["runtime"]["runs_dir"]) / _run_id(variant, seed)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", _copy_config(config, variant, device_name, seed))
    _write_json(output / "file_hashes.json", data.file_hashes)
    _write_json(output / "data_contract.json", {
        "train_rows": int(len(data.train_ids)),
        "validation_rows": {split: int((data.metadata["split_final"] == split).sum()) for split in VALIDATION_SPLITS},
        "raw_proteins": int(len(data.missing_rate)),
        "kept_proteins": int(len(data.proteins)),
        "missing_rate_threshold": float(data_config["missing_rate_threshold"]),
        "observed_fraction": float(data.mask.to_numpy(dtype=bool).mean()),
        "target_scale": "log2",
        "output_contract": "local_4422_proteins",
    })

    if variant == "mean":
        predictions = _mean_predictions(data)
        metrics, proteins = evaluate_splits(data, predictions)
        metrics.to_csv(output / "metrics.csv", index=False)
        proteins.to_csv(output / "per_protein_metrics.csv", index_label="protein")
        np.savez_compressed(output / "validation_predictions.npz", **{split: predictions[split].to_numpy(dtype=np.float32) for split in VALIDATION_SPLITS})
        _write_json(output / "summary.json", {"variant": variant, "note": "training-mean pipeline check"})
        print(metrics.to_string(index=False), flush=True)
        print(f"Wrote run: {output.resolve()}", flush=True)
        return output

    set_seed(seed)
    device = _device(device_name)
    state = FeatureState.fit(data, float(data_config["target_std_floor"]), bool(feature_config["normalize_descriptor_blocks"]))
    train_metadata = data.metadata.loc[data.train_ids]
    x_train = state.transform_metadata(train_metadata, descriptor_mode=variant, shuffle_seed=seed + 100003)
    y_train, mask_train = state.transform_targets(data.y_log2.loc[data.train_ids])
    expected_dim = 4096 + 512 + 2 + 6 + 2
    if x_train.shape[1] != expected_dim:
        raise ValueError(f"feature width {x_train.shape[1]} does not match expected {expected_dim}")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(mask_train)),
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = DescriptorMLP(
        input_dim=x_train.shape[1],
        output_dim=len(data.proteins),
        hidden_dim=int(model_config["hidden_dim"]),
        latent_dim=int(model_config["latent_dim"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]), weight_decay=float(model_config["weight_decay"]))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(model_config["epochs"]) + 1):
        model.train()
        losses = []
        for features, targets, masks in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(features), targets, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "masked_standardized_mse": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(model_config["epochs"]):
            print(f"variant={variant} seed={seed} epoch={epoch:03d} masked_standardized_mse={epoch_loss:.6f}", flush=True)

    _write_json(output / "feature_contract.json", state.contract())
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {"input_dim": x_train.shape[1], "output_dim": len(data.proteins), "hidden_dim": int(model_config["hidden_dim"]), "latent_dim": int(model_config["latent_dim"]), "dropout": float(model_config["dropout"])},
        "feature_state": state,
        "proteins": data.proteins,
        "variant": variant,
        "seed": seed,
        "selected_epoch": int(model_config["epochs"]),
    }
    torch.save(checkpoint, output / "checkpoint.pt")

    predictions: dict[str, pd.DataFrame] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    for split in VALIDATION_SPLITS:
        ids = data.metadata.index[data.metadata["split_final"].eq(split)]
        inputs = state.transform_metadata(data.metadata.loc[ids], descriptor_mode=variant, shuffle_seed=seed + 100003)
        standardized = _predict(model, inputs, device, int(model_config["batch_size"]))
        values = state.inverse_targets(standardized).astype(np.float32)
        predictions[split] = pd.DataFrame(values, index=ids, columns=data.proteins)
        prediction_arrays[split] = values
    metrics, proteins = evaluate_splits(data, predictions)
    metrics.to_csv(output / "metrics.csv", index=False)
    proteins.to_csv(output / "per_protein_metrics.csv", index_label="protein")
    np.savez_compressed(output / "validation_predictions.npz", **prediction_arrays)
    summary = {
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "input_dim": int(x_train.shape[1]),
        "output_dim": int(len(data.proteins)),
        "selected_epoch": int(model_config["epochs"]),
        "descriptor_mode": variant,
        "descriptor_normalization": "per-row L2" if feature_config["normalize_descriptor_blocks"] else "none",
    }
    _write_json(output / "summary.json", summary)
    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote run: {output.resolve()}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("mean", "real", "zero", "shuffle"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    train(args.config, args.variant, args.device, args.seed, args.run_dir)


if __name__ == "__main__":
    main()
