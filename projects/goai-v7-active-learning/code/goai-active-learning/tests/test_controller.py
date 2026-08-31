from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import goai_al.experiment as experiment
from goai_al.data import (
    CHEMICAL,
    CONDITION_ID,
    GROUP_FIELDS,
    INTERPOLATION_SPLIT,
    MEDIUM,
    PROTOCOL_VERSION,
    STRAIN,
    TEMPERATURE,
    TIME,
    TIME_UNIT,
    VALIDATION_SPLITS,
    BenchmarkSplit,
    GroupedDataset,
    PoolFeatureEncoder,
    stable_condition_id,
)


@dataclass
class _TinyFit:
    n_proteins: int
    seed: int
    n_train: int
    kind: str

    def predict(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        del batch_size
        baseline = features.sum(axis=1, keepdims=True) * 0.01
        return np.repeat(baseline, self.n_proteins, axis=1).astype(np.float32)

    def uncertainty(
        self, features: np.ndarray, passes: int, batch_size: int = 512
    ) -> np.ndarray:
        del passes, batch_size
        return np.square(features).sum(axis=1, dtype=np.float64)

    def fit_summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "n_train": self.n_train,
            "n_features": 7,
            "n_proteins": self.n_proteins,
            "n_observed_values": self.n_train * self.n_proteins,
            "requested_response_rank": 64 if self.kind == "low_rank" else None,
            "response_rank": min(64, self.n_train, self.n_proteins) if self.kind == "low_rank" else 0,
            "basis_hash": "tiny-basis" if self.kind == "low_rank" else None,
            "explained_energy": 0.5 if self.kind == "low_rank" else None,
            "final_loss": 1.0,
        }


def _tiny_dataset() -> tuple[GroupedDataset, PoolFeatureEncoder, np.ndarray]:
    records: list[dict[str, object]] = []
    ids: list[str] = []
    assignments: list[str] = []
    split_names = [INTERPOLATION_SPLIT, *VALIDATION_SPLITS]
    for index in range(116):
        values = {
            STRAIN: f"s{index % 4}",
            CHEMICAL: f"c{(index // 4) % 7}",
            MEDIUM: f"m{(index // 28) % 2}",
            TEMPERATURE: 30 + (index // 56) % 2,
            TIME: 15 + 15 * ((index // 112) % 4),
            TIME_UNIT: "minutes",
        }
        row_id = stable_condition_id(values)
        ids.append(row_id)
        if index < 96:
            assignment = "candidate_pool"
            provenance = "train"
        else:
            assignment = split_names[(index - 96) // 4]
            provenance = "train" if assignment == INTERPOLATION_SPLIT else assignment
        assignments.append(assignment)
        records.append(
            {
                **values,
                "split_provenance": provenance,
                "replicate_count": 1,
            }
        )
    index = pd.Index(ids, name=CONDITION_ID)
    metadata = pd.DataFrame(records, index=index)
    response_values = np.arange(116 * 3, dtype=np.float32).reshape(116, 3) / 50.0
    response = pd.DataFrame(response_values, index=index, columns=("p0", "p1", "p2"))
    pool_ids = tuple(row_id for row_id, role in zip(ids, assignments) if role == "candidate_pool")
    validation_ids = {
        name: tuple(row_id for row_id, role in zip(ids, assignments) if role == name)
        for name in split_names
    }
    split = BenchmarkSplit(
        candidate_pool_ids=pool_ids,
        interpolation_ids=validation_ids[INTERPOLATION_SPLIT],
        validation_ids=validation_ids,
        removed_validation_overlap={name: () for name in VALIDATION_SPLITS},
        official_train_ids=(*pool_ids, *validation_ids[INTERPOLATION_SPLIT]),
        seed=42,
        interpolation_fraction=0.20,
    )
    dataset = GroupedDataset(
        metadata=metadata,
        response=response,
        proteins=("p0", "p1", "p2"),
        train_ids=pd.Index(pool_ids, name=CONDITION_ID),
        validation_ids={
            name: pd.Index(values, name=CONDITION_ID)
            for name, values in validation_ids.items()
        },
        protein_missing_rate=pd.Series([0.0, 0.0, 0.0], index=("p0", "p1", "p2")),
        benchmark_split=split,
    )
    encoder = PoolFeatureEncoder().fit(metadata.loc[list(pool_ids)])
    return dataset, encoder, encoder.transform(metadata)


def _write_tiny_config(path, metadata_path, proteome_path) -> None:
    path.write_text(
        f"""data:
  metadata: {metadata_path}
  proteome: {proteome_path}
  missing_rate_threshold: 0.80
  interpolation_fraction: 0.20
model:
  kind: low_rank
  response_rank: 64
  svd_niter: 1
  hidden_dim: 4
  dropout: 0.1
  learning_rate: 0.001
  weight_decay: 0.0
  epochs: 80
  batch_size: 16
  target_scale_floor: 0.05
  device: cpu
protocol:
  version: {PROTOCOL_VERSION}
  seed: 42
  strategies: [random, coreset, uncertainty]
  formal:
    initial_budget: 128
    acquisition_batch_size: 128
    checkpoints: [128, 256, 512, 1024]
    epochs: 80
    mc_passes: 8
  smoke:
    initial_budget: 32
    acquisition_batch_size: 32
    checkpoints: [32, 64, 96]
    epochs: 2
    mc_passes: 2
runtime:
  output_dir: pilot_v2
""",
        encoding="utf-8",
    )


def test_tiny_controller_smoke_writes_complete_leakage_safe_attempt(tmp_path, monkeypatch) -> None:
    metadata_path = tmp_path / "metadata.csv"
    proteome_path = tmp_path / "proteome.csv"
    metadata_path.write_text("fixture\n", encoding="utf-8")
    proteome_path.write_text("fixture\n", encoding="utf-8")
    config_path = tmp_path / "pilot.yaml"
    _write_tiny_config(config_path, metadata_path, proteome_path)
    dataset, encoder, features = _tiny_dataset()

    monkeypatch.setattr(
        experiment,
        "_load_dataset_and_features",
        lambda config_path, config: (dataset, encoder, features),
    )

    def fake_fit(features, response, settings, seed):
        assert response.shape[0] == features.shape[0]
        return _TinyFit(response.shape[1], seed, len(response), settings.kind)

    monkeypatch.setattr(experiment, "fit_response_model", fake_fit)

    def fake_audit(dataset, output_dir):
        experiment._atomic_json(output_dir / "data_audit.json", {"audit": "tiny"})
        experiment._atomic_csv(output_dir / "tensor_coverage.csv", pd.DataFrame({"x": [1]}))
        experiment._atomic_csv(output_dir / "low_rank_spectrum.csv", pd.DataFrame({"rank": [1]}))
        return {}

    monkeypatch.setattr(experiment, "write_audit_outputs", fake_audit)
    output_dir = tmp_path / "attempt"
    experiment.run_controller(
        config_path,
        smoke=True,
        output_dir=output_dir,
        command=("python", "-m", "goai_al.experiment", "--smoke"),
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "smoke"
    assert manifest["scientific"] is False
    acquisitions = pd.read_csv(output_dir / "acquisitions.csv")
    assert not {"label", "labels", "response", "impact", "oracle_impact"} & set(acquisitions)
    initial = acquisitions[acquisitions["round"].eq(0)]
    initial_sets = {
        strategy: tuple(values.sort_values("rank_in_batch")[CONDITION_ID])
        for strategy, values in initial.groupby("strategy")
    }
    assert len(set(initial_sets.values())) == 1
    assert set(acquisitions.groupby("strategy")["budget_after"].max()) == {96}
    assert (output_dir / "representation_metrics.csv").is_file()
    assert (output_dir / "learning_curve_delta_skill_zero.png").is_file()

    forbidden_receipt_keys = {
        "impact",
        "label",
        "labels",
        "oracle_impact",
        "oracle_response",
        "oracle_responses",
        "response",
        "responses",
    }

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    def assert_label_free(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_receipt_keys.isdisjoint(
                str(key).casefold() for key in value
            )
            for item in value.values():
                assert_label_free(item)
        elif isinstance(value, list):
            for item in value:
                assert_label_free(item)

    for strategy in experiment.STRATEGIES:
        receipt_paths = sorted((output_dir / "round_receipts" / strategy).glob("*.json"))
        assert [path.name for path in receipt_paths] == [
            "round_000.json",
            "round_001.json",
            "round_002.json",
        ]
        receipts = [
            json.loads(
                path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite
            )
            for path in receipt_paths
        ]
        assert [receipt["budget_after"] for receipt in receipts] == [32, 64, 96]
        assert receipts[0]["acquisition_seed"] is None
        for receipt in receipts:
            assert receipt["global_seed"] == 42
            assert receipt["model_seed"] is not None
            assert receipt["checkpoint"] is True
            assert len(receipt["labelled_ids"]) == receipt["budget_after"]
            assert receipt["labelled_ids_sha256"] == experiment._hash_ids(
                tuple(receipt["labelled_ids"])
            )
            assert receipt["model_fit_summary"]["seed"] == receipt["model_seed"]
            assert len(receipt["split_metrics"]) == 5
            assert receipt["train_seconds"] >= 0.0
            assert_label_free(receipt)
        final_receipt = receipts[-1]
        assert final_receipt["budget_after"] == 96
        assert len(final_receipt["labelled_ids"]) == 96
        assert final_receipt["model_seed"] is not None
        assert final_receipt["checkpoint"] is True
        assert len(final_receipt["split_metrics"]) == 5
        assert final_receipt["train_seconds"] >= 0.0


def test_controller_refuses_nonempty_output(tmp_path) -> None:
    target = tmp_path / "attempt"
    target.mkdir()
    (target / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        experiment._reserve_output(target)
