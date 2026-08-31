import csv
import json

from reproduction.run_matrix import run_name, validate_attempt, validate_pilot_config


def load_config():
    with open("reproduction/configs/pilot_il2_3seed.json", encoding="utf-8") as stream:
        return json.load(stream)


def write_complete_attempt(attempt, config, method="random", seed=1000):
    common = config["common"]
    dataset = config["datasets"][0]
    name = run_name(dataset, method, seed, common)
    result = attempt / "artifacts" / name
    result.mkdir(parents=True)
    (result / "run_results.pickle").write_bytes(b"complete")
    for cycle in range(common["num_active_learning_cycles"]):
        directory = result / f"cycle_{cycle}"
        directory.mkdir()
        (directory / "run_results.pickle").write_bytes(b"cycle")
    fields = [
        "dataset_name", "feature_set_name", "topk_percent", "model_name",
        "acquisition_function_name", "acquisition_batch_size",
        "num_active_learning_cycles", "seed", "num_topk_clusters",
        "acquisition_cycle", "aggregated_recall_topk",
        "aggregated_proportion_top_clusters_recovered", "bax_topk_kvalue",
        "bax_subset_select_subset_size", "bax_level_set_c",
        "bax_num_samples_EIG", "bax_num_samples_entropy",
        "bax_subset_select_num_samples", "device",
    ]
    with (attempt / "performance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cycle in range(1, common["num_active_learning_cycles"] + 1):
            writer.writerow(
                {
                    "dataset_name": dataset,
                    "feature_set_name": common["feature_set_name"],
                    "topk_percent": common["topk_percent"],
                    "model_name": common["model_name"],
                    "acquisition_function_name": method,
                    "acquisition_batch_size": common["acquisition_batch_size"],
                    "num_active_learning_cycles": common["num_active_learning_cycles"],
                    "seed": seed,
                    "num_topk_clusters": common["num_topk_clusters"],
                    "acquisition_cycle": cycle,
                    "aggregated_recall_topk": 0.1,
                    "aggregated_proportion_top_clusters_recovered": 0.5,
                    "bax_topk_kvalue": common["bax_topk_kvalue"],
                    "bax_subset_select_subset_size": common["bax_subset_select_subset_size"],
                    "bax_level_set_c": common["bax_level_set_c"],
                    "bax_num_samples_EIG": common["bax_num_samples_EIG"],
                    "bax_num_samples_entropy": common["bax_num_samples_entropy"],
                    "bax_subset_select_num_samples": common["bax_subset_select_num_samples"],
                    "device": common["device"],
                }
            )
    (attempt / "job.log").write_text("successful run\n", encoding="utf-8")
    (attempt / "args.json").write_text("{}\n", encoding="utf-8")
    return name, dataset, method, seed, common


def test_pilot_config_contract():
    validate_pilot_config(load_config())


def test_strict_attempt_validator_accepts_exact_25_cycles(tmp_path):
    values = write_complete_attempt(tmp_path, load_config())

    assert validate_attempt(tmp_path, *values) == []


def test_strict_attempt_validator_rejects_partial_and_nan(tmp_path):
    values = write_complete_attempt(tmp_path, load_config())
    name = values[0]
    (tmp_path / "artifacts" / name / "cycle_24" / "run_results.pickle").unlink()
    performance = tmp_path / "performance.csv"
    text = performance.read_text(encoding="utf-8").replace("0.1,0.5", "nan,0.5", 1)
    performance.write_text(text, encoding="utf-8")

    errors = validate_attempt(tmp_path, *values)

    assert any("cycle_24 missing" in error for error in errors)
    assert any("not finite" in error for error in errors)
