#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import statistics

try:
    from reproduction.run_matrix import run_name, validate_attempt
except ModuleNotFoundError:  # Direct script execution without editable install.
    from run_matrix import run_name, validate_attempt


PAPER_IL2 = {
    "random": (0.052, 0.328, 0.131),
    "topuncertain": (0.124, 0.655, 0.285),
    "coreset": (0.107, 0.563, 0.245),
    "badge": (0.096, 0.543, 0.228),
    "ucb": (0.131, 0.673, 0.296),
    "topk_bax": (0.136, 0.698, 0.308),
    "levelset_bax": (0.142, 0.700, 0.315),
    "discobax": (0.157, 0.750, 0.343),
}


def mean_std_sem(values):
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std, std / math.sqrt(len(values))


def atomic_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer columns for empty output {path}")
        fieldnames = list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_performance(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def legacy_summary(performance_csv, output=None):
    rows = read_performance(performance_csv)
    if not rows:
        raise ValueError(f"No performance rows found in {performance_csv}")
    final_by_run = {}
    for row in rows:
        key = (row["dataset_name"], row["acquisition_function_name"], row["seed"])
        if key not in final_by_run or int(row["acquisition_cycle"]) > int(
            final_by_run[key]["acquisition_cycle"]
        ):
            final_by_run[key] = row
    grouped = defaultdict(list)
    for (dataset, method, _), row in final_by_run.items():
        grouped[(dataset, method)].append(
            (float(row["aggregated_recall_topk"]), float(row["aggregated_proportion_top_clusters_recovered"]), int(row["acquisition_cycle"]))
        )
    summary = []
    for (dataset, method), values in sorted(grouped.items()):
        r_mean, r_std, r_sem = mean_std_sem([value[0] for value in values])
        d_mean, d_std, d_sem = mean_std_sem([value[1] for value in values])
        summary.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": len(values),
                "final_cycle": max(value[2] for value in values),
                "topk_recall_mean": r_mean,
                "topk_recall_std": r_std,
                "topk_recall_sem": r_sem,
                "diversity_mean": d_mean,
                "diversity_std": d_std,
                "diversity_sem": d_sem,
                "overall_mean": math.sqrt(r_mean * d_mean),
            }
        )
    destination = output or Path(performance_csv).with_name(Path(performance_csv).stem + "_summary.csv")
    atomic_csv(destination, summary)
    return 0


def summarize_pilot(pilot_root, config_path, allow_partial=False):
    pilot_root = Path(pilot_root).resolve()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    common = config["common"]
    completion_rows = []
    run_rows = []
    all_cycle_rows = []
    for dataset in config["datasets"]:
        for method in config["methods"]:
            for seed in config["seeds"]:
                name = run_name(dataset, method, seed, common)
                job = pilot_root / "jobs" / name
                pointer_path = job / "complete.json"
                errors = []
                attempt = None
                if pointer_path.is_file():
                    try:
                        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                        attempt = Path(pointer["attempt"])
                        errors = validate_attempt(attempt, name, dataset, method, seed, common)
                    except (OSError, KeyError, json.JSONDecodeError) as exc:
                        errors = [f"invalid complete pointer: {exc}"]
                else:
                    errors = ["missing complete.json"]
                status = "COMPLETE" if attempt is not None and not errors else "INCOMPLETE"
                completion_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "status": status,
                        "attempt": str(attempt) if attempt else "",
                        "validation_errors": " | ".join(errors),
                    }
                )
                if status != "COMPLETE":
                    continue
                rows = read_performance(attempt / "performance.csv")
                for row in rows:
                    cycle = int(row["acquisition_cycle"])
                    recall = float(row["aggregated_recall_topk"])
                    diversity = float(row["aggregated_proportion_top_clusters_recovered"])
                    item = {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "cycle": cycle,
                        "recall": recall,
                        "diversity": diversity,
                        "overall_per_seed": math.sqrt(recall * diversity),
                    }
                    all_cycle_rows.append(item)
                    if cycle == common["num_active_learning_cycles"]:
                        run_rows.append(item)

    prefix = "pilot_il2_3seed"
    atomic_csv(pilot_root / f"{prefix}_completion.csv", completion_rows)
    per_seed_fields = ["dataset", "method", "seed", "cycle", "recall", "diversity", "overall_per_seed"]
    atomic_csv(pilot_root / f"{prefix}_final_per_seed.csv", run_rows, per_seed_fields)

    summary_rows = []
    final_groups = defaultdict(list)
    for row in run_rows:
        final_groups[(row["dataset"], row["method"])].append(row)
    for (dataset, method), rows in sorted(final_groups.items()):
        recalls = [row["recall"] for row in rows]
        diversities = [row["diversity"] for row in rows]
        per_seed_overall = [row["overall_per_seed"] for row in rows]
        r_mean, r_std, r_sem = mean_std_sem(recalls)
        d_mean, d_std, d_sem = mean_std_sem(diversities)
        _, o_std, o_sem = mean_std_sem(per_seed_overall)
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": len(rows),
                "recall_mean": r_mean,
                "recall_std_ddof0": r_std,
                "recall_sem": r_sem,
                "diversity_mean": d_mean,
                "diversity_std_ddof0": d_std,
                "diversity_sem": d_sem,
                "overall_mean_paper": math.sqrt(r_mean * d_mean),
                "overall_per_seed_std_ddof0_descriptive": o_std,
                "overall_per_seed_sem_descriptive": o_sem,
            }
        )
    summary_fields = [
        "dataset", "method", "seeds", "recall_mean", "recall_std_ddof0", "recall_sem",
        "diversity_mean", "diversity_std_ddof0", "diversity_sem", "overall_mean_paper",
        "overall_per_seed_std_ddof0_descriptive", "overall_per_seed_sem_descriptive",
    ]
    atomic_csv(pilot_root / f"{prefix}_summary.csv", summary_rows, summary_fields)

    curve_groups = defaultdict(list)
    for row in all_cycle_rows:
        curve_groups[(row["dataset"], row["method"], row["cycle"])].append(row)
    curve_summary = []
    for (dataset, method, cycle), rows in sorted(curve_groups.items()):
        recalls = [row["recall"] for row in rows]
        diversities = [row["diversity"] for row in rows]
        r_mean, r_std, r_sem = mean_std_sem(recalls)
        d_mean, d_std, d_sem = mean_std_sem(diversities)
        curve_summary.append(
            {
                "dataset": dataset,
                "method": method,
                "cycle": cycle,
                "seeds": len(rows),
                "recall_mean": r_mean,
                "recall_std_ddof0": r_std,
                "recall_sem": r_sem,
                "diversity_mean": d_mean,
                "diversity_std_ddof0": d_std,
                "diversity_sem": d_sem,
                "overall_mean_paper": math.sqrt(r_mean * d_mean),
            }
        )
    curve_fields = [
        "dataset", "method", "cycle", "seeds", "recall_mean", "recall_std_ddof0",
        "recall_sem", "diversity_mean", "diversity_std_ddof0", "diversity_sem", "overall_mean_paper",
    ]
    atomic_csv(pilot_root / f"{prefix}_curves.csv", curve_summary, curve_fields)

    paper_rows = []
    summary_by_method = {row["method"]: row for row in summary_rows}
    for method in config["methods"]:
        reference = PAPER_IL2[method]
        observed = summary_by_method.get(method, {})
        paper_rows.append(
            {
                "method": method,
                "paper_reference_recall": reference[0],
                "paper_reference_diversity": reference[1],
                "paper_reference_overall": reference[2],
                "pilot_recall_mean": observed.get("recall_mean", ""),
                "pilot_diversity_mean": observed.get("diversity_mean", ""),
                "pilot_overall_mean_paper": observed.get("overall_mean_paper", ""),
                "provenance_note": "PAPER_REFERENCE is not a local run result",
            }
        )
    atomic_csv(pilot_root / f"{prefix}_paper_comparison.csv", paper_rows)

    complete = sum(row["status"] == "COMPLETE" for row in completion_rows)
    print(f"Validated FULL RUN jobs: {complete}/{len(completion_rows)}")
    for row in summary_rows:
        print(
            row["method"],
            f"n={row['seeds']}",
            f"recall={100 * row['recall_mean']:.2f}%",
            f"diversity={100 * row['diversity_mean']:.2f}%",
            f"overall={100 * row['overall_mean_paper']:.2f}%",
        )
    if complete != len(completion_rows) and not allow_partial:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Strictly summarize final-cycle DiscoBAX metrics.")
    parser.add_argument("performance_csv", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "pilot_il2_3seed.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.pilot_root:
        return summarize_pilot(args.pilot_root, args.config, args.allow_partial)
    if args.performance_csv:
        return legacy_summary(args.performance_csv, args.output)
    parser.error("provide performance_csv or --pilot-root")


if __name__ == "__main__":
    raise SystemExit(main())
