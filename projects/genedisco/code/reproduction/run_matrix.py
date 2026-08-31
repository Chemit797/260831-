#!/usr/bin/env python3
"""Run an auditable, resumable DiscoBAX experiment matrix.

Each attempt owns its output directory, performance CSV, command record and log.
Only an attempt that passes :func:`validate_attempt` is promoted to COMPLETE.
"""

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "release" / "data"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "pilot_il2_3seed"
DEFAULT_LOGS = REPO_ROOT / "logs" / "pilot_il2_3seed"
PILOT_METHODS = [
    "random",
    "topuncertain",
    "coreset",
    "badge",
    "ucb",
    "topk_bax",
    "levelset_bax",
    "discobax",
]
PILOT_SEEDS = [1000, 2000, 3000]
LOG_FAILURE_RE = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|out of memory|"
    r"\bNaN\b|device mismatch|silent fallback|Killed$",
    re.IGNORECASE | re.MULTILINE,
)
CURRENT_CHILD = None
STOP_REQUESTED = False


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_name(dataset, method, seed, common):
    if method == "discobax":
        hyper = common["bax_subset_select_subset_size"]
    elif method == "topk_bax":
        hyper = common["bax_topk_kvalue"]
    elif method == "levelset_bax":
        hyper = common["bax_level_set_c"]
    else:
        hyper = ""
    return "_".join(
        map(
            str,
            [
                dataset,
                common["feature_set_name"],
                common["topk_percent"],
                common["model_name"],
                method,
                hyper,
                seed,
            ],
        )
    )


def validate_pilot_config(config):
    if config.get("name") != "pilot_il2_3seed":
        return
    errors = []
    common = config.get("common", {})
    expected = {
        "datasets": ["schmidt_2021_il2"],
        "methods": PILOT_METHODS,
        "seeds": PILOT_SEEDS,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"{key} must equal {value!r}")
    required_common = {
        "feature_set_name": "achilles",
        "model_name": "bayesian_mlp",
        "acquisition_batch_size": 32,
        "num_active_learning_cycles": 25,
        "topk_percent": 0.01,
        "num_topk_clusters": 20,
        "bax_topk_kvalue": 2,
        "bax_level_set_c": 1.5,
        "bax_subset_select_subset_size": 10,
        "bax_noise_type": "additive",
        "bax_noise_lengthscale": 1.0,
        "bax_noise_outputscale": 1.0,
        "bax_num_samples_EIG": 20,
        "bax_num_samples_entropy": 20,
        "bax_entropy_average_mode": "arithmetic",
        "bax_batch_selection_mode": "topk_EIG",
        "bax_subset_select_num_samples": 20,
        "device": "cuda",
    }
    for key, value in required_common.items():
        if common.get(key) != value:
            errors.append(f"common.{key} must equal {value!r}")
    if errors:
        raise ValueError("Invalid pilot configuration: " + "; ".join(errors))


def required_cache_files(config):
    common = config["common"]
    files = ["achilles.h5"]
    for dataset in config["datasets"]:
        files.extend(
            [
                f"{dataset}.h5",
                f"clusters_{dataset}_{common['feature_set_name']}_{common['topk_percent']}_topk_{common['num_topk_clusters']}_clusters_to_items.pkl",
                f"clusters_{dataset}_{common['feature_set_name']}_{common['topk_percent']}_topk_items_to_{common['num_topk_clusters']}_clusters.pkl",
            ]
        )
    return files


def build_command(
    python,
    cache,
    artifact_base,
    performance,
    scratch,
    dataset,
    method,
    seed,
    common,
):
    app = REPO_ROOT / "discobax" / "apps" / "genedisco_experiment.py"
    params = dict(common)
    params.update(
        {
            "cache_directory": str(cache),
            "output_directory": str(artifact_base),
            "performance_file_location": str(performance),
            "scratch_directory": str(scratch),
            "dataset_name": dataset,
            "acquisition_function_name": method,
            "seed": seed,
        }
    )
    command = [str(python), str(app)]
    command.extend(f"--{key}={value}" for key, value in params.items())
    return command, params


def _same_value(actual, expected):
    if isinstance(expected, bool):
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def validate_attempt(attempt, name, dataset, method, seed, common):
    attempt = Path(attempt)
    result_directory = attempt / "artifacts" / name
    performance = attempt / "performance.csv"
    log_path = attempt / "job.log"
    errors = []
    if not (result_directory / "run_results.pickle").is_file():
        errors.append("missing top-level run_results.pickle")
    expected_cycles = int(common["num_active_learning_cycles"])
    actual_cycle_names = sorted(
        (path.name for path in result_directory.glob("cycle_*") if path.is_dir()),
        key=lambda value: int(value.split("_")[-1])
        if value.split("_")[-1].isdigit()
        else math.inf,
    )
    wanted_cycle_names = [f"cycle_{index}" for index in range(expected_cycles)]
    if actual_cycle_names != wanted_cycle_names:
        errors.append(
            f"cycle directories differ: expected {wanted_cycle_names}, got {actual_cycle_names}"
        )
    for cycle_name in wanted_cycle_names:
        if not (result_directory / cycle_name / "run_results.pickle").is_file():
            errors.append(f"{cycle_name} missing run_results.pickle")
    rows = []
    if not performance.is_file():
        errors.append("missing performance.csv")
    else:
        try:
            with performance.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        except Exception as exc:  # pragma: no cover - defensive audit path
            errors.append(f"cannot read performance.csv: {exc}")
    if rows:
        try:
            cycles = [int(row["acquisition_cycle"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid acquisition_cycle values: {exc}")
            cycles = []
        if cycles != list(range(1, expected_cycles + 1)):
            errors.append(f"performance cycles must be 1..{expected_cycles}, got {cycles}")
        for index, row in enumerate(rows, start=1):
            expected_row = {
                "dataset_name": dataset,
                "acquisition_function_name": method,
                "seed": seed,
                "feature_set_name": common["feature_set_name"],
                "model_name": common["model_name"],
                "acquisition_batch_size": common["acquisition_batch_size"],
                "num_active_learning_cycles": expected_cycles,
                "topk_percent": common["topk_percent"],
                "num_topk_clusters": common["num_topk_clusters"],
                "bax_topk_kvalue": common["bax_topk_kvalue"],
                "bax_subset_select_subset_size": common[
                    "bax_subset_select_subset_size"
                ],
                "bax_level_set_c": common["bax_level_set_c"],
                "bax_num_samples_EIG": common["bax_num_samples_EIG"],
                "bax_num_samples_entropy": common["bax_num_samples_entropy"],
                "bax_subset_select_num_samples": common[
                    "bax_subset_select_num_samples"
                ],
                "device": common["device"],
            }
            for key, expected in expected_row.items():
                if not _same_value(row.get(key), expected):
                    errors.append(
                        f"row {index} {key}: expected {expected!r}, got {row.get(key)!r}"
                    )
            for metric in (
                "aggregated_recall_topk",
                "aggregated_proportion_top_clusters_recovered",
            ):
                try:
                    if not math.isfinite(float(row[metric])):
                        errors.append(f"row {index} {metric} is not finite")
                except (KeyError, TypeError, ValueError):
                    errors.append(f"row {index} {metric} is invalid")
    elif performance.is_file():
        errors.append("performance.csv has no rows")
    if not log_path.is_file():
        errors.append("missing job.log")
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        match = LOG_FAILURE_RE.search(log_text)
        if match:
            errors.append(f"failure marker in log: {match.group(0)!r}")
    args_path = attempt / "args.json"
    if not args_path.is_file():
        errors.append("missing args.json")
    return errors


def completed_attempt(job_directory, name, dataset, method, seed, common):
    pointer = job_directory / "complete.json"
    if not pointer.is_file():
        return None, ["missing complete.json"]
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        attempt = Path(payload["attempt"])
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        return None, [f"invalid complete.json: {exc}"]
    errors = validate_attempt(attempt, name, dataset, method, seed, common)
    if errors:
        return None, errors
    return attempt, []


def next_attempt(job_directory):
    attempts = job_directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in attempts.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_")[-1]))
        except ValueError:
            pass
    number = max(numbers, default=0) + 1
    path = attempts / f"attempt_{number:04d}"
    path.mkdir(parents=False, exist_ok=False)
    return number, path


def gpu_snapshot(gpu_id):
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if query.returncode != 0:
        raise RuntimeError(query.stderr.strip() or "nvidia-smi GPU query failed")
    parts = [part.strip() for part in query.stdout.strip().split(",")]
    if len(parts) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {query.stdout!r}")
    process_query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    pids = []
    if process_query.returncode == 0:
        pids = [int(line.strip()) for line in process_query.stdout.splitlines() if line.strip().isdigit()]
    return {
        "timestamp": parts[0],
        "name": parts[1],
        "memory_used_mib": float(parts[2]),
        "memory_total_mib": float(parts[3]),
        "utilization_percent": float(parts[4]),
        "power_watts": float(parts[5]),
        "compute_pids": pids,
    }


def wait_for_gpu(gpu_id, idle_seconds, poll_seconds, state_file=None):
    stable_since = None
    while True:
        if STOP_REQUESTED:
            raise InterruptedError("stop requested while waiting for GPU")
        try:
            snapshot = gpu_snapshot(gpu_id)
            idle = (
                not snapshot["compute_pids"]
                and snapshot["memory_used_mib"] <= 512
                and snapshot["utilization_percent"] <= 5
            )
        except Exception as exc:
            snapshot = {"error": str(exc)}
            idle = False
        if idle:
            if stable_since is None:
                stable_since = time.monotonic()
            stable_for = time.monotonic() - stable_since
        else:
            stable_since = None
            stable_for = 0
        if state_file:
            atomic_json(
                state_file,
                {
                    "state": "WAITING_GPU",
                    "updated_utc": utc_now(),
                    "stable_idle_seconds": round(stable_for, 1),
                    "required_idle_seconds": idle_seconds,
                    "gpu": snapshot,
                },
            )
        if idle and stable_for >= idle_seconds:
            return snapshot
        time.sleep(poll_seconds)


class GpuMonitor:
    def __init__(self, path, gpu_id, interval):
        self.path = Path(path)
        self.gpu_id = gpu_id
        self.interval = interval
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.peak = 0.0
        self.thread = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(
                "observed_utc,gpu_timestamp,name,memory_used_mib,memory_total_mib,utilization_percent,power_watts,compute_pids\n",
                encoding="utf-8",
            )
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                item = gpu_snapshot(self.gpu_id)
                with self.lock:
                    self.peak = max(self.peak, item["memory_used_mib"])
                with self.path.open("a", encoding="utf-8") as stream:
                    values = [
                        utc_now(),
                        item["timestamp"],
                        item["name"],
                        item["memory_used_mib"],
                        item["memory_total_mib"],
                        item["utilization_percent"],
                        item["power_watts"],
                        "|".join(map(str, item["compute_pids"])),
                    ]
                    stream.write(",".join(map(str, values)) + "\n")
            except Exception as exc:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(f"{utc_now()},ERROR,{str(exc).replace(',', ';')}\n")
            self.stop_event.wait(self.interval)

    def reset_peak(self):
        with self.lock:
            self.peak = 0.0

    def peak_memory(self):
        with self.lock:
            return self.peak

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=min(self.interval + 1, 5))


def signal_handler(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    child = CURRENT_CHILD
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "pilot_il2_3seed.json",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--scratch", type=Path, default=Path("/dev/shm/chenyuming-discobax/jobs"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--datasets")
    parser.add_argument("--methods")
    parser.add_argument("--seeds")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-idle-seconds", type=int, default=300)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--monitor-seconds", type=int, default=30)
    parser.add_argument("--profile-kind", choices=("FULL_RUN", "SMOKE"), default="FULL_RUN")
    parser.add_argument("--state-file", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_pilot_config(config)
    datasets = parse_csv(args.datasets) or config["datasets"]
    methods = parse_csv(args.methods) or config["methods"]
    seeds = parse_csv(args.seeds, int) or config["seeds"]
    jobs = list(itertools.product(datasets, methods, seeds))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if config["name"] == "pilot_il2_3seed" and not any(
        [args.datasets, args.methods, args.seeds, args.limit]
    ) and len(jobs) != 24:
        parser.error(f"pilot must contain exactly 24 jobs, got {len(jobs)}")

    cache = args.cache.resolve()
    missing = [name for name in required_cache_files(config) if not (cache / name).is_file()]
    if missing:
        parser.error(f"required exact-cache files are missing at {cache}: {missing}")
    output = args.output.resolve()
    logs = args.logs.resolve()
    scratch_root = args.scratch.resolve()
    common = config["common"]
    manifest = output / f"{config['name']}_manifest.jsonl"
    state_file = args.state_file or output / "control" / "matrix_state.json"

    header = {
        "profile": config["name"],
        "profile_kind": args.profile_kind,
        "jobs": len(jobs),
        "cache": str(cache),
        "output": str(output),
        "logs": str(logs),
        "scratch": str(scratch_root),
        "device": common["device"],
        "config_sha256": sha256(args.config),
    }
    print(json.dumps(header, indent=2), flush=True)
    if args.dry_run:
        for index, (dataset, method, seed) in enumerate(jobs, start=1):
            name = run_name(dataset, method, seed, common)
            attempt = output / "jobs" / name / "attempts" / "DRY_RUN"
            command, params = build_command(
                args.python,
                cache,
                attempt / "artifacts",
                attempt / "performance.csv",
                scratch_root / name / "DRY_RUN",
                dataset,
                method,
                seed,
                common,
            )
            print(json.dumps({"index": index, "name": name, "params": params, "command": command}, sort_keys=True))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(WORKSPACE_ROOT), env.get("PYTHONPATH", "")]
    )

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_handler)
    monitor = GpuMonitor(logs / "gpu_monitor.csv", args.gpu_id, args.monitor_seconds)
    monitor.start()
    failures = 0
    completed_count = 0
    global CURRENT_CHILD
    try:
        for index, (dataset, method, seed) in enumerate(jobs, start=1):
            if STOP_REQUESTED:
                break
            name = run_name(dataset, method, seed, common)
            job_directory = output / "jobs" / name
            prior_attempt, prior_errors = completed_attempt(
                job_directory, name, dataset, method, seed, common
            )
            if prior_attempt is not None and not args.rerun:
                completed_count += 1
                print(f"[{index}/{len(jobs)}] skip validated COMPLETE {name}", flush=True)
                continue
            if (job_directory / "complete.json").exists() and prior_errors:
                print(f"[{index}/{len(jobs)}] stale completion for {name}: {prior_errors}", flush=True)
            if args.wait_for_gpu:
                print(f"[{index}/{len(jobs)}] wait for GPU {args.gpu_id}: {name}", flush=True)
                wait_for_gpu(
                    args.gpu_id,
                    args.gpu_idle_seconds,
                    args.gpu_poll_seconds,
                    state_file,
                )
            attempt_number, attempt = next_attempt(job_directory)
            artifact_base = attempt / "artifacts"
            attempt_scratch = scratch_root / name / f"attempt_{attempt_number:04d}"
            attempt_scratch.mkdir(parents=True, exist_ok=True)
            performance = attempt / "performance.csv"
            log_path = logs / "jobs" / name / f"attempt_{attempt_number:04d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command, params = build_command(
                args.python,
                cache,
                artifact_base,
                performance,
                attempt_scratch,
                dataset,
                method,
                seed,
                common,
            )
            args_record = {
                **header,
                "name": name,
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "attempt": attempt_number,
                "params": params,
                "command": command,
                "started_utc": utc_now(),
            }
            atomic_json(attempt / "args.json", args_record)
            atomic_json(
                state_file,
                {
                    "state": "RUNNING",
                    "profile": config["name"],
                    "profile_kind": args.profile_kind,
                    "progress_index": index,
                    "total": len(jobs),
                    "completed": completed_count,
                    "failed": failures,
                    "current": {"name": name, "dataset": dataset, "method": method, "seed": seed, "attempt": attempt_number},
                    "started_utc": args_record["started_utc"],
                    "updated_utc": utc_now(),
                },
            )
            print(f"[{index}/{len(jobs)}] run {name} attempt={attempt_number}", flush=True)
            monitor.reset_peak()
            started_monotonic = time.monotonic()
            with log_path.open("w", encoding="utf-8") as stream:
                CURRENT_CHILD = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                returncode = CURRENT_CHILD.wait()
            CURRENT_CHILD = None
            # CIFS on the approved persistent store does not support creating
            # symlinks.  Keep the canonical job log under logs/ and copy the
            # immutable completed stream into the attempt for self-contained
            # validation/audit.
            shutil.copyfile(log_path, attempt / "job.log")
            runtime = round(time.monotonic() - started_monotonic, 3)
            errors = []
            if returncode != 0:
                errors.append(f"subprocess return code {returncode}")
            errors.extend(validate_attempt(attempt, name, dataset, method, seed, common))
            status = "COMPLETE" if not errors else ("INTERRUPTED" if STOP_REQUESTED else "FAILED")
            record = {
                **args_record,
                "status": status,
                "returncode": returncode,
                "runtime_seconds": runtime,
                "finished_utc": utc_now(),
                "peak_gpu_memory_mib": monitor.peak_memory(),
                "attempt_path": str(attempt),
                "log": str(log_path),
                "result": str(artifact_base / name / "run_results.pickle"),
                "validation_errors": errors,
            }
            atomic_json(attempt / "manifest.json", record)
            append_jsonl(manifest, record)
            if not errors:
                atomic_json(
                    job_directory / "complete.json",
                    {
                        "status": "COMPLETE",
                        "profile_kind": args.profile_kind,
                        "attempt": str(attempt),
                        "manifest": str(attempt / "manifest.json"),
                        "validated_utc": utc_now(),
                    },
                )
                completed_count += 1
                print(f"complete {name} in {runtime:.1f}s", flush=True)
            else:
                failures += 1
                print(f"{status}: {name}: {'; '.join(errors)}", flush=True)
                if not args.keep_going or STOP_REQUESTED:
                    break
            # The scratch checkpoint is reproducible and not an audit artifact.
            shutil.rmtree(attempt_scratch, ignore_errors=True)
    except InterruptedError:
        pass
    finally:
        monitor.stop()
        CURRENT_CHILD = None

    final_state = "INTERRUPTED" if STOP_REQUESTED else ("FAILED" if failures else "COMPLETE")
    atomic_json(
        state_file,
        {
            "state": final_state,
            "profile": config["name"],
            "profile_kind": args.profile_kind,
            "total": len(jobs),
            "completed": completed_count,
            "failed": failures,
            "updated_utc": utc_now(),
        },
    )
    if STOP_REQUESTED:
        return 130
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
