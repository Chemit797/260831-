#!/usr/bin/env python3
"""Background lifecycle controller for the A100 IL-2 pilot."""

import argparse
import datetime as dt
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "reproduction" / "configs" / "pilot_il2_3seed.json"
SMOKE_CONFIG = REPO_ROOT / "reproduction" / "configs" / "pilot_il2_smoke.json"
CACHE = REPO_ROOT / "data" / "release" / "data"
ARCHIVE = REPO_ROOT / "data" / "DiscoBAX_GeneDisco_datasets.zip"
DEFAULT_STORE = Path("/mnt/Omics_GPU/chenyuming/DiscoBAX/pilot_il2_3seed")
DEFAULT_RUNTIME = Path("/dev/shm/chenyuming-discobax")
MIN_STORE_FREE = 20 * 1024**3
MIN_RUNTIME_FREE = 15 * 1024**3
MIN_ROOT_FREE = 2 * 1024**3
STOP_REQUESTED = False
CURRENT_CHILD = None


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def store_root():
    return Path(os.environ.get("DISCOBAX_PILOT_STORE", DEFAULT_STORE)).resolve()


def runtime_root():
    return Path(os.environ.get("DISCOBAX_PILOT_RUNTIME", DEFAULT_RUNTIME)).resolve()


def runtime_control():
    preferred = Path(f"/run/user/{os.getuid()}")
    base = preferred if preferred.is_dir() and os.access(preferred, os.W_OK) else Path("/tmp")
    return base / "discobax-pilot-il2"


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


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_state(state, **extra):
    payload = {"state": state, "updated_utc": utc_now(), **extra}
    runtime_path = runtime_control() / "state.json"
    atomic_json(runtime_path, payload)
    persistent = store_root() / "control" / "state.json"
    try:
        atomic_json(persistent, payload)
    except OSError:
        pass
    return payload


def log(message):
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    try:
        path = store_root() / "logs" / "controller.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except OSError:
        pass


def proc_start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        return None


def identity_path():
    return runtime_control() / "controller.json"


def valid_identity(payload=None):
    payload = payload or read_json(identity_path())
    if not payload:
        return False
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if proc_start_ticks(pid) != str(payload.get("start_ticks")):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "pilot_controller.py run" in command and payload.get("repo_root") == str(REPO_ROOT)


def signal_handler(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    child = CURRENT_CHILD
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def free_bytes(path):
    return shutil.disk_usage(path).free


def storage_snapshot():
    store = store_root()
    runtime = runtime_root()
    mount = Path("/mnt/Omics_GPU")
    return {
        "store": str(store),
        "store_mount_present": mount.is_mount(),
        "store_writable": mount.is_dir() and os.access(mount, os.W_OK),
        "store_free_bytes": free_bytes(mount) if mount.is_dir() else 0,
        "runtime": str(runtime),
        "runtime_free_bytes": free_bytes("/dev/shm"),
        "root_free_bytes": free_bytes(REPO_ROOT),
    }


def storage_ready(snapshot):
    return bool(
        snapshot["store_mount_present"]
        and snapshot["store_writable"]
        and snapshot["store_free_bytes"] >= MIN_STORE_FREE
        and snapshot["runtime_free_bytes"] >= MIN_RUNTIME_FREE
        and snapshot["root_free_bytes"] >= MIN_ROOT_FREE
    )


def wait_for_storage():
    while not STOP_REQUESTED:
        snapshot = storage_snapshot()
        if storage_ready(snapshot):
            store_root().mkdir(parents=True, exist_ok=True)
            runtime_root().mkdir(parents=True, exist_ok=True)
            return snapshot
        write_state(
            "WAITING_STORAGE",
            storage=snapshot,
            requirements={
                "store_free_bytes": MIN_STORE_FREE,
                "runtime_free_bytes": MIN_RUNTIME_FREE,
                "root_free_bytes": MIN_ROOT_FREE,
            },
        )
        log(f"waiting for storage: {snapshot}")
        for _ in range(60):
            if STOP_REQUESTED:
                return None
            time.sleep(1)
    return None


def ensure_link(link, target):
    link = Path(link)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"refusing to replace symlink {link} -> {link.resolve()}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing path {link}")
    link.symlink_to(target, target_is_directory=True)


def configure_storage_links():
    store = store_root()
    ensure_link(REPO_ROOT / "results" / "pilot_il2_3seed", store / "results")
    ensure_link(REPO_ROOT / "logs" / "pilot_il2_3seed", store / "logs")


def find_conda():
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
        str(Path.home() / "anaconda3" / "bin" / "conda"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("conda executable not found")


def command_env():
    runtime = runtime_root()
    env = os.environ.copy()
    directories = {
        "CONDA_PKGS_DIRS": runtime / "conda-pkgs",
        "PIP_CACHE_DIR": runtime / "pip-cache",
        "TMPDIR": runtime / "tmp",
    }
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_command(command, label, output=None, check=True):
    global CURRENT_CHILD
    if STOP_REQUESTED:
        raise InterruptedError(f"stop requested before {label}")
    log(f"{label}: {' '.join(map(str, command))}")
    path = Path(output) if output else store_root() / "logs" / "controller.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if output else "a"
    with path.open(mode, encoding="utf-8") as stream:
        CURRENT_CHILD = subprocess.Popen(
            list(map(str, command)),
            cwd=REPO_ROOT,
            env=command_env(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        returncode = CURRENT_CHILD.wait()
    CURRENT_CHILD = None
    if check and returncode != 0:
        raise RuntimeError(f"{label} failed with return code {returncode}; see {path}")
    return returncode


def environment_prefix():
    return runtime_root() / "envs" / "genedisco-repro"


def environment_python():
    return environment_prefix() / "bin" / "python"


def environment_is_valid():
    python = environment_python()
    if not python.is_file():
        return False
    probe = (
        "import sys,torch,gpytorch,numpy,sklearn,importlib.metadata as m;"
        "assert sys.version_info[:2]==(3,8);"
        "assert torch.__version__=='2.4.1+cu124';"
        "assert gpytorch.__version__=='1.11';"
        "assert numpy.__version__=='1.24.4';"
        "assert sklearn.__version__=='1.3.2';"
        "assert m.version('genedisco')=='1.0.5';"
        "assert m.version('slingpy')=='0.2.11';"
        "assert torch.cuda.is_available();"
        "assert 'A100' in torch.cuda.get_device_name(0)"
    )
    return subprocess.run(
        [str(python), "-c", probe],
        cwd=REPO_ROOT,
        env=command_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def ensure_environment():
    if environment_is_valid():
        log(f"validated existing environment {environment_prefix()}")
        return
    prefix = environment_prefix()
    python = environment_python()
    reusable = False
    if python.is_file():
        reusable = subprocess.run(
            [str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 8)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    if prefix.exists() and not reusable:
        # The prefix is disposable scratch; persistent experiment artifacts are
        # never placed under /dev/shm.
        shutil.rmtree(prefix)
    write_state("PREPARING_ENVIRONMENT", environment=str(prefix))
    if not reusable:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        conda = find_conda()
        run_command(
            [
                conda,
                "create",
                "-y",
                "--override-channels",
                "--channel",
                "conda-forge",
                "-p",
                prefix,
                "python=3.8.20",
                "pip",
            ],
            "create pinned conda environment",
        )
        python = environment_python()
    else:
        log(f"reusing partial Python 3.8 environment {prefix}")
    run_command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
            "torch==2.4.1+cu124",
        ],
        "install CUDA PyTorch",
    )
    run_command(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            REPO_ROOT / "reproduction" / "requirements-gpu.txt",
        ],
        "install pinned reproduction dependencies",
    )
    run_command(
        [python, "-m", "pip", "install", "--no-deps", "-e", REPO_ROOT],
        "install workspace DiscoBAX",
    )
    if not environment_is_valid():
        raise RuntimeError("created environment did not pass the pinned version/CUDA probe")


def run_audit_and_tests():
    python = environment_python()
    results = store_root() / "results"
    results.mkdir(parents=True, exist_ok=True)
    write_state("VALIDATING", phase="environment_audit")
    run_command(
        [
            python,
            REPO_ROOT / "reproduction" / "audit_environment.py",
            "--cache",
            CACHE,
            "--archive",
            ARCHIVE,
            "--storage",
            store_root(),
            "--scratch",
            runtime_root(),
            "--output",
            results / "pilot_il2_3seed_environment.json",
        ],
        "environment and provenance audit",
    )
    write_state("VALIDATING", phase="tests")
    run_command(
        [python, "-m", "pytest", "-q", REPO_ROOT / "tests"],
        "complete unit test suite",
        output=store_root() / "logs" / "pytest.log",
    )


def run_dry_run():
    python = environment_python()
    output = store_root() / "results" / "control" / "pilot_dry_run.jsonl"
    write_state("VALIDATING", phase="dry_run")
    run_command(
        [
            python,
            REPO_ROOT / "reproduction" / "run_matrix.py",
            "--config",
            CONFIG,
            "--cache",
            CACHE,
            "--output",
            store_root() / "results",
            "--logs",
            store_root() / "logs",
            "--scratch",
            runtime_root() / "jobs",
            "--python",
            python,
            "--dry-run",
        ],
        "24-job pilot dry-run",
        output=output,
    )
    records = []
    for line in output.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "index" in item and "params" in item:
            records.append(item)
    if len(records) != 24 or [item["index"] for item in records] != list(range(1, 25)):
        raise RuntimeError(f"dry-run must contain exactly 24 indexed jobs; got {len(records)}")


def matrix_command(config, output, logs, profile_kind, monitor_seconds):
    python = environment_python()
    return [
        python,
        REPO_ROOT / "reproduction" / "run_matrix.py",
        "--config",
        config,
        "--cache",
        CACHE,
        "--output",
        output,
        "--logs",
        logs,
        "--scratch",
        runtime_root() / "jobs" / profile_kind.lower(),
        "--python",
        python,
        "--keep-going",
        "--wait-for-gpu",
        "--gpu-id",
        "0",
        "--gpu-idle-seconds",
        os.environ.get("DISCOBAX_GPU_IDLE_SECONDS", "300"),
        "--gpu-poll-seconds",
        os.environ.get("DISCOBAX_GPU_POLL_SECONDS", "30"),
        "--monitor-seconds",
        str(monitor_seconds),
        "--profile-kind",
        profile_kind,
        "--state-file",
        output / "control" / "matrix_state.json",
    ]


def run_smoke():
    output = store_root() / "smoke" / "results"
    logs = store_root() / "smoke" / "logs"
    write_state("SMOKE", phase="full_mc_one_cycle")
    run_command(
        matrix_command(SMOKE_CONFIG, output, logs, "SMOKE", 1),
        "IL-2 DiscoBAX full-MC one-cycle smoke",
    )


def run_formal_matrix():
    output = store_root() / "results"
    logs = store_root() / "logs"
    write_state("RUNNING", phase="formal_24_job_matrix")
    return run_command(
        matrix_command(CONFIG, output, logs, "FULL_RUN", 30),
        "24-job formal pilot matrix",
        check=False,
    )


def run_summary():
    output = store_root() / "results"
    return run_command(
        [
            sys.executable,
            REPO_ROOT / "reproduction" / "summarize_performance.py",
            "--pilot-root",
            output,
            "--config",
            CONFIG,
        ],
        "strict pilot summary",
        check=False,
    )


def controller_run():
    runtime_control().mkdir(parents=True, exist_ok=True)
    lock_path = runtime_control() / "runner.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: another pilot controller holds the runner lock", file=sys.stderr)
        return 2
    controller_id = str(uuid.uuid4())
    identity = {
        "pid": os.getpid(),
        "start_ticks": proc_start_ticks(os.getpid()),
        "controller_id": controller_id,
        "repo_root": str(REPO_ROOT),
        "started_utc": utc_now(),
    }
    atomic_json(identity_path(), identity)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_handler)
    try:
        write_state("STARTING", controller=identity)
        snapshot = wait_for_storage()
        if snapshot is None:
            raise InterruptedError("stopped while waiting for storage")
        configure_storage_links()
        log(f"storage ready: {snapshot}")
        ensure_environment()
        run_audit_and_tests()
        run_dry_run()
        run_smoke()
        matrix_rc = run_formal_matrix()
        summary_rc = run_summary()
        if matrix_rc == 0 and summary_rc == 0:
            write_state("COMPLETE", controller=identity)
            log("pilot completed and strict summary succeeded")
            return 0
        write_state(
            "FAILED",
            controller=identity,
            matrix_returncode=matrix_rc,
            summary_returncode=summary_rc,
        )
        return 1
    except InterruptedError as exc:
        write_state("INTERRUPTED", controller=identity, reason=str(exc))
        log(f"controller interrupted: {exc}")
        return 130
    except Exception as exc:
        write_state("FAILED", controller=identity, reason=str(exc))
        log(f"controller failed: {exc}")
        return 1
    finally:
        try:
            identity_path().unlink()
        except FileNotFoundError:
            pass
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


def all_jobs(config):
    return list(itertools.product(config["datasets"], config["methods"], config["seeds"]))


def run_name(dataset, method, seed, common):
    hyper = ""
    if method == "discobax":
        hyper = common["bax_subset_select_subset_size"]
    elif method == "topk_bax":
        hyper = common["bax_topk_kvalue"]
    elif method == "levelset_bax":
        hyper = common["bax_level_set_c"]
    return "_".join(map(str, [dataset, common["feature_set_name"], common["topk_percent"], common["model_name"], method, hyper, seed]))


def status_payload():
    persistent = store_root() / "control" / "state.json"
    state = read_json(persistent) or read_json(runtime_control() / "state.json") or {"state": "NOT_STARTED"}
    identity = read_json(identity_path())
    running = valid_identity(identity)
    config = read_json(CONFIG, {})
    matrix_state = read_json(store_root() / "results" / "control" / "matrix_state.json", {})
    jobs = all_jobs(config) if config else []
    common = config.get("common", {})
    completed = 0
    failed = 0
    durations = {}
    manifest = store_root() / "results" / "pilot_il2_3seed_manifest.jsonl"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            durations[item.get("name")] = item
    for dataset, method, seed in jobs:
        name = run_name(dataset, method, seed, common)
        if (store_root() / "results" / "jobs" / name / "complete.json").is_file():
            completed += 1
        elif durations.get(name, {}).get("status") == "FAILED":
            failed += 1
    current = matrix_state.get("current")
    current_cycle = None
    if current:
        attempt = current.get("attempt")
        name = current.get("name")
        if attempt and name:
            directory = store_root() / "results" / "jobs" / name / "attempts" / f"attempt_{int(attempt):04d}" / "artifacts" / name
            current_cycle = sum(
                1
                for path in directory.glob("cycle_*")
                if path.is_dir() and (path / "run_results.pickle").is_file()
            )
    method_averages = {}
    for item in durations.values():
        if item.get("status") == "COMPLETE" and item.get("runtime_seconds") is not None:
            method_averages.setdefault(item["method"], []).append(float(item["runtime_seconds"]))
    eta = 0.0
    eta_known = True
    for dataset, method, seed in jobs:
        name = run_name(dataset, method, seed, common)
        if (store_root() / "results" / "jobs" / name / "complete.json").is_file():
            continue
        values = method_averages.get(method)
        if values:
            eta += sum(values) / len(values)
        else:
            eta_known = False
    return {
        "profile": "pilot_il2_3seed",
        "state": matrix_state.get("state") if running and matrix_state else state.get("state"),
        "controller_running": running,
        "pid": identity.get("pid") if running else None,
        "progress": f"{completed}/{len(jobs) or 24}",
        "completed": completed,
        "failed": failed,
        "remaining": (len(jobs) or 24) - completed,
        "current": current,
        "current_cycle": current_cycle,
        "eta_seconds": round(eta) if eta_known and eta else None,
        "storage": storage_snapshot(),
        "controller_log": str(store_root() / "logs" / "controller.log"),
        "matrix_state": matrix_state,
    }


def print_status():
    payload = status_payload()
    print(f"profile          {payload['profile']}")
    print(f"state            {payload['state']}")
    print(f"controller       {'RUNNING' if payload['controller_running'] else 'STOPPED'} (PID {payload['pid'] or '-'})")
    print(f"progress         {payload['progress']}")
    print(f"completed        {payload['completed']}")
    print(f"failed           {payload['failed']}")
    print(f"remaining        {payload['remaining']}")
    if payload["current"]:
        item = payload["current"]
        print(f"current          {item.get('dataset')} / {item.get('method')} / seed={item.get('seed')}")
        print(f"current cycle    {payload['current_cycle'] or 0}/25")
    if payload["eta_seconds"] is not None:
        print(f"ETA              {dt.timedelta(seconds=payload['eta_seconds'])}")
    else:
        print("ETA              pending same-method timing")
    storage = payload["storage"]
    print(f"store free       {storage['store_free_bytes'] / 1024**3:.1f} GiB")
    print(f"runtime free     {storage['runtime_free_bytes'] / 1024**3:.1f} GiB")
    print(f"controller log   {payload['controller_log']}")
    return 0


def stop_controller():
    identity = read_json(identity_path())
    if not valid_identity(identity):
        print("No validated active pilot controller.")
        return 0
    pid = int(identity["pid"])
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to validated pilot controller PID {pid}.")
    return 0


def doctor():
    runtime_control().mkdir(parents=True, exist_ok=True)
    lock_stream = (runtime_control() / "runner.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("pilot controller/doctor is already active")
    try:
        snapshot = storage_snapshot()
        if not storage_ready(snapshot):
            print(json.dumps(snapshot, indent=2))
            raise RuntimeError("storage gates are not satisfied")
        configure_storage_links()
        ensure_environment()
        run_audit_and_tests()
        run_dry_run()
        write_state("READY", phase="doctor_complete")
        return 0
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "stop", "doctor", "summarize"))
    args = parser.parse_args()
    if args.command == "run":
        return controller_run()
    if args.command == "status":
        return print_status()
    if args.command == "stop":
        return stop_controller()
    if args.command == "doctor":
        return doctor()
    if args.command == "summarize":
        return run_summary()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
