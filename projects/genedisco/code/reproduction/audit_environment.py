#!/usr/bin/env python3
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import gpytorch
import numpy
import sklearn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DECLARED_UPSTREAM_COMMIT = "84f01283bc7f6ab5f66b5ea2a63632b401cc0402"
EXPECTED_ARCHIVE_SIZE = 246500391
EXPECTED_ARCHIVE_MD5 = "9f2fb895e32c85377e4cf1b2d2658ed9"
HASHED_SOURCE_FILES = [
    "discobax/methods/bax_acquisition/bax_sampling.py",
    "discobax/models/consistent_mc_dropout.py",
    "discobax/models/meta_models.py",
    "discobax/apps/genedisco_experiment.py",
    "reproduction/run_matrix.py",
    "reproduction/pilot_controller.py",
    "reproduction/configs/pilot_il2_3seed.json",
    "run_reproduction.sh",
]


def digest(path, algorithm):
    value = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_state():
    if not (REPO_ROOT / ".git").exists():
        return {
            "kind": "uploaded_source_without_git",
            "declared_upstream_commit": DECLARED_UPSTREAM_COMMIT,
            "head": None,
            "dirty": None,
        }
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=REPO_ROOT, text=True
            ).strip()
        )
        return {
            "kind": "git_checkout",
            "declared_upstream_commit": DECLARED_UPSTREAM_COMMIT,
            "head": head,
            "dirty": dirty,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "kind": "git_unreadable",
            "declared_upstream_commit": DECLARED_UPSTREAM_COMMIT,
            "error": str(exc),
        }


def disk_payload(path):
    usage = shutil.disk_usage(path)
    return {
        "path": str(Path(path).resolve()),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / "data" / "release" / "data"
    )
    parser.add_argument(
        "--archive", type=Path, default=REPO_ROOT / "data" / "DiscoBAX_GeneDisco_datasets.zip"
    )
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--hash-data", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cache = args.cache.resolve()
    required_files = [
        "achilles.h5",
        "schmidt_2021_il2.h5",
        "clusters_schmidt_2021_il2_achilles_0.01_topk_20_clusters_to_items.pkl",
        "clusters_schmidt_2021_il2_achilles_0.01_topk_items_to_20_clusters.pkl",
    ]
    files = {}
    for name in required_files:
        path = cache / name
        files[name] = {
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
        }
        if args.hash_data and path.is_file():
            files[name]["md5"] = digest(path, "md5")

    archive = args.archive.resolve()
    archive_payload = {
        "path": str(archive),
        "exists": archive.is_file(),
        "expected_size": EXPECTED_ARCHIVE_SIZE,
        "expected_md5": EXPECTED_ARCHIVE_MD5,
    }
    if archive.is_file():
        archive_payload.update(
            {
                "size": archive.stat().st_size,
                "md5": digest(archive, "md5"),
            }
        )
        archive_payload["valid"] = (
            archive_payload["size"] == EXPECTED_ARCHIVE_SIZE
            and archive_payload["md5"] == EXPECTED_ARCHIVE_MD5
        )
    else:
        archive_payload["valid"] = False

    source_hashes = {}
    for relative in HASHED_SOURCE_FILES:
        path = REPO_ROOT / relative
        source_hashes[relative] = digest(path, "sha256") if path.is_file() else None

    cuda_available = torch.cuda.is_available()
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpytorch": gpytorch.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "genedisco": package_version("genedisco"),
            "slingpy": package_version("slingpy"),
        },
        "cuda": {
            "available": cuda_available,
            "device_count_visible": torch.cuda.device_count() if cuda_available else 0,
            "device_0": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source": git_state(),
        "source_sha256": source_hashes,
        "cache": str(cache),
        "archive": archive_payload,
        "files": files,
        "disk": {"project": disk_payload(REPO_ROOT)},
    }
    if args.storage:
        payload["disk"]["storage"] = disk_payload(args.storage)
    if args.scratch:
        payload["disk"]["scratch"] = disk_payload(args.scratch)
    payload["valid"] = bool(
        archive_payload["valid"]
        and all(item["exists"] for item in files.values())
        and cuda_available
        and payload["packages"]["genedisco"] == "1.0.5"
        and payload["packages"]["slingpy"] == "0.2.11"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
