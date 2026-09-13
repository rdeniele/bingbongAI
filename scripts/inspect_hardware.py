"""Inspect the local machine and report what BingBongAI can realistically train.

This script only *measures*. It never estimates silently: anything it could not
determine is reported as "unknown" rather than guessed.

Usage:
    python scripts/inspect_hardware.py
    python scripts/inspect_hardware.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(cmd: list[str]) -> str | None:
    """Run a command, returning stripped stdout, or None if it is unavailable."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def collect_os() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def collect_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "64bit": sys.maxsize > 2**32,
    }


def collect_cpu() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cores": "unknown", "physical_cores": "unknown"}
    try:
        import os

        info["logical_cores"] = os.cpu_count()
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        import psutil  # optional dependency

        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return info


def collect_memory() -> dict[str, Any]:
    """Total/available system RAM in GiB, or 'unknown' when psutil is absent."""
    try:
        import psutil
    except ImportError:
        return {"total_gib": "unknown", "available_gib": "unknown", "source": "psutil not installed"}
    vm = psutil.virtual_memory()
    return {
        "total_gib": round(vm.total / 1024**3, 2),
        "available_gib": round(vm.available / 1024**3, 2),
        "source": "psutil",
    }


def collect_disk(path: str = ".") -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gib": round(usage.total / 1024**3, 2),
        "free_gib": round(usage.free / 1024**3, 2),
    }


def collect_nvidia_smi() -> dict[str, Any]:
    """Query the NVIDIA driver directly. Works even when PyTorch is not installed."""
    query = "name,memory.total,memory.used,driver_version,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if out is None:
        return {"available": False, "reason": "nvidia-smi not found or returned an error"}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "name": parts[0],
                "memory_total": parts[1],
                "memory_used": parts[2],
                "driver_version": parts[3],
                "compute_capability": parts[4],
            }
        )
    return {"available": True, "gpus": gpus}


def collect_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False, "reason": "torch is not installed in this interpreter"}

    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_built_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    return info


def collect_numpy() -> dict[str, Any]:
    try:
        import numpy
    except ImportError:
        return {"installed": False}
    return {"installed": True, "version": numpy.__version__}


def collect() -> dict[str, Any]:
    return {
        "os": collect_os(),
        "python": collect_python(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "disk": collect_disk(),
        "nvidia_smi": collect_nvidia_smi(),
        "torch": collect_torch(),
        "numpy": collect_numpy(),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def print_report(report: dict[str, Any]) -> None:
    print("=" * 56)
    print("  BingBongAI - Environment Report")
    print("=" * 56)

    for section, data in report.items():
        print(f"\n[{section}]")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    if not value:
                        print(f"  {key}: (none)")
                    for item in value:
                        print(f"  {key}:")
                        for k, v in item.items():
                            print(f"      {k}: {_fmt(v)}")
                else:
                    print(f"  {key}: {_fmt(value)}")

    print("\n" + "-" * 56)
    torch_info = report["torch"]
    if not torch_info["installed"]:
        print("PyTorch is NOT installed. Training is not possible yet.")
    elif torch_info["cuda_available"]:
        print("PyTorch + CUDA are available. GPU training is possible.")
    else:
        print("PyTorch is installed but CUDA is NOT available: CPU-only training.")
    print("-" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the BingBongAI development environment.")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a report")
    args = parser.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
