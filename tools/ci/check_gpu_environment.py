# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate and record the CUDA runtime expected by GPU CI jobs."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when torch cannot see a CUDA device.",
    )
    parser.add_argument(
        "--min-device-count",
        type=int,
        default=1,
        help="Minimum CUDA device count required when --require-cuda is set.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON path for environment metadata.",
    )
    return parser.parse_args()


def _nvidia_smi() -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "error": "nvidia-smi not found"}

    proc = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "available": proc.returncode == 0,
        "return_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _torch_cuda_metadata() -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:
        metadata["torch_import_error"] = repr(exc)
        metadata["cuda_available"] = False
        metadata["cuda_device_count"] = 0
        return metadata

    metadata["torch_version"] = getattr(torch, "__version__", "unknown")
    metadata["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["cuda_device_count"] = (
        int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    )
    devices: List[Dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory": int(properties.total_memory),
                }
            )
    metadata["devices"] = devices
    return metadata


def _write_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    metadata: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi": _nvidia_smi(),
    }
    metadata.update(_torch_cuda_metadata())

    if args.metadata:
        _write_metadata(args.metadata, metadata)

    print(json.dumps(metadata, indent=2, sort_keys=True))

    if not args.require_cuda:
        return 0

    if not metadata.get("cuda_available"):
        print(
            "CUDA is required but torch.cuda.is_available() is false.", file=sys.stderr
        )
        return 1

    device_count = int(metadata.get("cuda_device_count", 0))
    if device_count < args.min_device_count:
        print(
            f"CUDA device count {device_count} is below required "
            f"{args.min_device_count}.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
