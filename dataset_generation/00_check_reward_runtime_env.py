from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_output(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return "missing"
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def main() -> None:
    result: dict[str, object] = {
        "python": sys.executable,
        "version": sys.version.replace("\n", " "),
        "modules": {},
    }
    modules = [
        "torch",
        "open_clip",
        "clip",
        "torchvision",
        "cv2",
        "numpy",
        "pandas",
        "sklearn",
        "mani_skill",
        "sapien",
        "mplib",
    ]
    for name in modules:
        result["modules"][name] = module_available(name)

    if module_available("torch"):
        import torch

        torch_info: dict[str, object] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            torch_info["cuda_device_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3),
                2,
            )
        result["torch_info"] = torch_info

    result["nvidia_smi"] = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
        ]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
