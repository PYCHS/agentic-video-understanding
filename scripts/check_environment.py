"""Read-only environment inventory; does not load models or certify compatibility."""
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import subprocess


def inventory():
    packages = {}
    for name in ("torch", "transformers", "accelerate", "av", "Pillow", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True)
        gpu = {"status":"reported", "csv":result.stdout.strip(), "error":None}
    except (OSError, subprocess.SubprocessError) as exc:
        gpu = {"status":"unavailable", "csv":None, "error":str(exc)}
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, nvidia_smi=gpu, inference_verified=False)


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
