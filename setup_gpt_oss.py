#!/usr/bin/env python3
"""
setup_gpt_oss.py — Automated environment setup for OpenAI’s gpt oss models

This script performs an end‑to‑end setup for working with OpenAI’s open‑weight
GPT‑OSS models.  It will:

1. Inspect your system to determine available GPU VRAM and system RAM.
2. Decide which GPT‑OSS model(s) to download (20b or 120b) based on available
   resources.
3. Ensure required Python dependencies are installed (via an existing
   ``requirements.txt`` file or direct installation).
4. Create a directory structure under the specified target directory to store
   downloaded models and accompanying files.
5. Download the selected model(s) from Hugging Face using the
   ``huggingface_hub`` library.
6. Provide guidance on next steps after setup completes.

Usage::

    python setup_gpt_oss.py [--target_dir TARGET] [--repo_root PATH]

If ``--repo_root`` points at the repository you just cloned (e.g.,
``IntuiTek-gpt-oss``), the script will look for a ``requirements.txt`` file
there.  You can override any of the default paths via command‑line options.

Note: Running gpt‑oss‑120b locally requires a high‑end GPU with ≠60 GB VRAM.
If your GPU memory is below that threshold, the script will download the
smaller 20b model by default.  You can force downloading both models by
setting the environment variable ``GPT_OSS_DOWNLOAD_ALL=1``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# Third‑party imports; these may not yet be installed.  The script will
# attempt to install missing dependencies if a requirements file exists.
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore
try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore
try:
    from huggingface_hub import snapshot_download  # type: ignore
except ImportError:
    snapshot_download = None  # type: ignore


def detect_gpu_memory_gb() -> float:
    """Return available GPU memory (in GiB) on device 0, or 0.0 if no GPU."""
    try:
        if torch is not None and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def detect_system_ram_gb() -> float:
    """Return total system RAM (in GiB), or 0.0 if psutil is unavailable."""
    try:
        if psutil is not None:
            return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def install_requirements(requirements_file: Path) -> None:
    """Install Python packages listed in a requirements.txt file."""
    print(f"Installing Python dependencies from {requirements_file}…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])

    # Ensure huggingface_hub is available for snapshot_download
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])



def decide_models_to_download(gpu_mem_gb: float) -> List[str]:
    """Decide which GPT‑OSS models to download based on available GPU memory."""
    # Environment override: download both models if GPT_OSS_DOWNLOAD_ALL is set
    if os.environ.get("GPT_OSS_DOWNLOAD_ALL"):
        return ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

    if gpu_mem_gb >= 60:
        # High‑end GPU – download the larger 120b model.  Users can still
        # manually download the 20b model later if desired.
        return ["openai/gpt-oss-120b"]

    # Default: download the smaller 20b model, which fits on consumer GPUs.
    return ["openai/gpt-oss-20b"]


def download_model(model_id: str, dest_dir: Path) -> None:
    """Download a GPT‑OSS model from Hugging Face into dest_dir."""
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed; cannot download models. Install it via pip."
        )
    print(f"\nDownloading {model_id} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md"],
    )
    print(f"Finished downloading {model_id} to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up OpenAI GPT‑OSS models and environment.")
    parser.add_argument(
        "--target_dir",
        default="models",
        help="Directory under repo_root where models will be downloaded",
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Path to the root of your project/repository (used to locate requirements.txt)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    target_dir = repo_root / args.target_dir

    # Install dependencies if a requirements.txt file is present
    req_file = repo_root / "requirements.txt"
    if req_file.exists():
        install_requirements(req_file)
    else:
        print("No requirements.txt found; skipping dependency installation.")

    # Detect system resources
    gpu_mem = detect_gpu_memory_gb()
    ram = detect_system_ram_gb()
    print(f"Detected GPU memory: {gpu_mem:.1f} GiB")
    print(f"Detected system RAM: {ram:.1f} GiB")

    # Decide which model(s) to download
    models_to_download = decide_models_to_download(gpu_mem)
    print(f"Models selected for download: {', '.join(models_to_download)}")

    # Download selected models
    for model_id in models_to_download:
        model_name = model_id.split("/")[-1]
        dest = target_dir / model_name
        download_model(model_id, dest)

    print("\nAll selected models downloaded successfully!")
    print("You can now run example_inference.py to test the models.")


if __name__ == "__main__":
    main()
