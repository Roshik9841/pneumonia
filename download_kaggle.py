"""
download_kaggle.py
-------------------
Helper script to download the Kaggle chest X‑ray pneumonia dataset
(`paultimothymooney/chest-xray-pneumonia`) into the local `archive/`
folder using the Kaggle API.

Usage (from project root):
    python download_kaggle.py

Requirements:
    - Kaggle account
    - kaggle.json API token placed in:
        %USERPROFILE%\\.kaggle\\kaggle.json   (Windows)
    - `pip install kaggle`
"""

import os
import zipfile
import subprocess
from pathlib import Path

DATASET = "paultimothymooney/chest-xray-pneumonia"


def ensure_kaggle_config():
    """Check that kaggle.json exists and Kaggle CLI is installed."""
    home = Path.home()
    kaggle_json = home / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print(
            "\n[ERROR] kaggle.json not found.\n"
            "1) Go to https://www.kaggle.com/ -> Account -> Create New API Token\n"
            "2) Download kaggle.json\n"
            f"3) Place it at: {kaggle_json}\n"
        )
        return False

    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except Exception:
        print(
            "\n[ERROR] Kaggle CLI not found.\n"
            "Run:  pip install kaggle\n"
            "Then try again.\n"
        )
        return False

    return True


def download_dataset():
    """Download the Kaggle zip into ./archive if not already present."""
    if not ensure_kaggle_config():
        return

    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)

    zip_path = archive_dir / "chest-xray-pneumonia.zip"
    if zip_path.exists():
        print(f"[INFO] Zip already exists at {zip_path}")
    else:
        print(f"[INFO] Downloading {DATASET} to {zip_path} ...")
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(archive_dir),
        ]
        subprocess.run(cmd, check=True)
        print("[INFO] Download complete.")

    # Extract if needed
    extract_dir = archive_dir / "chest_xray"
    if extract_dir.exists():
        print(f"[INFO] Extract folder already exists at {extract_dir}")
    else:
        print(f"[INFO] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(archive_dir)
        print(f"[INFO] Extracted to {extract_dir}")

    print(
        "\n[INFO] Next step:\n"
        "Run `python setup_dataset.py` to move images into "
        "`archive/train`, `archive/val`, and `archive/test`.\n"
    )


if __name__ == "__main__":
    try:
        download_dataset()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Kaggle CLI failed: {e}")


