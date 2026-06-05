"""
download_dataset.py
-------------------
High-level helper to prepare the `archive/` dataset directory.

This script:
  1. Ensures `archive/` exists.
  2. (Optionally) calls `download_kaggle.py` to fetch the Kaggle dataset.
  3. Prints instructions to run `setup_dataset.py` to arrange the data
     into `archive/train`, `archive/val`, and `archive/test`.

You can use this as a single entrypoint when setting up the project
on a new machine.
"""

from pathlib import Path
import subprocess


def main():
    archive_dir = Path("archive")
    archive_dir.mkdir(exist_ok=True)
    print(f"[INFO] Using dataset root: {archive_dir.resolve()}")

    chest_dir = archive_dir / "chest_xray"
    if not chest_dir.exists():
        print("\n[INFO] `archive/chest_xray` not found; attempting Kaggle download...")
        try:
            subprocess.run(
                ["python", "download_kaggle.py"], check=True
            )
        except Exception as e:
            print(f"[WARN] Could not auto-download dataset: {e}")
            print(
                "\nManual option:\n"
                "  1) Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
                "  2) Extract the zip so that you have `archive/chest_xray/train`, `val`, `test`.\n"
            )
    else:
        print(f"[INFO] Found existing Kaggle layout at {chest_dir}")

    print(
        "\nNext steps:\n"
        "  1) Run:  python setup_dataset.py\n"
        "     This will move images into `archive/train`, `archive/val`, `archive/test`.\n"
        "  2) Then you can train:\n"
        "       python improved_train.py\n"
    )


if __name__ == "__main__":
    main()


