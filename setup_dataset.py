"""
setup_dataset.py
----------------
Organize the Kaggle chest X‑ray dataset into the structure expected
by the training scripts in this project.

Expected input layout (after running `download_kaggle.py` manually or via
`download_dataset.py`):
    archive/chest_xray/
        train/NORMAL, train/PNEUMONIA
        val/NORMAL,   val/PNEUMONIA
        test/NORMAL,  test/PNEUMONIA

This script copies/moves those into:
    archive/train/NORMAL
    archive/train/PNEUMONIA
    archive/val/NORMAL
    archive/val/PNEUMONIA
    archive/test/NORMAL
    archive/test/PNEUMONIA

If that structure already exists (as in your current project), this script
will simply print the image counts and do nothing destructive.

Usage:
    python setup_dataset.py
"""

import os
import shutil
from pathlib import Path


def count_images(folder: Path) -> int:
    return sum(
        1
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def ensure_subdirs(base: Path):
    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            (base / split / cls).mkdir(parents=True, exist_ok=True)


def move_if_needed():
    archive = Path("archive")
    chest_root = archive / "chest_xray" / "chest_xray"
    if not chest_root.exists():
        # Some zips use archive/chest_xray/train directly.
        chest_root = archive / "chest_xray"

    target_root = archive
    ensure_subdirs(target_root)

    # If target already populated, just report and exit.
    train_norm = count_images(target_root / "train" / "NORMAL")
    train_pneu = count_images(target_root / "train" / "PNEUMONIA")
    val_total = count_images(target_root / "val")
    test_total = count_images(target_root / "test")

    if train_norm + train_pneu > 0 and val_total > 0 and test_total > 0:
        print("[INFO] archive/train, val, test already populated.")
        print(
            f"  train NORMAL={train_norm}, PNEUMONIA={train_pneu}, "
            f"val={val_total}, test={test_total}"
        )
        return

    if not chest_root.exists():
        print(
            "[ERROR] Could not find `archive/chest_xray`.\n"
            "Make sure you have extracted the Kaggle zip so that "
            "`archive/chest_xray/train`, `val`, `test` exist."
        )
        return

    print(f"[INFO] Moving images from {chest_root} into archive/train|val|test ...")

    for split in ("train", "val", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            src = chest_root / split / cls
            dst = target_root / split / cls
            if not src.exists():
                print(f"[WARN] Missing source folder: {src}")
                continue
            for img_name in os.listdir(src):
                src_path = src / img_name
                if not src_path.is_file():
                    continue
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                dst_path = dst / img_name
                if dst_path.exists():
                    continue
                shutil.copy2(src_path, dst_path)

    print("[INFO] Finished copying images.")
    print(
        f"  train NORMAL={count_images(target_root / 'train' / 'NORMAL')}, "
        f"PNEUMONIA={count_images(target_root / 'train' / 'PNEUMONIA')}"
    )
    print(f"  val total={count_images(target_root / 'val')}")
    print(f"  test total={count_images(target_root / 'test')}")


if __name__ == "__main__":
    move_if_needed()


