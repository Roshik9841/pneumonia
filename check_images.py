"""
Check and clean image files in dataset
"""

import os
from PIL import Image

def check_images(folder):
    bad_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()  # Check if image is valid
                    img.close()
                except Exception as e:
                    print(f"Bad image: {path} - {e}")
                    bad_files.append(path)
    return bad_files

# Check both train and val
train_bad = check_images("improved_xray_detector_data//train")
val_bad = check_images("improved_xray_detector_data//val")

print(f"Bad images in train: {len(train_bad)}")
print(f"Bad images in val: {len(val_bad)}")

# Remove bad files
for path in train_bad + val_bad:
    try:
        os.remove(path)
        print(f"Removed: {path}")
    except Exception as e:
        print(f"Failed to remove {path}: {e}")