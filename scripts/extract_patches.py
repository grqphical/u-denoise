"""Decodes the camera RAW files in data/raw, splits them into patches, and outputs .npy files containing the patches in data/processed"""

from udenoise import extract_patches
from pathlib import Path
import os

PATCH_SIZE = 256
PATCH_STRIDE = 128

images = os.listdir("data/raw")
total_image_count = len(images)
print(f"Patching {total_image_count} images...")

image_count = 0
total_patch_count = 0
for image in images:
    patch_count = extract_patches(
        Path("data/raw") / image, PATCH_SIZE, PATCH_STRIDE, Path("data/processed")
    )

    if patch_count > 0:
        image_count += 1
        total_patch_count += patch_count
        print(f"Patched {image_count}/{total_image_count} ({patch_count} patches)")
    else:
        print("Skipped image")
        total_image_count -= 1

print(f"Generated {total_patch_count} patches")
