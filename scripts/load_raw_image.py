from udenoise import extract_patches
from pathlib import Path

extract_patches(Path("data/raw/IMG_5117.CR2"), 512, 512, Path("data/processed"))
