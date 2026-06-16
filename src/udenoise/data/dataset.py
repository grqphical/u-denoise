from pathlib import Path
import rawpy

def load_raw_bayer(raw_path: Path):
    """Loads a camera RAW image and extracts the Bayer data"""
    raw_bayer = None
    with rawpy.imread(str(raw_path)) as raw:
        raw_bayer = raw.raw_image_visible.copy()

    return raw_bayer

def extract_patches(raw_path: Path, stride: int, patch_size: int, out_directory: Path):
    """Takes in a raw image and chunks into patches, saving each as a .npy file in the specificed output directory"""
    out_directory.mkdir(parents=True, exist_ok=True)

    raw_bayer = load_raw_bayer(raw_path)
    print(raw_bayer.shape)

