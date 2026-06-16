from pathlib import Path
from typing import Tuple
import rawpy
import numpy as np


def load_raw_bayer(raw_path: Path) -> Tuple[np.ndarray, dict]:
    """Loads a camera RAW image and extracts the Bayer data, returning the Bayer array and metadata"""
    raw_bayer = None
    with rawpy.imread(str(raw_path)) as raw:
        raw_bayer = raw.raw_image_visible.astype(np.float32).copy()

        # normalize the black level as it can be in inconsistent formats across
        # various RAW formats
        black_raw = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        black = black_raw.flatten()[:4]

        # If only 1 unique value came back, broadcast to all 4 channels
        if black.size == 1:
            black = np.repeat(black, 4)

        white = float(raw.white_level)

        meta = {
            "black": black,
            "white": white,
            "whitebalance": raw.camera_whitebalance,
            "daylight_wb": raw.daylight_whitebalance,
        }
        pattern = raw.raw_pattern
        if pattern is not None:
            meta["pattern"] = pattern.tolist()

    h, w = raw_bayer.shape
    raw_bayer = raw_bayer[: h - h % 2, : w - w % 2]

    black_map = np.empty_like(raw_bayer)
    black_map[0::2, 0::2] = black[0]  # R
    black_map[0::2, 1::2] = black[1]  # G1
    black_map[1::2, 0::2] = black[2]  # G2
    black_map[1::2, 1::2] = black[3]  # B

    raw_bayer = (raw_bayer - black_map) / (white - black.min())
    raw_bayer = np.clip(raw_bayer, 0.0, 1.0)

    return raw_bayer, meta


def pack_bayer(bayer: np.ndarray) -> np.ndarray:
    """Pack a HxW Bayer mosaic into a 4x(H/2)x(W/2) array"""
    if bayer.ndim != 2:
        raise ValueError(f"Expected 2D Bayer array, got shape {bayer.shape}")
    if bayer.shape[0] % 2 != 0 or bayer.shape[1] % 2 != 0:
        raise ValueError(
            f"Bayer array dims must be even, got {bayer.shape}. "
            "Trim one row/col if needed."
        )

    R = bayer[0::2, 0::2]  # every even row,  even col
    G1 = bayer[0::2, 1::2]  # every even row,  odd  col
    G2 = bayer[1::2, 0::2]  # every odd  row,  even col
    B = bayer[1::2, 1::2]  # every odd  row,  odd  col

    packed = np.stack([R, G1, G2, B], axis=0)

    return packed.astype(np.float32)


def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_bayer, unpacks a 4x(H/2)x(W/2) array back into a HxW Bayer mosaic"""
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise ValueError(f"Expected shape (4, H, W), got {packed.shape}")

    _, h, w = packed.shape
    bayer = np.empty((h * 2, w * 2), dtype=np.float32)

    bayer[0::2, 0::2] = packed[0]  # R
    bayer[0::2, 1::2] = packed[1]  # G1
    bayer[1::2, 0::2] = packed[2]  # G2
    bayer[1::2, 1::2] = packed[3]  # B

    return bayer


def extract_patches(
    raw_path: Path,
    stride: int,
    patch_size: int,
    out_directory: Path,
    min_mean: float = 0.01,
    max_mean: float = 0.99,
    min_var: float = 1e-5,
) -> int:
    """Takes in a raw image and chunks into patches, saving each as a .npy file in the specificed output directory.

    Returns the number of patches extracted"""
    out_directory.mkdir(parents=True, exist_ok=True)
    try:
        raw_bayer, metadata = load_raw_bayer(raw_path)
    except Exception as e:
        print(f"    Skipping {raw_path.name}: Exception occured: {e}")
        return 0

    packed = pack_bayer(raw_bayer)
    _, H, W = packed.shape
    p, s = patch_size // 2, stride // 2
    stem = raw_path.stem

    if H < p or W < p:
        print(f"  Skipping {raw_path.name}: too small ({H}x{W})")
        return 0

    count = 0
    for y in range(0, H - p + 1, s):
        for x in range(0, W - p + 1, s):
            patch = packed[:, y : y + p, x : x + p]

            # Make sure we don't have patches of just black, just white, etc.
            if patch.mean() < min_mean or patch.mean() > max_mean:
                continue
            if patch.var() < min_var:
                continue

            np.save(out_directory / f"{stem}_{count:05d}.npy", patch.astype(np.float32))
            count += 1

    return count
