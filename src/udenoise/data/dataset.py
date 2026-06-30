import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path
from .noise_generator import add_noise, sample_noise_params
from typing import List


class RawImageDataset(Dataset):
    """PyTorch Dataset implementation for U-Denoise's custom dataset"""

    def __init__(self, directory: Path = Path("data/processed")) -> None:
        self.directory = directory
        self.images: List[Path] = list(self.directory.glob("*"))
        self.image_count = len(self.images)

    def __len__(self):
        return self.image_count

    def __getitem__(self, index):
        clean_img_path = self.images[index]
        if not clean_img_path.exists():
            raise ValueError("image index does not exist")

        clean_img_data = np.load(str(clean_img_path))

        shot_scale, read_scale = sample_noise_params()
        noisy_img_data = add_noise(clean_img_data, shot_scale, read_scale)

        return (
            torch.from_numpy(noisy_img_data).float(),
            torch.from_numpy(clean_img_data).float(),
        )
