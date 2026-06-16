import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path
from .noise_generator import add_noise, sample_noise_params

FILE_EXTENSION = "CR2"

device = "cuda" if torch.cuda.is_available() else "cpu"


class RawImageDataset(Dataset):
    """PyTorch Dataset implementation for U-Denoise's custom dataset"""

    def __init__(self, directory: Path = Path("data/processed")) -> None:
        self.directory = directory
        self.image_count = len(list(self.directory.glob("*")))

    def __len__(self):
        return self.image_count

    def __getitem__(self, index):
        clean_img_path = self.directory / f"{index:05d}.{FILE_EXTENSION}"
        if not clean_img_path.exists():
            raise ValueError("image index does not exist")

        clean_img_data = np.load(str(clean_img_path))

        shot_scale, read_scale = sample_noise_params()
        noisy_img_data = add_noise(clean_img_data, shot_scale, read_scale)

        return (
            torch.from_numpy(noisy_img_data).to(device),
            torch.from_numpy(clean_img_data).to(device),
        )
