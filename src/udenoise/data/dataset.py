import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import Path

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
        img_path = self.directory / f"{index:05d}.{FILE_EXTENSION}"
        if not img_path.exists():
            raise ValueError("image index does not exist")

        raw_img_data = np.load(str(img_path))
        noise_img_data = None  # TODO: add noise generation

        return torch.tensor(noise_img_data).to(device), torch.tensor(raw_img_data).to(
            device
        )
