import numpy as np


def sample_noise_params():
    log_shot = np.random.uniform(np.log(1e-4), np.log(1e-2))
    log_read = np.random.uniform(np.log(1e-6), np.log(1e-4))
    return np.exp(log_shot), np.exp(log_read)


def add_noise(
    clean: np.ndarray, shot_noise_scale: float, read_noise_scale: float
) -> np.ndarray:
    """Adds Poisson-Gaussian noise to a RAW image array, to simulate DSLR sensor noise. Clean should be a normalized float tensor"""
    shot_noise = np.random.poisson(clean / shot_noise_scale) * shot_noise_scale

    read_noise = np.random.randn(*clean.shape) * read_noise_scale

    return shot_noise + read_noise
