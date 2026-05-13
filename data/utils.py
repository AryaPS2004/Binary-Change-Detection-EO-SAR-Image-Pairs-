import numpy as np


def normalize_image(image, mean=None, std=None):
    """
    Normalize image per channel
    image shape: (C, H, W)
    """

    image = image.astype(np.float32)

    if mean is None:
        mean = image.mean(axis=(1, 2), keepdims=True)

    if std is None:
        std = image.std(axis=(1, 2), keepdims=True)

    std = np.where(std == 0, 1e-6, std)

    image = (image - mean) / std

    return image


def remap_labels(mask):
    """
    Convert:
    0,1 -> 0 (no change)
    2,3 -> 1 (change)
    """

    mask = np.where(mask >= 2, 1, 0)

    return mask.astype(np.int64)