import numpy as np
from scipy.ndimage import median_filter


def sar_to_db(sar_image, eps=1e-6):
    """
    Convert SAR to dB scale
    """

    sar_image = 10 * np.log10(sar_image + eps)

    return sar_image


def reduce_speckle_noise(sar_image):
    """
    Simple median filtering for SAR
    """

    filtered = np.zeros_like(sar_image)

    for c in range(sar_image.shape[0]):
        filtered[c] = median_filter(sar_image[c], size=3)

    return filtered