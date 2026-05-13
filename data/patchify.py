import numpy as np


def extract_patches(image, patch_size=256, stride=256):

    """
    Extract patches from image

    image shape:
    (C, H, W)
    """

    patches = []

    _, H, W = image.shape

    for y in range(0, H - patch_size + 1, stride):

        for x in range(0, W - patch_size + 1, stride):

            patch = image[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(patch)

    return np.array(patches)