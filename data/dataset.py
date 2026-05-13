from pathlib import Path

import rasterio
import numpy as np
import torch

from torch.utils.data import Dataset

from data.utils import normalize_image, remap_labels
from data.preprocessing import sar_to_db, reduce_speckle_noise


class EOSARDataset(Dataset):

    def __init__(self, root_dir, split="train", transforms=None):

        self.root_dir = Path(root_dir)

        self.split = split

        self.transforms = transforms

        self.eo_dir = self.root_dir / split / "pre-event"

        self.sar_dir = self.root_dir / split / "post-event"

        self.target_dir = self.root_dir / split / "target"

        self.file_names = sorted([
            f.name for f in self.target_dir.glob("*.tif")
        ])


        self.positive_indices = []

        self.negative_indices = []

        for idx, file_name in enumerate(self.file_names):

            mask_path = self.target_dir / file_name

            with rasterio.open(mask_path) as src:

                mask = src.read(1)

            mask = remap_labels(mask)

            if mask.sum() > 0:

                self.positive_indices.append(idx)

            else:

                self.negative_indices.append(idx)

        print(
            f"[{split}] "
            f"Positive samples: {len(self.positive_indices)} | "
            f"Negative samples: {len(self.negative_indices)}"
        )

    def __len__(self):

        return len(self.file_names)

    def load_tif(self, path):

        with rasterio.open(path) as src:

            image = src.read().astype(np.float32)

        return image

    def __getitem__(self, idx):

  
        # Balanced Sampling
     

        if self.split == "train":

            if np.random.rand() < 0.7:

                idx = np.random.choice(
                    self.positive_indices
                )

            else:

                idx = np.random.choice(
                    self.negative_indices
                )

        file_name = self.file_names[idx]

        eo_path = self.eo_dir / file_name

        sar_path = self.sar_dir / file_name

        target_path = self.target_dir / file_name

   
        # Load EO, SAR, and Mask
  

        eo_image = self.load_tif(eo_path)

        sar_image = self.load_tif(sar_path)

        mask = self.load_tif(target_path)[0]


        # SAR preprocessing


        sar_image = sar_to_db(sar_image)

        sar_image = reduce_speckle_noise(sar_image)

        # Normalize EO and SAR separately
     

        eo_image = normalize_image(eo_image)

        sar_image = normalize_image(sar_image)

      
        # Label Remapping


        mask = remap_labels(mask)


        # Reduce empty-mask dominance
  

        if self.split == "train":

            if mask.sum() == 0:

                if np.random.rand() < 0.7:

                    new_idx = np.random.randint(0, len(self))

                    return self.__getitem__(new_idx)


        # Data Augmentation


        if self.transforms is not None:

            transformed = self.transforms(
                image=eo_image.transpose(1, 2, 0),
                sar=sar_image.transpose(1, 2, 0),
                mask=mask
            )

            eo_image = transformed["image"].transpose(2, 0, 1)

            sar_image = transformed["sar"].transpose(2, 0, 1)

            mask = transformed["mask"]

  
        # Convert to tensors
 

        eo_image = torch.tensor(
            eo_image,
            dtype=torch.float32
        )

        sar_image = torch.tensor(
            sar_image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )

        return {
            "eo": eo_image,
            "sar": sar_image,
            "mask": mask
        }