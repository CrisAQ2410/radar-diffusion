import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class RadarVAEDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
        max_rainfall=260.0,
        train_ratio=0.8,
        test_year=2023,
    ):
        self.image_size = image_size
        self.max_rainfall = max_rainfall

        files = sorted(Path(root_dir).rglob("*.tif"))

        # VAE train/val không được dùng test 2023
        train_val_files = [
            p for p in files
            if self._get_year(p) is not None
            and self._get_year(p) < test_year
        ]

        test_files = [
            p for p in files
            if self._get_year(p) == test_year
        ]

        n_train = int(len(train_val_files) * train_ratio)

        if split == "train":
            self.files = train_val_files[:n_train]
        elif split == "val":
            self.files = train_val_files[n_train:]
        elif split == "test":
            self.files = test_files
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"VAE {split}: {len(self.files)} frames")

    @staticmethod
    def _get_year(path):
        match = re.search(r"(20\d{2})\d{6,8}", path.name)
        if match:
            return int(match.group(1))

        for part in path.parts:
            if part.isdigit() and len(part) == 4:
                return int(part)

        return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        with rasterio.open(path) as src:
            img = src.read(1, masked=True)

            if np.ma.isMaskedArray(img):
                img = img.filled(0.0)

        img = np.asarray(img, dtype=np.float32)

        img = np.nan_to_num(
            img,
            nan=0.0,
            posinf=self.max_rainfall,
            neginf=0.0,
        )

        img = np.clip(
            img,
            0.0,
            self.max_rainfall,
        )

        if img.shape != (self.image_size, self.image_size):
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )

        # [0, 1]
        img = np.log1p(img) / np.log1p(self.max_rainfall)

        # C,H,W
        img = torch.from_numpy(img).float().unsqueeze(0)

        return img