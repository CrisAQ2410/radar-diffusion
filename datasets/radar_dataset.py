import os
import glob
import re
from datetime import datetime

import cv2
import numpy as np
import rasterio
import torch

from torch.utils.data import Dataset


class RadarDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        input_frames=6,
        output_frames=6,
        image_size=256,
        max_rainfall=260.0,
        val_ratio=0.1,
        test_year=2023,
    ):

        self.input_frames = input_frames
        self.output_frames = output_frames
        self.total_frames = input_frames + output_frames

        self.image_size = image_size
        self.max_rainfall = max_rainfall

        # --------------------------------------------------
        # Load all tif
        # --------------------------------------------------

        all_files = sorted(
            glob.glob(
                os.path.join(
                    root_dir,
                    "**",
                    "*.tif",
                ),
                recursive=True,
            ),
            key=self._timestamp,
        )

        # --------------------------------------------------
        # Split by year
        # --------------------------------------------------

        train_val_files = [
            f for f in all_files
            if self._timestamp(f).year < test_year
        ]

        test_files = [
            f for f in all_files
            if self._timestamp(f).year == test_year
        ]

        n_train = int(
            len(train_val_files)
            * (1.0 - val_ratio)
        )

        if split == "train":
            self.files = train_val_files[:n_train]

        elif split == "val":
            self.files = train_val_files[n_train:]

        elif split == "test":
            self.files = test_files

        else:
            raise ValueError(
                f"Invalid split: {split}"
            )

        # --------------------------------------------------
        # Create ONLY continuous sequences
        # --------------------------------------------------

        self.samples = []

        for i in range(
            len(self.files)
            - self.total_frames
            + 1
        ):

            seq = self.files[
                i:i + self.total_frames
            ]

            if self._is_continuous(seq):

                self.samples.append(seq)

        print(
            f"{split}: "
            f"{len(self.files)} frames | "
            f"{len(self.samples)} sequences"
        )

    # ======================================================
    # Timestamp
    # ======================================================

    @staticmethod
    def _timestamp(path):

        name = os.path.basename(path)

        match = re.search(
            r"(\d{10})",
            name
        )

        if match is None:
            raise ValueError(
                f"Cannot parse timestamp: {path}"
            )

        return datetime.strptime(
            match.group(1),
            "%Y%m%d%H"
        )

    # ======================================================
    # continuity check
    # ======================================================

    def _is_continuous(self, seq):

        times = [
            self._timestamp(f)
            for f in seq
        ]

        for a, b in zip(
            times[:-1],
            times[1:]
        ):

            diff = (
                b - a
            ).total_seconds() / 3600

            # radar của bạn hiện cadence = 1 hour
            if diff != 1:
                return False

        return True

    # ======================================================

    def __len__(self):

        return len(self.samples)

    # ======================================================

    def __getitem__(self, idx):

        seq = self.samples[idx]

        frames = []

        for f in seq:

            with rasterio.open(f) as src:

                img = src.read(1)

            img = np.asarray(
                img,
                dtype=np.float32
            )

            img = np.nan_to_num(
                img,
                nan=0.0,
                posinf=self.max_rainfall,
                neginf=0.0,
            )

            img = np.clip(
                img,
                0,
                self.max_rainfall,
            )

            if img.shape != (
                self.image_size,
                self.image_size,
            ):

                img = cv2.resize(
                    img,
                    (
                        self.image_size,
                        self.image_size,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            # SAME normalization as VAE
            img = (
                np.log1p(img)
                /
                np.log1p(
                    self.max_rainfall
                )
            )

            frames.append(img)

        frames = np.stack(
            frames,
            axis=0,
        )

        past = frames[
            :self.input_frames
        ]

        future = frames[
            self.input_frames:
        ]

        return (
            torch.from_numpy(
                past
            ).float(),

            torch.from_numpy(
                future
            ).float(),
        )