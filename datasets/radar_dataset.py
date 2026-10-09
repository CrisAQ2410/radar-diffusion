import os
import glob
import re

from datetime import datetime, timedelta

import numpy as np
import rasterio
import torch

from rasterio.enums import Resampling
from torch.utils.data import Dataset

from datasets.rain_events import RAIN_EVENTS


class RadarDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        input_frames=6,
        output_frames=6,
        image_size=80,
        max_rainfall=260.0,
    ):

        if split not in RAIN_EVENTS:
            raise ValueError(
                f"Invalid split: {split}"
            )

        self.root_dir = root_dir
        self.split = split

        self.input_frames = input_frames
        self.output_frames = output_frames

        self.total_frames = (
            input_frames
            + output_frames
        )

        self.image_size = image_size
        self.max_rainfall = max_rainfall

        # ==================================================
        # Load all TIFF files
        # ==================================================

        all_files = glob.glob(
            os.path.join(
                root_dir,
                "**",
                "*.tif",
            ),
            recursive=True,
        )

        all_files = sorted(
            all_files,
            key=self._timestamp,
        )

        # ==================================================
        # Build samples PER RAIN EVENT
        #
        # Important:
        # Không nối sequence giữa hai đợt mưa khác nhau.
        # ==================================================

        self.samples = []
        self.event_files = []

        event_ranges = RAIN_EVENTS[split]

        for start_str, end_str in event_ranges:

            start = datetime.strptime(
                start_str,
                "%d/%m/%Y",
            )

            # inclusive end date
            end = (
                datetime.strptime(
                    end_str,
                    "%d/%m/%Y",
                )
                + timedelta(
                    days=1
                )
            )

            files = [
                f
                for f in all_files
                if start
                <= self._timestamp(f)
                < end
            ]

            files = sorted(
                files,
                key=self._timestamp,
            )

            self.event_files.append(
                files
            )

            # ----------------------------------------------
            # continuous sequences inside this event only
            # ----------------------------------------------

            for i in range(
                len(files)
                - self.total_frames
                + 1
            ):

                seq = files[
                    i:
                    i + self.total_frames
                ]

                if self._is_continuous(
                    seq
                ):
                    self.samples.append(
                        seq
                    )

        num_frames = sum(
            len(x)
            for x in self.event_files
        )

        print(
            f"{split}: "
            f"{len(event_ranges)} events | "
            f"{num_frames} frames | "
            f"{len(self.samples)} sequences"
        )

    # ======================================================
    # timestamp
    # ======================================================

    @staticmethod
    def _timestamp(path):

        name = os.path.basename(
            path
        )

        match = re.search(
            r"(\d{10})",
            name,
        )

        if match is None:
            raise ValueError(
                f"Cannot parse timestamp: "
                f"{path}"
            )

        return datetime.strptime(
            match.group(1),
            "%Y%m%d%H",
        )

    # ======================================================
    # Check hourly continuity
    # ======================================================

    def _is_continuous(
        self,
        seq,
    ):

        times = [
            self._timestamp(f)
            for f in seq
        ]

        for a, b in zip(
            times[:-1],
            times[1:],
        ):

            diff = (
                b - a
            ).total_seconds() / 3600

            if diff != 1:
                return False

        return True

    # ======================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ======================================================
    # Read + spatial average resampling
    # ======================================================

    def _read_frame(
        self,
        path,
    ):

        with rasterio.open(
            path
        ) as src:

            img = src.read(
                1,

                out_shape=(
                    self.image_size,
                    self.image_size,
                ),

                resampling=(
                    Resampling.average
                ),
            )

        img = np.asarray(
            img,
            dtype=np.float32,
        )

        img = np.nan_to_num(
            img,
            nan=0.0,
            posinf=self.max_rainfall,
            neginf=0.0,
        )

        # ==================================================
        # Min-Max normalization [0, 1]
        #
        # x_min = 0
        # x_max = 260
        # ==================================================

        img = np.clip(
            img,
            0.0,
            self.max_rainfall,
        )

        img = (
            img
            /
            self.max_rainfall
        )

        return img

    # ======================================================

    def __getitem__(
        self,
        idx,
    ):

        seq = self.samples[
            idx
        ]

        frames = [
            self._read_frame(f)
            for f in seq
        ]

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