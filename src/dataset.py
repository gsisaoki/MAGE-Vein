import os

import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class GroupedFingerVeinDataset(Dataset):
    """PyTorch dataset for grouped finger-vein age and gender estimation.

    Expected CSV columns:

        file_path (str):
            Image path relative to ``data_root``.
        age (int):
            Subject age in years, precomputed by the dataset provider.
        gender (int):
            Binary label where ``0`` = male and ``1`` = female.
        group_id (str):
            Identifier shared by images captured from the same finger/session.
            Rows with the same ``group_id`` are grouped; every
            ``images_per_group`` consecutive rows inside a group form one sample.

    Loading behavior:
        For each sample, ``images_per_group`` grayscale BMP/PNG images are read
        from disk, normalized to ``[0, 1]``, replicated to 3 channels, center-
        cropped to ``crop_size``, and stacked into a tensor of shape
        ``(images_per_group, 3, crop_size, crop_size)``. During training, a
        single vertical-flip decision is sampled once per group so that all
        finger images in the group receive identical augmentation.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        mode: str,
        crop_size: int = 224,
        images_per_group: int = 3,
    ) -> None:
        """Initialize the dataset.

        Args:
            csv_path: Path to the CSV manifest file.
            data_root: Root directory prepended to each ``file_path`` entry.
            mode: ``"train"`` enables random vertical flip; any other value
                uses deterministic preprocessing only.
            crop_size: Side length for square center crop.
            images_per_group: Number of finger images per sample.
        """
        self.data_root = data_root
        self.mode = mode
        self.crop_size = crop_size
        self.images_per_group = images_per_group

        df = pd.read_csv(csv_path)
        required_columns = {"file_path", "age", "gender", "group_id"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing_columns)}"
            )

        self.data_df = df[list(required_columns)].copy()
        self.groups = self._build_groups()

    def _build_groups(self) -> list[list[int]]:
        """Build sample index lists from explicit ``group_id`` values."""
        groups: list[list[int]] = []
        grouped_indices = self.data_df.groupby("group_id", sort=False).groups

        for indices in grouped_indices.values():
            idx_list = list(indices)
            usable_length = len(idx_list) - (len(idx_list) % self.images_per_group)
            for start in range(0, usable_length, self.images_per_group):
                groups.append(idx_list[start:start + self.images_per_group])

        return groups

    def __len__(self) -> int:
        return len(self.groups)

    def _resolve_image_path(self, relative_path: str) -> str:
        """Return the absolute image path for a CSV ``file_path`` entry."""
        return os.path.join(self.data_root, relative_path)

    def _load_and_crop(self, relative_path: str) -> torch.Tensor:
        """Load a grayscale image and return a center-cropped 3-channel tensor."""
        img_path = self._resolve_image_path(relative_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        return TF.center_crop(tensor, self.crop_size)

    def __getitem__(self, group_idx: int) -> dict[str, torch.Tensor]:
        member_indices = self.groups[group_idx]
        images = [
            self._load_and_crop(str(self.data_df.iloc[idx]["file_path"]))
            for idx in member_indices
        ]

        if self.mode == "train":
            do_flip = torch.rand(1).item() < 0.4
            if do_flip:
                images = [TF.vflip(image) for image in images]

        row = self.data_df.iloc[member_indices[0]]
        gender = int(row["gender"])
        age = int(row["age"])

        return {
            "image": torch.stack(images, dim=0),
            "gender": torch.tensor(gender, dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.long),
        }
