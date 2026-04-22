import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import DefaultDataCollator

from .transforms import get_transforms


def rgb_to_class_mask_numpy(
    mask_np: np.ndarray,
    damage_color: Tuple[int, int, int],
    tolerance: int = 10,
) -> np.ndarray:
    diff = np.abs(mask_np - damage_color)
    is_damage = np.all(diff <= tolerance, axis=-1)
    return is_damage.astype(np.uint8)


def split_data(
    data_dir: str,
    val_percent: float,
    test_percent: float,
    seed: int,
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Tuple[Tuple[List[Path], List[Path]], ...]:
    images_dir = Path(data_dir) / "images"
    masks_dir = Path(data_dir) / "masks"

    image_paths: List[Path] = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(f"*{ext}"))
        image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
    image_paths = sorted(image_paths)

    mask_paths: List[Path] = []
    valid_images: List[Path] = []
    for img_path in image_paths:
        mask_path = masks_dir / img_path.name.replace(img_path.suffix, ".png")
        if mask_path.exists():
            valid_images.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"Warning: mask for {img_path.name} not found, skipping.")

    assert len(valid_images) > 0, "No valid image-mask pairs found."

    indices = list(range(len(valid_images)))
    random.seed(seed)
    random.shuffle(indices)

    n_test = int(len(indices) * test_percent)
    n_val = int(len(indices) * val_percent)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    def _select(lst: List, idx: List[int]) -> List:
        return [lst[i] for i in idx]

    return (
        (_select(valid_images, train_idx), _select(mask_paths, train_idx)),
        (_select(valid_images, val_idx), _select(mask_paths, val_idx)),
        (_select(valid_images, test_idx), _select(mask_paths, test_idx)),
    )


class CarDamageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform,
        damage_color: Tuple[int, int, int],
        color_tolerance: int = 10,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.damage_color = damage_color
        self.color_tolerance = color_tolerance

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))
        mask_class = rgb_to_class_mask_numpy(mask_rgb, self.damage_color, self.color_tolerance)

        augmented = self.transform(image=image, mask=mask_class)
        return {
            "pixel_values": augmented["image"],
            "labels": augmented["mask"].long(),
        }


data_collator = DefaultDataCollator()
