# data_utils.py
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Optional
import random

def rgb_to_class_mask(mask_pil: Image.Image, damage_color: Tuple[int, int, int], tolerance: int = 10) -> Image.Image:
    """
    Преобразует RGB-маску в бинарную карту классов (0 - фон, 1 - повреждение).
    Использует допуск tolerance для сравнения цветов.
    """
    mask_np = np.array(mask_pil)
    diff = np.abs(mask_np - damage_color)
    is_damage = np.all(diff <= tolerance, axis=-1)
    class_mask = is_damage.astype(np.uint8)
    return Image.fromarray(class_mask, mode='L')

def split_data(
    data_dir: Path,
    val_percent: float,
    test_percent: float,
    seed: int,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> Tuple[Tuple[List[Path], List[Path]], ...]:
    """
    Разделяет данные на train/val/test.
    Возвращает кортежи (список изображений, список масок) для каждой выборки.
    """
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Wrong path to dataset")

    # Собираем все изображения с допустимыми расширениями
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(f"*{ext}"))
        image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
    image_paths = sorted(image_paths)

    # Ищем соответствующие маски (предполагается .png)
    mask_paths = []
    valid_images = []
    for img_path in image_paths:
        mask_path = masks_dir / img_path.name.replace(img_path.suffix, ".png")
        if mask_path.exists():
            valid_images.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"Warning: Mask for {img_path.name} not found. Skipping.")

    assert len(valid_images) > 0, "No valid image-mask pairs found."

    indices = list(range(len(valid_images)))
    random.seed(seed)
    random.shuffle(indices)

    n_test = int(len(indices) * test_percent)
    n_val = int(len(indices) * val_percent)
    n_train = len(indices) - n_test - n_val

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test+n_val]
    train_idx = indices[n_test+n_val:]

    train_images = [valid_images[i] for i in train_idx]
    train_masks  = [mask_paths[i] for i in train_idx]
    val_images   = [valid_images[i] for i in val_idx]
    val_masks    = [mask_paths[i] for i in val_idx]
    test_images  = [valid_images[i] for i in test_idx]
    test_masks   = [mask_paths[i] for i in test_idx]

    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

def get_transforms(image_size: int):
    """Возвращает трансформации для изображений и масок."""
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()  # -> torch.int64 (1, H, W)
    ])
    return image_transform, mask_transform

class CarDamageDataset(Dataset):
    """Кастомный датасет для сегментации повреждений."""
    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        image_transform,
        mask_transform,
        damage_color: Tuple[int, int, int],
        color_tolerance: int = 10
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.damage_color = damage_color
        self.color_tolerance = color_tolerance

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_rgb = Image.open(self.mask_paths[idx]).convert("RGB")
        mask_class = rgb_to_class_mask(mask_rgb, self.damage_color, self.color_tolerance)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask_class = self.mask_transform(mask_class)  # (1, H, W)

        # Убираем лишнюю размерность и приводим к long
        mask_class = mask_class.squeeze(0).long()
        return {"pixel_values": image, "labels": mask_class}

# Можно определить свой коллатор, если нужно, но DefaultDataCollator подходит
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()