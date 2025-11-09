import cv2
import math
import numpy as np
from torch.utils.data import Dataset


class CarDamageDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Dataset для сегментации деталей автомобиля и повреждений

        Args:
            df: DataFrame с колонками ['image_path', 'mask_path']
            transform: Albumentations трансформации
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Возвращаем элемент по индексу"""
        img_path, mask_path = self.df.iloc[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        annotation = self._mask_to_annotation(mask)

        res = {"image": img, "annotation": annotation}

        if self.transform is not None:
            res = self.transform(res)

        return res

    def _mask_to_annotation(self, mask):
        """Конвертирует RGB маску в аннотацию с индексами классов"""
        r, g, b = cv2.split(mask)
        out32u = (np.int32(b) << 16) + (np.int32(g) << 8) + np.int32(r)
        colors32u = np.unique(out32u.reshape(-1))
        mask_colors = [(color % 256, color % (256 ** 2) // 256, color // (256 ** 2))
                       for color in colors32u]

        annotation = np.zeros(mask.shape[:2], dtype="uint8")

        for mask_color in mask_colors:
            best_color = self._find_best_color(mask_color)

            color_mask = ((r == mask_color[0]) &
                          (g == mask_color[1]) &
                          (b == mask_color[2])).astype(bool)
            if best_color in self.rgb2id:
                annotation[color_mask] = self.rgb2id[best_color]

        return annotation

    def _find_best_color(self, target_color):
        """Находит ближайший цвет в палитре"""
        if target_color == (255, 255, 255):  # background
            return (255, 255, 255)

        best_color = None
        best_loss = math.inf

        for color in self.rgb2id.keys():
            loss = sum([abs(target_color[i] - color[i]) for i in range(3)])
            if loss < best_loss:
                best_color = color
                best_loss = loss

        return best_color

    def set_rgb_mapping(self, rgb2id):
        """Устанавливает mapping цветов к ID классов"""
        self.rgb2id = rgb2id

    def set_transform(self, transform):
        self.transform = transform
