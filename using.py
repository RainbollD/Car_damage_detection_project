import sys
import os

from src.inference.detect_parts import CarPartsDetector
from src.config import TrainingConfig
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))

path_img = "dataset/classification_dent/images/Car damages 101.png"

# Загрузка модели
config = TrainingConfig()
detector = CarPartsDetector(
    "/home/lev/PycharmProjects/Car_damage_detection_project/models/nvidia_mit-b5/dent/checkpoint-700", config)

# Предсказание
mask, image = detector.predict(path_img)

# Визуализация
result = detector.visualize_prediction(image, mask)
plt.imshow(result)
plt.show()

# Сохранение результата визуализации
# plt.imsave(f"path_to_folder_save/{path_img.split("/")[-1]}", result)
