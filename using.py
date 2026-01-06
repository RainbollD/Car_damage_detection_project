import sys
import os

from src.inference.detect_parts import CarPartsDetector
from src.config.traning_config import TrainingConfig
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__)))

path_img = "path_to_image"

# Загрузка модели
config = TrainingConfig()
detector = CarPartsDetector("path_to_model", config)

# Предсказание
mask, image = detector.predict(path_img)

# Визуализация
result = detector.visualize_prediction(image, mask)
plt.imshow(result)
plt.show()

# Сохранение результата визуализации
plt.imsave(f"path_to_folder_save/{path_img.split("/")[-1]}", result)
