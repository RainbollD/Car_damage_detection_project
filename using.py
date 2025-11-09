from src.inference.detect_parts import CarPartsDetector
from src.config.traning_config import TrainingConfig
import matplotlib.pyplot as plt

# Загрузка модели
config = TrainingConfig()
detector = CarPartsDetector("./models/car_damage_segmentation/final", config)

# Предсказание
mask, image = detector.predict("path_to_your_image.jpg")

# Визуализация
result = detector.visualize_prediction(image, mask)
plt.imshow(result)
plt.show()