from src.inference.detect_parts import CarPartsDetector
from src.config.traning_config import TrainingConfig
import matplotlib.pyplot as plt

# Загрузка модели
config = TrainingConfig()
detector = CarPartsDetector("/models/car_damage_segmentation/final_small_dataset", config)

# Предсказание
mask, image = detector.predict("/home/lev/PycharmProjects/Car_damage_detection_project/data/classification_details/img/IMG_9215.jpg")

# Визуализация
result = detector.visualize_prediction(image, mask)
plt.imshow(result)
plt.show()
