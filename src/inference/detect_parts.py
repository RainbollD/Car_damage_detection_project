import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


class CarPartsDetector:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        """Предсказание для одного изображения"""
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits.cpu()

        mask = self._postprocess_mask(logits, image.shape)

        return mask, image

    def _postprocess_mask(self, logits, original_shape):
        """Постобработка маски"""
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=original_shape[:-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        return pred_seg.numpy()

    def visualize_prediction(self, image, mask, alpha=0.7):
        """Визуализация предсказания"""
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        for label_id in np.unique(mask):
            if label_id < len(self.config.id2label):
                color = self._get_color_for_label(label_id)
                color_mask[mask == label_id] = color

        blended = image * (1 - alpha) + color_mask * alpha
        blended = blended.astype(np.uint8)

        return blended
    def _get_color_for_label(self, label_id):
        """Генерирует цвет для метки"""
        colors = [
            (255, 0, 0),    # красный
            (0, 255, 0),    # зеленый
            (0, 0, 255),    # синий
            (255, 255, 0),  # желтый
            (255, 0, 255),  # пурпурный
            (0, 255, 255),  # голубой
            (128, 0, 0),    # темно-красный
            (0, 128, 0),    # темно-зеленый
            (0, 0, 128),    # темно-синий
            (128, 128, 0),  # оливковый
            (128, 0, 128),  # фиолетовый
            (0, 128, 128),  # темно-голубой
        ]
        return colors[label_id % len(colors)]