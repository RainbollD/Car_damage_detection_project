import os
import evaluate
import torch
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    UperNetForSemanticSegmentation,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import ImageColor

from ..data_preprocessing.dataset import CarSegmentationDataset
from ..data_preprocessing.augmentations import get_color_augmentations, get_shape_augmentations


class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_data(self, data_dir):
        """Подготовка данных и создание datasets"""
        # Загрузка информации о классах
        labels_df = pd.read_csv(os.path.join(data_dir, "masks_info.csv"), sep=",", header = 0)
        labels_df["elements"] = labels_df["elements"].str.replace(" ", "_")

        # Добавляем background класс
        labels_df.loc[len(labels_df)] = ["background", "#ffffff"]
        self.background_id = len(labels_df) - 1
        # Создаем маппинги
        self.rgb2id = {
            tuple(ImageColor.getrgb(labels_df.iloc[i]["color"])): i
            for i in range(len(labels_df))
        }
        self.id2label = {i: labels_df.iloc[i]["elements"] for i in range(len(labels_df))}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(self.id2label)

        # Создаем DataFrame с путями
        paths_df = self._create_paths_df(data_dir)

        # Разделяем на train/val/test
        train_df, nontrain_df = train_test_split(
            paths_df,
            test_size=self.config.val_percent + self.config.test_percent,
            random_state=42
        )
        eval_df, test_df = train_test_split(
            nontrain_df,
            test_size=self.config.test_percent / (self.config.val_percent + self.config.test_percent),
            random_state=42
        )

        # Создаем datasets
        self.train_dataset = CarSegmentationDataset(train_df)
        self.eval_dataset = CarSegmentationDataset(eval_df)
        self.test_dataset = CarSegmentationDataset(test_df)  # Устанавливаем маппинг цветов
        for dataset in [self.train_dataset, self.eval_dataset, self.test_dataset]:
            dataset.set_rgb_mapping(self.rgb2id)

    def _create_paths_df(self, data_dir):
        """Создает DataFrame с путями к изображениям и маскам"""
        paths_df = pd.DataFrame(columns=["image_path", "mask_path"])

        images_dir = os.path.join(data_dir, "img")
        masks_dir = os.path.join(data_dir, "masks")

        for img_name in os.listdir(images_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_name)
                mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join(masks_dir, mask_name)

                if os.path.exists(mask_path):
                    paths_df.loc[len(paths_df)] = [img_path, mask_path]

        return paths_df

    def setup_model(self):
        """Инициализация модели и процессора"""
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.model_name,
            reduce_labels=True
        )

        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            self.config.model_name,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

    def _get_model_class(self):
        """Определяет класс модели по имени"""
        if "mit-b" in self.config.model_name:
            return AutoModelForSemanticSegmentation
        elif "upernet" in self.config.model_name:
            return UperNetForSemanticSegmentation
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")

    def get_transforms(self, is_train=False):
        """Возвращает функцию трансформаций для dataset"""

        def transformations(data):
            if is_train:
                # Аугментации формы
                data = get_shape_augmentations()(
                    image=data["image"],
                    annotation=data["annotation"]
                )
                # Аугментации цвета
                data["image"] = get_color_augmentations()(image=data["image"])["image"]

            # Препроцессинг для модели
            inputs = self.image_processor(
                [data["image"]],
                [data["annotation"]],
                return_tensors="pt"
            )

            # Убираем batch dimension т.к. работаем с одним изображением
            inputs['pixel_values'] = inputs['pixel_values'][0]
            inputs['labels'] = inputs['labels'][0]

            return inputs

        return transformations

    def compute_metrics(self, pred):
        """Вычисление метрик для оценки"""
        metric = evaluate.load("mean_iou")

        logits, labels = pred
        logits_tensor = torch.from_numpy(logits)

        # Интерполяция к исходному размеру
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()

        # Вычисление метрик
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=self.num_labels,
            ignore_index=self.background_id,
            reduce_labels=False,
        )

        # Конвертация numpy arrays в lists

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()

        return metrics

    def train(self):
        """Запуск обучения"""
        # Применяем трансформации
        self.train_dataset.set_transform(self.get_transforms(is_train=True))
        self.eval_dataset.set_transform(self.get_transforms(is_train=False))
        self.test_dataset.set_transform(self.get_transforms(is_train=False))

        # Настройка аргументов обучения
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_strategy="steps",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            report_to=None,
        )

        # Создаем тренер
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Запускаем обучение
        print("Starting training...")
        trainer.train()

        # Сохраняем модель
        trainer.save_model(os.path.join(self.config.output_dir, "final"))

        # Оценка на тестовой выборке
        test_results = trainer.predict(self.test_dataset)
        print("Test results:", test_results.metrics)

        return trainer
