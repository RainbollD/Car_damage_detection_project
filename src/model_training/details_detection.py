import os
import torch
import numpy as np
import pandas as pd
import evaluate

from PIL import ImageColor
from sklearn.model_selection import train_test_split
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    UperNetForSemanticSegmentation,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from ..data_preprocessing.dataset import CarSegmentationDataset
from ..data_preprocessing.augmentations import get_color_augmentations, get_shape_augmentations


class SegmentationTrainer:
    """
    Бинарная сегментация:
      - background (0) — сталь/фон
      - dent (1) — вмятина

    ВАЖНО:
    - masks_info.csv может содержать много цветов/классов — мы "схлопываем" всё НЕ background в dent.
    - background определяется как elements == 'background' И/ИЛИ цвет #ffffff.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Метрика грузится один раз (а не на каждом eval)
        self.metric = evaluate.load("mean_iou")

        # ignore_index для void/pad; background НЕ игнорируем
        self.ignore_index = 255

        # Заполним позже
        self.image_processor = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # Бинарные маппинги
        self.id2label = {0: "background", 1: "dent"}
        self.label2id = {"background": 0, "dent": 1}
        self.num_labels = 2

        # для инфо/визуализации (не критично)
        self.label_colors = [
            [255, 255, 255],  # background
            [255, 0, 0],      # dent
        ]

        self.rgb2id = {}

    def setup_data(self, data_dir):
        """Подготовка данных, создание datasets, настройка rgb->id (строго 2 класса)."""

        labels_df = pd.read_csv(os.path.join(data_dir, "masks_info.csv"), sep=",", header=0)
        labels_df["elements"] = labels_df["elements"].astype(str).str.replace(" ", "_")
        labels_df["color"] = labels_df["color"].astype(str).str.lower()

        # === rgb -> binary_id ===
        bg_rgb = (255, 255, 255)  # #ffffff
        self.rgb2id = {}

        for _, row in labels_df.iterrows():
            rgb = tuple(ImageColor.getrgb(row["color"]))
            is_bg = (row["elements"] == "background") or (rgb == bg_rgb)
            self.rgb2id[rgb] = 0 if is_bg else 1

        # гарантируем, что белый всегда считается фоном
        self.rgb2id[bg_rgb] = 0

        paths_df = self._create_paths_df(data_dir)

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

        self.train_dataset = CarSegmentationDataset(train_df)
        self.eval_dataset = CarSegmentationDataset(eval_df)
        self.test_dataset = CarSegmentationDataset(test_df)

        for dataset in [self.train_dataset, self.eval_dataset, self.test_dataset]:
            dataset.set_rgb_mapping(self.rgb2id)

    def _create_paths_df(self, data_dir):
        """Создает DataFrame с путями к изображениям и маскам."""
        paths_df = pd.DataFrame(columns=["image_path", "mask_path"])

        images_dir = os.path.join(data_dir, "img")
        masks_dir = os.path.join(data_dir, "masks")

        for img_name in os.listdir(images_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(images_dir, img_name)
                mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join(masks_dir, mask_name)

                if os.path.exists(mask_path):
                    paths_df.loc[len(paths_df)] = [img_path, mask_path]

        return paths_df

    def setup_model(self):
        """Инициализация модели и процессора."""
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.config.model_name,
            reduce_labels=False
        )

        model_class = self._get_model_class()
        self.model = model_class.from_pretrained(
            self.config.model_name,
            num_labels=self.num_labels,      # <-- 2 класса
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # (опционально) для визуализации/логов
        self.model.config.label_colors = self.label_colors

        # ignore_index для loss, если модель поддерживает
        if hasattr(self.model.config, "semantic_loss_ignore_index"):
            self.model.config.semantic_loss_ignore_index = self.ignore_index
        if hasattr(self.model.config, "ignore_index"):
            self.model.config.ignore_index = self.ignore_index

    def _get_model_class(self):
        """Определяет класс модели по имени."""
        name = self.config.model_name.lower()
        if "segformer" in name:
            return SegformerForSemanticSegmentation
        elif "upernet" in name:
            return UperNetForSemanticSegmentation
        else:
            return AutoModelForSemanticSegmentation

    def get_transforms(self, is_train=False):
        """Возвращает функцию трансформаций для dataset."""

        def transformations(data):
            # ожидаем, что dataset дает:
            # data["image"] -> PIL/np/torch image
            # data["annotation"] -> mask (уже в id после rgb2id внутри dataset)
            if is_train:
                data = get_shape_augmentations()(image=data["image"], annotation=data["annotation"])
                data["image"] = get_color_augmentations()(image=data["image"])["image"]

            # ВАЖНО: используем корректный аргумент segmentation_maps
            inputs = self.image_processor(
                images=data["image"],
                segmentation_maps=data["annotation"],
                return_tensors="pt"
            )

            # снимаем batch=1
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # На всякий случай: приводим void/pad к ignore_index, если dataset где-то оставил 255 — ок
            if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
                inputs["labels"] = inputs["labels"].long()

            return inputs

        return transformations

    def compute_metrics(self, pred):
        """Вычисление метрик для оценки."""
        logits = pred.predictions   # numpy: (N, C, H, W) обычно
        labels = pred.label_ids     # numpy: (N, H, W)

        logits_tensor = torch.from_numpy(logits)

        # приводим логиты к размеру labels
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        pred_labels = logits_tensor.argmax(dim=1).cpu().numpy()

        metrics = self.metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=self.num_labels,        # 2
            ignore_index=self.ignore_index,    # 255 (не фон!)
            reduce_labels=False,
        )

        # удобная метрика именно для класса dent (1)
        if "per_category_iou" in metrics and metrics["per_category_iou"] is not None:
            ious = metrics["per_category_iou"]
            # ious[0]=background, ious[1]=dent
            metrics["iou_dent"] = float(ious[1])

        # numpy -> list для сериализации
        for key, value in list(metrics.items()):
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()

        return metrics

    def train(self):
        """Запуск обучения."""
        self.train_dataset.set_transform(self.get_transforms(is_train=True))
        self.eval_dataset.set_transform(self.get_transforms(is_train=False))
        self.test_dataset.set_transform(self.get_transforms(is_train=False))

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            save_strategy="steps",
            eval_strategy="steps",
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,

            load_best_model_at_end=True,
            remove_unused_columns=False,  # <-- обычно нужно для segmentation

            metric_for_best_model="iou_dent",  # или "mean_iou"
            greater_is_better=True,

            seed=self.config.seed,
            data_seed=self.config.data_seed,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )

        print("Starting training...")
        trainer.train()

        trainer.save_model(os.path.join(self.config.output_dir, "final_model"))
        self.image_processor.save_pretrained(os.path.join(self.config.output_dir, "final_image_processor"))

        test_results = trainer.predict(self.test_dataset)
        print("Test results:", test_results.metrics)

        return trainer
