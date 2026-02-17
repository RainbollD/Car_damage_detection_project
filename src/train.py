# train.py
import argparse
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from config import TrainingConfig
from data_utils import split_data, get_transforms, CarDamageDataset, data_collator
from metrics import compute_iou_metrics
from utils import set_seed
import os


def main():
    config = TrainingConfig()  # использовать значения по умолчанию
    print("Using default config.")

    set_seed(config.seed)

    # 1. Подготовка данных
    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_data(
        config.data_dir, config.val_percent, config.test_percent, config.data_seed
    )
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    image_transform, mask_transform = get_transforms(config.image_size)

    train_dataset = CarDamageDataset(
        train_images, train_masks,
        image_transform, mask_transform,
        config.damage_color, config.color_tolerance
    )
    val_dataset = CarDamageDataset(
        val_images, val_masks,
        image_transform, mask_transform,
        config.damage_color, config.color_tolerance
    )
    test_dataset = CarDamageDataset(
        test_images, test_masks,
        image_transform, mask_transform,
        config.damage_color, config.color_tolerance
    )

    # 2. Загрузка модели и процессора
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True
    )

    # 3. Настройка TrainingArguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",  # Изменено: добавили префикс eval_
        greater_is_better=True,
        remove_unused_columns=config.remove_unused_columns,
        seed=config.seed,
        report_to="none",  # или "wandb" при желании
        dataloader_num_workers=4,  # Добавлено для ускорения загрузки данных
    )

    # 4. Trainer - ИСПРАВЛЕНО: используем image_processor, а не tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,  # В новых версиях transformers нужно использовать processing_class
        data_collator=data_collator,
        compute_metrics=compute_iou_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )

    # 5. Обучение
    print("Starting training...")
    trainer.train()

    # 6. Сохранение модели
    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    # 7. Оценка на тесте
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
