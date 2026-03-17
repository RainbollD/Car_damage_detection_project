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
from hugging_face_tools import *

import argparse


def main():
    config = TrainingConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", help="HF token")
    args = parser.parse_args()
    if args.hf_token:
        config.hf_token = args.hf_token
        print(f"HF token: {config.hf_token}")
    else:
        print("NOT FOUND HF TOKEN")

    print("Using default config.")

    set_seed(config.seed)

    # 1. Подготовка данных
    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_data(
        str(config.data_dir), config.val_percent, config.test_percent, config.data_seed
    )
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Создаём трансформации: для train с аугментациями, для val/test без
    train_transform = get_transforms(config.image_size, is_training=True)
    val_transform = get_transforms(config.image_size, is_training=False)

    train_dataset = CarDamageDataset(
        train_images, train_masks,
        train_transform,
        config.damage_color,
        config.color_tolerance
    )
    val_dataset = CarDamageDataset(
        val_images, val_masks,
        val_transform,
        config.damage_color,
        config.color_tolerance
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
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=config.seed,
        report_to="none",
        dataloader_num_workers=4,
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_iou_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )

    # 5. Обучение
    print("Starting training...")
    trainer.train()

    # 6. Сохранение локально
    print("💾 Saving model locally...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    print(f"✅ Model saved to {config.output_dir}")

    # 7.  Загрузка на Hugging Face (если включено в конфиге)
    if hasattr(config, 'push_to_hub') and config.push_to_hub:
        print("\n🚀 Uploading to Hugging Face...")
        try:
            push_to_huggingface(
                model_path=str(config.output_dir),
                repo_id=config.hf_repo_id,
                token=getattr(config, 'hf_token', None),
                tag=getattr(config, 'hf_tag', None),
                private=getattr(config, 'hf_private', False)
            )
            print("✅ Model uploaded to Hugging Face!")
        except Exception as e:
            print(f"❌ Failed to upload to Hugging Face: {e}")
            print("💡 Model saved locally, you can upload manually later")
    else:
        print("\n💡 To upload to Hugging Face, set push_to_hub=True in config")


if __name__ == "__main__":
    main()
