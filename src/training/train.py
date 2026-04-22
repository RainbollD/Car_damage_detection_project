import argparse

from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import TrainingConfig
from src.data import CarDamageDataset, data_collator, get_transforms, split_data
from src.training.metrics import compute_iou_metrics
from src.utils import set_seed


def main() -> None:
    config = TrainingConfig()

    parser = argparse.ArgumentParser(description="Train car damage segmentation model")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--hf_repo_id", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()

    if args.hf_token:
        config.hf_token = args.hf_token
    else:
        print("HF token not provided. Public models only, upload disabled.")

    if args.dataset_name:
        config.data_dir = config.data_dir / args.dataset_name
    elif not config.data_dir.exists():
        print("Dataset not found.")
        return

    if args.model_name:
        config.model_name = args.model_name
    if args.hf_repo_id:
        config.hf_repo_id = args.hf_repo_id

    set_seed(config.seed)

    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), _ = split_data(
        str(config.data_dir), config.val_percent, config.test_percent, config.data_seed
    )
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    train_dataset = CarDamageDataset(
        train_images,
        train_masks,
        get_transforms(config.image_size, is_training=True),
        config.damage_color,
        config.color_tolerance,
    )
    val_dataset = CarDamageDataset(
        val_images,
        val_masks,
        get_transforms(config.image_size, is_training=False),
        config.damage_color,
        config.color_tolerance,
    )

    print(f"Loading model: {config.model_name}")
    processor = AutoImageProcessor.from_pretrained(config.model_name, token=config.hf_token)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,
        token=config.hf_token,
        use_safetensors=True,
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_iou_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")

    if config.push_to_hub and config.hf_token:
        print(f"Pushing to Hugging Face Hub: {config.hf_repo_id}")
        try:
            trainer.push_to_hub(
                repo_id=config.hf_repo_id,
                token=config.hf_token,
                commit_message="Upload trained model",
                private=config.hf_private,
            )
            print("Model uploaded to Hugging Face.")
        except Exception as exc:
            print(f"Upload failed: {exc}")
    else:
        print("Push to Hub skipped (disabled or no token).")


if __name__ == "__main__":
    main()
