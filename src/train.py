from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from huggingface_hub import HfApi, RepositoryNotFoundError
from config import TrainingConfig
from data_utils import split_data, get_transforms, CarDamageDataset, data_collator
from metrics import compute_iou_metrics
from utils import set_seed
from hugging_face_tools import *

import argparse


def check_model_exists(repo_id, token):
    api = HfApi()
    try:
        api.model_info(repo_id=repo_id, token=token)
        return True
    except RepositoryNotFoundError:
        return False
    except Exception as e:
        print(f"Warning while checking model existence: {e}")
        return False


def main():
    config = TrainingConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", help="HF token")
    parser.add_argument("--dataset_name", help="dataset name")
    parser.add_argument("--model_name", help="Target HF Repo ID (e.g., username/my-model)")

    args = parser.parse_args()

    if args.hf_token:
        config.hf_token = args.hf_token
        print(f"HF token: {config.hf_token}")
    else:
        print("NOT FOUND HF TOKEN. Public models only, cannot upload.")

    if args.dataset_name:
        config.data_dir = config.data_dir / args.dataset_name
    else:
        print("NOT FOUND DATASET")
        exit(0)

    if args.model_name:
        config.model_name = args.model_name

    if not hasattr(config, 'hf_repo_id'):
        config.hf_repo_id = config.model_name

    print(f"Target Model Repo: {config.model_name}")
    set_seed(config.seed)

    print("Splitting dataset...")
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = split_data(
        str(config.data_dir), config.val_percent, config.test_percent, config.data_seed
    )
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

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

    print("Checking model availability on Hugging Face...")
    hf_token = getattr(config, 'hf_token', None)

    is_new_repo = check_model_exists(config.model_name, hf_token)

    load_model_name = config.model_name
    print(f"✅ Model '{config.model_name}'")

    processor = AutoImageProcessor.from_pretrained(
        load_model_name,
        token=hf_token
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        load_model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,
        token=hf_token
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )

    print("Starting training...")
    trainer.train()

    print("Saving model locally...")
    trainer.save_model()
    processor.save_pretrained(config.output_dir)
    print(f"Model saved to {config.output_dir}")

    push_target_repo = config.model_name

    should_push = getattr(config, 'push_to_hub', False) or is_new_repo

    if should_push:
        if not hf_token:
            print("❌ Cannot upload without HF Token.")
        else:
            print(f"\nPreparing to upload to {push_target_repo}...")
            try:
                api = HfApi()
                api.create_repo(
                    repo_id=push_target_repo,
                    token=hf_token,
                    repo_type="model",
                    exist_ok=True,
                    private=getattr(config, 'hf_private', False)
                )
                print(f"Repository '{push_target_repo}' verified/created.")

                if 'push_to_huggingface' in globals():
                    push_to_huggingface(
                        model_path=str(config.output_dir),
                        repo_id=push_target_repo,
                        token=hf_token,
                        tag=getattr(config, 'hf_tag', None),
                        private=getattr(config, 'hf_private', False)
                    )
                else:
                    trainer.push_to_hub(push_target_repo, token=hf_token)

                print("✅ Model uploaded to Hugging Face!")
            except Exception as e:
                print(f"❌ Failed to upload to Hugging Face: {e}")
                print("💡 Model saved locally, you can upload manually later")
    else:
        print("\n💡 Push to Hub disabled in config and repo already existed.")


if __name__ == "__main__":
    main()
