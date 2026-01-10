import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.traning_config import TrainingConfig
from src.model_training.details_detection import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str, help="Path to the data directory")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name="nvidia/segformer-b5-finetuned-ade-512-512",
        data_dir=args.data_dir,
        output_dir="models/segformer-b5/car_damage_segmentation",
        batch_size=4,
        num_epochs=50,
        learning_rate=1e-5
    )

    print("Setting trainer...")
    trainer = SegmentationTrainer(config)

    print("Setup dataloader...")
    trainer.setup_data(config.data_dir)

    print("Setting up model...")
    trainer.setup_model()
    print("Starting training...")
    trainer.train()

    print("Training completed!")


if __name__ == "__main__":
    main()
