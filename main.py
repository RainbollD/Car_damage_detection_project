import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.traning_config import TrainingConfig
from src.model_training.details_detection import SegmentationTrainer


def main():
    config = TrainingConfig(
        model_name="nvidia/mit-b5",
        data_dir="./data/classification_details",
        output_dir="./models/car_damage_segmentation",
        num_epochs=1,
        batch_size=2,
        learning_rate=6e-5,
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