import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.traning_config import TrainingConfig
from src.model_training.details_detection import SegmentationTrainer


def main():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    # args = parser.parse_args()

    config = TrainingConfig()

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
