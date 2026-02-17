# config.py
from dataclasses import dataclass, field
from typing import Tuple, Optional
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "nvidia/mit-b5"
    num_classes: int = 2
    image_size: int = 512

    # Paths
    root = Path(__file__).parent.parent
    output_dir: Path = root / "models" / "car_damage_segmentation"
    data_dir: Path = root / "dataset" / "classification_dent"

    # Data split
    val_percent: float = 0.13
    test_percent: float = 0.02

    # Training
    num_epochs: int = 50
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 100
    save_total_limit: int = 4
    early_stopping_patience: int = 5
    remove_unused_columns: bool = False
    seed: int = 42
    data_seed: int = 42

    # Data specific
    damage_color: Tuple[int, int, int] = (255, 51, 255)  # RGB код повреждения
    color_tolerance: int = 10  # допуск при сравнении цветов

    def save_to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
