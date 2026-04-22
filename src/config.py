import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_ROOT = Path(__file__).parent.parent


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "nvidia/mit-b5"
    num_classes: int = 2
    image_size: int = int(os.getenv("IMAGE_SIZE", 512))

    # Paths
    output_dir: Path = _ROOT / "models" / "car_damage_segmentation"
    data_dir: Path = _ROOT / "dataset"

    # Hugging Face
    push_to_hub: bool = True
    hf_repo_id: str = os.getenv("HF_REPO_ID", "car_scratch_segment")
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    hf_tag: str = "v1.0.1"
    hf_private: bool = False

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
    damage_color: Tuple[int, int, int] = (0, 0, 255)
    color_tolerance: int = 10

    def save_to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
