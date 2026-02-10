from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Конфигурационный класс для обучения модели семантической сегментации
    """

    model_name: str = "nvidia/mit-b5"

    output_dir: str = "./models/car_damage_segmentation"
    data_dir: str = "./dataset/classification_dent"

    val_percent: float = 0.1
    test_percent: float = 0.05

    num_epochs: int = 50
    learning_rate: float = 2e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2

    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 100
    save_total_limit: int = 4

    early_stopping_patience: int = 5

    remove_unused_columns: bool = False

    seed: int = 42
    data_seed: int = 42
