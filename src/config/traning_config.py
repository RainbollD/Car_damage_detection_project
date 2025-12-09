from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Конфигурационный класс для обучения модели семантической сегментации
    """

    model_name: str = "nvidia/mit-b5"

    output_dir: str = "./models/car_damage_segmentation"
    data_dir: str = "./data"

    val_percent: float = 0.1
    test_percent: float = 0.05

    num_epochs: int = 20
    learning_rate: float = 3e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2

    save_steps: int = 25
    eval_steps: int = 10
    logging_steps: int = 10
    save_total_limit: int = 4