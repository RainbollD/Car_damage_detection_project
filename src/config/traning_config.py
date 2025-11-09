from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Модель
    model_name: str = "nvidia/mit-b5"

    # Директории
    output_dir: str = "./car_damage_segmentation"
    data_dir: str = "./data"

    # Параметры данных
    val_percent: float = 0.1
    test_percent: float = 0.05

    # Параметры обучения
    num_epochs: int = 20
    learning_rate: float = 6e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 2

    # Параметры логирования и сохранения
    save_steps: int = 25
    eval_steps: int = 10
    logging_steps: int = 10
    save_total_limit: int = 4

    # HuggingFace token (опционально)
    hf_token: str = None