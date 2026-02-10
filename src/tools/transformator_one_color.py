import os
from PIL import Image
import numpy as np
from pathlib import Path


def filter_mask_color(
        input_path: Path,
        output_dir: Path,
        target_color: tuple,
        tolerance: int = 0
):
    """
    Оставляет только target_color, остальные пиксели делает черными.

    :param input_path: путь к исходной картинке
    :param output_dir: папка для сохранения
    :param target_color: (R, G, B) нужного цвета
    :param tolerance: допустимое отклонение цвета (0 = точное совпадение)
    """

    # Загружаем изображение
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)

    # Разделяем каналы
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    tr, tg, tb = target_color

    # Маска совпадения цвета
    mask = (
            (np.abs(r - tr) <= tolerance) &
            (np.abs(g - tg) <= tolerance) &
            (np.abs(b - tb) <= tolerance)
    )

    # Новое изображение (черное)
    result = np.zeros_like(data)

    # Оставляем нужный цвет
    result[mask] = data[mask]

    # Проверка: всё ли черное
    if not np.any(result):
        print("❌ Изображение полностью черное — не сохранено ", input_path)
        return False

    # Создаем папку, если нет
    os.makedirs(output_dir, exist_ok=True)

    # Имя файла
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    # Сохраняем
    Image.fromarray(result).save(output_path)

    print(f"✅ Сохранено: {output_path}")
    return True


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent

    path_img = ROOT_DIR / "dataset" / "car_damage_detection_with_detectron2" / "masks"
    path_dir_save = ROOT_DIR / "dataset" / "masks"

    for filename in os.listdir(path_img):
        filter_mask_color(Path(str(path_img) + f"/{filename}"), path_dir_save, target_color=(255, 51, 255))
