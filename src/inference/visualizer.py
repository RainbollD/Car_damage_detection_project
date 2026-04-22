from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def overlay_mask(
    original_image_path: str,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    original = np.array(Image.open(original_image_path).convert("RGB"))
    mask_rgb = np.zeros_like(original)
    mask_rgb[mask == 1] = [0, 0, 255]
    blended = (original * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    Image.fromarray(blended).save(output_path)


def visualize_damage(
    orig_img: Image.Image,
    yolo_results,
    damage_mask: np.ndarray,
    analysis: List[dict],
    save_path: str = "result_visualized.jpg",
) -> None:
    img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    damage_bool = damage_mask.astype(np.uint8)

    res_obj = yolo_results[0]
    if res_obj.masks is None or len(analysis) == 0:
        cv2.imwrite(save_path, img)
        return

    masks_np = res_obj.masks.data.cpu().numpy()
    boxes_xyxy = res_obj.boxes.xyxy.cpu().numpy()

    for res in analysis:
        if res["damage_percentage"] <= 0:
            continue

        i = res["instance_id"]
        part_mask = cv2.resize(
            (masks_np[i] > 0.5).astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        damage_intersection = (part_mask & damage_bool).astype(np.uint8)

        overlay_part = img.copy()
        overlay_part[part_mask > 0] = (0, 255, 0)
        img = cv2.addWeighted(overlay_part, 0.35, img, 0.65, 0)

        overlay_dmg = img.copy()
        overlay_dmg[damage_intersection > 0] = (0, 0, 255)
        img = cv2.addWeighted(overlay_dmg, 0.55, img, 0.45, 0)

        x1, y1, x2, y2 = map(int, boxes_xyxy[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        text = f"{res['part_name']}: {res['damage_percentage']}%"
        y_text = min(max(y1 + 25, 25), h - 10)
        cv2.putText(img, text, (x1 + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(save_path, img)
