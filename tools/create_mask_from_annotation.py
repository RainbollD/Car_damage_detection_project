"""
Utility script: convert COCO-format segmentation annotations into binary PNG masks.

Usage (edit the constants at the bottom, then run):
    python tools/create_mask_from_annotation.py
"""
import json
import uuid
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

# Root of the project so relative paths resolve correctly when called from any cwd.
_PROJECT_ROOT = Path(__file__).parent.parent


def create_mask_from_annotations(
    annotations: list,
    image_width: int = 640,
    image_height: int = 640,
) -> Image.Image:
    """Draw segmentation polygons onto a black RGB canvas; damage areas are blue (0, 0, 255)."""
    mask = Image.new("RGB", (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        for poly in annotation.get("segmentation", []):
            points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(points, fill=(0, 0, 255))
    return mask


def convert_coco_to_masks(
    coco_json_path: str,
    images_source_dir: str,
    output_dir: str,
    category_id: int,
) -> None:
    """
    Parse a COCO JSON file and produce image+mask pairs under `output_dir`.

    Args:
        coco_json_path:     Path to `_annotations.coco.json`.
        images_source_dir:  Directory that contains the raw images referenced in the JSON.
        output_dir:         Destination root; `images/` and `masks/` subdirs are created automatically.
        category_id:        COCO category id to extract (e.g. 8 for scratch).
    """
    coco_path = Path(coco_json_path)
    images_dir = Path(images_source_dir)
    save_dir = Path(output_dir)

    mask_dir = save_dir / "masks"
    img_dir = save_dir / "images"
    mask_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco_data = json.load(f)

    images_dict = {img["id"]: img for img in coco_data["images"]}

    grouped: defaultdict = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann["category_id"] == category_id:
            grouped[(ann["image_id"], ann["category_id"])].append(ann)

    for (image_id, _), ann_list in grouped.items():
        image_info = images_dict.get(image_id)
        if not image_info:
            print(f"Image id {image_id} not found in JSON, skipping.")
            continue

        width = image_info.get("width", 640)
        height = image_info.get("height", 640)
        mask = create_mask_from_annotations(ann_list, image_width=width, image_height=height)

        filename = f"{uuid.uuid4().hex}_{image_id}.png"
        mask.save(mask_dir / filename)

        src_img_path = images_dir / image_info["file_name"]
        if src_img_path.exists():
            img = Image.open(src_img_path).convert("RGB")
            img.save(img_dir / filename)
            print(f"Saved: {filename} ({len(ann_list)} annotations)")
        else:
            print(f"Source image not found: {src_img_path}")


if __name__ == "__main__":
    # ── Edit these before running ──────────────────────────────────────────────
    CATEGORY_ID = 8
    COCO_SPLIT = "valid"
    COCO_ROOT = Path("/home/lev/Downloads/carsss 3.v1i.coco-segmentation") / COCO_SPLIT
    OUTPUT_DIR = _PROJECT_ROOT / "dataset" / "classification_scratch"
    # ───────────────────────────────────────────────────────────────────────────

    convert_coco_to_masks(
        coco_json_path=str(COCO_ROOT / "_annotations.coco.json"),
        images_source_dir=str(COCO_ROOT),
        output_dir=str(OUTPUT_DIR),
        category_id=CATEGORY_ID,
    )
