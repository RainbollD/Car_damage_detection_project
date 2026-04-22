#!/usr/bin/env python3
"""
Batch inference script.

Usage:
    python scripts/infer.py --input path/to/images --output path/to/results \
        --seg_model models/car_damage_segmentation \
        --yolo_model best.pt
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.inference import CarDamageDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch car damage inference")
    parser.add_argument(
        "--input", required=True, help="Path to a single image or a directory of images"
    )
    parser.add_argument("--output", required=True, help="Directory to save overlay results")
    parser.add_argument(
        "--seg_model",
        default=os.getenv("SEG_MODEL_PATH"),
        help="Path or HF repo id for segmentation model",
    )
    parser.add_argument(
        "--yolo_model",
        default=os.getenv("YOLO_MODEL_PATH"),
        help="Path to YOLO .pt weights file (optional)",
    )
    parser.add_argument("--image_size", type=int, default=int(os.getenv("IMAGE_SIZE", 512)))
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.seg_model:
        print("Error: --seg_model is required (or set SEG_MODEL_PATH env var).")
        sys.exit(1)

    detector = CarDamageDetector(
        seg_model_dir=args.seg_model,
        yolo_model_path=args.yolo_model,
        image_size=args.image_size,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        [input_path]
        if input_path.is_file()
        else [p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if not image_paths:
        print(f"No images found at: {input_path}")
        sys.exit(1)

    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        try:
            result = detector.predict(img_path, conf=args.conf)
            overlay_path = output_dir / f"{img_path.stem}_overlay.png"
            result["overlay_image"].save(str(overlay_path))

            analysis = result["analysis"]
            if analysis:
                print(f"  {'Part':<15} | {'Conf':>5} | {'Damage %':>8}")
                print(f"  {'-'*35}")
                for p in analysis:
                    print(f"  {p['part_name']:<15} | {p['confidence']:>5.2f} | {p['damage_percentage']:>7.1f}%")
            else:
                mask = result["damage_mask"]
                ratio = (mask > 0).sum() / mask.size * 100
                print(f"  Damage coverage: {ratio:.1f}%")
        except Exception as exc:
            print(f"  Error: {exc}")

    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
