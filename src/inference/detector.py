import base64
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from src.utils import get_device


def _build_inference_transform(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class CarDamageDetector:
    """
    Combined car damage detector that runs:
    1. SegFormer-based damage segmentation (scratch / dent mask).
    2. YOLO-based car part segmentation.
    3. Per-part damage percentage analysis.
    """

    def __init__(
        self,
        seg_model_dir: str,
        yolo_model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        image_size: int = 512,
    ):
        self.device = device or get_device()
        self.image_size = image_size

        self.processor = AutoImageProcessor.from_pretrained(seg_model_dir)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(seg_model_dir)
        self.seg_model.to(self.device)
        self.seg_model.eval()

        self.yolo_model = None
        if yolo_model_path and Path(yolo_model_path).exists():
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_model_path)

        self._transform = _build_inference_transform(image_size)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def predict_mask(self, image_path: Union[str, Path]) -> np.ndarray:
        """Return binary damage mask (0/1) at original image resolution."""
        original = Image.open(image_path).convert("RGB")
        orig_w, orig_h = original.size

        image_np = np.array(original)
        tensor = self._transform(image=image_np)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.seg_model(pixel_values=tensor).logits
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        return mask.astype(np.uint8)

    # ------------------------------------------------------------------
    # YOLO part analysis
    # ------------------------------------------------------------------

    def analyze_parts(
        self,
        image_path: Union[str, Path],
        damage_mask: np.ndarray,
        conf: float = 0.25,
    ) -> List[dict]:
        """Return per-part damage analysis. Requires yolo_model_path to be set."""
        if self.yolo_model is None:
            return []

        orig = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig.size
        damage_bool = (damage_mask > 0).astype(np.uint8)

        yolo_results = self.yolo_model.predict(str(image_path), conf=conf)
        analysis: List[dict] = []

        for r in yolo_results:
            if r.masks is None or len(r.masks) == 0:
                continue

            masks_np = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(masks_np.shape[0]):
                part_mask = cv2.resize(
                    (masks_np[i] > 0.5).astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                part_area = int(np.count_nonzero(part_mask))
                if part_area == 0:
                    continue
                damage_pixels = int(np.count_nonzero(part_mask & damage_bool))
                analysis.append({
                    "part_name": self.yolo_model.names[class_ids[i]],
                    "instance_id": i,
                    "confidence": float(confs[i]),
                    "total_pixels": part_area,
                    "damage_pixels": damage_pixels,
                    "damage_percentage": round(damage_pixels / part_area * 100.0, 2),
                })

        return analysis

    # ------------------------------------------------------------------
    # Overlay helper
    # ------------------------------------------------------------------

    def build_overlay(self, image_path: Union[str, Path], damage_mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
        original = np.array(Image.open(image_path).convert("RGB"))
        mask_rgb = np.zeros_like(original)
        mask_rgb[damage_mask == 1] = [0, 0, 255]
        blended = (original * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
        return Image.fromarray(blended)

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        image_path: Union[str, Path],
        conf: float = 0.25,
        overlay_alpha: float = 0.5,
    ) -> dict:
        """
        Full pipeline: segmentation → part analysis → overlay.

        Returns a dict with:
            damage_mask   (np.ndarray)
            analysis      (list[dict])
            overlay_image (PIL.Image)
            overlay_b64   (str) — base64-encoded PNG for API responses
        """
        image_path = Path(image_path)
        damage_mask = self.predict_mask(image_path)
        analysis = self.analyze_parts(image_path, damage_mask, conf=conf)
        overlay = self.build_overlay(image_path, damage_mask, alpha=overlay_alpha)

        # Encode overlay to base64 for easy transport over HTTP
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            overlay.save(tmp.name)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            overlay_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp_path)

        return {
            "damage_mask": damage_mask,
            "analysis": analysis,
            "overlay_image": overlay,
            "overlay_b64": overlay_b64,
        }
