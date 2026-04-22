import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas import DetectionResponse, HealthResponse, PartAnalysis

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _get_detector(request: Request):
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set SEG_MODEL_PATH.")
    return detector


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    detector = request.app.state.detector
    return HealthResponse(
        status="ok",
        seg_model_loaded=detector is not None,
        yolo_model_loaded=detector is not None and detector.yolo_model is not None,
    )


@router.post("/detect", response_model=DetectionResponse, tags=["detection"])
async def detect(
    request: Request,
    file: UploadFile = File(..., description="Car image (JPEG or PNG)"),
    conf: float = 0.25,
    include_overlay: bool = True,
) -> DetectionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG or PNG.",
        )

    detector = _get_detector(request)

    contents = await file.read()
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        result = detector.predict(tmp_path, conf=conf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    damage_mask: np.ndarray = result["damage_mask"]
    total_pixels = int(damage_mask.size)
    damaged_pixels = int((damage_mask > 0).sum())

    return DetectionResponse(
        filename=file.filename or "",
        damage_detected=damaged_pixels > 0,
        damaged_pixel_count=damaged_pixels,
        total_pixel_count=total_pixels,
        damage_ratio=round(damaged_pixels / total_pixels, 6) if total_pixels else 0.0,
        parts_analysis=[PartAnalysis(**p) for p in result["analysis"]],
        overlay_b64=result["overlay_b64"] if include_overlay else None,
    )
