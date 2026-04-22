from typing import List, Optional

from pydantic import BaseModel, Field


class PartAnalysis(BaseModel):
    part_name: str
    instance_id: int
    confidence: float
    total_pixels: int
    damage_pixels: int
    damage_percentage: float


class DetectionResponse(BaseModel):
    filename: str
    damage_detected: bool
    damaged_pixel_count: int
    total_pixel_count: int
    damage_ratio: float = Field(description="Fraction of image pixels classified as damaged")
    parts_analysis: List[PartAnalysis] = Field(default_factory=list)
    overlay_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG overlay image (damage highlighted in blue)",
    )


class HealthResponse(BaseModel):
    status: str
    seg_model_loaded: bool
    yolo_model_loaded: bool
