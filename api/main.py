import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from api.routes.detection import router as detection_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    seg_model_path = os.getenv("SEG_MODEL_PATH")
    yolo_model_path = os.getenv("YOLO_MODEL_PATH")

    if seg_model_path:
        from src.inference import CarDamageDetector
        app.state.detector = CarDamageDetector(
            seg_model_dir=seg_model_path,
            yolo_model_path=yolo_model_path,
            image_size=int(os.getenv("IMAGE_SIZE", 512)),
        )
        print(f"Segmentation model loaded from: {seg_model_path}")
        if yolo_model_path:
            print(f"YOLO model loaded from: {yolo_model_path}")
    else:
        app.state.detector = None
        print("WARNING: SEG_MODEL_PATH not set. /detect endpoint will return 503.")

    yield

    app.state.detector = None


app = FastAPI(
    title="Car Damage Detection API",
    description=(
        "Detects car damage using a SegFormer segmentation model "
        "and optionally maps damage to specific car parts via YOLO."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router, prefix="/api/v1")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_static / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
