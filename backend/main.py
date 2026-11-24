"""FastAPI application for the Fruit Classifier (PyTorch VGG16).

Serves the REST API and the static frontend from the same origin/port.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

try:  # supports both `uvicorn main:app` and `uvicorn backend.main:app`
    from .model import CLASS_NAMES, IMG_SIZE, FruitClassifier
except ImportError:  # pragma: no cover - direct script execution
    from model import CLASS_NAMES, IMG_SIZE, FruitClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fruit-classifier")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}

classifier: FruitClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained weights exactly once, at application startup."""
    global classifier
    logger.info("Loading VGG16 checkpoint...")
    classifier = FruitClassifier()
    logger.info("Model loaded on %s", classifier.device)
    yield
    classifier = None
    logger.info("Model released.")


app = FastAPI(
    title="Fruit Classifier API",
    description="Fruit classification with a fine-tuned VGG16 model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Local-only tool: allow the frontend served from file://, Live Server,
# Vite, or any localhost port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_classifier() -> FruitClassifier:
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return classifier


@app.get("/status")
def status() -> dict:
    return {
        "status": "ok",
        "service": "Fruit Classifier API",
        "model": "VGG16",
        "framework": "PyTorch",
        "endpoints": ["/", "/status", "/health", "/predict", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    model = get_classifier()
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "VGG16",
        "checkpoint": model.checkpoint_path.name,
        "device": str(model.device),
        "image_size": IMG_SIZE,
        "num_classes": len(model.class_names),
        "classes": list(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    model = get_classifier()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPEG, PNG, WEBP, BMP or GIF image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 10 MB).")

    try:
        result = model.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from None
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read the image: {exc}"
        ) from None
    except Exception as exc:  # pragma: no cover - unexpected inference failure
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail="Prediction failed. Please try another image."
        ) from exc
    finally:
        await file.close()

    predicted_class = result["predicted_class"]

    return {
        "success": True,
        "filename": file.filename,
        "predicted_class": predicted_class,
        "display_name": (
            "No Fruit Detected"
            if predicted_class == "no fruit"
            else predicted_class.replace(" fruit", "").title()
        ),
        "class_index": result["class_index"],
        "confidence": round(result["confidence"], 6),
        "probabilities": {
            name: round(probability, 6)
            for name, probability in sorted(
                result["probabilities"].items(), key=lambda item: item[1], reverse=True
            )
        },
    }


# Serve the existing frontend (index.html, style.css, app.js) from the same
# origin. Mounted last so the API routes above always take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
