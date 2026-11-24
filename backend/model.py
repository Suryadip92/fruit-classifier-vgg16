"""VGG16 inference wrapper for the Fruit Classifier.

The architecture, class order and preprocessing mirror notebook.ipynb exactly:

* Backbone: torchvision ``vgg16`` (ImageNet VGG16-BN-free variant).
* Head:      ``classifier[3] = Linear(4096, 512)``
             ``classifier[6] = Linear(512, 16)``
* Classes:   ``sorted(p.name for p in dataset/image if p.is_dir())``
* Eval transform: Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize
* Inference: ``model.eval()`` + ``torch.no_grad()`` + softmax over 16 logits.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from threading import Lock

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMG_SIZE = 224
NUM_CLASSES = 16

# Identical to MEAN / STD in notebook.ipynb
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Identical to ``class_names = sorted(...)`` in notebook.ipynb (alphabetical).
CLASS_NAMES: tuple[str, ...] = (
    "apple fruit",
    "banana fruit",
    "cherry fruit",
    "chickoo fruit",
    "coconut fruit",
    "dates fruit",
    "grapes fruit",
    "guava fruit",
    "kiwi fruit",
    "mango fruit",
    "no fruit",
    "orange fruit",
    "pineapple fruit",
    "pomegranate fruit",
    "strawberry fruit",
    "watermelon fruit",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "best_fruit_vgg16.pth"

# Deterministic evaluation pipeline from the notebook (no augmentation).
eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def build_model() -> nn.Module:
    """Recreate the exact VGG16 architecture used during training.

    ``weights=None`` avoids downloading ImageNet weights: every parameter is
    overwritten by the trained checkpoint.
    """
    model = models.vgg16(weights=None)

    # Same head as notebook.ipynb (cells 27-29).
    model.classifier[3] = nn.Linear(4096, 512)
    model.classifier[6] = nn.Linear(512, NUM_CLASSES)
    return model


def resolve_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FruitClassifier:
    """Loads the checkpoint once and exposes a thread-safe predict method."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.device = resolve_device()
        self.checkpoint_path = Path(
            checkpoint_path or os.getenv("MODEL_PATH") or DEFAULT_CHECKPOINT
        ).expanduser()
        self.class_names = CLASS_NAMES
        self._lock = Lock()
        self.model = self._load()

    def _load(self) -> nn.Module:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}"
            )

        model = build_model()
        state_dict = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _prepare_image(image_bytes: bytes) -> Image.Image:
        """Decode, EXIF-orient, and convert an upload to RGB."""
        with Image.open(io.BytesIO(image_bytes)) as raw:
            raw.verify()  # raises on truncated / corrupted files
        with Image.open(io.BytesIO(image_bytes)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return image

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes and return real model output."""
        image = self._prepare_image(image_bytes)
        tensor = eval_transform(image).unsqueeze(0).to(self.device)

        with self._lock, torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        top_probability, top_index = torch.max(probabilities, dim=0)
        index = int(top_index.item())

        return {
            "predicted_class": self.class_names[index],
            "class_index": index,
            "confidence": float(top_probability.item()),
            "probabilities": {
                name: float(probabilities[i].item())
                for i, name in enumerate(self.class_names)
            },
        }
