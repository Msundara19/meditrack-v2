"""
ML inference wrapper for wound type classification.

Loads the EfficientNet-B0 checkpoint trained by training/train.py and provides
a single predict() call that returns the predicted class and per-class
confidence scores.

If the model checkpoint is not found (e.g. in a fresh clone before training),
the classifier degrades gracefully: is_available() returns False and cv_service
falls back to the heuristic rule-based classifier.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Checkpoint is stored at project_root/models/wound_classifier.pt
# Backend lives at project_root/backend/, so go up two levels.
_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent.parent / "models" / "wound_classifier.pt"
)

WOUND_CLASSES = [
    "surgical_incision",
    "laceration",
    "burn",
    "pressure_ulcer",
    "diabetic_ulcer",
    "abrasion",
    "venous_ulcer",
]

# Input size must match what was used during training (config.yaml → data.image_size)
_INPUT_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class MLWoundClassifier:
    """
    Singleton-style inference wrapper around the trained EfficientNet-B0.

    Usage
    -----
    classifier = MLWoundClassifier()          # loads model once
    if classifier.is_available():
        result = classifier.predict(image_path)
        # result = {"wound_type": "burn", "confidence": 0.91,
        #           "all_scores": {"burn": 0.91, "abrasion": 0.06, ...}}
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._device = None
        self._transform = None
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._model is not None

    def predict(self, image_path: str) -> Dict:
        """
        Run inference on a single wound image.

        Parameters
        ----------
        image_path : str
            Path to the input image (any format readable by PIL).

        Returns
        -------
        dict with keys:
            wound_type    : str   — top-1 predicted class name
            confidence    : float — softmax probability of top-1 class
            all_scores    : dict  — {class_name: probability} for all 7 classes
        """
        if not self.is_available():
            raise RuntimeError(
                "ML classifier is not available. Train the model first with "
                "training/train.py and ensure the checkpoint exists at "
                f"{self._model_path}"
            )

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            top_idx = int(probs.argmax())
            return {
                "wound_type": WOUND_CLASSES[top_idx],
                "confidence": float(probs[top_idx]),
                "all_scores": {
                    cls: round(float(p), 4)
                    for cls, p in zip(WOUND_CLASSES, probs)
                },
            }

        except Exception as exc:
            logger.error(f"ML inference failed for {image_path}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Attempt to load model; sets self._model = None on failure."""
        if not self._model_path.exists():
            logger.info(
                f"ML model checkpoint not found at {self._model_path}. "
                "Heuristic classifier will be used instead. "
                "Run training/train.py to train and save the model."
            )
            return

        try:
            import torch
            import timm
            import torch.nn as nn
            import torchvision.transforms as T

            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            checkpoint = torch.load(
                self._model_path, map_location=self._device, weights_only=False
            )

            # Rebuild the same architecture used during training
            cfg = checkpoint.get("config", {})
            num_classes = cfg.get("model", {}).get("num_classes", len(WOUND_CLASSES))
            dropout = cfg.get("model", {}).get("dropout", 0.3)
            arch = cfg.get("model", {}).get("architecture", "efficientnet_b0")

            model = timm.create_model(arch, pretrained=False, num_classes=0)
            in_features = model.num_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self._device).eval()

            self._model = model
            self._transform = T.Compose([
                T.Resize(int(_INPUT_SIZE * 1.14)),
                T.CenterCrop(_INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])

            epoch = checkpoint.get("epoch", "?")
            val_acc = checkpoint.get("val_accuracy", "?")
            logger.info(
                f"ML wound classifier loaded from epoch {epoch} "
                f"(val_accuracy={val_acc})"
            )

        except ImportError as exc:
            logger.warning(
                f"Could not import ML dependencies ({exc}). "
                "Install training/requirements.txt to enable the ML classifier."
            )
        except Exception as exc:
            logger.error(f"Failed to load ML model: {exc}")


# Module-level singleton — loaded once at import time
_classifier: Optional[MLWoundClassifier] = None


def get_ml_classifier() -> MLWoundClassifier:
    """Return (or lazily create) the module-level classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = MLWoundClassifier()
    return _classifier
