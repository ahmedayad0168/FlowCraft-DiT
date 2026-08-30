import colorsys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from utils.torch_utils import default_device

# Hue centers (degrees) used for the coarse color check.
COLOR_HUES = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 55.0,
    "green": 120.0,
    "cyan": 180.0,
    "blue": 225.0,
    "purple": 275.0,
    "magenta": 310.0,
    "pink": 330.0,
}


class GenEvaluator:
    """GenEval-style compositional check: object presence, count and color."""

    def __init__(self, device: Optional[torch.device] = None, score_threshold: float = 0.5):
        self.device = device or default_device()
        self.score_threshold = score_threshold

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.categories = weights.meta["categories"]
        self.to_tensor = T.ToTensor()

    @staticmethod
    def _dominant_color(crop: np.ndarray) -> str:
        """Nearest named hue of a crop's mean saturated pixel."""
        pixels = crop.reshape(-1, 3) / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(*px) for px in pixels])
        saturated = hsv[(hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.15)]
        if saturated.size == 0:
            return "gray"
        hue = float(np.median(saturated[:, 0]) * 360.0)
        return min(COLOR_HUES, key=lambda name: min(
            abs(hue - COLOR_HUES[name]), 360.0 - abs(hue - COLOR_HUES[name])
        ))

    @torch.no_grad()
    def evaluate_image(
        self,
        image_path: str,
        target_object: str,
        expected_count: int = 1,
        expected_color: Optional[str] = None,
    ) -> Dict[str, Any]:
        img = Image.open(image_path).convert("RGB")
        img_np = np.asarray(img)
        img_tensor = self.to_tensor(img).unsqueeze(0).to(self.device)

        predictions = self.detector(img_tensor)[0]

        boxes: List[List[float]] = []
        colors: List[str] = []
        for score, label, box in zip(
            predictions["scores"], predictions["labels"], predictions["boxes"]
        ):
            if score.item() < self.score_threshold:
                continue
            category_name = self.categories[label.item()]
            if category_name.lower() != target_object.lower():
                continue
            box_list = box.cpu().numpy().tolist()
            boxes.append(box_list)
            if expected_color is not None:
                x0, y0, x1, y1 = (int(max(0, v)) for v in box_list)
                crop = img_np[y0:y1, x0:x1]
                colors.append(self._dominant_color(crop) if crop.size else "gray")

        detected_count = len(boxes)
        result: Dict[str, Any] = {
            "image_path": image_path,
            "target_object": target_object,
            "expected_count": expected_count,
            "detected_count": detected_count,
            "count_matched": float(detected_count == expected_count),
            "boxes": boxes,
        }
        if expected_color is not None:
            result["expected_color"] = expected_color
            result["detected_colors"] = colors
            result["color_matched"] = float(
                bool(colors) and any(c == expected_color.lower() for c in colors)
            )
        return result

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """samples: dicts of kwargs for evaluate_image. Returns per-sample + aggregate scores."""
        results = [self.evaluate_image(**sample) for sample in samples]
        aggregate = {
            "count_accuracy": float(np.mean([r["count_matched"] for r in results])) if results else 0.0
        }
        color_scores = [r["color_matched"] for r in results if "color_matched" in r]
        if color_scores:
            aggregate["color_accuracy"] = float(np.mean(color_scores))
        return {"results": results, "aggregate": aggregate}
