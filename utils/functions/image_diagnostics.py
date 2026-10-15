"""Contour measurements for image/trajectory diagnostics; no centroid extraction."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class ContourMeasurement:
    contour: Optional[np.ndarray]
    count: int
    area: float
    circularity: float
    border_touch: bool


def measure_contour(image: np.ndarray, threshold: int = 120) -> ContourMeasurement:
    """Measure the largest external contour with the analysis threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ContourMeasurement(None, 0, np.nan, np.nan, False)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    return ContourMeasurement(
        contour,
        len(contours),
        area,
        4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan,
        x == 0 or y == 0 or x + width >= gray.shape[1] or y + height >= gray.shape[0],
    )
