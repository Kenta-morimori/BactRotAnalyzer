"""Register a cropped binary reference to an enlarged grayscale source.

Registration similarity describes image correspondence, never centroid accuracy.
No estimator results or centroid tracks are used to fit the transform.
"""

from pathlib import Path

import cv2
import numpy as np

INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def verify_labels(labels, timeline):
    """Require independently saved filenames and timestamps for every row."""
    if len(labels) != len(timeline):
        raise ValueError("Stack/source counts differ")
    for label, row in zip(labels, timeline):
        lines = label.splitlines()
        if not lines or lines[0] != Path(row["tiff_path"]).name:
            raise ValueError("Stack source filename differs")
        if f"DateTime: {row['source_timestamp']}" not in lines:
            raise ValueError("Stack source timestamp differs")


def enlarge(image, scale=10, interpolation="linear"):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=INTERPOLATIONS[interpolation])


def optimal_threshold(image, reference):
    """Maximize foreground IoU with a strict intensity > threshold rule."""
    if image.shape != reference.shape or not reference.any() or reference.all():
        raise ValueError("Registration requires equal shapes and mixed reference labels")
    order = np.argsort(image.ravel(), kind="stable")
    values = image.ravel()[order]
    truth = reference.ravel()[order].astype(np.int64)
    tp = truth.sum() - np.cumsum(truth)
    fp = len(truth) - np.arange(1, len(truth) + 1) - tp
    scores = tp / (truth.sum() + fp)
    scores[:-1][values[:-1] == values[1:]] = -1
    index = int(scores.argmax())
    all_foreground_score = float(truth.mean())
    if all_foreground_score > scores[index]:
        return float(np.nextafter(float(values[0]), -np.inf)), all_foreground_score
    return float(values[index]), float(scores[index])


def register_crop(image, reference, scale=10, radius=5):
    """Search interpolations and integer crop locations around texture matching.

    Local IoU search follows a full-image normalized correlation search for each
    interpolation. This is a candidate transform; other frames must validate it.
    """
    h, w = reference.shape
    candidates = []
    for interpolation in INTERPOLATIONS:
        expanded = enlarge(image, scale, interpolation)
        if h > expanded.shape[0] or w > expanded.shape[1]:
            raise ValueError("Reference exceeds expanded source")
        correlation = cv2.matchTemplate(expanded.astype(np.float32), reference.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        _, _, _, (cx, cy) = cv2.minMaxLoc(correlation)
        for y in range(max(0, cy - radius), min(expanded.shape[0] - h, cy + radius) + 1):
            for x in range(max(0, cx - radius), min(expanded.shape[1] - w, cx + radius) + 1):
                threshold, iou = optimal_threshold(expanded[y : y + h, x : x + w], reference)
                candidates.append(dict(interpolation=interpolation, x=x, y=y, threshold=threshold, iou=iou))
    return sorted(candidates, key=lambda row: row["iou"], reverse=True)


def crop_image(image, transform):
    expanded = enlarge(image, transform["scale"], transform["interpolation"])
    x, y, w, h = (transform[key] for key in ("x", "y", "width", "height"))
    if x < 0 or y < 0 or x + w > expanded.shape[1] or y + h > expanded.shape[0]:
        raise ValueError("Crop lies outside the enlarged source")
    return expanded[y : y + h, x : x + w]


def reference_to_native(points, transform):
    """OpenCV resize pixel centers: native=(crop pixel+origin+0.5)/scale-0.5."""
    return (np.asarray(points, dtype=float) + [transform["x"], transform["y"]] + 0.5) / transform["scale"] - 0.5


def mask_iou(left, right):
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0
