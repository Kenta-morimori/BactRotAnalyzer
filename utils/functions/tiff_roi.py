"""One-target TIFF tracking inside a fixed, optionally inferred ROI.

Missing coordinates never become predictions. The last measured position is
only a search reference; an ambiguous contact needs separation or a new click.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def validate(self, shape):
        if any(type(v) is not int for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("ROI coordinates must be integer native-pixel bounds")
        if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
            raise ValueError("Invalid ROI bounds")
        if self.x + self.width > shape[1] or self.y + self.height > shape[0]:
            raise ValueError("ROI lies outside the source image")


@dataclass
class Region:
    label: int
    point: np.ndarray
    area: float
    box: tuple
    edge: bool
    mask: np.ndarray


def binary_regions(image, roi, threshold, scale=10):
    """Resize original intensities before thresholding; preserve pixel centers."""
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale TIFF plane")
    roi.validate(image.shape)
    if not np.isfinite(threshold) or threshold <= 0 or type(scale) is not int or scale < 1:
        raise ValueError("Positive threshold and integer scale required")
    expanded = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    binary = (
        expanded[roi.y * scale : (roi.y + roi.height) * scale, roi.x * scale : (roi.x + roi.width) * scale] > threshold
    )
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    objects = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 3 * scale**2:
            continue
        objects.append(
            Region(
                label=label,
                point=(centers[label] + 0.5) / scale - 0.5 + [roi.x, roi.y],
                area=float(area / scale**2),
                box=(roi.x + x / scale, roi.y + y / scale, roi.x + (x + w) / scale, roi.y + (y + h) / scale),
                edge=bool(x == 0 or y == 0 or x + w == binary.shape[1] or y + h == binary.shape[0]),
                mask=labels == label,
            )
        )
    return binary, labels, objects


@dataclass
class Observation:
    point: Optional[np.ndarray]
    mask: np.ndarray
    foreground: np.ndarray
    reason: str
    reacquired: bool = False
    region: Optional[Region] = None


class Tracker:
    """Nearest-region tracking; no appearance model, watershed or object IDs."""

    def __init__(self, roi, threshold=2000, scale=10, max_dt=0.0075):
        if not np.isfinite(max_dt) or max_dt <= 0:
            raise ValueError("max_dt must be positive")
        self.roi, self.threshold, self.scale, self.max_dt = roi, threshold, scale, max_dt
        self.point: Optional[np.ndarray] = None
        self.area: Optional[float] = None
        self.last_row = None
        self.previous_labels = None
        self.previous_areas = {}
        self.missing = False
        self.contact = False
        self.contact_mask = None

    def step(self, image, source_frame, stamp, segment=0, seed=None):
        if type(source_frame) is not int or source_frame < 1:
            raise ValueError("Original frame must be a positive integer")
        boundary = False
        if self.last_row is not None:
            frame, time, recording = self.last_row
            boundary = (
                source_frame != frame + 1
                or recording != segment
                or not (np.isfinite(time) and np.isfinite(stamp) and 0 < stamp - time <= self.max_dt)
            )
        self.last_row = source_frame, stamp, segment
        if boundary:
            self.point = self.area = None
            self.contact = False
            self.contact_mask = None
            self.previous_labels = None
            self.previous_areas = {}
        binary, labels, objects = binary_regions(image, self.roi, self.threshold, self.scale)
        result = Observation(None, np.zeros(binary.shape, bool), binary, "initial_target_required")
        previous_labels, previous_areas = self.previous_labels, self.previous_areas
        self.previous_labels = labels
        self.previous_areas = {obj.label: int(obj.mask.sum()) for obj in objects}
        if not np.isfinite(stamp):
            self.point = self.area = None
            result.reason = "invalid_time"
            return result
        chosen = None
        if seed is not None:
            seed = np.asarray(seed, dtype=float)
            if seed.shape != (2,) or not np.isfinite(seed).all():
                raise ValueError("Seed must be a finite native-image x,y point")
            local = np.rint((seed - [self.roi.x, self.roi.y] + 0.5) * self.scale - 0.5).astype(int)
            x, y = local
            choices = [
                obj for obj in objects if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and obj.mask[y, x]
            ]
            self.point = self.area = None
            self.contact = self.missing = False
            self.contact_mask = None
            if len(choices) != 1:
                result.reason = "seed_not_foreground"
                return result
            chosen = choices[0]
            self.area = chosen.area
        elif self.point is None:
            return result
        else:
            reference_area = self.area
            assert reference_area is not None
            radius = float(np.sqrt(reference_area / np.pi))
            possible = [obj for obj in objects if np.linalg.norm(obj.point - self.point) <= 2 * radius]
            if not possible:
                self.missing = True
                result.reason = "not_found"
                return result
            possible.sort(key=lambda obj: np.linalg.norm(obj.point - self.point))
            chosen = possible[0]
            # Area is a broad identity guard, not a shape classifier.
            if chosen.area > 2 * reference_area:
                self.contact = True
                self.contact_mask = chosen.mask.copy()
                self.missing = True
                result.reason = "merged_or_too_large"
                return result
            if chosen.area < max(3, 0.25 * reference_area):
                self.missing = True
                result.reason = "too_small"
                return result
            if previous_labels is not None:
                overlaps = np.bincount(previous_labels[chosen.mask], minlength=int(previous_labels.max()) + 1)
                contributors = sum(
                    overlaps[label] >= max(self.scale**2, 0.25 * area) for label, area in previous_areas.items()
                )
                if contributors >= 2:
                    self.contact = True
                    self.contact_mask = chosen.mask.copy()
                    self.missing = True
                    result.reason = "merged"
                    return result
            if self.contact:
                # A merged blob staying merged must not silently become valid
                # next frame just because the previous mask is now one region.
                split_parts = [
                    obj
                    for obj in objects
                    if self.contact_mask is not None and np.count_nonzero(obj.mask & self.contact_mask) >= self.scale**2
                ]
                if len(split_parts) < 2:
                    if split_parts:
                        self.contact_mask = split_parts[0].mask.copy()
                    self.missing = True
                    result.reason = "contact_unresolved"
                    return result
            eligible = [
                obj for obj in possible if not obj.edge and 0.25 * reference_area <= obj.area <= 2 * reference_area
            ]
            if not eligible:
                self.missing = True
                result.reason = "roi_edge"
                return result
            eligible.sort(key=lambda obj: np.linalg.norm(obj.point - self.point))
            if len(eligible) > 1 and np.linalg.norm(eligible[1].point - self.point) - np.linalg.norm(
                eligible[0].point - self.point
            ) < max(1, 0.25 * radius):
                self.missing = True
                result.reason = "ambiguous"
                return result
            chosen = eligible[0]
        if chosen.edge:
            self.missing = True
            result.reason = "roi_edge"
            return result
        reacquired = bool(self.missing and seed is None)
        self.point = chosen.point.copy()
        self.missing = self.contact = False
        self.contact_mask = None
        return Observation(
            chosen.point, chosen.mask, binary, "reacquired" if reacquired else "accepted", reacquired, chosen
        )


def nominal_max_dt(rows):
    times = np.array([np.nan if row["time_sec"] is None else row["time_sec"] for row in rows], float)
    dt = np.diff(times)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    return 1.5 * float(np.median(positive)) if len(positive) else 0.0075


def infer_roi(images, rows, seed, threshold=2000, scale=10):
    """Use at most one contiguous second / 200 frames; never consume a gap."""
    if not rows or len(images) != len(rows):
        raise ValueError("Calibration images and rows must be nonempty and aligned")
    h, w = images[0].shape
    full = ROI(0, 0, w, h)
    tracker = Tracker(full, threshold, scale, nominal_max_dt(rows))
    boxes = []
    areas = []
    used = 0
    for index, (image, row) in enumerate(zip(images, rows)):
        if index >= 200:
            break
        if row["time_sec"] is None or rows[0]["time_sec"] is None:
            break
        if row["time_sec"] - rows[0]["time_sec"] >= 1:
            break
        if index and (
            row["segment_id"] != rows[0]["segment_id"]
            or row["source_frame_1based"] != rows[index - 1]["source_frame_1based"] + 1
            or not 0 < row["time_sec"] - rows[index - 1]["time_sec"] <= tracker.max_dt
        ):
            break
        used += 1
        result = tracker.step(
            image,
            row["source_frame_1based"],
            row["time_sec"],
            row["segment_id"],
            seed if index == 0 else None,
        )
        if result.region is not None:
            boxes.append(result.region.box)
            areas.append(result.region.area)
    summary = dict(
        calibration_frames=used,
        detected_frames=len(boxes),
        accepted_fraction=len(boxes) / used if used else 0,
    )
    if len(boxes) < 20 or len(boxes) < 0.8 * used:
        return None, dict(summary, reason="manual_roi_required")
    margin = max(2, float(np.sqrt(np.median(areas) / np.pi)) / 2)
    bounds = np.asarray(boxes)
    x, y = np.floor(bounds[:, :2].min(axis=0) - margin).astype(int)
    right, bottom = np.ceil(bounds[:, 2:].max(axis=0) + margin).astype(int)
    x, y = max(0, int(x)), max(0, int(y))
    right, bottom = min(w, int(right)), min(h, int(bottom))
    return ROI(x, y, right - x, bottom - y), dict(summary, reason="auto", margin_native_px=margin)
