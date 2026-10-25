"""Experimental segmentation and selection comparisons, never production extraction.

The selector sees images, frame/time identity and explicit user seeds only.
Synthetic object masks/IDs are available exclusively to the scoring functions.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.functions.centroid_diagnostics import position_metrics


@dataclass
class Component:
    coverage: np.ndarray
    center: np.ndarray
    area: float
    contour_area: float
    aspect: float
    border: bool
    eligible: bool
    external_rank: int


def settings_digest(settings):
    """Bind cache and freeze records to settings and the implementation."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(f"opencv={getattr(cv2, '__version__')};numpy={np.__version__}".encode())
    return digest.hexdigest()


def segment(image, method):
    """Apply explicit intensity/resize conditions without normalizing uint16."""
    if image.ndim != 2:
        raise ValueError("Expected a grayscale plane")
    threshold = method["threshold"]
    if method["threshold_mode"] == "background_relative":
        threshold += float(np.median(image))
    elif method["threshold_mode"] != "fixed":
        raise ValueError("Unknown threshold mode")
    scale = method["scale"]
    if scale != 1:
        interpolation = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation[method["interpolation"]])
    return image > threshold


def cached_masks(images, method, cache, measurements=None):
    """Use a private, content-addressed cache with code and settings identity."""
    key = hashlib.sha256(images.tobytes() + str(images.shape).encode() + str(images.dtype).encode())
    key.update(settings_digest(method).encode())
    path = Path(cache) / f"{key.hexdigest()}.npz"
    if path.exists():
        clock = time.perf_counter()
        with np.load(path, allow_pickle=False) as saved:
            masks = saved["masks"]
            if str(saved["identity"]) != key.hexdigest():
                raise ValueError("Cache identity mismatch")
        shape = (len(images), images.shape[1] * method["scale"], images.shape[2] * method["scale"])
        if masks.shape != shape or masks.dtype != bool:
            raise ValueError("Malformed segmentation cache")
        if measurements is not None:
            measurements.update(cache_read_sec=time.perf_counter() - clock, segmentation_core_sec=0.0)
        return masks, True
    clock = time.perf_counter()
    masks = np.stack([segment(image, method) for image in images])
    core_time = time.perf_counter() - clock
    path.parent.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    np.savez_compressed(path, masks=masks, identity=key.hexdigest())
    if measurements is not None:
        measurements.update(segmentation_core_sec=core_time, cache_write_sec=time.perf_counter() - clock)
    return masks, False


def components(binary, scale):
    """Describe connected foreground without losing holes or resize offsets."""
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    external, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rank = {int(labels[c[0, 0, 1], c[0, 0, 0]]): i for i, c in enumerate(external)}
    result = []
    for label in range(1, count):
        mask = labels == label
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        # Fractional occupancy compares masks on the original raster without
        # hiding subpixel changes introduced by interpolation before threshold.
        coverage = cv2.resize(mask.astype(np.float32), None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        x, y, width, height, area = stats[label]
        result.append(
            Component(
                coverage=coverage,
                center=(centers[label] + 0.5) / scale - 0.5,
                area=area / scale**2,
                contour_area=float(cv2.contourArea(contour)) / scale**2,
                aspect=width / height,
                border=bool(x == 0 or y == 0 or x + width == binary.shape[1] or y + height == binary.shape[0]),
                eligible=len(contour) >= 5,
                external_rank=rank.get(label, -1),
            )
        )
    return result


class ConstrainedSelector:
    """Conservative identity tracking; losing identity requires another seed."""

    def __init__(self, settings):
        self.settings = settings
        self.previous = None
        self.initial_area = None
        self.last_frame = None
        self.last_time = float("nan")
        self.segment = None

    def step(self, candidates, frame, timestamp, segment_id, seed=None):
        boundary = self.last_frame is not None and (
            frame != self.last_frame + 1
            or not np.isfinite(timestamp)
            or not np.isfinite(self.last_time)
            or timestamp <= self.last_time
            or timestamp - self.last_time > self.settings["max_dt_sec"]
            or segment_id != self.segment
        )
        if boundary:
            self.previous = None
        self.last_frame, self.last_time, self.segment = frame, timestamp, segment_id
        if not np.isfinite(timestamp):
            self.previous = None
            return None, "invalid_timestamp"
        if seed is None and self.previous is None:
            return None, "reidentification_required"
        reference = np.asarray(seed if seed is not None else self.previous, dtype=float)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ValueError("A seed must be a finite native-pixel point")
        plausible = []
        for candidate in candidates:
            if candidate.area < self.settings["min_area_px"] or candidate.border:
                continue
            if not self.settings["min_aspect"] <= candidate.aspect <= self.settings["max_aspect"]:
                continue
            if seed is not None:
                x, y = np.rint(reference).astype(int)
                if not (0 <= y < candidate.coverage.shape[0] and 0 <= x < candidate.coverage.shape[1]):
                    continue
                if candidate.coverage[y, x] <= 0:
                    continue
            else:
                ratio = candidate.area / self.initial_area
                if not self.settings["min_area_ratio"] <= ratio <= self.settings["max_area_ratio"]:
                    continue
                if np.linalg.norm(candidate.center - reference) > self.settings["max_displacement_px"]:
                    continue
            plausible.append(candidate)
        if len(plausible) != 1:
            self.previous = None
            return None, "ambiguous_or_unavailable"
        chosen = plausible[0]
        self.previous = chosen.center.copy()
        if seed is not None:
            self.initial_area = chosen.area
        return chosen, "accepted"


def select_sequence(masks, method, settings, frames, times, segments, seeds):
    """Hold segmentation fixed while comparing largest-contour and tracking."""
    n = len(masks)
    if not all(len(axis) == n for axis in (frames, times, segments)):
        raise ValueError("Source axes and masks must have the same number of rows")
    tracker = ConstrainedSelector(settings)
    outputs: dict[str, dict[str, Any]] = {
        name: dict(points=np.full((n, 2), np.nan), masks=[], reasons=[], processing_sec=0.0)
        for name in ("largest_external_contour", "constrained")
    }
    descriptor_time = 0.0
    descriptions = []
    for index, mask in enumerate(masks):
        started = time.perf_counter()
        objects = components(mask, method["scale"])
        descriptions.append(objects)
        descriptor_time += time.perf_counter() - started
        for name, output in outputs.items():
            started = time.perf_counter()
            if name == "constrained":
                chosen, reason = tracker.step(
                    objects, int(frames[index]), float(times[index]), int(segments[index]), seeds.get(index)
                )
            else:
                chosen = max(
                    (item for item in objects if item.external_rank >= 0),
                    key=lambda item: (item.contour_area, -item.external_rank),
                    default=None,
                )
                if chosen is not None and not chosen.eligible:
                    chosen = None
                reason = "accepted" if chosen is not None else "no_eligible_contour"
            output["processing_sec"] += time.perf_counter() - started
            if chosen is not None:
                output["points"][index] = chosen.center
            output["masks"].append(
                chosen.coverage
                if chosen is not None
                else np.zeros((masks.shape[1] // method["scale"], masks.shape[2] // method["scale"]), dtype=np.float32)
            )
            output["reasons"].append(reason)
    for output in outputs.values():
        output["masks"] = np.stack(output["masks"])
        output["processing_sec"] += descriptor_time
    return outputs, descriptions


def separation_metrics(descriptions, truth_masks):
    """Measure separation on all components independently of target selection."""
    merged, split, lost = 0, 0, 0
    for objects, truth in zip(descriptions, truth_masks):
        overlaps = np.array([[(obj.coverage * mask).sum() for mask in truth] for obj in objects]).reshape(-1, 2)
        merged += bool(np.any(np.all(overlaps > 0, axis=1)))
        split += int(np.sum(overlaps[:, 0] > 0)) > 1
        lost += bool(truth[0].any() and not np.any(overlaps[:, 0] > 0))
    return dict(merge_frames=int(merged), oversplit_frames=int(split), target_loss_frames=int(lost))


def score_sequence(output, truth_masks, truth_points, observable, frames):
    """Use truth for scoring only, including mixed/wrong foreground regions."""
    predicted_ids, ious, merge, missing, oversplit = [], [], [], [], []
    for selected, objects in zip(output["masks"], truth_masks):
        overlap = np.array([(selected * mask).sum() for mask in objects])
        predicted_ids.append("target" if overlap[0] > 0 and overlap[0] >= overlap[1] else "distractor_or_background")
        union = np.maximum(selected, objects[0]).sum()
        ious.append(float(overlap[0] / union) if union else None)
        merge.append(bool(np.all(overlap > 0)))
        missing.append(bool(objects[0].any() and overlap[0] == 0))
        count, _ = cv2.connectedComponents(((selected > 0) & objects[0]).astype(np.uint8), connectivity=8)
        oversplit.append(count > 2)
    radii = np.sqrt(truth_masks[:, 0].sum(axis=(1, 2)) / np.pi)
    metrics = position_metrics(output["points"], truth_points, observable, radii, predicted_ids, frames, frames)
    metrics.update(
        mask_iou_mean=float(np.mean([v for v in ious if v is not None])),
        merge_frames=int(np.sum(merge)),
        target_loss_frames=int(np.sum(missing)),
        oversplit_frames=int(np.sum(oversplit)),
    )
    return metrics, dict(predicted_ids=predicted_ids, mask_iou=ious, merge=merge)


def acceptance_failures(metrics, criteria):
    """Apply the agreed gates to each condition, never a pooled success rate."""
    failures = []
    if metrics["identifiable_frames"]:
        if metrics["identifiable_accepted_fraction"] < criteria["minimum_accepted_fraction_per_identifiable_condition"]:
            failures.append("accepted_fraction")
        for name, limit in (
            ("position_native_px", "identifiable_synthetic_position_p95_native_px"),
            ("position_over_radius", "identifiable_synthetic_position_p95_over_radius"),
        ):
            if metrics[name]["p95"] is None or metrics[name]["p95"] > criteria[limit]:
                failures.append(name)
    if metrics["wrong_target_count"] > criteria["maximum_wrong_target_count"]:
        failures.append("wrong_target")
    if metrics["unidentifiable_finite_count"] > criteria["unidentifiable_finite_output_count"]:
        failures.append("unidentifiable_finite_output")
    return failures
