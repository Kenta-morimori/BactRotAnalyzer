"""Read-only image diagnostics, separate from extraction and canonical caches."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

CAUSES = (
    "diagnostic_merge_or_selection",
    "existing_mask_merge_loss_or_deformation",
    "wrong_object",
    "centroid_implementation",
    "frame_or_coordinate_mapping",
    "undetermined",
)


def private_output_root(repository, requested):
    """Keep diagnostic artifacts inside the ignored output area, including symlinks."""
    repository = Path(repository).resolve()
    output = Path(requested).resolve()
    private = repository / "outputs"
    public = (private / "test_data").resolve()
    if private not in output.parents or output == public or public in output.parents:
        raise ValueError("Use a new private subdirectory under outputs/, outside outputs/test_data")
    return output


def fingerprint(path):
    """Identify input contents without copying measured data to public reports."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def selected_component(binary):
    """Reconstruct the largest external contour's 8-connected foreground mask.

    The ellipse eligibility gate is deliberately separate: a selected mask can
    exist even when the normal extraction returns a missing centroid.
    """
    binary = np.asarray(binary, dtype=bool).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = np.zeros(binary.shape, dtype=bool)
    if not contours:
        return selected, False
    contour = max(contours, key=cv2.contourArea)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    x, y = contour[0, 0]
    return labels == labels[y, x], len(contour) >= 5


def pixel_mean(mask):
    """Equal-weight area centroid in image pixels, independent of moments."""
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()]) if len(x) else np.full(2, np.nan)


def audit_frame(frame, extractor):
    """Compare normal extraction to an independent selected-pixel average."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = gray > 120
    selected, eligible = selected_component(binary)
    expected = pixel_mean(selected) if eligible else np.full(2, np.nan)
    x, y, _ = extractor(frame.copy())
    actual = np.array([x, y])
    matches = np.allclose(actual, expected, atol=1e-10, rtol=0, equal_nan=True)
    return binary, selected, actual, bool(matches)


def original_coordinates(points, transform):
    """Map AVI pixels to TIFF pixels using an explicitly verified affine map.

    The 2x3 matrix includes crop offsets and pixel-center resize conventions;
    knowing only the nominal tenfold scale is insufficient.
    """
    points = np.asarray(points, dtype=float)
    if not transform or transform.get("status") != "verified" or not transform.get("evidence"):
        return np.full(points.shape, np.nan)
    matrix = np.asarray(transform["avi_to_tiff_affine"], dtype=float)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 2x3 AVI-to-TIFF affine matrix")
    return points @ matrix[:, :2].T + matrix[:, 2]


def timeline(paths, nominal_dt):
    """Keep TIFF suffixes and timestamps explicit; never infer AVI row identity."""
    if nominal_dt <= 0:
        raise ValueError("nominal_dt must be positive")
    rows = []
    segment = 0
    previous: Optional[tuple[int, Optional[datetime]]] = None
    base = None

    def suffix_number(path):
        match = re.search(r"_(\d+)\.tif$", path.name)
        if match is None:
            raise ValueError(f"TIFF has no explicit source suffix: {path.name}")
        return int(match.group(1))

    for path in sorted(paths, key=suffix_number):
        suffix = suffix_number(path)
        if previous is not None and suffix == previous[0]:
            raise ValueError("Duplicate TIFF suffix: source identity is ambiguous")
        with Image.open(path) as image:
            stamp = image.tag_v2.get(306)
        try:
            timestamp = datetime.strptime(stamp, "%m/%d/%Y %H:%M:%S.%f")
        except (TypeError, ValueError):
            timestamp = None
        if base is None and timestamp is not None:
            base = timestamp
        delta = (
            (timestamp - previous[1]).total_seconds()
            if timestamp is not None and previous is not None and previous[1] is not None
            else None
        )
        boundary = previous is not None and (
            suffix != previous[0] + 1 or delta is None or delta <= 0 or delta > 1.5 * nominal_dt
        )
        segment += int(boundary)
        rows.append(
            dict(
                tiff_path=str(path.resolve()),
                tiff_suffix_0based=suffix,
                source_timestamp=stamp,
                time_sec=(timestamp - base).total_seconds() if timestamp and base else None,
                segment_id=segment,
                gap_before=boundary,
                avi_source_frame_1based=None,
                frame_mapping_status="unverified",
            )
        )
        previous = (suffix, timestamp)
    return rows


def representative_indices(rows):
    """Choose temporal coverage plus both sides of every recording boundary."""
    if not rows:
        return []
    indices = {0, len(rows) // 2, len(rows) - 1}
    for index, row in enumerate(rows):
        if row["gap_before"]:
            indices.update((index - 1, index))
    return sorted(indices)


def diagnostic_mask(image, threshold):
    """Threshold in native intensity units; no implicit uint16-to-uint8 cast."""
    if image.ndim != 2:
        raise ValueError("Expected a single grayscale TIFF plane")
    return image > threshold


def save_panel(path, image, binary, selected, avi=None, centroid=None):
    """Save local evidence with original-image and AVI axes kept separate."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 4))
    FigureCanvasAgg(figure)
    for index, (title, data) in enumerate(
        [("Original TIFF", image), ("Diagnostic: all / selected", binary), ("Saved AVI (unmapped if unverified)", avi)]
    ):
        axis = figure.add_subplot(1, 3, index + 1)
        axis.set_title(title)
        if data is not None:
            axis.imshow(data, cmap="gray")
        if index == 1 and selected.any():
            axis.contour(selected, levels=[0.5], colors=["orange"])
        if index == 2 and centroid is not None and np.isfinite(centroid).all():
            axis.plot(*centroid, "+", color="red")
    figure.tight_layout()
    figure.savefig(path, dpi=120)


def write_json(path, value):
    """Write strict JSON, so missing values cannot silently become NaN tokens."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config):
    """Reject accidental holdout reuse or a change of the preregistered split."""
    expected = {"development": list(range(20)), "validation": list(range(1000, 1020))}
    if config["synthetic_seeds"] != expected:
        raise ValueError("Changing the seed split requires a new evaluation protocol")
    if config["sample_roles"] != {"1": "development", "2": "validation"}:
        raise ValueError("Measured sample roles must remain fixed")
    if config["stage"] not in ("A", "B") or (config["stage"] == "B" and config["acceptance"]["status"] != "agreed"):
        raise ValueError("Stage B requires separately agreed numerical acceptance criteria")
    if config["coverage_stride"] < 1 or config["nominal_dt_sec"] <= 0:
        raise ValueError("Coverage stride and nominal time step must be positive")
    if not np.isfinite(config["diagnostic_threshold_native"]):
        raise ValueError("Diagnostic threshold must be finite")


def position_metrics(predicted, truth, identifiable, radii, predicted_ids, source_frames, expected_frames):
    """Score all target frames as well as conditionally accepted positions.

    Callers must supply the isolated-mask centroid as truth when evaluating an
    area-centroid estimator. Generated geometric centers have a different meaning.
    This helper does not run methods or decide numerical acceptance thresholds.
    """
    if not np.array_equal(source_frames, expected_frames):
        raise ValueError("Evaluation requires exact source-frame correspondence")
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    identifiable, radii = np.asarray(identifiable, bool), np.asarray(radii, float)
    n = len(expected_frames)
    if predicted.shape != (n, 2) or truth.shape != (n, 2):
        raise ValueError("Positions must retain every source-frame row")
    if identifiable.shape != (n,) or radii.shape != (n,) or np.shape(predicted_ids) != (n,):
        raise ValueError("Identity, radius and observability must share the frame axis")
    accepted = np.isfinite(predicted).all(axis=1)
    scored = accepted & identifiable & np.isfinite(truth).all(axis=1)
    error = np.linalg.norm(predicted[scored] - truth[scored], axis=1)
    valid_radius = np.isfinite(radii[scored]) & (radii[scored] > 0)
    normalized = error[valid_radius] / radii[scored][valid_radius]

    def statistics(values):
        if not len(values):
            return dict(median=None, p95=None, maximum=None)
        return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), maximum=float(values.max()))

    return dict(
        all_target_frames=n,
        identifiable_frames=int(identifiable.sum()),
        accepted_frames=int(accepted.sum()),
        accepted_fraction=float(accepted.mean()) if n else None,
        identifiable_accepted_fraction=float(accepted[identifiable].mean()) if identifiable.any() else None,
        wrong_target_count=int(np.sum(accepted & (np.asarray(predicted_ids) != "target"))),
        unidentifiable_finite_count=int(np.sum(accepted & ~identifiable)),
        scored_frames=int(scored.sum()),
        position_native_px=statistics(error),
        position_over_radius=statistics(normalized),
        radius_scored_frames=int(valid_radius.sum()),
    )


def regression(avi, extractor, expected_frames, fps, calibration, cached=None):
    """Audit every decoded frame without writing to normal extraction caches."""
    capture = cv2.VideoCapture(str(avi))
    if not capture.isOpened():
        raise OSError(f"Cannot open {avi}")
    records: list[list[float]] = []
    mismatches = 0
    cache_mismatches = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            _, _, xy, matches = audit_frame(image, extractor)
            index = len(records)
            mismatches += not matches
            if cached is not None:
                cache_mismatches += index >= len(cached) or not np.allclose(
                    xy, cached[index], atol=1e-10, rtol=0, equal_nan=True
                )
            records.append([index + 1, index / fps, *xy, *(xy * calibration)])
    finally:
        capture.release()
    if cached is not None and len(cached) != len(records):
        cache_mismatches += abs(len(cached) - len(records))
    return np.asarray(records), dict(
        frame_count=len(records),
        expected_frames=expected_frames,
        frame_count_matches=len(records) == expected_frames,
        pixel_mean_mismatches=int(mismatches),
        cached_coordinate_mismatches=int(cache_mismatches) if cached is not None else None,
        cached_comparison_status="checked" if cached is not None else "unavailable",
    )
