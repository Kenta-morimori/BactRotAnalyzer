"""Conditional centroid-orbit center assessment, revised with known truth."""

from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch

from utils.functions.rotation_center_candidates import estimate_center, robust_radius
from utils.functions.rotation_center_diagnostics import evaluate_center


def block_extent_ratio(points: np.ndarray) -> float:
    """Smallest versus largest observed cloud radius across four time blocks."""
    points = np.asarray(points, dtype=float)
    if len(points) < 4 or not np.isfinite(points).all():
        return np.nan
    extents = [robust_radius(block) for block in np.array_split(points, 4)]
    return float(min(extents) / max(extents)) if max(extents) > 0 else np.nan


def half_extent_ratio(points: np.ndarray) -> float:
    """Detect abrupt changes in the observed orbit size within a window."""
    halves = np.array_split(np.asarray(points, dtype=float), 2)
    radii = [robust_radius(part) for part in halves]
    return float(min(radii) / max(radii)) if max(radii) > 0 else np.nan


def segmented_quadrature_score(points: np.ndarray, sample_rate_hz: float) -> float:
    """Median absolute x/y quadrature in short overlapping time blocks.

    Averaging the complex cross spectrum over opposite rotation directions
    cancels its imaginary component. Taking its magnitude within each block
    before aggregating retains evidence of both directions.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 64 or not np.isfinite(points).all():
        return np.nan
    width = 64
    scores = []
    for start in range(0, len(points) - width + 1, width // 2):
        block = points[start : start + width]
        frequencies, xx = welch(block[:, 0], fs=sample_rate_hz, nperseg=width)
        _, yy = welch(block[:, 1], fs=sample_rate_hz, nperseg=width)
        _, cross = csd(block[:, 0], block[:, 1], fs=sample_rate_hz, nperseg=width)
        band = (frequencies >= 2) & (frequencies <= min(80, 0.45 * sample_rate_hz))
        if not band.any():
            continue
        peak = np.flatnonzero(band)[np.argmax((xx + yy)[band])]
        denominator = np.sqrt(xx[peak] * yy[peak])
        if denominator > 0:
            scores.append(abs(cross[peak].imag) / denominator)
    return float(np.median(scores)) if scores else np.nan


def center_gate(row: dict) -> tuple[bool, str]:
    """Historical issue #39 gate, retained for reproducible old comparisons."""
    checks = (
        ("missing", row["n_used"] == row["n_input"]),
        ("quadrature", row["rotation_quadrature_score"] >= 0.8),
        ("spectral_energy", row["peak_energy_fraction"] >= 0.15),
        ("block_extent", row["block_extent_ratio"] >= 0.25),
        ("fit_status", row["status"] == "computed"),
        ("block_fits", row.get("block_cv_computationally_ok") == 4),
        ("heldout_residual", row.get("block_cv_p95_R", np.nan) <= 0.55),
        ("center_stability", row.get("center_max_shift_R", np.nan) <= 0.10),
        ("origin_stability", row.get("origin_shift_error_R", np.nan) <= 0.01),
    )
    failures = [name for name, passed in checks if not passed]
    return not failures, ";".join(failures)


def center_gate_v2(row: dict) -> tuple[bool, str]:
    """Known-truth revision: wide coverage and reversal-aware local quadrature."""
    checks = (
        ("missing", row["n_used"] == row["n_input"]),
        ("segmented_quadrature", row.get("segmented_quadrature_score", np.nan) >= 0.8),
        ("angular_coverage", row.get("coverage_deg", np.nan) >= 270),
        ("spectral_energy", row["peak_energy_fraction"] >= 0.15),
        ("block_extent", row["block_extent_ratio"] >= 0.25),
        ("half_extent", row.get("half_extent_ratio", np.nan) >= 0.8),
        ("fit_status", row["status"] == "computed"),
        ("block_fits", row.get("block_cv_computationally_ok") == 4),
        ("heldout_residual", row.get("block_cv_p95_R", np.nan) <= 0.55),
        ("center_stability", row.get("center_max_shift_R", np.nan) <= 0.10),
        ("origin_stability", row.get("origin_shift_error_R", np.nan) <= 0.01),
    )
    failures = [name for name, passed in checks if not passed]
    return not failures, ";".join(failures)


def center_gate_v3(row: dict) -> tuple[bool, str]:
    """Reject low signal-to-noise orbits exposed by the first holdout."""
    passed, failure = center_gate_v2(row)
    if row.get("fit_residual_p95_R", np.nan) > 0.4 or not np.isfinite(row.get("fit_residual_p95_R", np.nan)):
        return False, ";".join(filter(None, (failure, "orbit_residual")))
    return passed, failure


def assess_center_window(points: np.ndarray, sample_rate_hz: float) -> dict:
    """Return a center only when the full, frame-aligned window passes the gate.

    The output retains all diagnostic fields plus ``gate_pass`` and
    ``gate_failure``. A rejected window has NaN center coordinates, so no
    downstream code can silently turn a merely computed center into a valid
    angular velocity.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be positive and finite")
    if len(points) < 30 or not np.isfinite(points).all():
        return {
            "n_input": len(points),
            "n_used": int(np.isfinite(points).all(axis=1).sum()),
            "status": "insufficient_points" if len(points) < 30 else "missing",
            "gate_pass": False,
            "gate_failure": "insufficient_points" if len(points) < 30 else "missing",
            "center_x_um": np.nan,
            "center_y_um": np.nan,
        }
    row = evaluate_center(points, "ellipse_constrained", sample_rate_hz=sample_rate_hz)
    row["candidate_center_x_um"] = row.get("center_x_um", np.nan)
    row["candidate_center_y_um"] = row.get("center_y_um", np.nan)
    row["block_extent_ratio"] = block_extent_ratio(points)
    row["half_extent_ratio"] = half_extent_ratio(points)
    row["segmented_quadrature_score"] = segmented_quadrature_score(points, sample_rate_hz)
    row["gate_pass"], row["gate_failure"] = center_gate_v3(row)
    left = estimate_center(points[: len(points) // 2], "ellipse_constrained")
    right = estimate_center(points[len(points) // 2 :], "ellipse_constrained")
    row["half_center_shift_R"] = (
        np.linalg.norm(left.center - right.center) / robust_radius(points)
        if left.computationally_ok and right.computationally_ok and left.center is not None and right.center is not None
        else np.nan
    )
    if not np.isfinite(row["half_center_shift_R"]) or row["half_center_shift_R"] > 0.12:
        row["gate_pass"] = False
        row["gate_failure"] = ";".join(filter(None, [row["gate_failure"], "half_center_shift"]))
    if not row["gate_pass"]:
        row["center_x_um"] = np.nan
        row["center_y_um"] = np.nan
    return row


def local_period_frames(points: np.ndarray, sample_rate_hz: float) -> int | None:
    """Estimate a center-independent orbital period from a local x/y block."""
    points = np.asarray(points, dtype=float)
    if len(points) < 60:
        return None
    if not np.isfinite(points).all():
        finite = np.isfinite(points).all(axis=1)
        boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        runs = [(start, end) for start, end in zip(boundaries[::2], boundaries[1::2])]
        start, end = max(runs, key=lambda pair: pair[1] - pair[0])
        points = points[start:end]
        if len(points) < 60:
            return None
    frequencies, px = welch(points[:, 0], fs=sample_rate_hz, nperseg=min(256, len(points)))
    _, py = welch(points[:, 1], fs=sample_rate_hz, nperseg=min(256, len(points)))
    band = (frequencies >= 2) & (frequencies <= min(80, 0.45 * sample_rate_hz))
    power = px[band] + py[band]
    if not power.size or power.sum() <= 0:
        return None
    peak_index = int(np.argmax(power))
    if power[max(0, peak_index - 1) : peak_index + 2].sum() / power.sum() < 0.15:
        return None
    return max(1, int(round(sample_rate_hz / frequencies[band][peak_index])))


def estimate_center_series(
    time: np.ndarray,
    points: np.ndarray,
    *,
    rotations_per_window: int = 5,
    step_frames: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Assess centered local windows and preserve NaNs where support is absent.

    Each successful center applies only to the Voronoi cell around its anchor.
    We never carry a center across a rejected window or the unsupported ends.
    The returned array keeps the original source-frame indexing.
    """
    time = np.asarray(time, dtype=float)
    points = np.asarray(points, dtype=float)
    if time.ndim != 1 or points.shape != (len(time), 2):
        raise ValueError("time and points must have shapes (n,) and (n, 2)")
    if rotations_per_window < 2:
        raise ValueError("rotations_per_window must be at least two")
    dt = np.diff(time)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(finite_dt):
        return np.full((len(time), 2), np.nan), []
    sample_rate_hz = 1 / float(np.median(finite_dt))
    step = max(1, int(round(sample_rate_hz * 0.2))) if step_frames is None else step_frames
    if step < 1:
        raise ValueError("step_frames must be positive")
    context_half = max(60, int(round(2.5 * sample_rate_hz)))
    centers = np.full((len(time), 2), np.nan)
    records: list[dict] = []
    for anchor in range(0, len(time), step):
        context = points[max(0, anchor - context_half) : min(len(time), anchor + context_half)]
        period = local_period_frames(context, sample_rate_hz)
        record: dict = {"anchor_frame_0based": anchor, "period_frames": period}
        if period is None:
            record.update(status="period_unavailable", gate_pass=False, gate_failure="period_unavailable")
            records.append(record)
            continue
        width = max(60, min(int(round(rotations_per_window * period)), int(round(5 * sample_rate_hz))))
        start = anchor - width // 2
        end = start + width
        record.update(window_start_frame_0based=start, window_end_frame_exclusive=end)
        if start < 0 or end > len(time):
            record.update(status="incomplete_window", gate_pass=False, gate_failure="incomplete_window")
            records.append(record)
            continue
        window_time = time[start:end]
        window_dt = np.diff(window_time)
        if (
            not np.isfinite(window_time).all()
            or not np.isfinite(window_dt).all()
            or np.any((window_dt <= 0) | (window_dt > 1.5 / sample_rate_hz))
        ):
            record.update(status="time_gap", gate_pass=False, gate_failure="time_gap")
            records.append(record)
            continue
        record.update(assess_center_window(points[start:end], sample_rate_hz))
        records.append(record)
    change_candidates = [
        record
        for record in records
        if record.get("half_center_shift_R", np.nan) > 0.12 or record.get("half_extent_ratio", np.nan) < 0.8
    ]
    change_regions = [
        (
            record["anchor_frame_0based"],
            (record["window_end_frame_exclusive"] - record["window_start_frame_0based"]) // 2,
        )
        for record in change_candidates
        if any(
            other is not record and abs(other["anchor_frame_0based"] - record["anchor_frame_0based"]) <= step
            for other in change_candidates
        )
    ]
    for record in records:
        if record["gate_pass"] and any(
            abs(record["anchor_frame_0based"] - change_anchor) <= half_width
            for change_anchor, half_width in change_regions
        ):
            record["gate_pass"] = False
            record["gate_failure"] = "near_center_change"
    for record in records:
        if record["gate_pass"]:
            anchor = record["anchor_frame_0based"]
            cell_start = max(0, anchor - step // 2)
            cell_end = min(len(time), anchor + (step + 1) // 2)
            centers[cell_start:cell_end] = [record["center_x_um"], record["center_y_um"]]
    return centers, records
