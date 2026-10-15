"""Reusable trajectory diagnostics, independent of sample, paths and plotting.

These measures describe computation and reproducibility, not physical validity.
The sampled distance is retained only for compatibility with early diagnostic
reports; candidate comparisons use the refined signed nearest-point distance.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import csd, welch
from scipy.spatial import cKDTree

from utils.functions.rotation_center_candidates import (
    CenterEstimate,
    Ellipse,
    ellipse_distance,
    estimate_center,
    robust_radius,
)


def fit_legacy_ellipse(points: np.ndarray) -> Ellipse | None:
    """Fit the same quadratic form as the current algorithm, then resolve its geometry."""
    x, y = points.T
    if len(x) < 30 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    result = estimate_center(points, "legacy")
    if result.ellipse is None:
        return None
    fit = result.ellipse
    # Normalization prevents microscope-origin offsets from dominating the condition number.
    scale = max(float(np.median(np.hypot(x - np.median(x), y - np.median(y)))), 1e-6)
    u, v = (x - np.median(x)) / scale, (y - np.median(y)) / scale
    normalized_design = np.column_stack((u * u, u * v, v * v, u, v))
    condition = float(np.linalg.cond(normalized_design))
    return Ellipse(fit.center, fit.axes, fit.basis, condition)


def ellipse_points(ellipse: Ellipse, count: int = 1440, *, endpoint: bool = False) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, count, endpoint=endpoint)
    return ellipse.center + np.column_stack((np.cos(phase), np.sin(phase))) @ np.diag(ellipse.axes) @ ellipse.basis.T


def sampled_ellipse_distance(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    # A dense polyline approximates the shortest Euclidean distance to the ellipse.
    return cKDTree(ellipse_points(ellipse)).query(points)[0]


def angular_coverage(points: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    """Union of observed polar directions, regardless of revisiting the same arc."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return np.nan, np.nan
    phase = np.sort(np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * np.pi))
    biggest = float(np.degrees(np.diff(np.r_[phase, phase[0] + 2 * np.pi]).max()))
    return 360 - biggest, biggest


def observed_extent(points: np.ndarray) -> dict:
    """Center-free cloud extent in µm; not a physical rotation radius."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    quantiles = np.quantile(points, [0.05, 0.95], axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    return {
        "x_half_span_um": (quantiles[1, 0] - quantiles[0, 0]) / 2,
        "y_half_span_um": (quantiles[1, 1] - quantiles[0, 1]) / 2,
        "median_x_um": np.median(points[:, 0]),
        "median_y_um": np.median(points[:, 1]),
        "cloud_pca_ratio": np.sqrt(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else np.nan,
        "cloud_R_um": robust_radius(points),
    }


def block_cross_validation(
    points: np.ndarray, method: str, *, n_blocks: int = 4, fit: CenterEstimate | None = None
) -> dict:
    """Leave out contiguous time blocks without compacting missing frame rows.

    Every returned ellipse contributes residuals, including flagged fits. Counts
    distinguish geometry returned from successful computation. Caller selects
    physical acceptance criteria separately; no thresholds are imposed here.
    """
    points = np.asarray(points, dtype=float)
    if n_blocks < 2 or len(points) < n_blocks:
        raise ValueError("Need at least two nonempty time blocks")
    finite = np.isfinite(points).all(axis=1)
    radius = robust_radius(points[finite]) if finite.any() else np.nan
    result = estimate_center(points, method) if fit is None else fit
    centers, heldout, bounds, ok_counts = [], [], [], []
    for omit in np.array_split(np.arange(len(points)), n_blocks):
        keep = np.ones(len(points), bool)
        keep[omit] = False
        part = estimate_center(points[keep], method)
        ok_counts.append(part.computationally_ok)
        if part.ellipse is not None:
            centers.append(part.ellipse.center)
            heldout.extend(np.abs(ellipse_distance(points[omit][finite[omit]], part.ellipse)).tolist())
            bounds.append(part.boundary_reached or part.converged is False)
    return {
        "block_cv_valid": len(centers),
        "block_cv_computationally_ok": sum(ok_counts),
        "block_cv_bound_count": sum(bounds),
        "block_cv_p95_R": np.quantile(heldout, 0.95) / radius if heldout else np.nan,
        "center_max_shift_R": (
            np.max(np.linalg.norm(np.asarray(centers) - result.center, axis=1)) / radius
            if centers and result.center is not None
            else np.nan
        ),
    }


def evaluate_center(
    points: np.ndarray, method: str, *, sample_rate_hz: float, frequency_band_hz: tuple[float, float] = (2.0, 80.0)
) -> dict:
    """Compare one window in µm with an explicit sampling rate and frequency band.

    Missing rows are kept for time-block validation. Spectral scores are left
    unavailable when rows are missing rather than compacting the time axis.
    """
    points = np.asarray(points, dtype=float)
    result = estimate_center(points, method)
    finite = np.isfinite(points).all(axis=1)
    radius = robust_radius(points[finite]) if finite.any() else np.nan
    fit = result.ellipse
    bounded = result.boundary_reached or result.converged is False
    row = {
        "method": method,
        "ellipse_returned": fit is not None,
        "bound_or_nonconvergence": bounded,
        "radius_um": radius,
        "status": result.status,
        "computationally_ok": result.computationally_ok,
        "converged": result.converged,
        "boundary_reached": result.boundary_reached,
        "n_input": result.n_input,
        "n_used": result.n_used,
        "scientific_assessment": "not_assessed",
    }
    if len(points) >= 4 and finite.all():
        freq, xx = welch(points[:, 0], fs=sample_rate_hz, nperseg=min(256, len(points)))
        _, yy = welch(points[:, 1], fs=sample_rate_hz, nperseg=min(256, len(points)))
        _, cross = csd(points[:, 0], points[:, 1], fs=sample_rate_hz, nperseg=min(256, len(points)))
        band = (freq >= frequency_band_hz[0]) & (freq <= frequency_band_hz[1])
        if not band.any():
            raise ValueError("No spectral bins in the requested frequency band")
        peak = np.flatnonzero(band)[np.argmax((xx + yy)[band])]
        row["rotation_quadrature_score"] = abs(cross[peak].imag) / np.sqrt(xx[peak] * yy[peak])
        row["peak_energy_fraction"] = np.sum((xx + yy)[max(1, peak - 1) : peak + 2]) / np.sum((xx + yy)[band])
    else:
        row["rotation_quadrature_score"] = row["peak_energy_fraction"] = np.nan
    if fit is None:
        return row
    row.update(
        center_x_um=fit.center[0],
        center_y_um=fit.center[1],
        fit_residual_p95_R=np.quantile(np.abs(ellipse_distance(points[finite], fit)), 0.95) / radius,
        axis_ratio=max(fit.axes) / min(fit.axes),
        major_axis_R=max(fit.axes) / radius,
        coverage_deg=angular_coverage(points, fit.center)[0],
    )
    row.update(block_cross_validation(points, method, fit=result))
    # Origin perturbation is a diagnostic of the algebraic objective, not ground truth.
    shifted = estimate_center(points + np.array([2.0, -1.0]), method).ellipse
    row["origin_shift_error_R"] = (
        np.linalg.norm(shifted.center - np.array([2.0, -1.0]) - fit.center) / radius if shifted else np.nan
    )
    return row


def geometric_sensitivity(points: np.ndarray, *, max_nfev: int = 80, retry_nfev: int = 320) -> list[dict]:
    """Compare initial centers and 1×/2× search ranges; retry nonconvergence only.

    Returned records do not select a candidate or impose physical error limits.
    """
    records = []
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    radius = robust_radius(points[finite]) if finite.any() else np.nan
    baseline = estimate_center(points, "geometric_ellipse", max_nfev=max_nfev)
    for initial in ("ellipse", "median"):
        for scale in (1.0, 2.0):
            result = (
                baseline
                if (initial, scale) == ("ellipse", 1.0)
                else estimate_center(
                    points, "geometric_ellipse", initial_center=initial, bounds_scale=scale, max_nfev=max_nfev
                )
            )
            for budget, fit in [(max_nfev, result)] + (
                [
                    (
                        retry_nfev,
                        estimate_center(
                            points, "geometric_ellipse", initial_center=initial, bounds_scale=scale, max_nfev=retry_nfev
                        ),
                    )
                ]
                if result.converged is False
                else []
            ):
                records.append(
                    {
                        "n_input": fit.n_input,
                        "n_used": fit.n_used,
                        "initial_center": initial,
                        "bounds_scale": scale,
                        "max_nfev": budget,
                        "status": fit.status,
                        "converged": fit.converged,
                        "boundary_reached": fit.boundary_reached,
                        "center_x_um": fit.center[0] if fit.center is not None else np.nan,
                        "center_y_um": fit.center[1] if fit.center is not None else np.nan,
                        "center_difference_from_baseline_R": (
                            np.linalg.norm(fit.center - baseline.center) / radius
                            if fit.center is not None and baseline.center is not None
                            else np.nan
                        ),
                        "fit_residual_p95_R": (
                            np.quantile(np.abs(ellipse_distance(points[finite], fit.ellipse)), 0.95) / radius
                            if fit.ellipse is not None
                            else np.nan
                        ),
                        "axis_ratio": (max(fit.axes) / min(fit.axes) if fit.axes is not None else np.nan),
                    }
                )
    return records
