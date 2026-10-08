"""Compare center estimators on identical No.14 windows and known-center trials."""

from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from issue38_paths import FIGURES, TABLES, ensure_output_dirs

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.signal import csd, welch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from issue38_center_diagnostics import Ellipse, fit_ellipse, angular_coverage  # noqa: E402
from issue38_no14_center_evaluation import robust_radius  # noqa: E402

METHODS = ("legacy", "centered_algebraic", "ellipse_constrained", "geometric_ellipse", "robust_circle")


def distance(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """Signed Euclidean distance; seed on a polyline then refine the closest angle."""
    local = (points - ellipse.center) @ ellipse.basis
    a, b = ellipse.axes
    theta_grid = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    curve = np.column_stack((a * np.cos(theta_grid), b * np.sin(theta_grid)))
    indices = cKDTree(curve).query(local)[1]
    theta = theta_grid[indices]
    u, v = local.T
    for _ in range(8):
        s, c = np.sin(theta), np.cos(theta)
        grad = (b * b - a * a) * s * c + a * u * s - b * v * c
        hessian = (b * b - a * a) * (c * c - s * s) + a * u * c + b * v * s
        step = np.divide(grad, hessian, out=np.zeros_like(grad), where=np.abs(hessian) > 1e-12)
        theta -= np.clip(step, -0.1, 0.1)
    closest = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    sign = np.where((u / a)**2 + (v / b)**2 >= 1, 1.0, -1.0)
    return np.linalg.norm(local - closest, axis=1) * sign


def estimate(points: np.ndarray, method: str) -> tuple[Ellipse | None, bool]:
    x, y = points.T
    radius = robust_radius(x, y)
    if radius <= 1e-9 or len(points) < 30:
        return None, False
    middle = np.median(points, axis=0)
    scaled = (points - middle) / radius
    if method == "legacy":
        return fit_ellipse(x, y), False
    if method == "centered_algebraic":
        fit = fit_ellipse(scaled[:, 0], scaled[:, 1])
        if fit is None:
            return None, False
        return Ellipse(middle + radius * fit.center, radius * fit.axes, fit.basis, fit.condition), False
    if method == "robust_circle":
        initial_radius = np.median(np.linalg.norm(points - middle, axis=1))
        result = least_squares(
            lambda z: np.linalg.norm(points - z[:2], axis=1) - z[2],
            np.r_[middle, initial_radius],
            bounds=(np.r_[middle - radius, .1 * radius], np.r_[middle + radius, 2 * radius]),
            loss="soft_l1", f_scale=.2 * radius, max_nfev=150,
        )
        center, r = result.x[:2], result.x[2]
        bounded = (np.max(np.abs(center - middle)) >= 0.999 * radius
                   or r >= 1.999 * radius or r <= 0.1001 * radius or not result.success)
        return Ellipse(center, np.array([r, r]), np.eye(2), np.nan), bounded
    raw = cv2.fitEllipseDirect(scaled.astype(np.float32).reshape(-1, 1, 2))
    center, diameters, angle = raw
    axes = np.asarray(diameters) / 2 * radius
    theta = np.deg2rad(angle)
    basis = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    direct = Ellipse(middle + radius * np.asarray(center), axes, basis, np.nan)
    if method == "ellipse_constrained":
        return direct, False
    if method != "geometric_ellipse":
        raise ValueError(method)
    # Bounds prevent a near-line conic from escaping. A bound hit invalidates a candidate.
    lower = np.r_[middle - radius, np.log([0.1 * radius, 0.1 * radius]), -4 * np.pi]
    upper = np.r_[middle + radius, np.log([3 * radius, 3 * radius]), 4 * np.pi]

    def geometry(parameters: np.ndarray) -> Ellipse:
        c, s = np.cos(parameters[4]), np.sin(parameters[4])
        return Ellipse(parameters[:2], np.exp(parameters[2:4]), np.array([[c, -s], [s, c]]), np.nan)

    initial = np.r_[direct.center, np.log(direct.axes), theta]
    initial = np.clip(initial, lower + 1e-6, upper - 1e-6)
    result = least_squares(lambda z: distance(points, geometry(z)), initial,
                           bounds=(lower, upper), loss="soft_l1", f_scale=0.2 * radius,
                           max_nfev=80, xtol=1e-7, ftol=1e-7, gtol=1e-7)
    bounded = bool(np.any(result.active_mask) or not result.success)
    return geometry(result.x), bounded


def evaluate(points: np.ndarray, method: str) -> dict:
    radius = robust_radius(*points.T)
    fit, bounded = estimate(points, method)
    row = {"method": method, "valid": fit is not None, "bound_or_nonconvergence": bounded,
           "radius_um": radius}
    freq, xx = welch(points[:, 0], fs=200, nperseg=min(256, len(points)))
    _, yy = welch(points[:, 1], fs=200, nperseg=min(256, len(points)))
    _, cross = csd(points[:, 0], points[:, 1], fs=200, nperseg=min(256, len(points)))
    band = (freq >= 2) & (freq <= 80)
    peak = np.flatnonzero(band)[np.argmax((xx + yy)[band])]
    row["rotation_quadrature_score"] = abs(cross[peak].imag) / np.sqrt(xx[peak] * yy[peak])
    row["peak_energy_fraction"] = np.sum((xx + yy)[max(1, peak - 1):peak + 2]) / np.sum((xx + yy)[band])
    if fit is None:
        return row
    row.update(center_x_um=fit.center[0], center_y_um=fit.center[1],
               fit_residual_p95_R=np.quantile(np.abs(distance(points, fit)), .95) / radius,
               axis_ratio=max(fit.axes) / min(fit.axes), major_axis_R=max(fit.axes) / radius,
               coverage_deg=angular_coverage(*points.T, fit.center)[0])
    centers, heldout, bounds = [], [], []
    for omit in np.array_split(np.arange(len(points)), 4):
        keep = np.ones(len(points), bool)
        keep[omit] = False
        part, hit = estimate(points[keep], method)
        if part is not None:
            centers.append(part.center)
            heldout.extend(np.abs(distance(points[omit], part)).tolist())
            bounds.append(hit)
    row["block_cv_valid"] = len(centers)
    row["block_cv_bound_count"] = sum(bounds)
    row["block_cv_p95_R"] = np.quantile(heldout, .95) / radius if heldout else np.nan
    row["center_max_shift_R"] = np.max(np.linalg.norm(np.asarray(centers) - fit.center, axis=1)) / radius if centers else np.nan
    # Origin perturbation is a diagnostic of the algebraic objective, not ground truth.
    shifted, _ = estimate(points + np.array([2.0, -1.0]), method)
    row["origin_shift_error_R"] = np.linalg.norm(shifted.center - np.array([2.0, -1.0]) - fit.center) / radius if shifted else np.nan
    return row


def real_windows() -> pd.DataFrame:
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    windows = pd.read_csv(TABLES / "no14_center_window_metrics.csv")
    t = data.time_sec.to_numpy(float)
    xy = data[["x_um", "y_um"]].to_numpy(float)
    records = []
    for i, window in enumerate(windows.itertuples()):
        points = xy[(t >= window.start_time_sec) & (t < window.end_time_sec)]
        for method in METHODS:
            row = evaluate(points, method)
            row.update(start_time_sec=window.start_time_sec, end_time_sec=window.end_time_sec)
            records.append(row)
        if i % 20 == 0:
            print(f"real windows {i + 1}/{len(windows)}", flush=True)
    return pd.DataFrame(records)


def synthetic_trials(methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    rng = np.random.default_rng(38)
    records = []
    for scenario in ("circle", "ellipse", "thick_circle", "thick_ellipse", "short_arc", "stopped_cloud", "center_drift"):
        for trial in range(12):
            phase = np.linspace(0, 30 * 2 * np.pi, 360)
            if scenario == "short_arc":
                phase = np.linspace(-0.3, 0.3, 360)
            axes = np.array([.3, .3 if scenario in ("circle", "thick_circle") else .18])
            center = np.array([1.3, 1.45])
            points = center + np.column_stack((np.cos(phase), np.sin(phase))) * axes
            if scenario == "stopped_cloud":
                points = np.tile(center + [.3, 0], (360, 1))
            if scenario == "center_drift":
                points += np.column_stack((np.linspace(-.06, .06, 360), np.zeros(360)))
            noise = .08 if scenario.startswith("thick_") else (.008 if scenario == "stopped_cloud" else .02)
            points += rng.normal(0, noise, points.shape)
            for method in methods:
                row = evaluate(points, method)
                row.update(scenario=scenario, trial=trial)
                if row["valid"]:
                    row["true_center_error_R"] = np.linalg.norm([row["center_x_um"] - center[0],
                                                                row["center_y_um"] - center[1]]) / .3
                records.append(row)
    return pd.DataFrame(records)


def plot_results(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    for method in METHODS:
        d = data.loc[data.method == method]
        for axis, field in zip(axes, ("center_x_um", "center_y_um", "center_max_shift_R")):
            axis.plot(d.start_time_sec, d[field], lw=.85, label=method)
    axes[0].set_ylabel("Center x (µm)")
    axes[1].set_ylabel("Center y (µm)")
    axes[2].set_ylabel("Block exclusion shift / R")
    axes[0].set_ylim(.9, 1.8)
    axes[1].set_ylim(.8, 1.8)
    axes[2].set_ylim(0, 1)
    axes[0].legend(ncol=3, frameon=False)
    axes[2].set_xlabel("Window start time (s)")
    for axis in axes:
        axis.axvline(40, color="gray", ls=":")
        axis.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_center_method_comparison.png", dpi=150)
    plt.close(fig)


def interpolation_trials(real: pd.DataFrame) -> pd.DataFrame:
    """Hide stable-region center estimates; compare two-sided linear fill to them.

    This validates reproduction of a reference estimator, not true-center accuracy.
    """
    reference = real.loc[(real.method == "robust_circle") & (real.start_time_sec >= 65)].reset_index(drop=True)
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    t = data.time_sec.to_numpy(float)
    xy = data[["x_um", "y_um"]].to_numpy(float)
    records = []
    for count in (1, 2, 3, 4, 5, 8):
        for start in range(1, len(reference) - count):
            left, right = reference.iloc[start - 1], reference.iloc[start + count]
            errors, phase_errors = [], []
            for i in range(start, start + count):
                row = reference.iloc[i]
                fraction = (row.start_time_sec - left.start_time_sec) / (right.start_time_sec - left.start_time_sec)
                center = np.array([left.center_x_um, left.center_y_um]) * (1 - fraction) + np.array(
                    [right.center_x_um, right.center_y_um]) * fraction
                actual = np.array([row.center_x_um, row.center_y_um])
                errors.append(np.linalg.norm(center - actual) / row.radius_um)
                points = xy[(t >= row.start_time_sec) & (t < row.end_time_sec)]
                radius = np.linalg.norm(points - actual, axis=1)
                # No phase is defined at the center. Omit only machine-zero distances.
                nonzero = radius > 1e-12
                a = np.arctan2(points[nonzero, 1] - actual[1], points[nonzero, 0] - actual[0])
                b = np.arctan2(points[nonzero, 1] - center[1], points[nonzero, 0] - center[0])
                phase_errors.extend(np.abs(np.angle(np.exp(1j * (b - a)))) * 180 / np.pi)
            records.append({"omitted_center_samples": count,
                            "omitted_duration_sec": count * float(np.median(np.diff(reference.start_time_sec))),
                            "gap_start_sec": reference.start_time_sec.iloc[start],
                            "bracket_max_block_shift_R": max(left.center_max_shift_R, right.center_max_shift_R),
                            "bracket_max_cv_residual_R": max(left.block_cv_p95_R, right.block_cv_p95_R),
                            "bracket_min_coverage_deg": min(left.coverage_deg, right.coverage_deg),
                            "bracket_min_cv_valid": min(left.block_cv_valid, right.block_cv_valid),
                            "bracket_bound_count": int(left.bound_or_nonconvergence) + int(right.bound_or_nonconvergence)
                            + int(left.block_cv_bound_count) + int(right.block_cv_bound_count),
                            "max_center_error_R": max(errors),
                            "phase_error_p95_deg": np.quantile(phase_errors, .95),
                            "phase_error_max_deg": max(phase_errors)})
    return pd.DataFrame(records)


def main() -> None:
    ensure_output_dirs()
    real = real_windows()
    real.to_csv(TABLES / "no14_center_method_comparison.csv", index=False)
    interpolation_trials(real).to_csv(TABLES / "no14_center_interpolation_trials.csv", index=False)
    plot_results(real)
    synthetic = synthetic_trials()
    synthetic.to_csv(TABLES / "no14_center_method_synthetic.csv", index=False)
    print(real.groupby("method")[["block_cv_p95_R", "center_max_shift_R", "origin_shift_error_R"]].median().to_string())


if __name__ == "__main__":
    main()
