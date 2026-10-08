"""Evaluate No.14 moving-window centers against the canonical AVI centroid.

This is a diagnostic only. It does not alter the production center or angular
velocity. Run issue38_no14_center_timeseries.py first, then this script.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from issue38_center_diagnostics import (  # noqa: E402
    angular_coverage,
    ellipse_distance,
    ellipse_points,
    fit_ellipse,
)
from issue38_no14_center_timeseries import param, window_width  # noqa: E402
from issue38_paths import FIGURES, TABLES, ensure_output_dirs
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STRIDE_FRAMES = 100  # About 0.5 s; production center itself remains frame by frame.


def robust_radius(x: np.ndarray, y: np.ndarray) -> float:
    """Half the larger 5–95% coordinate span, independent of fitted center."""
    return 0.5 * max(np.ptp(np.quantile(x, [0.05, 0.95])), np.ptp(np.quantile(y, [0.05, 0.95])))


def fit_circle(points: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
    """Constrained robust circle used only as a simple geometric comparator."""
    x, y = points.T
    middle = np.median(points, axis=0)
    initial_radius = np.median(np.linalg.norm(points - middle, axis=1))

    def residual(parameters: np.ndarray) -> np.ndarray:
        return np.hypot(x - parameters[0], y - parameters[1]) - parameters[2]

    result = least_squares(
        residual,
        [middle[0], middle[1], initial_radius],
        bounds=(
            [middle[0] - scale, middle[1] - scale, 0.1 * scale],
            [middle[0] + scale, middle[1] + scale, 2.0 * scale],
        ),
        loss="soft_l1",
        f_scale=0.2 * scale,
        max_nfev=150,
    )
    return result.x[:2], result.x[2], float(np.quantile(np.abs(residual(result.x)), 0.95) / scale)


def evaluate(data: pd.DataFrame) -> pd.DataFrame:
    t = data.time_sec.to_numpy(float)
    xy = data[["x_um", "y_um"]].to_numpy(float)
    width_sec = window_width(t, xy[:, 0], xy[:, 1])
    saved = data[["center_x_um", "center_y_um"]].to_numpy(float)
    raw = data[["raw_center_x_um", "raw_center_y_um"]].to_numpy(float)
    dt = (t[-1] - t[0]) / len(t)
    rows = []
    for i in range(0, len(data), STRIDE_FRAMES):
        start_time = t[0] + i * dt
        end_time = start_time + width_sec
        if end_time >= t[-1]:
            continue
        start = int(np.searchsorted(t, start_time, side="left"))
        end = int(np.searchsorted(t, end_time, side="left"))
        points = xy[start:end]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        radius = robust_radius(points[:, 0], points[:, 1]) if len(points) >= 30 else np.nan
        row = {
            "center_index_0based": i,
            "source_frame_1based": int(data.source_frame_1based.iloc[i]),
            "start_time_sec": start_time,
            "end_time_sec": end_time,
            "n_points": len(points),
            "orbit_radius_um": radius,
            "current_center_x_um": saved[i, 0],
            "current_center_y_um": saved[i, 1],
            "raw_center_x_um": raw[i, 0],
            "raw_center_y_um": raw[i, 1],
            "fit_valid": False,
        }
        if len(points) < 30 or not np.isfinite(radius) or radius <= 0:
            rows.append(row)
            continue
        fit = fit_ellipse(points[:, 0], points[:, 1])
        if fit is None:
            rows.append(row)
            continue
        center = fit.center
        residual = ellipse_distance(points[:, 0], points[:, 1], fit)
        circle_center, circle_radius, circle_p95 = fit_circle(points, radius)
        smooth = gaussian_filter1d(points, sigma=3, axis=0)
        phase = np.unwrap(np.arctan2(smooth[:, 1] - center[1], smooth[:, 0] - center[0]))
        phase_steps = np.diff(phase)
        phase_path = float(np.sum(np.abs(phase_steps)))
        coverage, max_gap = angular_coverage(points[:, 0], points[:, 1], center)
        block_centers = []
        block_coverage = []
        block_residual = []
        for block in np.array_split(points, 4):
            part = fit_ellipse(block[:, 0], block[:, 1])
            if part is None:
                continue
            block_centers.append(part.center)
            block_coverage.append(angular_coverage(block[:, 0], block[:, 1], part.center)[0])
            block_residual.append(np.quantile(ellipse_distance(block[:, 0], block[:, 1], part), 0.95) / radius)
        jackknife = []
        for omit in np.array_split(np.arange(len(points)), 4):
            keep = np.ones(len(points), dtype=bool)
            keep[omit] = False
            part = fit_ellipse(points[keep, 0], points[keep, 1])
            if part is not None:
                jackknife.append(part.center)
        bc = np.asarray(block_centers)
        jk = np.asarray(jackknife)
        block_span = float(np.max(np.linalg.norm(bc[:, None] - bc[None, :], axis=2))) if len(bc) > 1 else np.nan
        row.update(
            {
                "fit_valid": True,
                "fit_center_x_um": center[0],
                "fit_center_y_um": center[1],
                "ellipse_axis_1_um": fit.axes[0],
                "ellipse_axis_2_um": fit.axes[1],
                "normalized_design_condition": fit.condition,
                "coverage_deg": coverage,
                "max_empty_angle_deg": max_gap,
                "smoothed_net_turns": abs(float(np.sum(phase_steps))) / (2 * np.pi),
                "smoothed_path_turns": phase_path / (2 * np.pi),
                "smoothed_directionality": abs(float(np.sum(phase_steps))) / phase_path if phase_path else np.nan,
                "residual_p50_over_radius": np.median(residual) / radius,
                "residual_p95_over_radius": np.quantile(residual, 0.95) / radius,
                "circle_center_x_um": circle_center[0],
                "circle_center_y_um": circle_center[1],
                "circle_radius_um": circle_radius,
                "circle_residual_p95_over_radius": circle_p95,
                "circle_vs_ellipse_center_over_radius": np.linalg.norm(circle_center - center) / radius,
                "raw_vs_refit_over_radius": np.linalg.norm(raw[i] - center) / radius,
                "correction_over_radius": np.linalg.norm(saved[i] - raw[i]) / radius,
                "current_vs_refit_over_radius": np.linalg.norm(saved[i] - center) / radius,
                "fit_center_offset_over_radius": np.linalg.norm(center - np.median(points, axis=0)) / radius,
                "block_valid": len(bc),
                "block_sd_x_um": float(np.std(bc[:, 0])) if len(bc) else np.nan,
                "block_sd_y_um": float(np.std(bc[:, 1])) if len(bc) else np.nan,
                "block_center_span_over_radius": block_span / radius,
                "block_min_coverage_deg": min(block_coverage) if block_coverage else np.nan,
                "block_max_residual_p95_over_radius": max(block_residual) if block_residual else np.nan,
                "jackknife_valid": len(jk),
                "jackknife_sd_x_um": float(np.std(jk[:, 0])) if len(jk) else np.nan,
                "jackknife_sd_y_um": float(np.std(jk[:, 1])) if len(jk) else np.nan,
                "jackknife_max_shift_over_radius": (
                    float(np.max(np.linalg.norm(jk - center, axis=1))) / radius if len(jk) else np.nan
                ),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    center_xy = result[["current_center_x_um", "current_center_y_um"]].to_numpy(float)
    adjacent_step = np.r_[np.nan, np.linalg.norm(np.diff(center_xy, axis=0), axis=1)]
    result["adjacent_center_step_over_radius"] = adjacent_step / result.orbit_radius_um
    # A relative, data-derived low-motion screen: the median extent before stimulus.
    baseline = result.loc[(result.start_time_sec >= 20) & (result.start_time_sec < 35), "orbit_radius_um"].median()
    result["extent_over_baseline"] = result.orbit_radius_um / baseline
    result["low_motion_flag"] = result.extent_over_baseline < 0.4
    result["poor_coverage_flag"] = result.coverage_deg < 270
    result["unstable_center_flag"] = (result.jackknife_valid < 4) | (result.jackknife_max_shift_over_radius > 0.25)
    result["nonellipse_flag"] = result.residual_p95_over_radius > 0.20
    result["drift_candidate_flag"] = (
        (result.block_valid == 4)
        & (result.block_min_coverage_deg >= 270)
        & (result.block_max_residual_p95_over_radius <= 0.20)
        & (result.block_center_span_over_radius > 0.25)
    )
    result["correction_large_flag"] = result.correction_over_radius > 0.25
    # These are conservative review labels, not validated acceptance criteria.
    result["assessment"] = "provisionally_supported"
    result.loc[result.nonellipse_flag, "assessment"] = "repeatable_but_nonelliptic"
    result.loc[result.correction_large_flag, "assessment"] = "correction_displaced"
    result.loc[result.drift_candidate_flag, "assessment"] = "possible_center_drift"
    result.loc[result.unstable_center_flag | result.poor_coverage_flag, "assessment"] = "center_unidentifiable"
    result.loc[result.low_motion_flag, "assessment"] = "low_motion_unidentifiable"
    result.loc[~result.fit_valid, "assessment"] = "fit_failed"
    return result


def plot_metrics(rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(18, 13), sharex=True)
    t = rows.start_time_sec
    axes[0].plot(t, rows.orbit_radius_um, ".-", ms=2, lw=0.6)
    axes[0].set_ylabel("Orbit radius (µm)")
    axes[1].plot(t, rows.coverage_deg, ".-", ms=2, lw=0.6, label="full window")
    axes[1].plot(t, rows.block_min_coverage_deg, ".-", ms=2, lw=0.6, label="least-covered quarter")
    axes[1].axhline(270, color="gray", ls=":")
    axes[1].set_ylabel("Angular coverage (deg)")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(t, rows.residual_p95_over_radius, ".-", ms=2, lw=0.6, label="ellipse residual P95")
    axes[2].plot(t, rows.jackknife_max_shift_over_radius, ".-", ms=2, lw=0.6, label="leave-quarter-out shift")
    axes[2].set_ylabel("Distance / orbit radius")
    axes[2].set_ylim(0, min(4, max(1, np.nanquantile(rows.jackknife_max_shift_over_radius, 0.95) * 1.2)))
    axes[2].legend(frameon=False, ncol=2)
    axes[3].plot(t, rows.smoothed_net_turns, ".-", ms=2, lw=0.6, label="net turns")
    axes[3].plot(t, rows.smoothed_path_turns, ".-", ms=2, lw=0.6, label="total angular path")
    axes[3].set_ylabel("Turns in forward window")
    axes[3].legend(frameon=False, ncol=2)
    axes[4].plot(t, rows.correction_over_radius.clip(upper=3), ".-", ms=2, lw=0.6, label="outlier correction")
    axes[4].plot(t, rows.block_center_span_over_radius.clip(upper=3), ".-", ms=2, lw=0.6, label="quarter-center span")
    axes[4].set_ylabel("Center change / orbit radius")
    axes[4].set_xlabel("Window start time (s)")
    axes[4].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.axvline(40, color="#7b3294", ls=":")
        axis.grid(alpha=0.18)
    axes[-1].set_xlim(15.78, 82.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_center_window_metrics.png", dpi=160)
    plt.close(fig)


def plot_examples(data: pd.DataFrame, rows: pd.DataFrame, times: list[float]) -> None:
    t = data.time_sec.to_numpy(float)
    xy = data[["x_um", "y_um"]].to_numpy(float)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for axis, target in zip(axes.flat, times):
        row = rows.iloc[(rows.start_time_sec - target).abs().argmin()]
        start, end = row.start_time_sec, row.end_time_sec
        points = xy[(t >= start) & (t < end)]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            c=np.linspace(0, 1, len(points)),
            cmap="viridis",
            s=4,
            alpha=0.45,
            rasterized=True,
        )
        if row.fit_valid:
            fit = fit_ellipse(points[:, 0], points[:, 1])
            if fit is not None:
                ellipse = ellipse_points(fit)
                axis.plot(ellipse[:, 0], ellipse[:, 1], color="#0072b2", lw=1.2, label="window fit")
                axis.plot(fit.center[0], fit.center[1], "+", color="#0072b2", ms=11, mew=2)
        axis.plot(
            row.current_center_x_um,
            row.current_center_y_um,
            "x",
            color="#d95f02",
            ms=11,
            mew=2,
            label="corrected center",
        )
        for quarter, block in enumerate(np.array_split(points, 4), start=1):
            part = fit_ellipse(block[:, 0], block[:, 1])
            if part is not None:
                axis.plot(part.center[0], part.center[1], ".", color="#6a3d9a", ms=5)
                axis.annotate(
                    str(quarter), part.center, fontsize=7, color="#6a3d9a", xytext=(3, 3), textcoords="offset points"
                )
        axis.set_title(f"{start:.1f}–{end:.1f} s | {row.assessment}", fontsize=10)
        axis.set_xlabel("x in AVI frame (µm)")
        axis.set_ylabel("y in AVI frame (µm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_center_window_examples.png", dpi=160)
    plt.close(fig)


def plot_video_checks(data: pd.DataFrame) -> None:
    """Check centroid and calculated center against selected source AVI frames."""
    cap = cv2.VideoCapture(str(ROOT / "data/repellent-response/23/2026_0802_182430.avi"))
    if not cap.isOpened():
        raise FileNotFoundError("No.14 source AVI")
    px2um_x, px2um_y = param.get_px2um_config("repellent-response/23")
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    times = data.time_sec.to_numpy(float)
    for row, start in enumerate((20.0, 38.0, 65.0, 82.5)):
        for col, offset in enumerate((0.0, 2.0, 4.0)):
            i = int(np.argmin(np.abs(times - (start + offset))))
            source = int(data.source_frame_1based.iloc[i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, source - 1)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read source frame {source}")
            axis = axes[row, col]
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.plot(data.x_px.iloc[i], data.y_px.iloc[i], "+", color="#1b9e77", ms=13, mew=2, label="centroid")
            axis.plot(
                data.center_x_um.iloc[i] / px2um_x,
                data.center_y_um.iloc[i] / px2um_y,
                "x",
                color="#d95f02",
                ms=11,
                mew=2,
                label="current center",
            )
            axis.set_title(f"{times[i]:.2f} s | AVI {source}", fontsize=9)
            axis.set_xlim(0, frame.shape[1])
            axis.set_ylim(frame.shape[0], 0)
            axis.set_xticks([])
            axis.set_yticks([])
    cap.release()
    axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_center_video_checks.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    assert data.source_frame_1based.iloc[[0, -1]].tolist() == [2049, 16384]
    assert np.all(np.diff(data.time_sec) > 0)
    rows = evaluate(data)
    if rows.loc[rows.fit_valid, "raw_vs_refit_over_radius"].max() > 1e-8:
        raise AssertionError("Diagnostic windows do not reproduce the production raw ellipse fits")
    rows.to_csv(TABLES / "no14_center_window_metrics.csv", index=False)
    plot_metrics(rows)
    plot_examples(data, rows, [20, 35, 38, 40, 65, 80])
    plot_video_checks(data)
    print(rows.assessment.value_counts().to_string())
    print(
        f"windows={len(rows)}; fit_valid={rows.fit_valid.sum()}; "
        f"max raw/refit delta={rows.raw_vs_refit_over_radius.max():.3g} R"
    )


if __name__ == "__main__":
    main()
