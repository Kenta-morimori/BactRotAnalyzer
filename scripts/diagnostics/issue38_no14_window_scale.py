"""Measure observed cloud size before interpreting fitted ellipse radius.

Compare the same three ellipse candidates and start times at 1/4, 1/2, 1,
and 2 times the existing forward window. No model is selected automatically.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from issue38_no14_image_trajectory import WINDOWS
from issue38_paths import FIGURES, TABLES, ensure_output_dirs

from utils.functions.rotation_center_candidates import (
    ELLIPSE_METHODS,
    ellipse_distance,
    estimate_center,
    robust_radius,
)

SCALES = (0.25, 0.5, 1.0, 2.0)
COLORS = dict(zip(ELLIPSE_METHODS, ("#0072b2", "#009e73", "#d55e00")))


def observed_extent(points):
    """Extent of sampled centroid cloud, not a physical rotation radius."""
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


def measure_images(data):
    """Measure morphology once; reuse saved centroid without extracting it again."""
    root = Path(__file__).resolve().parents[2]
    cap = cv2.VideoCapture(str(root / "data/repellent-response/23/2026_0802_182430.avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(data.source_frame_1based.iloc[0]) - 1)
    records = []
    try:
        for item in data.itertuples():
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read AVI {item.source_frame_1based}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = circularity = np.nan
            touches = False
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / perimeter**2 if perimeter else np.nan
                x, y, w, h = cv2.boundingRect(contour)
                touches = x == 0 or y == 0 or x + w >= gray.shape[1] or y + h >= gray.shape[0]
            records.append(
                {
                    "source_frame_1based": item.source_frame_1based,
                    "time_sec": item.time_sec,
                    "contour_area_px2": area,
                    "circularity": circularity,
                    "contour_count": len(contours),
                    "border_touch": touches,
                }
            )
    finally:
        cap.release()
    return pd.DataFrame(records)


def fit_metrics(points, method):
    radius = robust_radius(points)
    result = estimate_center(points, method)
    row = {
        "method": method,
        "status": result.status,
        "boundary_reached": result.boundary_reached,
        "converged": result.converged,
        "n_input": result.n_input,
        "n_used": result.n_used,
        "center_x_um": result.center[0] if result.center is not None else np.nan,
        "center_y_um": result.center[1] if result.center is not None else np.nan,
        "major_um": np.nan,
        "minor_um": np.nan,
        "major_angle_deg": np.nan,
        "fit_p95_R": np.nan,
        "cv_p95_R": np.nan,
        "center_cv_max_R": np.nan,
        "cv_computed_count": 0,
        "cv_ellipse_count": 0,
    }
    if result.ellipse is None:
        return row
    major_index = int(np.argmax(result.ellipse.axes))
    row.update(
        major_um=max(result.ellipse.axes),
        minor_um=min(result.ellipse.axes),
        major_angle_deg=np.rad2deg(
            np.arctan2(result.ellipse.basis[1, major_index], result.ellipse.basis[0, major_index])
        )
        % 180,
        fit_p95_R=np.quantile(np.abs(ellipse_distance(points, result.ellipse)), 0.95) / radius,
    )
    centers, residuals = [], []
    for omitted in np.array_split(np.arange(len(points)), 4):
        keep = np.ones(len(points), bool)
        keep[omitted] = False
        part = estimate_center(points[keep], method)
        row["cv_computed_count"] += int(part.computationally_ok)
        if part.ellipse is not None:
            centers.append(part.center)
            residuals.extend(np.abs(ellipse_distance(points[omitted], part.ellipse)))
    row["cv_ellipse_count"] = len(centers)
    if centers:
        row["center_cv_max_R"] = np.max(np.linalg.norm(np.asarray(centers) - result.center, axis=1)) / radius
        row["cv_p95_R"] = np.quantile(residuals, 0.95) / radius
    return row


def analyze(data, windows, morphology):
    times = data.time_sec.to_numpy(float)
    width = float(windows.end_time_sec.iloc[0] - windows.start_time_sec.iloc[0])
    starts = windows.start_time_sec.tolist()
    # Include the held terminal interval explicitly, without forcing a full fit.
    tail = float(data.loc[data.center_source == "terminal_hold", "time_sec"].iloc[0])
    starts.append(tail)
    rows, raw_rows, block_rows = [], [], []
    for index, start in enumerate(starts):
        baseline = data.loc[(times >= start) & (times < start + width)]
        baseline_R = robust_radius(baseline[["x_um", "y_um"]].to_numpy(float))
        for scale in SCALES:
            end = start + width * scale
            frame = data.loc[(times >= start) & (times < end)]
            points = frame[["x_um", "y_um"]].to_numpy(float)
            common = {
                "anchor_time_sec": start,
                "window_scale": scale,
                "window_width_sec": width * scale,
                "window_end_sec": end,
                "complete_forward_window": end < times[-1],
                "available_first_frame": int(frame.source_frame_1based.iloc[0]),
                "available_last_frame": int(frame.source_frame_1based.iloc[-1]),
                "n_available": len(frame),
                "baseline_cloud_R_um": baseline_R,
            }
            if end >= times[-1]:
                for method in ELLIPSE_METHODS:
                    rows.append({**common, "method": method, "status": "incomplete_window"})
                continue
            raw = {**common, **observed_extent(points)}
            image_rows = morphology.loc[morphology.source_frame_1based.isin(frame.source_frame_1based)]
            raw.update(
                area_median_px2=image_rows.contour_area_px2.median(),
                area_p05_px2=image_rows.contour_area_px2.quantile(0.05),
                area_p95_px2=image_rows.contour_area_px2.quantile(0.95),
            )
            blocks = []
            for block, indices in enumerate(np.array_split(np.arange(len(frame)), 4)):
                item = {
                    **common,
                    "time_block": block,
                    **observed_extent(points[indices]),
                    "block_start_sec": frame.time_sec.iloc[indices[0]],
                    "block_end_sec": frame.time_sec.iloc[indices[-1]],
                }
                blocks.append(item)
                block_rows.append(item)
            raw["block_R_max_over_min"] = max(b["cloud_R_um"] for b in blocks) / min(b["cloud_R_um"] for b in blocks)
            raw["block_median_position_span_R"] = (
                max(
                    np.hypot(a["median_x_um"] - b["median_x_um"], a["median_y_um"] - b["median_y_um"])
                    for a in blocks
                    for b in blocks
                )
                / raw["cloud_R_um"]
            )
            raw_rows.append(raw)
            for method in ELLIPSE_METHODS:
                rows.append({**common, **fit_metrics(points, method)})
        if index % 20 == 0:
            print(f"width analysis {index + 1}/{len(starts)}", flush=True)
    fits = pd.DataFrame(rows)
    for start in starts:
        for method in ELLIPSE_METHODS:
            selected = (fits.anchor_time_sec == start) & (fits.method == method)
            reference = fits.loc[selected & (fits.window_scale == 1)].iloc[0]
            if reference.status == "incomplete_window":
                continue
            idx = fits.index[selected & fits.complete_forward_window]
            fits.loc[idx, "center_difference_from_5s_R"] = (
                np.hypot(
                    fits.loc[idx, "center_x_um"] - reference.center_x_um,
                    fits.loc[idx, "center_y_um"] - reference.center_y_um,
                )
                / reference.baseline_cloud_R_um
            )
            fits.loc[idx, "major_over_5s"] = fits.loc[idx, "major_um"] / reference.major_um
            fits.loc[idx, "minor_over_5s"] = fits.loc[idx, "minor_um"] / reference.minor_um
    return fits, pd.DataFrame(raw_rows), pd.DataFrame(block_rows)


def figures(fits, raw, morphology, data, windows):
    fig, axes = plt.subplots(3, 1, figsize=(19, 10), sharex=True)
    for scale in SCALES:
        sample = raw[(raw.window_scale == scale) & (raw.anchor_time_sec < 82.5)]
        axes[0].plot(sample.anchor_time_sec, sample.cloud_R_um, label=f"{scale:g} × 5.080s", lw=1)
        axes[1].plot(sample.anchor_time_sec, sample.block_R_max_over_min, lw=1)
    axes[0].set_ylabel("Observed cloud R (µm)")
    axes[1].set_ylabel("Within-window block R max/min")
    axes[2].plot(morphology.time_sec, morphology.contour_area_px2, lw=0.6, color="#666666")
    axes[2].set_ylabel("Source contour area (px²)")
    axes[2].set_xlabel("Forward window start / source time (s)")
    axes[0].legend(ncol=4)
    for axis in axes:
        axis.axvline(40, color="gray", ls=":")
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_observed_orbit_size.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(20, 11), sharex=True)
    for column, method in enumerate(ELLIPSE_METHODS):
        for scale in SCALES:
            sample = fits[(fits.method == method) & fits.complete_forward_window]
            sample = sample[sample.window_scale == scale]
            for axis, field in zip(axes[:, column], ("major_um", "minor_um", "center_difference_from_5s_R")):
                axis.plot(sample.anchor_time_sec, sample[field], lw=0.8, label=f"{scale:g} ×")
        axes[0, column].set_title(method)
        axes[2, column].set_xlabel("Forward window start (s)")
    for axis, label in zip(
        axes[:, 0], ("Major semi-axis (µm)", "Minor semi-axis (µm)", "Center change / baseline cloud R")
    ):
        axis.set_ylabel(label)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
        axis.axvline(40, color="gray", ls=":")
    # Large conics remain in CSV, but clip plots to make supported intervals readable.
    for axis in axes[0]:
        axis.set_ylim(0, 1.7)
    for axis in axes[1]:
        axis.set_ylim(0, 0.6)
    for axis in axes[2]:
        axis.set_ylim(0, 2)
    axes[0, 0].legend(ncol=4)
    fig.suptitle("Computational results include flagged fits; clipped display, full values in CSV")
    fig.tight_layout()
    fig.savefig(FIGURES / "no14_window_width_comparison.png", dpi=150)
    plt.close(fig)
    for name, target in WINDOWS.items():
        start = float(fits.anchor_time_sec.iloc[(fits.anchor_time_sec - target).abs().argmin()])
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharex=True, sharey=True)
        for axis, scale in zip(axes, SCALES):
            selected = fits[(fits.anchor_time_sec == start) & (fits.window_scale == scale)]
            end = selected.window_end_sec.iloc[0]
            frame = data[(data.time_sec >= start) & (data.time_sec < end)]
            for block, indices in enumerate(np.array_split(np.arange(len(frame)), 4)):
                axis.scatter(
                    frame.x_px.iloc[indices],
                    frame.y_px.iloc[indices],
                    s=4,
                    color=plt.get_cmap("viridis")(block / 3),
                    alpha=0.45,
                )
            if selected.complete_forward_window.iloc[0]:
                points = frame[["x_um", "y_um"]].to_numpy(float)
                for method in ELLIPSE_METHODS:
                    result = estimate_center(points, method)
                    if result.ellipse is not None:
                        phase = np.linspace(0, 2 * np.pi, 720)
                        axis.plot(*(result.ellipse.center / 0.02), "+", color=COLORS[method], ms=10)
                        curve = (
                            result.center
                            + (np.column_stack((np.cos(phase), np.sin(phase))) * result.axes) @ result.ellipse.basis.T
                        )
                        axis.plot(
                            curve[:, 0] / 0.02,
                            curve[:, 1] / 0.02,
                            color=COLORS[method],
                            ls="-" if result.computationally_ok else "--",
                            label=f"{method}: {result.status}",
                        )
                    elif result.center is not None:
                        axis.plot(
                            *(result.center / 0.02), "+", color=COLORS[method], label=f"{method}: {result.status}"
                        )
            else:
                axis.text(0.05, 0.93, "Incomplete forward window\nNo candidate fitted", transform=axis.transAxes)
            axis.set(
                xlim=(30, 110),
                ylim=(110, 30),
                xlabel="AVI x (px)",
                title=f"{scale:g} × | {end-start:.3f}s\nrequested {start:.3f}–{end:.3f}s",
            )
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
            axis.legend(fontsize=6, loc="lower left") if selected.complete_forward_window.iloc[0] else None
        axes[0].set_ylabel("AVI y (px)")
        fig.suptitle(f"No.14 {name}: same start, changing forward width | four time blocks colored")
        fig.tight_layout()
        fig.savefig(FIGURES / f"no14_window_width_{name}.png", dpi=150)
        plt.close(fig)


def main():
    ensure_output_dirs()
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    windows = pd.read_csv(TABLES / "no14_center_window_metrics.csv")
    assert len(windows) == 134
    assert np.array_equal(data.source_frame_1based.to_numpy(int), np.arange(2049, 16385))
    assert np.all(np.diff(data.time_sec) > 0)
    assert np.allclose(data[["x_um", "y_um"]], data[["x_px", "y_px"]].to_numpy() * 0.02)
    assert np.isfinite(data[["x_um", "y_um"]].to_numpy()).all()
    morphology = measure_images(data)
    morphology.to_csv(TABLES / "no14_source_morphology.csv", index=False)
    fits, raw, blocks = analyze(data, windows, morphology)
    fits.to_csv(TABLES / "no14_window_width_fits.csv", index=False)
    raw.to_csv(TABLES / "no14_observed_orbit_extent.csv", index=False)
    blocks.to_csv(TABLES / "no14_observed_orbit_blocks.csv", index=False)
    figures(fits, raw, morphology, data, windows)
    print(fits.groupby(["window_scale", "method", "status"]).size().to_string())


if __name__ == "__main__":
    main()
