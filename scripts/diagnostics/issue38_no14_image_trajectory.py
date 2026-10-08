"""Pair No.14 source images with the centroid trajectory and current center.

Run after issue38_no14_center_timeseries.py and
issue38_no14_center_evaluation.py. Outputs only diagnostic files.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from PIL import Image  # noqa: E402

from issue38_center_diagnostics import ellipse_points, fit_ellipse  # noqa: E402
from issue38_no14_center_timeseries import param  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "docs/analysis/issue-38"
AVI = ROOT / "data/repellent-response/23/2026_0802_182430.avi"
TIFF = AVI.with_suffix(".tif")
TIFF_LOG_DIR = AVI.parent / "tiff_data" / AVI.stem
WINDOWS = {
    "initial_invalid": 15.781025,
    "early_displaced": 20.789,
    "rise_crossing": 38.318,
    "later_reference": 64.862,
    "terminal_hold": 82.500132,
}
BURST_SPACING_FRAMES = 8
BURST_LENGTH = 5
BURST_FRACTIONS = (0.10, 0.50, 0.90)
SEGMENT_COLORS = ("#2a9d8f", "#7b3294", "#e9a23b")


def select_window(data: pd.DataFrame, metrics: pd.DataFrame, name: str, target: float) -> tuple[pd.DataFrame, pd.Series | None]:
    if name == "terminal_hold":
        frame = data.loc[data.center_source == "terminal_hold"].copy()
        return frame, None
    row = metrics.iloc[(metrics.start_time_sec - target).abs().argmin()]
    if abs(float(row.start_time_sec) - target) > 0.05:
        raise ValueError(f"No matching diagnostic window for {name}: {target}")
    frame = data.loc[(data.time_sec >= row.start_time_sec) & (data.time_sec < row.end_time_sec)].copy()
    if len(frame) != int(row.n_points):
        raise ValueError(f"Window point count mismatch for {name}")
    return frame, row


def selected_positions(n: int) -> list[int]:
    span = (BURST_LENGTH - 1) * BURST_SPACING_FRAMES
    if n <= span:
        raise ValueError("Window is shorter than one diagnostic image burst")
    positions = []
    for fraction in BURST_FRACTIONS:
        start = int(round(fraction * (n - span - 1)))
        positions.extend(start + j * BURST_SPACING_FRAMES for j in range(BURST_LENGTH))
    if len(set(positions)) != len(positions):
        raise ValueError("Diagnostic image bursts overlap")
    return positions


def read_window_images(cap: cv2.VideoCapture, frame: pd.DataFrame, positions: list[int], tiff: tifffile.TiffFile,
                       name: str) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    first = int(frame.source_frame_1based.iloc[0])
    last = int(frame.source_frame_1based.iloc[-1])
    if not np.array_equal(frame.source_frame_1based.to_numpy(int), np.arange(first, last + 1)):
        raise ValueError(f"Nonconsecutive source frames for {name}")
    if not np.all(np.diff(frame.time_sec.to_numpy(float)) > 0):
        raise ValueError(f"Nonmonotonic TIFF times for {name}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    selected = set(positions)
    burst_positions = {j for start in positions[::BURST_LENGTH]
                       for j in range(start, start + (BURST_LENGTH - 1) * BURST_SPACING_FRAMES + 1)}
    image_map: dict[int, np.ndarray] = {}
    rows = []
    log_files = sorted(TIFF_LOG_DIR.glob("*.tif"),
                       key=lambda p: int(re.search(r"_(\d+)\.tif$", p.name).group(1)))
    if len(log_files) != 18432:
        raise ValueError("TIFF log frame count differs from AVI")
    with Image.open(log_files[0]) as log_image:
        base_time = datetime.strptime(log_image.tag_v2[306], "%m/%d/%Y %H:%M:%S.%f")
    px2um_x, px2um_y = param.get_px2um_config("repellent-response/23")
    for offset, item in enumerate(frame.itertuples(index=False)):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read AVI frame {item.source_frame_1based}")
        if offset in burst_positions:
            original = tiff.pages[item.source_frame_1based - 1].asarray()
            if not np.array_equal(image[:, :, 0], original):
                raise AssertionError(f"AVI/TIFF image mismatch at frame {item.source_frame_1based}")
            image_map[offset] = image[:, :, 0]
            with Image.open(log_files[item.source_frame_1based - 1]) as log_image:
                log_time = datetime.strptime(log_image.tag_v2[306], "%m/%d/%Y %H:%M:%S.%f")
            time_error = abs((log_time - base_time).total_seconds() - item.time_sec)
            if time_error > 1e-9:
                raise AssertionError(f"TIFF timestamp mismatch at frame {item.source_frame_1based}")
        else:
            time_error = np.nan
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                measured_x = float(np.mean(contour[:, 0, 0]))
                measured_y = float(np.mean(contour[:, 0, 1]))
            else:
                measured_x = measured_y = np.nan
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else np.nan
            x0, y0, width, height = cv2.boundingRect(contour)
            border_touch = x0 == 0 or y0 == 0 or x0 + width >= gray.shape[1] or y0 + height >= gray.shape[0]
        else:
            measured_x = measured_y = area = circularity = np.nan
            border_touch = False
        error = float(np.hypot(measured_x - item.x_px, measured_y - item.y_px))
        if bool(item.detected) != bool(np.isfinite(measured_x)):
            raise AssertionError(f"Detection status mismatch at frame {item.source_frame_1based}")
        if np.isfinite(error) and error > 1e-9:
            raise AssertionError(f"Saved/image centroid mismatch at frame {item.source_frame_1based}: {error} px")
        if not np.isclose(item.x_um, item.x_px * px2um_x, atol=1e-12, rtol=0) or not np.isclose(
            item.y_um, item.y_px * px2um_y, atol=1e-12, rtol=0
        ):
            raise AssertionError(f"Pixel-to-µm mismatch at frame {item.source_frame_1based}")
        rows.append({
            "window": name,
            "source_frame_1based": int(item.source_frame_1based),
            "time_sec": float(item.time_sec),
            "selected_image_number": positions.index(offset) + 1 if offset in selected else np.nan,
            "displayed_in_continuous_burst": offset in burst_positions,
            "tiff_log_time_error_sec": time_error,
            "saved_x_px": float(item.x_px),
            "saved_y_px": float(item.y_px),
            "measured_x_px": measured_x,
            "measured_y_px": measured_y,
            "centroid_error_px": error,
            "contour_count": len(contours),
            "max_contour_area_px2": area,
            "max_contour_circularity": circularity,
            "border_touch": border_touch,
        })
    return pd.DataFrame(rows), image_map


def continuous_burst_gif(name: str, frame: pd.DataFrame, image_map: dict[int, np.ndarray],
                         positions: list[int]) -> None:
    """Show all 33 consecutive frames for each burst, slowed 20 times for inspection."""
    px_x, px_y = param.get_px2um_config("repellent-response/23")
    center = frame.iloc[0]
    frames = []
    for step in range((BURST_LENGTH - 1) * BURST_SPACING_FRAMES + 1):
        panels = []
        for burst, start in enumerate(positions[::BURST_LENGTH]):
            item = frame.iloc[start + step]
            image = cv2.cvtColor(image_map[start + step], cv2.COLOR_GRAY2RGB)
            cv2.drawMarker(image, (round(item.x_px), round(item.y_px)), (27, 158, 119),
                           cv2.MARKER_CROSS, 9, 1)
            cv2.drawMarker(image, (round(center.center_x_um / px_x), round(center.center_y_um / px_y)),
                           (217, 95, 2), cv2.MARKER_TILTED_CROSS, 9, 1)
            image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST)
            panel = np.full((350, 300, 3), 255, np.uint8)
            panel[50:] = image
            cv2.putText(panel, f'{("early", "middle", "late")[burst]} AVI {int(item.source_frame_1based)}',
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
            cv2.putText(panel, f'{item.time_sec:.6f}s | 20x slower',
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
            panels.append(panel)
        frames.append(Image.fromarray(np.concatenate(panels, axis=1)))
    frames[0].save(DEST / f"no14_image_trajectory_{name}.gif", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)


def image_trajectory_figure(name: str, frame: pd.DataFrame, measurements: pd.DataFrame,
                            image_map: dict[int, np.ndarray], row: pd.Series | None) -> None:
    px2um_x, px2um_y = param.get_px2um_config("repellent-response/23")
    fig = plt.figure(figsize=(20, 11.2))
    grid = fig.add_gridspec(2, 2, width_ratios=(2.65, 1), height_ratios=(3.1, 1),
                            left=0.03, right=0.98, bottom=0.06, top=0.92, wspace=0.12, hspace=0.14)
    image_grid = grid[:, 0].subgridspec(3, 5, wspace=0.04, hspace=0.16)
    selected_rows = measurements.loc[measurements.selected_image_number.notna()].copy()
    first_center = frame.iloc[0]
    for _, selected in selected_rows.iterrows():
        number = int(selected.selected_image_number)
        burst, phase = divmod(number - 1, BURST_LENGTH)
        axis = fig.add_subplot(image_grid[burst, phase])
        image = image_map[int(selected.source_frame_1based - frame.source_frame_1based.iloc[0])]
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper", interpolation="nearest")
        axis.plot(selected.saved_x_px, selected.saved_y_px, "+", color="#1b9e77", ms=11, mew=1.6)
        axis.plot(first_center.center_x_um / px2um_x, first_center.center_y_um / px2um_y,
                  "x", color="#d95f02", ms=10, mew=1.5)
        if row is not None and np.isfinite(first_center.raw_center_x_um) and np.isfinite(first_center.raw_center_y_um):
            axis.plot(first_center.raw_center_x_um / px2um_x, first_center.raw_center_y_um / px2um_y,
                      "o", mfc="none", mec="#0072b2", ms=8, mew=1.2)
        axis.set_title(f"{number}: AVI {int(selected.source_frame_1based)} | {selected.time_sec:.3f}s",
                       fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        if phase == 0:
            axis.set_ylabel(("early", "middle", "late")[burst], fontsize=11)
    trajectory = fig.add_subplot(grid[0, 1])
    count = len(frame)
    for block, indices in enumerate(np.array_split(np.arange(count), 3)):
        trajectory.scatter(frame.x_px.iloc[indices], frame.y_px.iloc[indices], s=3, alpha=0.48,
                           color=SEGMENT_COLORS[block], rasterized=True, label=("early", "middle", "late")[block])
    for _, selected in selected_rows.iterrows():
        trajectory.plot(selected.saved_x_px, selected.saved_y_px, "o", ms=3.5, color="#161616")
        trajectory.annotate(str(int(selected.selected_image_number)),
                            (selected.saved_x_px, selected.saved_y_px), xytext=(3, 3),
                            textcoords="offset points", fontsize=7)
    if row is not None and bool(row.fit_valid):
        fit = fit_ellipse(frame.x_um.to_numpy(float), frame.y_um.to_numpy(float))
        if fit is None:
            raise AssertionError(f"Metrics mark {name} valid, but ellipse is invalid")
        curve = ellipse_points(fit)
        trajectory.plot(curve[:, 0] / px2um_x, curve[:, 1] / px2um_y,
                        color="#0072b2", lw=1.2, label="raw-fit ellipse")
    if row is not None:
        trajectory.plot(first_center.raw_center_x_um / px2um_x, first_center.raw_center_y_um / px2um_y,
                        "+", color="#0072b2", ms=14, mew=2, label="raw-fit center")
    trajectory.plot(first_center.center_x_um / px2um_x, first_center.center_y_um / px2um_y,
                    "x", color="#d95f02", ms=12, mew=2, label="corrected / held center")
    trajectory.set(xlim=(30, 110), ylim=(110, 30), xlabel="x in AVI (px)", ylabel="y in AVI (px)")
    trajectory.set_aspect("equal", adjustable="box")
    trajectory.grid(alpha=0.15)
    trajectory.legend(loc="upper left", fontsize=8, framealpha=0.9)
    morphology = fig.add_subplot(grid[1, 1])
    morphology.plot(measurements.time_sec, measurements.max_contour_area_px2,
                    color="#455468", lw=0.9, label="contour area")
    for burst in range(3):
        selected = selected_rows.iloc[burst * BURST_LENGTH:(burst + 1) * BURST_LENGTH]
        morphology.scatter(selected.time_sec, selected.max_contour_area_px2,
                           color=SEGMENT_COLORS[burst], s=12, zorder=3)
    morphology.set_xlabel("Time (s)")
    morphology.set_ylabel("Largest contour area (px²)")
    morphology.grid(alpha=0.15)
    circle_axis = morphology.twinx()
    circle_axis.plot(measurements.time_sec, measurements.max_contour_circularity,
                     color="#7b3294", lw=0.6, alpha=0.38)
    circle_axis.set_ylabel("Circularity", color="#7b3294")
    circle_axis.tick_params(axis="y", colors="#7b3294")
    fig.suptitle(f"No.14 {name}: AVI images and matched centroid trajectory | "
                 f"{frame.time_sec.iloc[0]:.3f}–{frame.time_sec.iloc[-1]:.3f} s\n"
                 "green + = frame centroid; orange × = window-start corrected/held center; "
                 "blue ○ = window-start raw center", fontsize=12)
    fig.savefig(DEST / f"no14_image_trajectory_{name}.png", dpi=140)
    plt.close(fig)


def summarize(name: str, frame: pd.DataFrame, images: pd.DataFrame, row: pd.Series | None) -> dict:
    finite = images.centroid_error_px.dropna()
    data = {
        "window": name,
        "source_first_frame_1based": int(frame.source_frame_1based.iloc[0]),
        "source_last_frame_1based": int(frame.source_frame_1based.iloc[-1]),
        "start_time_sec": float(frame.time_sec.iloc[0]),
        "end_time_sec": float(frame.time_sec.iloc[-1]),
        "n_frames": len(images),
        "n_displayed": int(images.selected_image_number.notna().sum()),
        "n_missing_contour": int(images.measured_x_px.isna().sum()),
        "n_multiple_contours": int((images.contour_count > 1).sum()),
        "n_border_touch": int(images.border_touch.sum()),
        "max_centroid_error_px": float(finite.max()) if len(finite) else np.nan,
        "area_median_px2": float(images.max_contour_area_px2.median()),
        "area_min_px2": float(images.max_contour_area_px2.min()),
        "area_max_px2": float(images.max_contour_area_px2.max()),
        "circularity_median": float(images.max_contour_circularity.median()),
        "fit_valid": bool(row.fit_valid) if row is not None else False,
        "center_source": "terminal_hold" if row is None else "window_fit",
    }
    if row is not None:
        data["correction_over_radius"] = float(row.correction_over_radius) if pd.notna(row.correction_over_radius) else np.nan
        data["jackknife_max_shift_over_radius"] = (float(row.jackknife_max_shift_over_radius)
                                                      if pd.notna(row.jackknife_max_shift_over_radius) else np.nan)
        if bool(row.fit_valid):
            axes = sorted((float(row.ellipse_axis_1_um), float(row.ellipse_axis_2_um)))
            data["ellipse_axis_ratio"] = axes[1] / axes[0]
            data["ellipse_major_axis_over_radius"] = axes[1] / float(row.orbit_radius_um)
        data["block_valid_fits"] = int(row.block_valid) if pd.notna(row.block_valid) else 0
        data["block_min_coverage_deg"] = (float(row.block_min_coverage_deg)
                                           if pd.notna(row.block_min_coverage_deg) else np.nan)
        data["block_center_span_over_radius"] = (float(row.block_center_span_over_radius)
                                                  if pd.notna(row.block_center_span_over_radius) else np.nan)
    cloud = frame[["x_px", "y_px"]].to_numpy(float)
    eigenvalues = np.linalg.eigvalsh(np.cov(cloud.T))
    data["cloud_pca_axis_ratio"] = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    if data["n_missing_contour"] or data["n_multiple_contours"] or data["n_border_touch"] or data[
        "max_centroid_error_px"
    ] > 1e-9:
        data["assessment"] = "centroid_detection_suspect"
    elif row is None:
        data["assessment"] = "unassessable_terminal_hold"
    elif not bool(row.fit_valid) or (
        data.get("ellipse_major_axis_over_radius", 0) > 2.0
        and data.get("ellipse_axis_ratio", 0) > 3.0
        and data["jackknife_max_shift_over_radius"] > 0.25
    ):
        data["assessment"] = "center_fit_suspect"
    elif (
        data["jackknife_max_shift_over_radius"] <= 0.1
        and data["block_valid_fits"] == 4
        and data["block_min_coverage_deg"] >= 270
    ):
        data["assessment"] = "trajectory_consistent_provisional"
    else:
        data["assessment"] = "unassessable"
    return data


def main() -> None:
    data = pd.read_csv(DEST / "no14_centroid_center_timeseries.csv")
    metrics = pd.read_csv(DEST / "no14_center_window_metrics.csv")
    if data.source_frame_1based.iloc[[0, -1]].tolist() != [2049, 16384]:
        raise ValueError("No.14 frame range changed")
    cap = cv2.VideoCapture(str(AVI))
    if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 18432:
        raise ValueError("Cannot read all 18,432 No.14 AVI frames")
    frame_tables = []
    summaries = []
    try:
        with tifffile.TiffFile(TIFF) as tiff:
            if len(tiff.pages) != 18432:
                raise ValueError("No.14 TIFF page count differs from AVI")
            for name, target in WINDOWS.items():
                frame, row = select_window(data, metrics, name, target)
                positions = selected_positions(len(frame))
                images, image_map = read_window_images(cap, frame, positions, tiff, name)
                image_trajectory_figure(name, frame, images, image_map, row)
                continuous_burst_gif(name, frame, image_map, positions)
                frame_tables.append(images)
                summaries.append(summarize(name, frame, images, row))
    finally:
        cap.release()
    all_frames = pd.concat(frame_tables, ignore_index=True)
    all_frames.to_csv(DEST / "no14_image_trajectory_frame_checks.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(DEST / "no14_image_trajectory_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
