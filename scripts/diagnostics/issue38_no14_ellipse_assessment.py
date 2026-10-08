"""Apply reusable ellipse candidates to No.14; compare numerical sensitivities."""

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
import tifffile
from issue38_no14_image_trajectory import (
    read_window_images,
    select_window,
    selected_positions,
)
from issue38_paths import FIGURES
from issue38_paths import METHOD_COLORS as COLORS
from issue38_paths import TABLES, WINDOWS, ensure_output_dirs

from utils.functions.rotation_center_candidates import ELLIPSE_METHODS, estimate_center
from utils.functions.rotation_center_diagnostics import (
    ellipse_points,
    geometric_sensitivity,
)


def sensitivity_table():
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    windows = pd.read_csv(TABLES / "no14_center_window_metrics.csv")
    records = []
    for index, window in enumerate(windows.itertuples()):
        selected = data.loc[(data.time_sec >= window.start_time_sec) & (data.time_sec < window.end_time_sec)]
        points = selected[["x_um", "y_um"]].to_numpy(float)
        for record in geometric_sensitivity(points):
            records.append(
                {
                    "start_time_sec": window.start_time_sec,
                    "end_time_sec": window.end_time_sec,
                    "source_first_frame_1based": int(selected.source_frame_1based.iloc[0]),
                    "source_last_frame_1based": int(selected.source_frame_1based.iloc[-1]),
                    **record,
                }
            )
        if index % 20 == 0:
            print(f"sensitivity windows {index + 1}/{len(windows)}", flush=True)
    return pd.DataFrame(records)


def representative_figures():
    data = pd.read_csv(TABLES / "no14_centroid_center_timeseries.csv")
    windows = pd.read_csv(TABLES / "no14_center_window_metrics.csv")
    comparison = pd.read_csv(TABLES / "no14_center_method_comparison.csv")
    root = Path(__file__).resolve().parents[2]
    avi = root / "data/repellent-response/23/2026_0802_182430.avi"
    cap = cv2.VideoCapture(str(avi))
    try:
        with tifffile.TiffFile(avi.with_suffix(".tif")) as tiff:
            for name, target in WINDOWS.items():
                frame, window = select_window(data, windows, name, target)
                positions = selected_positions(len(frame))
                checks, images = read_window_images(cap, frame, positions, tiff, name)
                points = frame[["x_um", "y_um"]].to_numpy(float)
                fits = (
                    {method: estimate_center(points, method) for method in ELLIPSE_METHODS}
                    if window is not None
                    else {}
                )
                fig = plt.figure(figsize=(15, 8))
                grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 2))
                axis = fig.add_subplot(grid[:, 2])
                for indices, color in zip(np.array_split(np.arange(len(frame)), 3), ("#999999", "#666666", "#222222")):
                    axis.scatter(frame.x_px.iloc[indices], frame.y_px.iloc[indices], s=4, color=color, alpha=0.4)
                for method, result in fits.items():
                    if result.ellipse is not None:
                        curve = ellipse_points(result.ellipse, count=720, endpoint=True)
                        axis.plot(
                            curve[:, 0] / 0.02,
                            curve[:, 1] / 0.02,
                            color=COLORS[method],
                            ls="-" if result.computationally_ok else "--",
                            label=f"{method}: {result.status}",
                        )
                    if result.center is not None:
                        axis.plot(
                            *(result.center / 0.02),
                            "+",
                            color=COLORS[method],
                            ms=12,
                            label=f"{method}: {result.status}" if result.ellipse is None else None,
                        )
                axis.plot(
                    frame.center_x_um.iloc[0] / 0.02,
                    frame.center_y_um.iloc[0] / 0.02,
                    "x",
                    color="#cc79a7",
                    ms=10,
                    label="current corrected / held",
                )
                axis.set(xlim=(30, 110), ylim=(110, 30), xlabel="AVI x (px)", ylabel="AVI y (px)")
                axis.set_aspect("equal")
                axis.legend(fontsize=8, loc="upper left")
                axis.grid(alpha=0.15)
                for j, position in enumerate((positions[0], positions[5], positions[10], positions[14])):
                    item = frame.iloc[position]
                    image_axis = fig.add_subplot(grid[j // 2, j % 2])
                    image_axis.imshow(images[position], cmap="gray", vmin=0, vmax=255)
                    image_axis.plot(item.x_px, item.y_px, "+", color="lime", ms=10)
                    image_axis.plot(
                        frame.center_x_um.iloc[0] / 0.02, frame.center_y_um.iloc[0] / 0.02, "x", color="#cc79a7", ms=10
                    )
                    axis.plot(item.x_px, item.y_px, "o", color="black", ms=4)
                    axis.annotate(str(j + 1), (item.x_px, item.y_px), xytext=(3, 3), textcoords="offset points")
                    for method, result in fits.items():
                        if result.center is not None:
                            image_axis.plot(*(result.center / 0.02), "+", color=COLORS[method], ms=10)
                    image_axis.set_title(
                        f"{j + 1}: AVI {int(item.source_frame_1based)} | {item.time_sec:.6f}s", fontsize=9
                    )
                    image_axis.set_xticks([])
                    image_axis.set_yticks([])
                if window is None:
                    details = "terminal hold: no complete forward window; no candidate ellipse fitted"
                else:
                    sample = comparison.loc[(comparison.start_time_sec - window.start_time_sec).abs() < 1e-8]
                    details = " | ".join(
                        f"{r.method}: CV P95/R={r.block_cv_p95_R:.3f}, center shift/R={r.center_max_shift_R:.3f}, "
                        f"CV computed={int(r.block_cv_computationally_ok)}/4"
                        for r in sample.itertuples()
                        if r.method in ELLIPSE_METHODS and pd.notna(r.block_cv_p95_R)
                    )
                fig.suptitle(
                    f"No.14 {name}: {frame.time_sec.iloc[0]:.3f}–{frame.time_sec.iloc[-1]:.3f}s\n"
                    "green + = centroid; colored + = candidate center; dashed = flagged computation",
                    fontsize=11,
                )
                fig.text(0.02, 0.015, details, fontsize=7, wrap=True)
                fig.tight_layout(rect=(0, 0.06, 1, 0.91))
                fig.subplots_adjust(hspace=0.3)
                fig.savefig(FIGURES / f"no14_ellipse_candidates_{name}.png", dpi=150)
                plt.close(fig)
                if checks.centroid_error_px.max() > 1e-9:
                    raise AssertionError("Displayed centroid differs from source")
    finally:
        cap.release()


def main():
    ensure_output_dirs()
    sensitivity_table().to_csv(TABLES / "no14_ellipse_sensitivity.csv", index=False)
    representative_figures()


if __name__ == "__main__":
    main()
