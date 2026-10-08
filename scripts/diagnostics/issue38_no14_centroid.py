"""Reproduce No.14 raw-centroid extraction and compare legacy trajectories."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.functions import raw_centroid, read_csv  # noqa: E402
from utils.functions.get_centroid_coordinate import contours  # noqa: E402

DAY = "repellent-response/23"
SAMPLE = 14
START, END = 2049, 16384
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / DAY
REPORT = ROOT / "docs/analysis/issue-38"


def legacy_series():
    initial_relative = pd.read_csv(OUTPUT / "centroid_coordinate.csv")
    initial_center = pd.read_csv(OUTPUT / "center_coordinate/bef_correction/center_coordinate.csv")
    stage_relative = pd.read_csv(
        OUTPUT / "repellent_response/00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv"
    )
    stage_center = pd.read_csv(
        OUTPUT / "repellent_response/00_all_rotational_analysis/center_coordinate/center_coordinate.csv"
    )
    a, b = START - 1, END
    initial = np.column_stack([
        initial_relative[f"{axis}_{SAMPLE}"].to_numpy(float)[a:b]
        + initial_center[f"No.{SAMPLE}_{axis}"].to_numpy(float)[a:b]
        for axis in "xy"
    ])
    stage = np.column_stack([
        stage_relative[f"No.{SAMPLE}_{axis}"].to_numpy(float)[: b - a]
        + stage_center[f"No.{SAMPLE}_{axis}"].to_numpy(float)[: b - a]
        for axis in "xy"
    ])
    return initial, stage


def summarize_difference(name: str, old: np.ndarray, new: np.ndarray) -> dict:
    delta = old - new
    offset = np.nanmedian(delta, axis=0)
    residual = delta - offset
    return {
        "series": name,
        "median_offset_x_um": offset[0],
        "median_offset_y_um": offset[1],
        "p95_abs_x_after_offset_um": np.nanquantile(abs(residual[:, 0]), 0.95),
        "p95_abs_y_after_offset_um": np.nanquantile(abs(residual[:, 1]), 0.95),
        "median_2d_after_offset_um": np.nanmedian(np.linalg.norm(residual, axis=1)),
        "p95_2d_after_offset_um": np.nanquantile(np.linalg.norm(residual, axis=1), 0.95),
    }


def checked_frames(frame: pd.DataFrame, comparison: pd.DataFrame) -> list[int]:
    times = frame.time_sec.to_numpy(float)
    selected = {1, START, END, len(frame)}
    for second in (20, 30, 38, 40, 42, 50, 75):
        selected.add(int(np.searchsorted(times, second)) + 1)
    stage_gap = comparison.stage_minus_initial_2d_um.to_numpy(float)
    selected.add(START + int(np.nanargmax(stage_gap)))
    step = np.hypot(np.diff(frame.x_px), np.diff(frame.y_px))
    selected.add(int(np.nanargmax(step)) + 2)
    return sorted(i for i in selected if 1 <= i <= len(frame))


def plot_outputs(frame: pd.DataFrame, comparison: pd.DataFrame, numbers: list[int]) -> None:
    xy = frame[["x_um", "y_um"]].to_numpy(float)
    t = frame.time_sec.to_numpy(float)
    subset = comparison
    sl = slice(START - 1, END)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for axis_idx, axis_name in enumerate("xy"):
        axes[axis_idx].plot(t, xy[:, axis_idx], lw=0.55, label="AVI canonical")
        for name in ("initial", "stage"):
            legacy = subset[f"{name}_{axis_name}_um"].to_numpy(float)
            offset = np.nanmedian(legacy - xy[sl, axis_idx])
            axes[axis_idx].plot(t[sl], legacy - offset, lw=0.4, alpha=0.65, label=f"{name} minus median offset")
        axes[axis_idx].set_ylabel(f"{axis_name} (um)")
        axes[axis_idx].legend(loc="upper right", fontsize=8)
    axes[2].plot(t[sl], subset.stage_minus_initial_2d_um, lw=0.5)
    axes[2].set_ylabel("legacy stage-initial (um)")
    axes[2].set_xlabel("TIFF time (s)")
    fig.tight_layout()
    fig.savefig(REPORT / "no14_raw_centroid_timeseries.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.plot(frame.x_um, frame.y_um, lw=0.5, color="black", alpha=0.5)
    axis.scatter(frame.x_um.iloc[np.array(numbers) - 1], frame.y_um.iloc[np.array(numbers) - 1],
                 c=frame.time_sec.iloc[np.array(numbers) - 1], cmap="viridis", s=28, zorder=3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("AVI local x (um)")
    axis.set_ylabel("AVI local y (um)")
    fig.tight_layout()
    fig.savefig(REPORT / "no14_raw_centroid_trajectory.png", dpi=140)
    plt.close(fig)

    capture = cv2.VideoCapture(str(raw_centroid.source_path(DAY, SAMPLE)))
    fig, axes = plt.subplots(5, 3, figsize=(11, 16))
    overlays = []
    try:
        for axis, number in zip(axes.ravel(), numbers):
            capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
            ok, image = capture.read()
            if not ok:
                raise OSError(f"Cannot read AVI frame {number}")
            x, y, _ = contours(image)
            saved = frame.iloc[number - 1]
            error = np.hypot(x - saved.x_px, y - saved.y_px)
            overlays.append({"source_frame_1based": number, "time_sec": saved.time_sec,
                             "saved_x_px": saved.x_px, "saved_y_px": saved.y_px,
                             "remeasured_x_px": x, "remeasured_y_px": y,
                             "error_px": error})
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.plot(saved.x_px, saved.y_px, marker="+", color="red", ms=10, mew=1.5)
            axis.set_title(f"frame {number}, {saved.time_sec:.2f}s", fontsize=9)
            axis.set_axis_off()
        for axis in axes.ravel()[len(numbers):]:
            axis.set_axis_off()
    finally:
        capture.release()
    fig.tight_layout()
    fig.savefig(REPORT / "no14_raw_centroid_video_overlay.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(overlays).to_csv(REPORT / "no14_raw_centroid_video_checks.csv", index=False)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    times = read_csv.get_timelist(DAY)[SAMPLE - 1]
    frame = raw_centroid.extract_or_load(DAY, SAMPLE, times)
    if len(frame) != 18432 or not frame.detected.all():
        raise ValueError("No.14 frame count or detection flags need manual review")
    initial, stage = legacy_series()
    canonical = frame[["x_um", "y_um"]].to_numpy(float)[START - 1:END]
    if len(initial) != len(stage) or len(initial) != len(canonical):
        raise ValueError("Legacy series do not match selected source frames")
    selected_time = frame.time_sec.to_numpy(float)[START - 1:END]
    stage_time = pd.read_csv(
        OUTPUT / "repellent_response/00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv",
        usecols=[f"No.{SAMPLE}_time"],
    ).iloc[:len(selected_time), 0].to_numpy(float)
    if not np.allclose(stage_time, selected_time, atol=1e-9, rtol=0):
        raise ValueError("Repellent timestamps are not aligned to the source AVI frames")
    comparison = pd.DataFrame({"source_frame_1based": np.arange(START, END + 1),
                               "time_sec": selected_time,
                               "canonical_x_um": canonical[:, 0], "canonical_y_um": canonical[:, 1],
                               "initial_x_um": initial[:, 0], "initial_y_um": initial[:, 1],
                               "stage_x_um": stage[:, 0], "stage_y_um": stage[:, 1],
                               "stage_minus_initial_2d_um": np.linalg.norm(stage - initial, axis=1)})
    comparison.to_csv(REPORT / "no14_raw_centroid_legacy_comparison.csv", index=False)
    stats = pd.DataFrame([summarize_difference("initial", initial, canonical),
                          summarize_difference("stage", stage, canonical)])
    stats.to_csv(REPORT / "no14_raw_centroid_summary.csv", index=False)
    numbers = checked_frames(frame, comparison)
    plot_outputs(frame, comparison, numbers)
    print(stats.to_string(index=False))
    print(f"checked frames: {numbers}")


if __name__ == "__main__":
    main()
