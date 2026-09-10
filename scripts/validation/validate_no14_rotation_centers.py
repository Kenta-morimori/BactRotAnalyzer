"""Create a qualitative No.14 rotation-center comparison video.

This is an intentionally standalone validation tool.  It does not modify the
repellent-response pipeline or its analytical outputs.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import param  # noqa: E402
from utils.functions import get_centroid_coordinate  # noqa: E402


WINDOW_WIDTHS_SEC = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_SAMPLE_NO = 14
DEFAULT_FRAME_STRIDE = 10
DEFAULT_OUTPUT_FPS = 20.0


def estimate_window_center(time, x, y, frame_index, window_width_sec):
    """Estimate one center using only finite points in the trailing time window."""
    t = np.asarray(time, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    end = min(int(frame_index) + 1, len(t), len(x_arr), len(y_arr))
    if end <= 0:
        return np.nan, np.nan, 0, False, np.empty((0, 2), dtype=float)

    start_time = t[end - 1] - float(window_width_sec)
    start = int(np.searchsorted(t[:end], start_time, side="left"))
    wx = x_arr[start:end]
    wy = y_arr[start:end]
    finite = np.isfinite(wx) & np.isfinite(wy)
    points = np.column_stack([wx[finite], wy[finite]])
    if len(points) < 5:
        return np.nan, np.nan, len(points), False, points

    center_x, center_y, _, _, warning = get_centroid_coordinate.calculate_ellipse_properties(
        points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1)
    )
    fit_ok = bool(np.isfinite(center_x) and np.isfinite(center_y) and not warning)
    if not fit_ok:
        center_x, center_y = np.nan, np.nan
    return float(center_x), float(center_y), len(points), fit_ok, points


def _draw_text(image, text, line, color=(255, 255, 255)):
    cv2.putText(image, text, (6, 18 + line * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _um_to_pixel(points_um, px2um_x, px2um_y):
    points = np.asarray(points_um, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    result = np.column_stack([points[:, 0] / px2um_x, points[:, 1] / px2um_y])
    return np.rint(result).astype(np.int32)


def _draw_candidate(frame, points_um, center_x, center_y, fit_ok, window_width_sec, px2um_x, px2um_y):
    panel = frame.copy()
    pixel_points = _um_to_pixel(points_um, px2um_x, px2um_y)
    if len(pixel_points) >= 2:
        cv2.polylines(panel, [pixel_points.reshape(-1, 1, 2)], False, (0, 220, 255), 1, cv2.LINE_AA)
    if len(pixel_points) >= 1:
        cv2.circle(panel, tuple(pixel_points[-1]), 3, (0, 0, 255), -1, cv2.LINE_AA)
    if fit_ok:
        center_px = _um_to_pixel([[center_x, center_y]], px2um_x, px2um_y)[0]
        cv2.drawMarker(panel, tuple(center_px), (0, 255, 0), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        status = "fit OK"
        status_color = (0, 255, 0)
    else:
        status = "fit unavailable"
        status_color = (0, 0, 255)
    _draw_text(panel, f"window {window_width_sec:.1f} s", 0)
    _draw_text(panel, status, 1, status_color)
    return panel


def _make_canvas(panels):
    panel_height, panel_width = panels[0].shape[:2]
    title_height = 0
    canvas = np.zeros((panel_height * 2 + title_height, panel_width * 3, 3), dtype=np.uint8)
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, 3)
        canvas[row * panel_height : (row + 1) * panel_height, col * panel_width : (col + 1) * panel_width] = panel
    return canvas


def load_no14_pre_rise_inputs(day, sample_no=DEFAULT_SAMPLE_NO):
    sample_index = int(sample_no) - 1
    if sample_index < 0:
        raise ValueError("sample_no must be 1-based and positive")
    output_root = Path(param.save_dir_bef) / day / "repellent_response"
    range_path = output_root / "00_time_list" / "analysis_frame_ranges.csv"
    rise_path = output_root / "01_brightness_change" / "rise_summary.csv"
    time_path = output_root / "00_all_rotational_analysis" / "time_list.csv"
    raw_centroid_path = Path(param.save_dir_bef) / day / "centroid_coordinate.csv"
    for path in (range_path, rise_path, time_path, raw_centroid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis output not found: {path}")

    range_row = pd.read_csv(range_path).query("sample_no == @sample_no")
    if len(range_row) != 1:
        raise ValueError(f"Expected exactly one frame-range row for No.{sample_no}")
    range_row = range_row.iloc[0]
    source_start = int(range_row["source_start_index_0based"])
    source_end = int(range_row["source_end_index_0based"])

    time = pd.read_csv(time_path)[f"No.{sample_no}"].dropna().to_numpy(dtype=float)
    raw = pd.read_csv(raw_centroid_path)
    x = raw[f"x_{sample_no}"].to_numpy(dtype=float)[source_start : source_end + 1]
    y = raw[f"y_{sample_no}"].to_numpy(dtype=float)[source_start : source_end + 1]
    n = min(len(time), len(x), len(y))
    time, x, y = time[:n], x[:n], y[:n]

    rise_row = pd.read_csv(rise_path).query("sample_no == @sample_no")
    if len(rise_row) != 1 or not np.isfinite(rise_row.iloc[0]["rise_time"]):
        raise ValueError(f"No finite rise time found for No.{sample_no}")
    rise_time = float(rise_row.iloc[0]["rise_time"])
    pre_end = int(np.searchsorted(time, rise_time, side="left"))
    if pre_end < 5:
        raise ValueError(f"No.{sample_no} does not contain enough pre-rise frames")

    sample_name = param.get_tiffinfo_config(day)[sample_index]
    avi_path = Path(param.input_dir_bef) / day / f"{sample_name}.avi"
    if not avi_path.is_file():
        raise FileNotFoundError(f"Source AVI not found: {avi_path}")
    return time[:pre_end], x[:pre_end], y[:pre_end], source_start, avi_path


def create_validation_video(
    day, sample_no=DEFAULT_SAMPLE_NO, frame_stride=DEFAULT_FRAME_STRIDE, output_fps=DEFAULT_OUTPUT_FPS
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    time, x, y, source_start, avi_path = load_no14_pre_rise_inputs(day, sample_no)
    px2um_x, px2um_y = param.get_px2um_config(day)
    output_dir = Path(param.save_dir_bef) / day / "validation" / "no14_rotation_center_windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "pre_rise_window_comparison.mp4"
    csv_path = output_dir / "window_center_estimates.csv"

    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open AVI: {avi_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)
    ok, initial_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read source frame {source_start} from {avi_path}")
    frame_height, frame_width = initial_frame.shape[:2]
    panel_size = (frame_width * 2, frame_height * 2)
    output_size = (panel_size[0] * 3, panel_size[1] * 2)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, output_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video: {video_path}")

    rows = []
    try:
        for local_index in range(len(time)):
            frame = initial_frame if local_index == 0 else cap.read()[1]
            if frame is None:
                raise RuntimeError(f"AVI ended before analysis frame {local_index}")
            if local_index % frame_stride != 0:
                continue

            raw_panel = cv2.resize(frame, panel_size, interpolation=cv2.INTER_NEAREST)
            panels = [raw_panel]
            for width in WINDOW_WIDTHS_SEC:
                center_x, center_y, point_count, fit_ok, points = estimate_window_center(time, x, y, local_index, width)
                candidate = _draw_candidate(frame, points, center_x, center_y, fit_ok, width, px2um_x, px2um_y)
                candidate = cv2.resize(candidate, panel_size, interpolation=cv2.INTER_NEAREST)
                panels.append(candidate)
                rows.append(
                    {
                        "sample_no": sample_no,
                        "window_width_sec": width,
                        "analysis_frame_index_0based": local_index,
                        "source_frame_index_0based": source_start + local_index,
                        "source_frame_no_1based": source_start + local_index + 1,
                        "time_sec": time[local_index],
                        "finite_point_count": point_count,
                        "center_x_um": center_x,
                        "center_y_um": center_y,
                        "fit_ok": fit_ok,
                    }
                )
            writer.write(_make_canvas(panels))
    finally:
        cap.release()
        writer.release()

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return video_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Validate No.14 rotation centers across local window widths.")
    parser.add_argument("--day", default="repellent-response/23")
    parser.add_argument("--sample-no", type=int, default=DEFAULT_SAMPLE_NO)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--output-fps", type=float, default=DEFAULT_OUTPUT_FPS)
    args = parser.parse_args()
    video_path, csv_path = create_validation_video(args.day, args.sample_no, args.frame_stride, args.output_fps)
    print(f"Saved video: {video_path}")
    print(f"Saved centers: {csv_path}")


if __name__ == "__main__":
    main()
