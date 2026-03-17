import configparser
import glob
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import threading

from utils import param
from utils.functions import (
    fluctuation_analysis,
    frequency_analysis,
    get_angular_velocity,
    get_centroid_coordinate,
    get_tiff_info,
    make_graph,
    read_csv,
    rot_df_manage,
    save2csv,
)

_param_io_lock = threading.Lock()


class RotationalAnalysisResult(TypedDict):
    valid_indices: List[int]
    time_list: List[List[float]]
    angle_list: List[List[float]]
    angular_velocity_list: List[List[float]]


def _safe_extract_number(filename: str) -> float:
    try:
        return get_tiff_info.extract_number(filename)
    except Exception:
        return math.inf


def get_tiff_sample_names(day: str) -> List[str]:
    config_path = f"{param.input_dir_bef}/{day}/config.ini"
    if os.path.isfile(config_path):
        try:
            return param.get_tiffinfo_config(day)
        except Exception:
            pass

    tiff_root = f"{param.input_dir_bef}/{day}/tiff_data"
    if not os.path.isdir(tiff_root):
        raise FileNotFoundError(f"tiff_data directory not found: {tiff_root}")
    return sorted([name for name in os.listdir(tiff_root) if os.path.isdir(os.path.join(tiff_root, name))])


def ensure_repellent_config(day: str, default_frame_rate: float = 200.0, default_px2um: float = 1.0) -> str:
    config_path = f"{param.input_dir_bef}/{day}/config.ini"

    cfg = configparser.ConfigParser()
    if os.path.isfile(config_path):
        cfg.read(config_path)
        if cfg.has_section("Settings") and cfg.has_section("Tiff_info"):
            return config_path

    input_dir = f"{param.input_dir_bef}/{day}"
    avi_names = sorted([name for name in os.listdir(input_dir) if name.lower().endswith(".avi")])
    sample_num = len(avi_names)
    if sample_num == 0:
        raise FileNotFoundError(f"No .avi file found in {input_dir}")

    tiff_names = get_tiff_sample_names(day)
    if len(tiff_names) == 0:
        raise FileNotFoundError(f"No tiff sample directory found in {input_dir}/tiff_data")

    if len(tiff_names) < sample_num:
        tiff_names = tiff_names + [tiff_names[-1]] * (sample_num - len(tiff_names))
    else:
        tiff_names = tiff_names[:sample_num]

    cfg = configparser.ConfigParser()
    cfg["Settings"] = {
        "sample_num": str(sample_num),
        "FrameRate": str(default_frame_rate),
        "total_time": "1",
        "flag_use_tiff_log": "True",
        "px2um_x": str(default_px2um),
        "px2um_y": str(default_px2um),
    }
    cfg["Tiff_info"] = {"tiff_data": ", ".join(tiff_names)}
    cfg["RepellentResponse"] = {"post_rise_center_mode": "2"}
    with open(config_path, "w", encoding="utf-8") as fp:
        cfg.write(fp)
    return config_path


def get_post_rise_center_mode(day: str, default_mode: int = 2) -> int:
    config_path = f"{param.input_dir_bef}/{day}/config.ini"
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    section = "RepellentResponse"
    changed = False
    if not cfg.has_section(section):
        cfg.add_section(section)
        changed = True
    if not cfg.has_option(section, "post_rise_center_mode"):
        cfg.set(section, "post_rise_center_mode", str(default_mode))
        changed = True

    raw_mode = cfg.get(section, "post_rise_center_mode", fallback=str(default_mode))
    try:
        mode = int(raw_mode)
    except (ValueError, TypeError):
        mode = int(default_mode)
    if mode not in (1, 2, 3):
        mode = int(default_mode)
    if cfg.get(section, "post_rise_center_mode", fallback="") != str(mode):
        cfg.set(section, "post_rise_center_mode", str(mode))
        changed = True

    if changed:
        with open(config_path, "w", encoding="utf-8") as fp:
            cfg.write(fp)
    return mode


def get_background_intensity_time_series(day: str, roi_size: int = 10) -> List[List[float]]:
    tiff_root = f"{param.input_dir_bef}/{day}/tiff_data"
    sample_names = get_tiff_sample_names(day)
    background_list: List[List[float]] = []

    for sample_name in sample_names:
        sample_dir = os.path.join(tiff_root, sample_name)
        if not os.path.isdir(sample_dir):
            background_list.append([])
            continue

        frame_names = [
            name for name in os.listdir(sample_dir) if name.lower().endswith(".tif") or name.lower().endswith(".tiff")
        ]
        frame_names = sorted(frame_names, key=_safe_extract_number)

        sample_bg: List[float] = []
        for frame_name in frame_names:
            frame_path = os.path.join(sample_dir, frame_name)
            with Image.open(frame_path) as img:
                frame_arr = np.asarray(img)
            if frame_arr.ndim >= 3:
                frame_arr = frame_arr[..., 0]
            h, w = frame_arr.shape[:2]
            roi_h = min(roi_size, h)
            roi_w = min(roi_size, w)
            roi = frame_arr[:roi_h, w - roi_w : w]
            sample_bg.append(float(np.mean(roi)))
        background_list.append(sample_bg)
    return background_list


def detect_rise_index(
    values: Sequence[float],
    baseline_ratio: float = 0.5,
    sigma_threshold: float = 3.0,
    min_consecutive: int = 3,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "rise_index": np.nan,
            "baseline_mean": np.nan,
            "baseline_std": np.nan,
            "threshold": np.nan,
        }

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return {
            "rise_index": np.nan,
            "baseline_mean": np.nan,
            "baseline_std": np.nan,
            "threshold": np.nan,
        }

    # Interpolate NaN to avoid breaking whole-series change-point evaluation.
    arr_work = arr.copy()
    if not np.all(finite_mask):
        interp_idx = np.arange(arr_work.size)
        arr_work[~finite_mask] = np.interp(interp_idx[~finite_mask], interp_idx[finite_mask], arr_work[finite_mask])

    n = arr_work.size
    min_segment_ratio = max(0.05, min(0.25, baseline_ratio / 2.0))
    min_segment = max(min_consecutive, int(n * min_segment_ratio))
    min_segment = min(min_segment, max(1, n // 2))

    if n < 2 * min_segment + 1:
        min_segment = max(1, n // 4)
    if n < 3 or min_segment < 1 or (n - min_segment) <= min_segment:
        baseline = arr_work[np.isfinite(arr_work)]
        baseline_mean = float(np.mean(baseline)) if baseline.size else np.nan
        baseline_std = float(np.std(baseline)) if baseline.size else np.nan
        return {
            "rise_index": np.nan,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "threshold": np.nan,
        }

    candidate_indices = range(min_segment, n - min_segment + 1)
    scores = []
    for k in candidate_indices:
        before = arr_work[:k]
        after = arr_work[k:]
        scores.append(float(np.mean(after) - np.mean(before)))

    scores_arr = np.asarray(scores, dtype=float)
    best_local_idx = int(np.argmax(scores_arr))
    best_k = min_segment + best_local_idx
    best_score = float(scores_arr[best_local_idx])

    if not np.isfinite(best_score) or best_score <= 0:
        baseline = arr_work[:best_k]
        baseline_mean = float(np.mean(baseline)) if baseline.size else np.nan
        baseline_std = float(np.std(baseline)) if baseline.size else np.nan
        return {
            "rise_index": np.nan,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "threshold": np.nan,
        }

    # Find the strongest rising region, then backtrack to its onset.
    smooth_window = min(11, n if n % 2 == 1 else n - 1)
    smooth_window = max(3, smooth_window)
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    smooth = np.convolve(arr_work, kernel, mode="same")
    diff = np.diff(smooth)

    search_half_width = max(5, smooth_window * 2)
    left = max(0, best_k - search_half_width)
    right = min(diff.size, best_k + search_half_width)
    if right > left and np.nanmax(diff[left:right]) > 0:
        peak_slope_idx = int(left + np.nanargmax(diff[left:right]))
    else:
        peak_slope_idx = int(max(0, best_k - 1))

    baseline_diff_end = max(min_segment, peak_slope_idx // 2)
    baseline_diff = diff[:baseline_diff_end]
    baseline_diff = baseline_diff[np.isfinite(baseline_diff)]
    if baseline_diff.size == 0:
        diff_threshold = float(np.nanpercentile(diff, 60))
    else:
        diff_threshold = float(np.mean(baseline_diff) + 0.5 * sigma_threshold * np.std(baseline_diff))
        diff_threshold = max(diff_threshold, float(np.nanpercentile(diff, 60)))

    backtrack_width = max(min_segment * 4, smooth_window * 3)
    start_search = max(0, peak_slope_idx - backtrack_width)
    local_diff = diff[start_search : peak_slope_idx + 1]

    rise_index = peak_slope_idx
    if local_diff.size > 0 and np.isfinite(np.nanmax(local_diff)):
        active_threshold = max(diff_threshold, 0.35 * float(np.nanmax(local_diff)))
        active_mask = local_diff > active_threshold

        runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for i, flag in enumerate(active_mask):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                runs.append((run_start, i - 1))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(active_mask) - 1))

        if runs:
            # Prefer the rising block closest to the peak slope.
            selected_run = max(runs, key=lambda x: x[1])
            if (selected_run[1] - selected_run[0] + 1) < min_consecutive:
                selected_run = max(runs, key=lambda x: np.nanmax(local_diff[x[0] : x[1] + 1]))
            rise_index = int(start_search + selected_run[0])

    # Fallback to first intensity crossing before the slope peak.
    if rise_index >= peak_slope_idx:
        rough_baseline = arr_work[: max(1, peak_slope_idx // 2)]
        rough_baseline = rough_baseline[np.isfinite(rough_baseline)]
        if rough_baseline.size > 0:
            base_mean = float(np.mean(rough_baseline))
            base_std = float(np.std(rough_baseline))
            intensity_threshold = base_mean + max(base_std, best_score / max(4.0, sigma_threshold))
            crossings = np.where(smooth[start_search : peak_slope_idx + 1] > intensity_threshold)[0]
            if crossings.size > 0:
                rise_index = int(start_search + crossings[0])

    baseline = arr_work[: max(1, rise_index)]
    baseline_mean = float(np.mean(baseline)) if baseline.size else np.nan
    baseline_std = float(np.std(baseline)) if baseline.size else np.nan
    threshold = baseline_mean + sigma_threshold * baseline_std
    return {
        "rise_index": float(rise_index),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold": float(threshold),
    }


def detect_rise_points(
    background_list: Sequence[Sequence[float]],
    baseline_ratio: float = 0.5,
    sigma_threshold: float = 3.0,
    min_consecutive: int = 3,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for i, values in enumerate(background_list):
        result = detect_rise_index(
            values=values,
            baseline_ratio=baseline_ratio,
            sigma_threshold=sigma_threshold,
            min_consecutive=min_consecutive,
        )
        result["sample_no"] = float(i + 1)
        result["num_frames"] = float(len(values))
        results.append(result)
    return results


def _normalize_rise_index(rise_index: float, max_len: int) -> int:
    if max_len <= 0:
        return 0
    if rise_index is None or not np.isfinite(rise_index):
        return max_len
    rise_idx = int(rise_index)
    return max(0, min(max_len, rise_idx))


def build_pre_rise_fluctuation_inputs(
    time_list: Sequence[Sequence[float]],
    angular_velocity_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]]]:
    n = min(len(time_list), len(angular_velocity_list), len(rise_indices))
    pre_time_list: List[List[float]] = []
    pre_angular_velocity_list: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        av_arr = np.asarray(angular_velocity_list[i], dtype=float)
        max_frame_len = min(time_arr.size, av_arr.size + 1)
        if max_frame_len <= 0:
            pre_time_list.append([])
            pre_angular_velocity_list.append([])
            continue

        time_arr = time_arr[:max_frame_len]
        av_arr = av_arr[: max_frame_len - 1]

        rise_idx = _normalize_rise_index(float(rise_indices[i]), max_frame_len)
        pre_time = time_arr[:rise_idx]
        pre_av = av_arr[: max(0, rise_idx - 1)]

        pre_time_list.append(pre_time.tolist())
        pre_angular_velocity_list.append(pre_av.tolist())

    return pre_time_list, pre_angular_velocity_list


def build_pre_rise_centroid_series(
    time_list: Sequence[Sequence[float]],
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    n = min(len(time_list), len(x_list), len(y_list), len(rise_indices))
    pre_time_list: List[List[float]] = []
    pre_x_list: List[List[float]] = []
    pre_y_list: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        max_len = min(time_arr.size, x_arr.size, y_arr.size)

        if max_len <= 0:
            pre_time_list.append([])
            pre_x_list.append([])
            pre_y_list.append([])
            continue

        time_arr = time_arr[:max_len]
        x_arr = x_arr[:max_len]
        y_arr = y_arr[:max_len]
        rise_idx = _normalize_rise_index(float(rise_indices[i]), max_len)

        pre_time_list.append(time_arr[:rise_idx].tolist())
        pre_x_list.append(x_arr[:rise_idx].tolist())
        pre_y_list.append(y_arr[:rise_idx].tolist())

    return pre_time_list, pre_x_list, pre_y_list


def load_centroid_coordinate_with_nan(day: str) -> Tuple[List[List[float]], List[List[float]]]:
    csv_path = f"{param.save_dir_bef}/{day}/centroid_coordinate.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"centroid_coordinate.csv not found: {csv_path}. "
            "Run centroid extraction first (e.g., rotation_analysis_main.py)."
        )

    df = pd.read_csv(csv_path)
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    for i, col_name in enumerate(df.columns.tolist()):
        col_arr = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
        if i % 2 == 0:
            x_list.append(col_arr.tolist())
        else:
            y_list.append(col_arr.tolist())
    return x_list, y_list


def load_rotation_center_with_nan(day: str) -> Tuple[List[List[float]], List[List[float]]]:
    candidate_paths = [
        f"{param.save_dir_bef}/{day}/center_coordinate/center_coordinate.csv",
        f"{param.save_dir_bef}/{day}/center_coordinate/bef_correction/center_coordinate.csv",
        f"{param.save_dir_bef}/{day}/center_coordinate/bef_correction/center_coordinate_bef_correct.csv",
    ]

    csv_path = ""
    for path in candidate_paths:
        if os.path.isfile(path):
            csv_path = path
            break
    if csv_path == "":
        return [], []

    df = pd.read_csv(csv_path)
    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    for col_name in df.columns.tolist():
        col_arr = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
        if col_name.endswith("_x"):
            x_list.append(col_arr.tolist())
        elif col_name.endswith("_y"):
            y_list.append(col_arr.tolist())
    return x_list, y_list


def _fill_center_nan_with_previous(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if arr.size == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros(arr.size, dtype=float)

    valid = np.where(np.isfinite(arr))[0]
    first_valid = int(valid[0])
    arr[:first_valid] = arr[first_valid]
    for i in range(first_valid + 1, arr.size):
        if not np.isfinite(arr[i]):
            arr[i] = arr[i - 1]
    return arr


def build_all_time_raw_centroid_series(
    day: str,
    time_list: Sequence[Sequence[float]],
    corrected_x_list: Sequence[Sequence[float]],
    corrected_y_list: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    center_x_list, center_y_list = load_rotation_center_with_nan(day)
    n = min(len(time_list), len(corrected_x_list), len(corrected_y_list))

    out_time_list: List[List[float]] = []
    out_x_raw_list: List[List[float]] = []
    out_y_raw_list: List[List[float]] = []

    for i in range(n):
        t = np.asarray(time_list[i], dtype=float)
        x_corr = np.asarray(corrected_x_list[i], dtype=float)
        y_corr = np.asarray(corrected_y_list[i], dtype=float)

        if i < len(center_x_list) and i < len(center_y_list):
            cx = _fill_center_nan_with_previous(center_x_list[i])
            cy = _fill_center_nan_with_previous(center_y_list[i])
            m = min(len(t), len(x_corr), len(y_corr), len(cx), len(cy))
            cx = cx[:m]
            cy = cy[:m]
        else:
            m = min(len(t), len(x_corr), len(y_corr))
            cx = np.zeros(m, dtype=float)
            cy = np.zeros(m, dtype=float)

        t = t[:m]
        x_corr = x_corr[:m]
        y_corr = y_corr[:m]

        out_time_list.append(t.tolist())
        out_x_raw_list.append((x_corr + cx).tolist())
        out_y_raw_list.append((y_corr + cy).tolist())

    return out_time_list, out_x_raw_list, out_y_raw_list


def build_post_rise_centroid_series(
    time_list: Sequence[Sequence[float]],
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    n = min(len(time_list), len(x_list), len(y_list), len(rise_indices))
    post_time_list: List[List[float]] = []
    post_x_list: List[List[float]] = []
    post_y_list: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        max_len = min(time_arr.size, x_arr.size, y_arr.size)

        if max_len <= 0:
            post_time_list.append([])
            post_x_list.append([])
            post_y_list.append([])
            continue

        time_arr = time_arr[:max_len]
        x_arr = x_arr[:max_len]
        y_arr = y_arr[:max_len]
        rise_idx = _normalize_rise_index(float(rise_indices[i]), max_len)

        post_time_list.append(time_arr[rise_idx:].tolist())
        post_x_list.append(x_arr[rise_idx:].tolist())
        post_y_list.append(y_arr[rise_idx:].tolist())

    return post_time_list, post_x_list, post_y_list


def build_post_rise_x_components(
    day: str,
    time_list: Sequence[Sequence[float]],
    corrected_x_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    center_x_list, _ = load_rotation_center_with_nan(day)
    flag_has_center = len(center_x_list) > 0

    n = min(len(time_list), len(corrected_x_list), len(rise_indices))
    post_time_list: List[List[float]] = []
    post_x_raw_list: List[List[float]] = []
    post_x_center_list: List[List[float]] = []
    post_x_corr_list: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_corr_arr = np.asarray(corrected_x_list[i], dtype=float)

        if flag_has_center and i < len(center_x_list):
            x_center_arr = np.asarray(center_x_list[i], dtype=float)
            m = min(len(time_arr), len(x_corr_arr), len(x_center_arr))
            time_arr = time_arr[:m]
            x_corr_arr = x_corr_arr[:m]
            x_center_arr = x_center_arr[:m]
            x_raw_arr = x_corr_arr + x_center_arr
        else:
            m = min(len(time_arr), len(x_corr_arr))
            time_arr = time_arr[:m]
            x_corr_arr = x_corr_arr[:m]
            x_center_arr = np.zeros(m, dtype=float)
            x_raw_arr = x_corr_arr.copy()

        if m <= 0:
            post_time_list.append([])
            post_x_raw_list.append([])
            post_x_center_list.append([])
            post_x_corr_list.append([])
            continue

        rise_idx = _normalize_rise_index(float(rise_indices[i]), m)
        post_time_list.append(time_arr[rise_idx:].tolist())
        post_x_raw_list.append(x_raw_arr[rise_idx:].tolist())
        post_x_center_list.append(x_center_arr[rise_idx:].tolist())
        post_x_corr_list.append(x_corr_arr[rise_idx:].tolist())

    return post_time_list, post_x_raw_list, post_x_center_list, post_x_corr_list


def build_post_rise_coordinate_components(
    day: str,
    time_list: Sequence[Sequence[float]],
    corrected_x_list: Sequence[Sequence[float]],
    corrected_y_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    center_x_list, center_y_list = load_rotation_center_with_nan(day)
    flag_has_center = (len(center_x_list) > 0) and (len(center_y_list) > 0)

    n = min(len(time_list), len(corrected_x_list), len(corrected_y_list), len(rise_indices))
    post_time_list: List[List[float]] = []
    post_x_before_list: List[List[float]] = []
    post_y_before_list: List[List[float]] = []
    post_x_center_list: List[List[float]] = []
    post_y_center_list: List[List[float]] = []
    post_x_corr_list: List[List[float]] = []
    post_y_corr_list: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_corr_arr = np.asarray(corrected_x_list[i], dtype=float)
        y_corr_arr = np.asarray(corrected_y_list[i], dtype=float)

        if flag_has_center and (i < len(center_x_list)) and (i < len(center_y_list)):
            x_center_arr = np.asarray(center_x_list[i], dtype=float)
            y_center_arr = np.asarray(center_y_list[i], dtype=float)
            m = min(len(time_arr), len(x_corr_arr), len(y_corr_arr), len(x_center_arr), len(y_center_arr))
            time_arr = time_arr[:m]
            x_corr_arr = x_corr_arr[:m]
            y_corr_arr = y_corr_arr[:m]
            x_center_arr = x_center_arr[:m]
            y_center_arr = y_center_arr[:m]
            if np.isfinite(x_center_arr).any():
                x_center_mean = float(np.nanmean(x_center_arr))
            elif np.isfinite(x_corr_arr).any():
                x_center_mean = float(np.nanmean(x_corr_arr))
            else:
                x_center_mean = 0.0
            if np.isfinite(y_center_arr).any():
                y_center_mean = float(np.nanmean(y_center_arr))
            elif np.isfinite(y_corr_arr).any():
                y_center_mean = float(np.nanmean(y_corr_arr))
            else:
                y_center_mean = 0.0
        else:
            m = min(len(time_arr), len(x_corr_arr), len(y_corr_arr))
            time_arr = time_arr[:m]
            x_corr_arr = x_corr_arr[:m]
            y_corr_arr = y_corr_arr[:m]

            # Temporary fallback requested by user: use sample-wide constant center.
            x_center_mean = float(np.nanmean(x_corr_arr)) if np.isfinite(x_corr_arr).any() else 0.0
            y_center_mean = float(np.nanmean(y_corr_arr)) if np.isfinite(y_corr_arr).any() else 0.0

        # Temporary behavior: always embed rotation center as a constant series.
        x_center_arr = np.full(m, x_center_mean, dtype=float)
        y_center_arr = np.full(m, y_center_mean, dtype=float)

        if m <= 0:
            post_time_list.append([])
            post_x_before_list.append([])
            post_y_before_list.append([])
            post_x_center_list.append([])
            post_y_center_list.append([])
            post_x_corr_list.append([])
            post_y_corr_list.append([])
            continue

        x_before_arr = x_corr_arr + x_center_arr
        y_before_arr = y_corr_arr + y_center_arr
        rise_idx = _normalize_rise_index(float(rise_indices[i]), m)

        post_time_list.append(time_arr[rise_idx:].tolist())
        post_x_before_list.append(x_before_arr[rise_idx:].tolist())
        post_y_before_list.append(y_before_arr[rise_idx:].tolist())
        post_x_center_list.append(x_center_arr[rise_idx:].tolist())
        post_y_center_list.append(y_center_arr[rise_idx:].tolist())
        post_x_corr_list.append(x_corr_arr[rise_idx:].tolist())
        post_y_corr_list.append(y_corr_arr[rise_idx:].tolist())

    return (
        post_time_list,
        post_x_before_list,
        post_y_before_list,
        post_x_center_list,
        post_y_center_list,
        post_x_corr_list,
        post_y_corr_list,
    )


def _write_temp_config(config_path: str, sample_num: int) -> None:
    cfg = configparser.ConfigParser()
    cfg["Settings"] = {
        "sample_num": str(sample_num),
        "FrameRate": "1",
        "total_time": "1",
        "flag_use_tiff_log": "True",
        "px2um_x": "1.0",
        "px2um_y": "1.0",
    }
    cfg["Tiff_info"] = {"tiff_data": ", ".join(["dummy"] * max(1, sample_num))}
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fp:
        cfg.write(fp)


def _write_dummy_angle_fft(day: str, sample_num: int) -> None:
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    for i in range(sample_num):
        data[f"No.{i + 1}_freq"] = pd.Series([0.0], dtype="float64")
        data[f"No.{i + 1}_Amp"] = pd.Series([1.0], dtype="float64")
    pd.DataFrame(data).to_csv(f"{save_dir}/angle_FFT.csv", index=False)


def _copytree_replace(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _save_centroid_coordinate_csv(
    day: str, x_list: Sequence[Sequence[float]], y_list: Sequence[Sequence[float]]
) -> None:
    n = min(len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/centroid_coordinate.csv"

    data = {}
    for i in range(n):
        data[f"x_{i + 1}"] = pd.Series(np.asarray(x_list[i], dtype=float))
        data[f"y_{i + 1}"] = pd.Series(np.asarray(y_list[i], dtype=float))
    pd.DataFrame(data).to_csv(csv_path, index=False)


def _prepare_valid_rotational_inputs(
    time_list: Sequence[Sequence[float]],
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[int]]:
    n = min(len(time_list), len(x_list), len(y_list))
    selected_time_list: List[List[float]] = []
    selected_x_list: List[List[float]] = []
    selected_y_list: List[List[float]] = []
    valid_indices: List[int] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)

        m = min(time_arr.size, x_arr.size, y_arr.size)
        if m < 3:
            continue
        time_arr = time_arr[:m]
        x_arr = x_arr[:m]
        y_arr = y_arr[:m]
        if not np.isfinite(time_arr[-1]) or not np.isfinite(time_arr[0]) or float(time_arr[-1]) <= float(time_arr[0]):
            continue

        selected_time_list.append(time_arr.tolist())
        selected_x_list.append(x_arr.tolist())
        selected_y_list.append(y_arr.tolist())
        valid_indices.append(i)

    return selected_time_list, selected_x_list, selected_y_list, valid_indices


def save_repellent_segment_centroid_series(
    time_list: Sequence[Sequence[float]],
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
    day: str,
    segment_subdir: str,
    csv_name: str = "centroid_time_series.csv",
) -> None:
    n = min(len(time_list), len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/{csv_name}"

    data = {}
    for i in range(n):
        t_arr = pd.Series(np.asarray(time_list[i], dtype=float), dtype="float64")
        x_arr = pd.Series(np.asarray(x_list[i], dtype=float), dtype="float64")
        y_arr = pd.Series(np.asarray(y_list[i], dtype=float), dtype="float64")
        m = min(len(t_arr), len(x_arr), len(y_arr))
        data[f"No.{i + 1}_time"] = t_arr.iloc[:m].reset_index(drop=True)
        data[f"No.{i + 1}_x"] = x_arr.iloc[:m].reset_index(drop=True)
        data[f"No.{i + 1}_y"] = y_arr.iloc[:m].reset_index(drop=True)
    pd.DataFrame(data).to_csv(csv_path, index=False)


def run_segment_rotational_analysis(
    day: str,
    segment_subdir: str,
    time_list: Sequence[Sequence[float]],
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
    run_fluctuation: bool = False,
) -> RotationalAnalysisResult:
    target_dir = f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}"
    os.makedirs(target_dir, exist_ok=True)

    selected_time_list, selected_x_list, selected_y_list, valid_indices = _prepare_valid_rotational_inputs(
        time_list=time_list,
        x_list=x_list,
        y_list=y_list,
    )
    if not valid_indices:
        pd.DataFrame([{"message": "No sample has enough data for rotational analysis.", "analyzed_samples": 0}]).to_csv(
            f"{target_dir}/summary.csv", index=False
        )
        return RotationalAnalysisResult(
            valid_indices=[],
            time_list=[],
            angle_list=[],
            angular_velocity_list=[],
        )

    tmp_day = f"repellent_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_input_root = os.path.join(tmp_root, "data")
        tmp_output_root = os.path.join(tmp_root, "outputs")
        os.makedirs(tmp_input_root, exist_ok=True)
        os.makedirs(tmp_output_root, exist_ok=True)
        _write_temp_config(f"{tmp_input_root}/{tmp_day}/config.ini", len(valid_indices))

        with _param_io_lock:
            orig_input_root = param.input_dir_bef
            orig_output_root = param.save_dir_bef
            try:
                param.input_dir_bef = tmp_input_root
                param.save_dir_bef = tmp_output_root

                save2csv.save_time_list(selected_time_list, tmp_day)
                _save_centroid_coordinate_csv(tmp_day, selected_x_list, selected_y_list)
                rot_df_manage.create_rot_df(tmp_day)
                _write_dummy_angle_fft(tmp_day, len(valid_indices))
                angle_list, angular_velocity_list = get_angular_velocity.get_angular_velocity(
                    selected_x_list, selected_y_list, tmp_day
                )

                if run_fluctuation:
                    fluctuation_analysis.main(angular_velocity_list, tmp_day)
                    make_graph.plot_rot_param(tmp_day)
            finally:
                param.input_dir_bef = orig_input_root
                param.save_dir_bef = orig_output_root

        src_base = f"{tmp_output_root}/{tmp_day}"
        if os.path.isdir(f"{src_base}/angular_velocity"):
            _copytree_replace(f"{src_base}/angular_velocity", f"{target_dir}/angular_velocity")
        if run_fluctuation and os.path.isdir(f"{src_base}/fluctuation_analysis"):
            _copytree_replace(f"{src_base}/fluctuation_analysis", f"{target_dir}/fluctuation_analysis")

    map_df = pd.DataFrame(
        {
            "segment_no": [i + 1 for i in range(len(valid_indices))],
            "original_sample_no": [idx + 1 for idx in valid_indices],
        }
    )
    map_df.to_csv(f"{target_dir}/sample_index_map.csv", index=False)

    time_data = {}
    for i, t_series in enumerate(selected_time_list):
        time_data[f"No.{i + 1}"] = pd.Series(np.asarray(t_series, dtype=float), dtype="float64")
    pd.DataFrame(time_data).to_csv(f"{target_dir}/time_list.csv", index=False)
    return RotationalAnalysisResult(
        valid_indices=valid_indices,
        time_list=selected_time_list,
        angle_list=[np.asarray(a, dtype=float).tolist() for a in angle_list],
        angular_velocity_list=[np.asarray(v, dtype=float).tolist() for v in angular_velocity_list],
    )


def _correct_center_outlier_like_standard(center_series: Sequence[float]) -> List[float]:
    data = np.asarray(center_series, dtype=float).copy()
    if data.size == 0:
        return []

    complement_index_set = set()
    if param.mode_correct_center_outlier == 1:
        mean = float(np.nanmean(data))
        std = float(np.nanstd(data))
        lower_th = mean - param.num_std_center * std
        upper_th = mean + param.num_std_center * std
        complement_index_set.update(np.where((data < lower_th) | (data > upper_th))[0].tolist())
    elif param.mode_correct_center_outlier == 2:
        med = float(np.nanmedian(data))
        mad = float(np.nanmedian(np.abs(data - med)))
        eps = np.finfo(float).eps
        mz = 0.6745 * (data - med) / (mad + eps)
        complement_index_set.update(np.where(np.abs(mz) > 10.0)[0].tolist())

    if not complement_index_set:
        return data.tolist()

    data_aft = data.copy()
    n_neighbors = 2
    for idx in sorted(complement_index_set):
        neighbors = []
        for k in range(1, n_neighbors + 1):
            left = idx - k
            right = idx + k
            if left >= 0 and left not in complement_index_set and np.isfinite(data[left]):
                neighbors.append(float(data[left]))
            if right < len(data) and right not in complement_index_set and np.isfinite(data[right]):
                neighbors.append(float(data[right]))
        if neighbors:
            data_aft[idx] = float(np.nanmean(neighbors))
        else:
            valid = [float(v) for j, v in enumerate(data) if (j not in complement_index_set) and np.isfinite(v)]
            data_aft[idx] = float(np.nanmean(valid)) if valid else np.nan
    return data_aft.tolist()


def estimate_rotation_center_like_standard(
    time_list: Sequence[Sequence[float]],
    x_raw_list: Sequence[Sequence[float]],
    y_raw_list: Sequence[Sequence[float]],
    rise_indices: Optional[Sequence[float]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    n = min(len(time_list), len(x_raw_list), len(y_raw_list))
    center_x_all: List[List[float]] = []
    center_y_all: List[List[float]] = []

    for i in range(n):
        time_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_raw_list[i], dtype=float)
        y_arr = np.asarray(y_raw_list[i], dtype=float)
        m = min(len(time_arr), len(x_arr), len(y_arr))
        time_arr = time_arr[:m]
        x_arr = x_arr[:m]
        y_arr = y_arr[:m]

        if m <= 0:
            center_x_all.append([])
            center_y_all.append([])
            continue

        finite_time = np.isfinite(time_arr)
        finite_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
        valid_mask = finite_time & finite_xy
        if np.count_nonzero(valid_mask) < max(3, param.min_ref_centroid_num):
            center_x_mean = float(np.nanmean(x_arr)) if np.isfinite(x_arr).any() else 0.0
            center_y_mean = float(np.nanmean(y_arr)) if np.isfinite(y_arr).any() else 0.0
            center_x_all.append([center_x_mean] * m)
            center_y_all.append([center_y_mean] * m)
            continue

        t = time_arr[valid_mask]
        x = x_arr[valid_mask]
        y = y_arr[valid_mask]
        total_duration_all = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if total_duration_all <= 0:
            center_x_mean = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
            center_y_mean = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
            center_x_all.append([center_x_mean] * m)
            center_y_all.append([center_y_mean] * m)
            continue

        rise_idx = m
        if rise_indices is not None and i < len(rise_indices):
            rise_idx = _normalize_rise_index(float(rise_indices[i]), m)
        pre_mask = valid_mask & (np.arange(m) < rise_idx)
        fft_mask = pre_mask if np.count_nonzero(pre_mask) >= max(3, param.min_ref_centroid_num) else valid_mask
        fft_source = "pre-rise" if fft_mask is pre_mask else "all-time"

        t_fft = time_arr[fft_mask]
        x_fft = x_arr[fft_mask]
        y_fft = y_arr[fft_mask]
        total_duration_fft = float(t_fft[-1] - t_fft[0]) if len(t_fft) > 1 else 0.0
        if total_duration_fft <= 0:
            total_duration_fft = total_duration_all
            t_fft = t
            x_fft = x
            y_fft = y
            fft_source = "all-time"

        frame_rate_fft = max(1e-6, float(len(t_fft)) / total_duration_fft)
        frame_rate_all = max(1e-6, float(len(t)) / total_duration_all)
        x_freq, x_amp = frequency_analysis.fft(x_fft, 1.0 / frame_rate_fft)
        y_freq, y_amp = frequency_analysis.fft(y_fft, 1.0 / frame_rate_fft)
        freq_th = 5.0
        x_mask = np.asarray(x_freq) > freq_th
        y_mask = np.asarray(y_freq) > freq_th
        x_freq = np.asarray(x_freq)[x_mask]
        x_amp = np.asarray(x_amp)[x_mask]
        y_freq = np.asarray(y_freq)[y_mask]
        y_amp = np.asarray(y_amp)[y_mask]

        peak_candidates = []
        if x_freq.size > 0 and x_amp.size > 0 and np.isfinite(x_amp).any():
            peak_candidates.append(float(x_freq[int(np.nanargmax(x_amp))]))
        if y_freq.size > 0 and y_amp.size > 0 and np.isfinite(y_amp).any():
            peak_candidates.append(float(y_freq[int(np.nanargmax(y_amp))]))
        if peak_candidates:
            peak_freq = max(peak_candidates)
        else:
            peak_freq = max(0.1, 1.0 / max(total_duration_fft, 1e-6))

        width_time = param.n_rotations / max(peak_freq, 1e-6)
        width_time = min(max(width_time, 1.0 / frame_rate_fft), total_duration_fft)
        print(f"No.{i + 1} width_time ({fft_source}): {width_time:.3f} s")

        center_x_series: List[float] = []
        center_y_series: List[float] = []
        start_time = float(t[0])
        dt = 1.0 / frame_rate_all
        while True:
            cond = (t >= start_time) & (t < start_time + width_time)
            x_win = x[cond]
            y_win = y[cond]
            if len(x_win) < param.min_ref_centroid_num:
                center_x = np.nan
                center_y = np.nan
            else:
                center_x, center_y, _, _, _ = get_centroid_coordinate.calculate_ellipse_properties(
                    x_win.reshape(-1, 1),
                    y_win.reshape(-1, 1),
                )
            center_x_series.append(float(center_x))
            center_y_series.append(float(center_y))

            start_time += dt
            if start_time + width_time >= float(t[-1]):
                rest = len(t) - len(center_x_series)
                cx_valid = np.asarray(center_x_series, dtype=float)
                cy_valid = np.asarray(center_y_series, dtype=float)
                if np.isnan(cx_valid).any():
                    mean_x = float(np.nanmean(cx_valid)) if np.isfinite(cx_valid).any() else 0.0
                    cx_valid = np.where(np.isnan(cx_valid), mean_x, cx_valid)
                if np.isnan(cy_valid).any():
                    mean_y = float(np.nanmean(cy_valid)) if np.isfinite(cy_valid).any() else 0.0
                    cy_valid = np.where(np.isnan(cy_valid), mean_y, cy_valid)
                center_x_series = cx_valid.tolist()
                center_y_series = cy_valid.tolist()
                if rest > 0 and center_x_series:
                    center_x_series.extend([center_x_series[-1]] * rest)
                    center_y_series.extend([center_y_series[-1]] * rest)
                break

        if param.flag_correct_center_outlier:
            center_x_series = _correct_center_outlier_like_standard(center_x_series)
            center_y_series = _correct_center_outlier_like_standard(center_y_series)

        # Expand back to original length (including invalid points) by nearest previous value.
        cx_full = np.full(m, np.nan, dtype=float)
        cy_full = np.full(m, np.nan, dtype=float)
        cx_full[valid_mask] = np.asarray(center_x_series, dtype=float)[: np.count_nonzero(valid_mask)]
        cy_full[valid_mask] = np.asarray(center_y_series, dtype=float)[: np.count_nonzero(valid_mask)]
        for arr in [cx_full, cy_full]:
            if np.isnan(arr).all():
                arr[:] = 0.0
            else:
                first_valid = np.where(~np.isnan(arr))[0][0]
                arr[:first_valid] = arr[first_valid]
                for k in range(first_valid + 1, len(arr)):
                    if np.isnan(arr[k]):
                        arr[k] = arr[k - 1]

        center_x_all.append(cx_full.tolist())
        center_y_all.append(cy_full.tolist())

    return center_x_all, center_y_all


def _estimate_window_frames_from_fft(
    time_arr: np.ndarray, x_arr: np.ndarray, y_arr: np.ndarray, rise_idx: Optional[int] = None
) -> int:
    finite_time = np.isfinite(time_arr)
    finite_xy = np.isfinite(x_arr) & np.isfinite(y_arr)
    valid_mask = finite_time & finite_xy
    if np.count_nonzero(valid_mask) < max(3, param.min_ref_centroid_num):
        base = min(len(time_arr), len(x_arr), len(y_arr))
        return max(3, min(base if base % 2 == 1 else max(3, base - 1), 301))

    if rise_idx is None:
        rise_idx = len(time_arr)
    rise_idx = max(0, min(int(rise_idx), len(time_arr)))
    pre_mask = valid_mask & (np.arange(len(time_arr)) < rise_idx)
    fft_mask = pre_mask if np.count_nonzero(pre_mask) >= max(3, param.min_ref_centroid_num) else valid_mask

    t = time_arr[fft_mask]
    x = x_arr[fft_mask]
    y = y_arr[fft_mask]
    total_duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if total_duration <= 0:
        return 3

    frame_rate = max(1e-6, float(len(t)) / total_duration)
    x_freq, x_amp = frequency_analysis.fft(x, 1.0 / frame_rate)
    y_freq, y_amp = frequency_analysis.fft(y, 1.0 / frame_rate)
    freq_th = 5.0
    x_mask = np.asarray(x_freq) > freq_th
    y_mask = np.asarray(y_freq) > freq_th
    x_freq = np.asarray(x_freq)[x_mask]
    x_amp = np.asarray(x_amp)[x_mask]
    y_freq = np.asarray(y_freq)[y_mask]
    y_amp = np.asarray(y_amp)[y_mask]

    peak_candidates = []
    if x_freq.size > 0 and x_amp.size > 0 and np.isfinite(x_amp).any():
        peak_candidates.append(float(x_freq[int(np.nanargmax(x_amp))]))
    if y_freq.size > 0 and y_amp.size > 0 and np.isfinite(y_amp).any():
        peak_candidates.append(float(y_freq[int(np.nanargmax(y_amp))]))
    if peak_candidates:
        peak_freq = max(peak_candidates)
    else:
        peak_freq = max(0.1, 1.0 / max(total_duration, 1e-6))

    width_time = param.n_rotations / max(peak_freq, 1e-6)
    width_time = min(max(width_time, 1.0 / frame_rate), total_duration)
    frames = max(3, int(round(width_time * frame_rate)))
    if frames % 2 == 0:
        frames += 1
    max_valid = max(3, int(np.count_nonzero(valid_mask)))
    frames = min(frames, max_valid if max_valid % 2 == 1 else max(3, max_valid - 1))
    return max(3, frames)


def _rolling_stat(arr: np.ndarray, window: int, stat: Literal["mean", "median"]) -> np.ndarray:
    s = pd.Series(arr, dtype="float64")
    if stat == "mean":
        out = s.rolling(window=window, center=True, min_periods=1).mean()
    else:
        out = s.rolling(window=window, center=True, min_periods=1).median()
    return out.to_numpy(dtype=float)


def apply_post_rise_center_strategy(
    time_list: Sequence[Sequence[float]],
    x_raw_list: Sequence[Sequence[float]],
    y_raw_list: Sequence[Sequence[float]],
    center_x_standard_list: Sequence[Sequence[float]],
    center_y_standard_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
    post_rise_center_mode: int = 2,
) -> Tuple[List[List[float]], List[List[float]]]:
    n = min(
        len(time_list),
        len(x_raw_list),
        len(y_raw_list),
        len(center_x_standard_list),
        len(center_y_standard_list),
        len(rise_indices),
    )
    out_center_x_list: List[List[float]] = []
    out_center_y_list: List[List[float]] = []

    for i in range(n):
        t = np.asarray(time_list[i], dtype=float)
        x_raw = np.asarray(x_raw_list[i], dtype=float)
        y_raw = np.asarray(y_raw_list[i], dtype=float)
        cx_std = np.asarray(center_x_standard_list[i], dtype=float)
        cy_std = np.asarray(center_y_standard_list[i], dtype=float)
        m = min(len(t), len(x_raw), len(y_raw), len(cx_std), len(cy_std))
        if m <= 0:
            out_center_x_list.append([])
            out_center_y_list.append([])
            continue

        t = t[:m]
        x_raw = x_raw[:m]
        y_raw = y_raw[:m]
        cx_std = _fill_center_nan_with_previous(cx_std[:m].tolist())
        cy_std = _fill_center_nan_with_previous(cy_std[:m].tolist())
        rise_idx = _normalize_rise_index(float(rise_indices[i]), m)

        cx_final = cx_std.copy()
        cy_final = cy_std.copy()
        if rise_idx < m:
            if post_rise_center_mode == 3:
                x_post = x_raw[rise_idx:]
                y_post = y_raw[rise_idx:]
                cx_post = float(np.nanmean(x_post)) if np.isfinite(x_post).any() else float(np.nanmean(x_raw))
                cy_post = float(np.nanmean(y_post)) if np.isfinite(y_post).any() else float(np.nanmean(y_raw))
                if not np.isfinite(cx_post):
                    cx_post = 0.0
                if not np.isfinite(cy_post):
                    cy_post = 0.0
                cx_final[rise_idx:] = cx_post
                cy_final[rise_idx:] = cy_post
            else:
                window = _estimate_window_frames_from_fft(t, x_raw, y_raw, rise_idx=rise_idx)
                stat_mode: Literal["mean", "median"] = "mean" if post_rise_center_mode == 1 else "median"
                cx_slide = _rolling_stat(x_raw, window=window, stat=stat_mode)
                cy_slide = _rolling_stat(y_raw, window=window, stat=stat_mode)
                cx_slide = _fill_center_nan_with_previous(cx_slide.tolist())
                cy_slide = _fill_center_nan_with_previous(cy_slide.tolist())
                cx_final[rise_idx:] = cx_slide[rise_idx:]
                cy_final[rise_idx:] = cy_slide[rise_idx:]

        out_center_x_list.append(cx_final.tolist())
        out_center_y_list.append(cy_final.tolist())

    return out_center_x_list, out_center_y_list


def save_segment_center_coordinate(
    day: str,
    segment_subdir: str,
    center_x_list: Sequence[Sequence[float]],
    center_y_list: Sequence[Sequence[float]],
) -> None:
    n = min(len(center_x_list), len(center_y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/center_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    data = {}
    for i in range(n):
        data[f"No.{i + 1}_x"] = pd.Series(np.asarray(center_x_list[i], dtype=float), dtype="float64")
        data[f"No.{i + 1}_y"] = pd.Series(np.asarray(center_y_list[i], dtype=float), dtype="float64")
    pd.DataFrame(data).to_csv(f"{save_dir}/center_coordinate.csv", index=False)


def build_pre_rise_raw_centroid_series(
    day: str,
    time_list: Sequence[Sequence[float]],
    corrected_x_list: Sequence[Sequence[float]],
    corrected_y_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    center_x_list, center_y_list = load_rotation_center_with_nan(day)
    n = min(len(time_list), len(corrected_x_list), len(corrected_y_list), len(rise_indices))
    pre_time_list: List[List[float]] = []
    pre_x_raw_list: List[List[float]] = []
    pre_y_raw_list: List[List[float]] = []

    for i in range(n):
        t = np.asarray(time_list[i], dtype=float)
        x_corr = np.asarray(corrected_x_list[i], dtype=float)
        y_corr = np.asarray(corrected_y_list[i], dtype=float)
        if i < len(center_x_list) and i < len(center_y_list):
            cx = np.asarray(center_x_list[i], dtype=float)
            cy = np.asarray(center_y_list[i], dtype=float)
            m = min(len(t), len(x_corr), len(y_corr), len(cx), len(cy))
            cx = cx[:m]
            cy = cy[:m]
            cx = np.where(np.isfinite(cx), cx, np.nanmean(cx) if np.isfinite(cx).any() else 0.0)
            cy = np.where(np.isfinite(cy), cy, np.nanmean(cy) if np.isfinite(cy).any() else 0.0)
        else:
            m = min(len(t), len(x_corr), len(y_corr))
            cx = np.zeros(m, dtype=float)
            cy = np.zeros(m, dtype=float)

        t = t[:m]
        x_corr = x_corr[:m]
        y_corr = y_corr[:m]
        rise_idx = _normalize_rise_index(float(rise_indices[i]), m)

        pre_time_list.append(t[:rise_idx].tolist())
        pre_x_raw_list.append((x_corr[:rise_idx] + cx[:rise_idx]).tolist())
        pre_y_raw_list.append((y_corr[:rise_idx] + cy[:rise_idx]).tolist())

    return pre_time_list, pre_x_raw_list, pre_y_raw_list


def subtract_center_from_raw(
    x_raw_list: Sequence[Sequence[float]],
    y_raw_list: Sequence[Sequence[float]],
    center_x_list: Sequence[Sequence[float]],
    center_y_list: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    n = min(len(x_raw_list), len(y_raw_list), len(center_x_list), len(center_y_list))
    x_corr_list: List[List[float]] = []
    y_corr_list: List[List[float]] = []
    for i in range(n):
        xr = np.asarray(x_raw_list[i], dtype=float)
        yr = np.asarray(y_raw_list[i], dtype=float)
        cx = np.asarray(center_x_list[i], dtype=float)
        cy = np.asarray(center_y_list[i], dtype=float)
        m = min(len(xr), len(yr), len(cx), len(cy))
        x_corr_list.append((xr[:m] - cx[:m]).tolist())
        y_corr_list.append((yr[:m] - cy[:m]).tolist())
    return x_corr_list, y_corr_list


def split_coordinate_components_by_rise(
    time_list: Sequence[Sequence[float]],
    x_before_list: Sequence[Sequence[float]],
    y_before_list: Sequence[Sequence[float]],
    x_center_list: Sequence[Sequence[float]],
    y_center_list: Sequence[Sequence[float]],
    x_corr_list: Sequence[Sequence[float]],
    y_corr_list: Sequence[Sequence[float]],
    rise_indices: Sequence[float],
    mode: str,
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
    List[List[float]],
]:
    n = min(
        len(time_list),
        len(x_before_list),
        len(y_before_list),
        len(x_center_list),
        len(y_center_list),
        len(x_corr_list),
        len(y_corr_list),
        len(rise_indices),
    )
    out_t: List[List[float]] = []
    out_x_before: List[List[float]] = []
    out_y_before: List[List[float]] = []
    out_x_center: List[List[float]] = []
    out_y_center: List[List[float]] = []
    out_x_corr: List[List[float]] = []
    out_y_corr: List[List[float]] = []

    for i in range(n):
        t = np.asarray(time_list[i], dtype=float)
        xb = np.asarray(x_before_list[i], dtype=float)
        yb = np.asarray(y_before_list[i], dtype=float)
        xc = np.asarray(x_center_list[i], dtype=float)
        yc = np.asarray(y_center_list[i], dtype=float)
        xg = np.asarray(x_corr_list[i], dtype=float)
        yg = np.asarray(y_corr_list[i], dtype=float)
        m = min(len(t), len(xb), len(yb), len(xc), len(yc), len(xg), len(yg))
        if m <= 0:
            out_t.append([])
            out_x_before.append([])
            out_y_before.append([])
            out_x_center.append([])
            out_y_center.append([])
            out_x_corr.append([])
            out_y_corr.append([])
            continue

        t = t[:m]
        xb = xb[:m]
        yb = yb[:m]
        xc = xc[:m]
        yc = yc[:m]
        xg = xg[:m]
        yg = yg[:m]
        rise_idx = _normalize_rise_index(float(rise_indices[i]), m)

        if mode == "pre":
            sl = slice(0, rise_idx)
        elif mode == "post":
            sl = slice(rise_idx, m)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out_t.append(t[sl].tolist())
        out_x_before.append(xb[sl].tolist())
        out_y_before.append(yb[sl].tolist())
        out_x_center.append(xc[sl].tolist())
        out_y_center.append(yc[sl].tolist())
        out_x_corr.append(xg[sl].tolist())
        out_y_corr.append(yg[sl].tolist())

    return out_t, out_x_before, out_y_before, out_x_center, out_y_center, out_x_corr, out_y_corr


def split_rotational_series_by_rise(
    time_list: Sequence[Sequence[float]],
    angle_list: Sequence[Sequence[float]],
    angular_velocity_list: Sequence[Sequence[float]],
    valid_indices: Sequence[int],
    rise_indices: Sequence[float],
    mode: str,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    out_time: List[List[float]] = []
    out_angle: List[List[float]] = []
    out_av: List[List[float]] = []

    n = min(len(time_list), len(angle_list), len(angular_velocity_list), len(valid_indices))
    for i in range(n):
        t = np.asarray(time_list[i], dtype=float)
        ang = np.asarray(angle_list[i], dtype=float)
        av = np.asarray(angular_velocity_list[i], dtype=float)
        m = min(len(t), len(ang), len(av) + 1)
        t = t[:m]
        ang = ang[:m]
        av = av[: max(0, m - 1)]

        orig_idx = int(valid_indices[i])
        rise_idx = _normalize_rise_index(float(rise_indices[orig_idx]), m) if orig_idx < len(rise_indices) else m

        if mode == "pre":
            t_seg = t[:rise_idx]
            ang_seg = ang[:rise_idx]
            av_seg = av[: max(0, rise_idx - 1)]
        elif mode == "post":
            t_seg = t[rise_idx:]
            ang_seg = ang[rise_idx:]
            av_seg = av[rise_idx:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out_time.append(t_seg.tolist())
        out_angle.append(ang_seg.tolist())
        out_av.append(av_seg.tolist())

    return out_time, out_angle, out_av


def save_segment_angular_velocity_outputs(
    day: str,
    segment_subdir: str,
    time_list: Sequence[Sequence[float]],
    angle_list: Sequence[Sequence[float]],
    angular_velocity_list: Sequence[Sequence[float]],
    original_sample_indices: Optional[Sequence[int]] = None,
    rise_time_list: Optional[Sequence[float]] = None,
) -> None:
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)

    n = min(len(time_list), len(angle_list), len(angular_velocity_list))
    angle_df = {}
    av_df = {}
    for i in range(n):
        a = np.asarray(angle_list[i], dtype=float)
        v = np.asarray(angular_velocity_list[i], dtype=float)
        angle_df[f"No.{i + 1}"] = pd.Series(a, dtype="float64")
        av_df[f"No.{i + 1}"] = pd.Series(v, dtype="float64")

    pd.DataFrame(angle_df).to_csv(f"{save_dir}/angle_time-series.csv", index=False)
    pd.DataFrame(av_df).to_csv(f"{save_dir}/angular-velocity_time-series.csv", index=False)
    time_df = {}
    for i in range(n):
        time_df[f"No.{i + 1}"] = pd.Series(np.asarray(time_list[i], dtype=float), dtype="float64")
    pd.DataFrame(time_df).to_csv(
        f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/time_list.csv", index=False
    )

    if original_sample_indices is not None:
        map_df = pd.DataFrame(
            {
                "segment_no": [i + 1 for i in range(n)],
                "original_sample_no": [int(original_sample_indices[i]) + 1 for i in range(n)],
            }
        )
        map_df.to_csv(
            f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/sample_index_map.csv", index=False
        )

    make_graph.plot_repellent_angular_velocity_onecol(
        time_list,
        angle_list,
        angular_velocity_list,
        save_dir,
        rise_time_list=rise_time_list,
    )


def _get_valid_pre_rise_indices(
    pre_time_list: Sequence[Sequence[float]],
    pre_angular_velocity_list: Sequence[Sequence[float]],
) -> List[int]:
    valid_indices: List[int] = []
    for i, (time_i, av_i) in enumerate(zip(pre_time_list, pre_angular_velocity_list)):
        if len(time_i) < 2 or len(av_i) < 3:
            continue
        if float(time_i[-1]) <= float(time_i[0]):
            continue
        valid_indices.append(i)
    return valid_indices


def run_pre_rise_fluctuation(
    pre_time_list: Sequence[Sequence[float]],
    pre_angular_velocity_list: Sequence[Sequence[float]],
    day: str,
    original_sample_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    target_dir = f"{param.save_dir_bef}/{day}/repellent_response/02_pre_rise_fluctuation"
    os.makedirs(target_dir, exist_ok=True)

    valid_indices = _get_valid_pre_rise_indices(pre_time_list, pre_angular_velocity_list)
    if not valid_indices:
        pd.DataFrame(
            [{"message": "No sample has enough pre-rise data for fluctuation analysis.", "analyzed_samples": 0}]
        ).to_csv(f"{target_dir}/summary.csv", index=False)
        return []

    selected_time_list = [list(pre_time_list[i]) for i in valid_indices]
    selected_av_list = [np.asarray(pre_angular_velocity_list[i], dtype=float) for i in valid_indices]

    tmp_day = f"repellent_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_input_root = os.path.join(tmp_root, "data")
        tmp_output_root = os.path.join(tmp_root, "outputs")
        os.makedirs(tmp_input_root, exist_ok=True)
        os.makedirs(tmp_output_root, exist_ok=True)

        _write_temp_config(f"{tmp_input_root}/{tmp_day}/config.ini", len(valid_indices))

        orig_input_root = param.input_dir_bef
        orig_output_root = param.save_dir_bef
        try:
            param.input_dir_bef = tmp_input_root
            param.save_dir_bef = tmp_output_root

            save2csv.save_time_list(selected_time_list, tmp_day)
            rot_df_manage.create_rot_df(tmp_day)
            _write_dummy_angle_fft(tmp_day, len(valid_indices))
            fluctuation_analysis.main(selected_av_list, tmp_day)
            make_graph.plot_rot_param(tmp_day)
        finally:
            param.input_dir_bef = orig_input_root
            param.save_dir_bef = orig_output_root

        src_dir = f"{tmp_output_root}/{tmp_day}/fluctuation_analysis"
        target_fluc_dir = f"{target_dir}/fluctuation_analysis"
        if os.path.isdir(target_fluc_dir):
            shutil.rmtree(target_fluc_dir)
        shutil.copytree(src_dir, target_fluc_dir)

    if original_sample_indices is None:
        original_indices = [idx + 1 for idx in valid_indices]
    else:
        original_indices = []
        for idx in valid_indices:
            if idx < len(original_sample_indices):
                original_indices.append(int(original_sample_indices[idx]) + 1)
            else:
                original_indices.append(int(idx) + 1)

    map_df = pd.DataFrame(
        {
            "pre_rise_fluctuation_no": [i + 1 for i in range(len(valid_indices))],
            "original_sample_no": original_indices,
        }
    )
    map_df.to_csv(f"{target_dir}/sample_index_map.csv", index=False)
    return valid_indices


def add_rise_time_to_results(results: List[Dict[str, float]], time_list: Sequence[Sequence[float]]) -> None:
    for i, result in enumerate(results):
        rise_index = result.get("rise_index", np.nan)
        if i >= len(time_list) or not np.isfinite(rise_index):
            result["rise_time"] = np.nan
            continue
        time_arr = np.asarray(time_list[i], dtype=float)
        if time_arr.size == 0:
            result["rise_time"] = np.nan
            continue
        idx = _normalize_rise_index(float(rise_index), time_arr.size)
        if idx >= time_arr.size:
            result["rise_time"] = np.nan
        else:
            result["rise_time"] = float(time_arr[idx])


def ensure_time_list(day: str) -> List[List[float]]:
    try:
        get_tiff_info.get_timelist(day)
        return read_csv.get_timelist(day)
    except Exception:
        # Fallback: derive time list directly from TIFF metadata if config-based path is unavailable.
        tiff_root = f"{param.input_dir_bef}/{day}/tiff_data"
        sample_names = get_tiff_sample_names(day)
        time_list_all: List[List[float]] = []

        for sample_name in sample_names:
            sample_dir = os.path.join(tiff_root, sample_name)
            # If the expected sample directory does not exist, append an empty/default
            # time list and continue. This prevents FileNotFoundError when sample
            # names from configuration are missing or misspelled.
            if not os.path.isdir(sample_dir):
                time_list_all.append([])
                continue

            frame_names = [
                name
                for name in os.listdir(sample_dir)
                if name.lower().endswith(".tif") or name.lower().endswith(".tiff")
            ]
            frame_names = sorted(frame_names, key=_safe_extract_number)

            base_time = None
            sample_time_list = []
            for frame_name in frame_names:
                frame_path = os.path.join(sample_dir, frame_name)
                with Image.open(frame_path) as img:
                    metadata = getattr(img, "tag_v2", None)
                    if metadata is None:
                        continue
                    time_raw = metadata.get(306, None)
                    if time_raw is None:
                        continue
                    frame_time = datetime.strptime(time_raw, "%m/%d/%Y %H:%M:%S.%f")
                    if base_time is None:
                        base_time = frame_time
                    sample_time_list.append((frame_time - base_time).total_seconds())
            if not sample_time_list:
                sample_time_list = [float(i) for i in range(len(frame_names))]
            time_list_all.append(sample_time_list)

        save2csv.save_time_list(time_list_all, day)
        return time_list_all


def _extract_centroid_from_frame(frame) -> Tuple[float, float]:
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.nan, np.nan
    max_contour = max(contours, key=cv2.contourArea)
    if max_contour is None or max_contour.size == 0:
        return np.nan, np.nan
    mean_x = np.mean(max_contour[:, 0, 0].astype(float))
    mean_y = np.mean(max_contour[:, 0, 1].astype(float))
    return float(mean_x), float(mean_y)


def _sort_avi_paths(avi_paths: Sequence[str]) -> List[str]:
    def _key(path: str) -> float:
        numbers = re.findall(r"\d+", os.path.basename(path))
        if not numbers:
            return math.inf
        return int(numbers[-1])

    return sorted(avi_paths, key=_key)


def generate_centroid_coordinate_simple(day: str) -> str:
    input_dir = f"{param.input_dir_bef}/{day}"
    save_path = f"{param.save_dir_bef}/{day}/centroid_coordinate.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    avi_paths = _sort_avi_paths(glob.glob(f"{input_dir}/*.avi"))
    if len(avi_paths) == 0:
        raise FileNotFoundError(f"No .avi file found in {input_dir}")

    try:
        px2um_x, px2um_y = param.get_px2um_config(day)
    except Exception:
        px2um_x, px2um_y = 1.0, 1.0

    x_list: List[List[float]] = []
    y_list: List[List[float]] = []
    for avi_path in avi_paths:
        cap = cv2.VideoCapture(avi_path)
        sample_x: List[float] = []
        sample_y: List[float] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            x, y = _extract_centroid_from_frame(frame)
            if np.isfinite(x):
                sample_x.append(float(x) * px2um_x)
            else:
                sample_x.append(np.nan)
            if np.isfinite(y):
                sample_y.append(float(y) * px2um_y)
            else:
                sample_y.append(np.nan)
        cap.release()
        x_list.append(sample_x)
        y_list.append(sample_y)

    data = {}
    for i in range(len(x_list)):
        data[f"x_{i+1}"] = pd.Series(x_list[i], dtype="float64")
        data[f"y_{i+1}"] = pd.Series(y_list[i], dtype="float64")
    pd.DataFrame(data).to_csv(save_path, index=False)
    return save_path


def align_coordinate_series_to_time(
    x_list: Sequence[Sequence[float]],
    y_list: Sequence[Sequence[float]],
    time_list: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    n = min(len(x_list), len(y_list), len(time_list))
    x_aligned: List[List[float]] = []
    y_aligned: List[List[float]] = []
    for i in range(n):
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        t_arr = np.asarray(time_list[i], dtype=float)
        m = min(len(x_arr), len(y_arr), len(t_arr))
        x_aligned.append(x_arr[:m].tolist())
        y_aligned.append(y_arr[:m].tolist())
    return x_aligned, y_aligned


def copy_center_coordinate_to_segment(day: str, segment_subdir: str) -> None:
    src = f"{param.save_dir_bef}/{day}/center_coordinate"
    if not os.path.isdir(src):
        return
    dst = f"{param.save_dir_bef}/{day}/repellent_response/{segment_subdir}/center_coordinate"
    _copytree_replace(src, dst)


def cleanup_legacy_repellent_outputs(day: str) -> None:
    root = f"{param.save_dir_bef}/{day}/repellent_response"
    legacy_files = [
        f"{root}/background_intensity_time_series.csv",
        f"{root}/background_intensity_time_series.png",
        f"{root}/post_rise_centroid_time_series.csv",
        f"{root}/rise_summary.csv",
        f"{root}/03_post_rise_analysis/centroid_coordinate/trajectory.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/x_coordinate.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/y_coordinate.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/x_centroid_before.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/x_rotation_center.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/x_centroid_corrected.png",
        f"{root}/03_post_rise_analysis/centroid_coordinate/x_components.png",
    ]
    legacy_dirs = [
        f"{root}/00_all_rotational_analysis",
        f"{root}/02_pre_rise_fluctuation",
        f"{root}/03_post_rise_analysis",
        f"{root}/center_coordinate",
        f"{root}/pre_rise_fluctuation",
    ]

    for path in legacy_files:
        if os.path.isfile(path):
            os.remove(path)
    for path in legacy_dirs:
        if os.path.isdir(path):
            shutil.rmtree(path)
