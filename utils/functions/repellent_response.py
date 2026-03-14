import configparser
import glob
import math
import os
import re
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils import param
from utils.functions import (
    fluctuation_analysis,
    get_tiff_info,
    make_graph,
    read_csv,
    rot_df_manage,
    save2csv,
)


def _safe_extract_number(filename: str) -> int:
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
    with open(config_path, "w", encoding="utf-8") as fp:
        cfg.write(fp)
    return config_path


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
        idx = np.arange(arr_work.size)
        arr_work[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], arr_work[finite_mask])

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

        runs = []
        run_start = None
        for idx, flag in enumerate(active_mask):
            if flag and run_start is None:
                run_start = idx
            elif not flag and run_start is not None:
                runs.append((run_start, idx - 1))
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

    map_df = pd.DataFrame(
        {
            "pre_rise_fluctuation_no": [i + 1 for i in range(len(valid_indices))],
            "original_sample_no": [idx + 1 for idx in valid_indices],
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
                    metadata = img.tag_v2
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
    def _key(path: str) -> int:
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
        f"{root}/center_coordinate",
        f"{root}/pre_rise_fluctuation",
    ]

    for path in legacy_files:
        if os.path.isfile(path):
            os.remove(path)
    for path in legacy_dirs:
        if os.path.isdir(path):
            shutil.rmtree(path)
