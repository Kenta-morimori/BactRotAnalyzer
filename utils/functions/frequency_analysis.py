import numpy as np
import pandas as pd

from utils import param
from utils.functions import get_angular_velocity, make_graph, save2csv


CONTINUITY_GAP_FACTOR = 1.5


def longest_continuous_segment(time, *series):
    """Return the longest finite, regularly sampled segment and its metadata."""
    arrays = [np.asarray(time, dtype=float)] + [np.asarray(values, dtype=float) for values in series]
    if not arrays:
        return tuple(), {
            "start_index": np.nan,
            "end_index": np.nan,
            "sample_count": 0,
            "effective_frame_rate_hz": np.nan,
        }
    n = min(len(values) for values in arrays)
    arrays = [values[:n] for values in arrays]
    valid = np.isfinite(arrays[0])
    for values in arrays[1:]:
        valid &= np.isfinite(values)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        empty = tuple(np.asarray([], dtype=float) for _ in arrays)
        return empty, {"start_index": np.nan, "end_index": np.nan, "sample_count": 0, "effective_frame_rate_hz": np.nan}

    valid_time = arrays[0][indices]
    positive_diffs = np.diff(valid_time)
    positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0)]
    nominal_dt = float(np.median(positive_diffs)) if len(positive_diffs) else np.nan
    split = np.zeros(len(indices), dtype=bool)
    if len(indices) > 1:
        diffs = np.diff(valid_time)
        split[1:] = (~np.isfinite(diffs)) | (diffs <= 0)
        if np.isfinite(nominal_dt):
            split[1:] |= diffs > nominal_dt * CONTINUITY_GAP_FACTOR
    starts = np.flatnonzero(split)
    starts = np.r_[0, starts]
    ends = np.r_[starts[1:], len(indices)]
    best_start, best_end = max(zip(starts, ends), key=lambda item: item[1] - item[0])
    chosen_indices = indices[best_start:best_end]
    chosen = tuple(values[chosen_indices] for values in arrays)
    chosen_diffs = np.diff(chosen[0])
    chosen_diffs = chosen_diffs[np.isfinite(chosen_diffs) & (chosen_diffs > 0)]
    effective_rate = 1.0 / float(np.median(chosen_diffs)) if len(chosen_diffs) else np.nan
    return chosen, {
        "start_index": int(chosen_indices[0]),
        "end_index": int(chosen_indices[-1]),
        "sample_count": int(len(chosen_indices)),
        "start_time_sec": float(chosen[0][0]),
        "end_time_sec": float(chosen[0][-1]),
        "effective_frame_rate_hz": effective_rate,
    }


def fft(data_bef, dt):
    data_arr = np.asarray(data_bef, dtype=float)
    if data_arr.size == 0 or not np.isfinite(data_arr).any():
        empty = np.asarray([], dtype=float)
        return empty, empty

    data_aft = data_arr - np.nanmean(data_arr)
    N = len(data_aft)
    # FFT
    F = np.fft.fft(data_aft)
    freq = np.fft.fftfreq(N, d=dt)
    F = F / (N / 2)
    Amp = np.abs(F)

    return freq[1 : N // 2], Amp[1 : N // 2]


def _save_fft_timebase_metadata(save_dir, save_name, metadata):
    pd.DataFrame(metadata).to_csv(f"{save_dir}/{save_name}_timebase.csv", index=False)


def fft_angle(angle_list, day, time_list=None):
    sample_num, FrameRate_list, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    metadata = []
    for i in range(sample_num):
        if param.flag_get_angle_with_cell_direcetion:
            # normalize angle to -π~π
            angle = get_angular_velocity.normalized_angle(angle_list[i])
        else:
            angle = angle_list[i]
        if time_list is not None and i < len(time_list):
            (segment_time, segment_angle), meta = longest_continuous_segment(time_list[i], angle)
            dt = (
                1.0 / meta["effective_frame_rate_hz"]
                if np.isfinite(meta["effective_frame_rate_hz"])
                else 1 / FrameRate_list[i]
            )
        else:
            segment_angle = angle
            meta = {
                "start_index": 0,
                "end_index": len(angle) - 1,
                "sample_count": len(angle),
                "start_time_sec": np.nan,
                "end_time_sec": np.nan,
                "effective_frame_rate_hz": FrameRate_list[i],
            }
            dt = 1 / FrameRate_list[i]
        meta["sample_no"] = i + 1
        metadata.append(meta)
        freq, Amp = fft(segment_angle, dt)
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    save_name = "angle_FFT"
    make_graph.plot_fft(freq_list, Amp_list, save_dir, save_name, day, flag_add_peak=True)
    save2csv.save_fft(save_dir, save_name, freq_list, Amp_list)
    _save_fft_timebase_metadata(save_dir, save_name, metadata)


def fft_angular_velocity(angular_velocity_list, day, time_list=None):
    sample_num, FrameRate_list, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    metadata = []
    for i in range(sample_num):
        av = np.asarray(angular_velocity_list[i], dtype=float)
        if time_list is not None and i < len(time_list):
            av_time = np.asarray(time_list[i], dtype=float)[1 : len(av) + 1]
            (segment_time, segment_av), meta = longest_continuous_segment(av_time, av)
            dt = (
                1.0 / meta["effective_frame_rate_hz"]
                if np.isfinite(meta["effective_frame_rate_hz"])
                else 1 / FrameRate_list[i]
            )
        else:
            segment_av = av[np.isfinite(av)]
            meta = {
                "start_index": 0,
                "end_index": len(av) - 1,
                "sample_count": len(segment_av),
                "start_time_sec": np.nan,
                "end_time_sec": np.nan,
                "effective_frame_rate_hz": FrameRate_list[i],
            }
            dt = 1 / FrameRate_list[i]
        meta["sample_no"] = i + 1
        metadata.append(meta)
        freq, Amp = fft(segment_av, dt)
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    save_name = "angular_velocity_FFT"
    make_graph.plot_fft(freq_list, Amp_list, save_dir, save_name, day)
    save2csv.save_fft(save_dir, save_name, freq_list, Amp_list)
    _save_fft_timebase_metadata(save_dir, save_name, metadata)


def fft_sd_list(df, day, flag_std=False):
    sample_num, FrameRate_list, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        add_freq_list, add_Amp_list = [], []
        for j in range(len(width_time_list)):
            if flag_std:
                sd_list = df[f"No.{i + 1}_{width_time_list[j]}s_sd_std"].dropna().tolist()
            else:
                sd_list = df[f"No.{i + 1}_{width_time_list[j]}s_sd"].dropna().tolist()
            freq, Amp = fft(sd_list, 1 / FrameRate_list[i])
            add_freq_list.append(freq.tolist())
            add_Amp_list.append(Amp.tolist())
        freq_list.append(add_freq_list)
        Amp_list.append(add_Amp_list)
    freq_arr = np.array(freq_list, dtype=object)
    Amp_arr = np.array(Amp_list, dtype=object)
    # plot
    make_graph.plot_SD_list_fft(freq_arr, Amp_arr, day, flag_std)
    make_graph.dev_plot_sd_FFT_with_rotation(freq_arr, Amp_arr, day)

    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    save2csv.save_sd_fft(save_dir, freq_arr, Amp_arr)

    return freq_arr, Amp_arr
