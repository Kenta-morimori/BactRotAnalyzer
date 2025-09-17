from collections import defaultdict
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import (
    frequency_analysis,
    make_graph,
    read_csv,
    rot_df_manage,
)


def get_weights_gaussian(
    start_time: float,
    window_width: float,
    time_list: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    edge_peak_divisor = param.edge_peak_divisor
    t = np.asarray(time_list, dtype=float)

    if t.size == 0:
        return np.array([], dtype=float)
    if window_width <= 0:
        raise ValueError("window_width must be > 0")
    if edge_peak_divisor <= 1.0:
        raise ValueError("edge_peak_divisor must be > 1")

    center_time_sec = start_time + window_width / 2.0
    sigma_sec = window_width / (2.0 * np.sqrt(2.0 * np.log(edge_peak_divisor)))

    weights = np.exp(-0.5 * ((t - center_time_sec) / sigma_sec) ** 2)

    weight_sum = weights.sum()
    return (weights / weight_sum) if weight_sum > 0 else np.zeros_like(weights)


def get_weighted_stats(
    data: Union[Sequence[float], np.ndarray],
    weights: Union[Sequence[float], np.ndarray],
) -> Tuple[float, float]:
    values_arr = np.asarray(data, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    if values_arr.size == 0 or weights_arr.size == 0:
        return np.nan, np.nan
    if values_arr.shape != weights_arr.shape:
        raise ValueError("`data` and `weights` must have the same shape.")

    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        return np.nan, np.nan

    # Normalize weights: Σ w_i = 1
    normalized_weights = weights_arr / total_weight
    # Weighted mean: μ = Σ (w_i * x_i) / Σ w_i
    weighted_mean = float(np.sum(normalized_weights * values_arr))
    # Weighted variance: σ² = Σ (w_i * (x_i - μ)²) / Σ w_i
    weighted_variance = float(np.sum(normalized_weights * ((values_arr - weighted_mean) ** 2)))
    # Weighted standard deviation: σ = sqrt(σ²)
    weighted_std = float(np.sqrt(weighted_variance))

    return weighted_std, weighted_mean


def get_sd_time_series(i, angular_velocity, day):
    _, FrameRate_list, total_time_list = param.get_config(day)
    width_time_list = param.SD_window_width_list

    time_step = 1 / FrameRate_list[i]
    total_time = float(total_time_list[i])

    # time-based window
    time_list = read_csv.get_timelist(day)
    df = pd.DataFrame(
        {
            "time": time_list[i][: len(angular_velocity)],
            "velocity": np.abs(angular_velocity),
        }
    )

    df_sd_stats_dict = defaultdict(list)
    times = [j * time_step for j in range(int(time_list[i][-1] / time_step) + 1)]
    df_sd_stats_dict[f"No.{i + 1}_time"] = times
    n_time = len(times)

    for width_time in width_time_list:
        prev_mean = np.nan
        prev_sd = np.nan

        key_sd = f"No.{i + 1}_{width_time}s_sd"
        key_mean = f"No.{i + 1}_{width_time}s_mean"
        key_n = f"No.{i + 1}_{width_time}s_data_num"

        # record Nan from start to width_time / 2
        start_nan_data_num = int((width_time / 2) // time_step)
        df_sd_stats_dict[key_sd] = [np.nan] * start_nan_data_num
        df_sd_stats_dict[key_mean] = [np.nan] * start_nan_data_num
        df_sd_stats_dict[key_n] = [np.nan] * start_nan_data_num

        start_time = 0.0
        while start_time + width_time < total_time:
            mask = (df["time"] >= start_time) & (df["time"] < start_time + width_time)
            data = df.loc[mask, "velocity"].to_numpy(dtype=float)
            times_win = df.loc[mask, "time"].to_numpy(dtype=float)

            if len(data) >= 3:  # at least 3 data to calculate SD
                if param.flag_apply_gaussian_window:
                    weights = get_weights_gaussian(start_time, width_time, times_win)
                    if weights.size == 0 or np.sum(weights) == 0:
                        sd_val, mean_val = np.nan, np.nan
                    else:
                        sd_val, mean_val = get_weighted_stats(data, weights)
                else:
                    mean_val = float(np.mean(data))
                    sd_val = float(np.std(data, ddof=1))

                # SD/Mean
                if param.mode_evaluate_SD_fluctuation == 1:
                    if np.isfinite(mean_val) and mean_val != 0.0:
                        sd_val = sd_val / mean_val
                    else:
                        print(f"[Warning] No.{i + 1}: mean is zero or NaN, cannot compute SD/Mean.")
                        sd_val = np.nan

                df_sd_stats_dict[key_sd].append(sd_val)
                df_sd_stats_dict[key_mean].append(mean_val)
                prev_sd, prev_mean = sd_val, mean_val
            else:
                df_sd_stats_dict[key_sd].append(prev_sd)
                df_sd_stats_dict[key_mean].append(prev_mean)
            df_sd_stats_dict[key_n].append(len(data))
            start_time += time_step

        # record Nan until the end time
        cur_len = len(df_sd_stats_dict[key_sd])
        if cur_len < n_time:
            pad = n_time - cur_len
            df_sd_stats_dict[key_sd].extend([np.nan] * pad)
            df_sd_stats_dict[key_mean].extend([np.nan] * pad)
            df_sd_stats_dict[key_n].extend([np.nan] * pad)

    df_sd_stats = pd.DataFrame(df_sd_stats_dict)

    return df_sd_stats


def add_standardize_sd_to_df(df, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list

    for i in range(sample_num):
        for width_time in width_time_list:
            key_sd = f"No.{i + 1}_{width_time}s_sd"
            sd = np.array(df[key_sd])
            mean = np.nanmean(sd)
            std = np.nanstd(sd, axis=0)
            adt_sd = (sd - mean) / std
            key_std_sd = f"No.{i + 1}_{width_time}s_sd_std"
            df[key_std_sd] = adt_sd

    return df


def evaluate_FFT(sd_freq_list, sd_Amp_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list

    decrease_list = []
    ratio_list = []
    ratio_reciprocal_list = []
    ref_point_list: List[List[float]] = [[] for _ in range(sample_num)]
    for i in range(sample_num):
        add_decrease = []
        add_ratio = []
        add_ratio_reciprocal = []
        for j in range(len(width_time_list)):
            add_ref_point = np.mean(sd_Amp_list[i][j][0:3])
            ref_point_list[i].append(add_ref_point)

            high_freq_indices = [k for k, x in enumerate(sd_freq_list[i][j]) if x >= 80]
            Amp_high_freq_arr = np.mean([sd_Amp_list[i][j][k] for k in high_freq_indices])

            add_decrease.append(add_ref_point - Amp_high_freq_arr)
            add_ratio.append(add_ref_point / Amp_high_freq_arr)
            add_ratio_reciprocal.append(Amp_high_freq_arr / add_ref_point)
        decrease_list.append(add_decrease)
        ratio_list.append(add_ratio)
        ratio_reciprocal_list.append(add_ratio_reciprocal)
    # plot
    make_graph.plot_SD_FFT_feats(ratio_list, ratio_reciprocal_list, decrease_list, ref_point_list, day)

    # save rot_df
    for j, width in enumerate(width_time_list):
        decrease_list_rot_df = []
        ratio_list_rot_df = []
        ratio_reciprocal_list_rot_df = []
        ref_point_list_rot_df = []
        for i in range(sample_num):
            decrease_list_rot_df.append(decrease_list[i][j])
            ratio_list_rot_df.append(ratio_list[i][j])
            ratio_reciprocal_list_rot_df.append(ratio_reciprocal_list[i][j])
            ref_point_list_rot_df.append(ref_point_list[i][j])
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_FFT_Amp_decrease}_{width}s", decrease_list_rot_df, day)
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_FFT_Amp_ratio}_{width}s", ratio_list_rot_df, day)
        rot_df_manage.update_rot_df(
            f"{ROTATION_FEATURES.SD_FFT_Amp_ratio_reciprocal}_{width}s", ratio_reciprocal_list_rot_df, day
        )
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_FFT_Amp_refpoints}_{width}s", ref_point_list_rot_df, day)


def main(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list

    df_list = []
    # get SD time-series
    for i in range(sample_num):
        add_df = get_sd_time_series(i, angular_velocity_list[i], day)
        df_list.append(add_df)
    df_all = pd.concat(df_list, axis=1)

    # dev plot
    if param.mode_evaluate_SD_fluctuation == 0:
        make_graph.dev_plot_av_with_mean_sd(angular_velocity_list, df_all, day)

    # save to rot_df
    for width in width_time_list:
        sd_mean_list = []
        for i in range(sample_num):
            sd_mean_list.append(np.nanmean(df_all[f"No.{i + 1}_{width}s_sd"]))
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_mean}_{width}s", sd_mean_list, day)

    # plot
    make_graph.plot_SD_list(df_all, day)
    make_graph.plot_sd_mean(df_all, day)

    # get standardized sd time-series
    df_all = add_standardize_sd_to_df(df_all, day)
    make_graph.plot_SD_list(df_all, day, flag_std=True)

    # FFT
    if param.flag_use_std_df_to_FFT:
        sd_freq_list, sd_Amp_list = frequency_analysis.fft_sd_list(df_all, day, flag_std=True)
    else:
        sd_freq_list, sd_Amp_list = frequency_analysis.fft_sd_list(df_all, day, flag_std=True)

    # save
    df_all.to_csv(f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/SD_time_series.csv", index=False)

    if param.flag_evaluate_SD_FFT:
        evaluate_FFT(sd_freq_list, sd_Amp_list, day)
