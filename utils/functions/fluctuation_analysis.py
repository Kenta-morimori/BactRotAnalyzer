from typing import List

import numpy as np
import pandas as pd

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import (
    frequency_analysis,
    make_graph,
    read_csv,
    rot_df_manage,
    save2csv,
)


def get_sd_time_series(i, angular_velocity, day):
    _, FrameRate_list, total_time_list = param.get_config(day)
    width_time_list = param.SD_window_width_list

    sd_list: List[List[float]] = []
    data_num_list: List[List[int]] = []  # develop

    # Obtaining SD time-series
    # time base
    time_list = read_csv.get_timelist(day)

    time_step = 1 / FrameRate_list[i]
    total_time = float(total_time_list[i])

    df = pd.DataFrame(
        {
            "time": time_list[i],
            "velocity": np.abs(angular_velocity),
        }
    )

    for width_time in width_time_list:
        add_sd = []
        prev_val = np.nan
        add_data_num = []  # develop

        start_time = 0.0
        while start_time + width_time < total_time:
            mask = (df["time"] >= start_time) & (df["time"] < start_time + width_time)
            data = df.loc[mask, "velocity"]
            if len(data) >= 3:  # at least 3 data to calculate SD
                if param.mode_evaluate_SD_fluctuation == 0:
                    val = data.std(ddof=1)
                elif param.mode_evaluate_SD_fluctuation == 1:
                    mean_val = data.mean()
                    val = data.std(ddof=1) / mean_val
                add_sd.append(data.std(ddof=1))
                prev_val = val
                add_data_num.append(len(data))
            else:
                add_sd.append(prev_val)
            start_time += time_step
        sd_list.append(add_sd)
        data_num_list.append(add_data_num)  # develop

    return sd_list, data_num_list


def standardize_sd_time_series(sd_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list

    std_sd_list = []
    for i in range(sample_num):
        add_std_sd_list = []
        for j in range(len(width_time_list)):
            sd = np.array(sd_list[i][j])
            mean = np.mean(sd)
            std = np.std(sd, axis=0)
            adt_sd = (sd - mean) / std
            add_std_sd_list.append(adt_sd.tolist())
        std_sd_list.append(add_std_sd_list)

    return std_sd_list


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

    # save CSV
    # save2csv.save_SD_FFT_decline(decrease_list, day)
    # save2csv.save_SD_FFT_refpoints(ref_point_list, day)

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
    sd_list = []
    data_num_list = []  # develop

    # get SD time-series
    for i in range(sample_num):
        add_sd_list, add_data_num_list = get_sd_time_series(i, angular_velocity_list[i], day)
        sd_list.append(add_sd_list)
        data_num_list.append(add_data_num_list)

    # save to rot_df
    for j, width in enumerate(width_time_list):
        sd_mean_list = []
        for i in range(sample_num):
            sd_mean_list.append(np.mean(sd_list[i][j]))
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_mean}_{width}s", sd_mean_list, day)

    """
    make_graph.dev_plot_sd_data_num(data_num_list, day)  # develop
    for j, width in enumerate(width_time_list):
        data_num_mean_list = []
        for i in range(sample_num):
            data_num_mean_list.append(np.mean(data_num_list[i][j]))
        rot_df_manage.update_rot_df(f"{ROTATION_FEATURES.SD_window_data_num_mean}_{width}s", data_num_mean_list, day)
    """

    # save
    save2csv.save_sd_time_series(sd_list, day, flag_std=False)

    # plot
    flag_std = False
    make_graph.plot_SD_list(sd_list, day, flag_std)
    make_graph.plot_sd_mean(sd_list, day)
    # FFT
    # frequency_analysis.fft_sd_list(sd_list, day, flag_std)

    # get standardized sd time-series
    flag_std = True
    std_sd_list = standardize_sd_time_series(sd_list, day)
    save2csv.save_sd_time_series(std_sd_list, day, flag_std)

    make_graph.plot_SD_list(std_sd_list, day, flag_std)
    sd_freq_list, sd_Amp_list = frequency_analysis.fft_sd_list(std_sd_list, day, flag_std)

    if param.flag_evaluate_SD_FFT:
        evaluate_FFT(sd_freq_list, sd_Amp_list, day)
