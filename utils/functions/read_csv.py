from typing import List

import pandas as pd

from utils import param

import numpy as np

def read_center_coordinates(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/center_coordinate/center_coordinate.csv"
    sample_num, _, _ = param.get_config(day)

    df = pd.read_csv(csv_save_dir)
    center_x_arr = np.array([df[f"No.{i + 1}_x"].values for i in range(sample_num)])
    center_y_arr = np.array([df[f"No.{i + 1}_y"].values for i in range(sample_num)])

    return center_x_arr, center_y_arr


def read_angle(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/angle.csv"

    df = pd.read_csv(csv_save_dir)
    angle_list = df.values.T.tolist()

    return angle_list


def get_timelist(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/time_list.csv"

    df = pd.read_csv(csv_save_dir)
    time_list = df.values.T.tolist()
    """
    csv_save_dir = f"{param.save_dir_bef}/{day}/time_list.csv"
    if os.path.isfile(csv_save_dir):
        time_list = read_time_list(day)
    else:
        sample_num, FrameRate_list, total_time_list = param.get_config(day)
        time_list = [
            np.linspace(0, total_time_list[i], int(total_time_list[i] * FrameRate_list[i])).tolist()
            for i in range(sample_num)
        ]
    return time_list
    """

    return time_list


def get_rot_axes(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate/rotation_axes.csv"
    df = pd.read_csv(csv_save_dir)

    long_axis_list = []
    short_axis_list = []
    aspect_ratio_list = []
    for col in df.columns:
        if "long_axis" in col:
            long_axis_list.append(df[col].values[0])
        elif "short_axis" in col:
            short_axis_list.append(df[col].values[0])

    for i in range(len(long_axis_list)):
        aspect_ratio_list.append(short_axis_list[i] / long_axis_list[i])

    return long_axis_list, short_axis_list, aspect_ratio_list


def get_SD_FFT_decline(day, ignore_data_no):
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/SD_FFT_Amp_decrease.csv"
    width_time_list = param.SD_window_width_list

    df = pd.read_csv(csv_save_dir)
    selected_No = [x for x in df["No"].unique().tolist() if x not in ignore_data_no]
    decrease_list: List[List[float]] = [[] for _ in range(len(selected_No))]
    for i, no in enumerate(selected_No):
        for width in width_time_list:
            decrease_list[i].append(float(df[(df["No"] == no) & (df["width"] == width)]["decrease"].values[0]))

    return decrease_list


def get_SD_FFT_refpoints(day, ignore_data_no):
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/SD_FFT_Amp_refpoints.csv"
    width_time_list = param.SD_window_width_list

    df = pd.read_csv(csv_save_dir)
    selected_No = [x for x in df["No"].unique().tolist() if x not in ignore_data_no]
    ref_point_list: List[List[float]] = [[] for _ in range(len(selected_No))]
    for i, no in enumerate(selected_No):
        for width in width_time_list:
            ref_point_list[i].append(float(df[(df["No"] == no) & (df["width"] == width)]["decrease"].values[0]))

    return ref_point_list


def get_angle_FFT(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angle_FFT.csv"
    sample_num, _, _ = param.get_config(day)

    df = pd.read_csv(csv_save_dir)
    freq_list = [df[f"No.{i + 1}_freq"].tolist() for i in range(sample_num)]
    Amp_list = [df[f"No.{i + 1}_Amp"].tolist() for i in range(sample_num)]

    return freq_list, Amp_list
