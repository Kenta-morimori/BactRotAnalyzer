import configparser
import os

import pandas as pd

# directory information
curr_dir = os.getcwd()
input_dir_bef = f"{curr_dir}/data"
save_dir_bef = f"{curr_dir}/outputs"


def get_flag_use_tiff_log(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    flag_use_tiff_log = config.getboolean("Settings", "flag_use_tiff_log")

    return flag_use_tiff_log


def get_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    flag_use_tiff_log = get_flag_use_tiff_log(day)
    sample_num = config.getint("Settings", "sample_num")
    # Frame Rate, Total Time
    FrameRate_list = []
    total_time_list = []
    if flag_use_tiff_log:
        csv_save_dir = f"{save_dir_bef}/{day}/time_list.csv"
        df = pd.read_csv(csv_save_dir)
        time_list = [[x for x in df[col].dropna().tolist()] for col in df.columns]
        for i in range(sample_num):
            FrameRate_list.append(len(time_list[i]) / time_list[i][-1])
            total_time_list.append(time_list[i][-1])
    else:
        try:
            FrameRate = config.getfloat("Settings", "FrameRate")
        except (ValueError, TypeError):
            FrameRate = config.getint("Settings", "FrameRate")
        try:
            total_time = config.getfloat("Settings", "total_time")
        except (ValueError, TypeError):
            total_time = config.getint("Settings", "total_time")
        FrameRate_list = [FrameRate for _ in range(sample_num)]
        total_time_list = [total_time for _ in range(sample_num)]

    return sample_num, FrameRate_list, total_time_list


def get_px2um_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)
    try:
        px2um_x = config.getfloat("Settings", "px2um_x")
    except (ValueError, TypeError):
        px2um_x = config.getint("Settings", "px2um_x")
    try:
        px2um_y = config.getfloat("Settings", "px2um_y")
    except (ValueError, TypeError):
        px2um_y = config.getint("Settings", "px2um_y")
    return px2um_x, px2um_y


def get_tiffinfo_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    items = config["Tiff_info"]["tiff_data"].split(", ")

    return items


# rotational analysis
## Determine the angle by the direction of the cell.
# flag_get_angle_with_cell_direcetion = True
flag_get_angle_with_cell_direcetion = False  # beads assay
n_rotations = 30
min_ref_centroid_num = 30

flag_evaluating_switching = True  # evaluate switching of rotation

flag_evaluate_rotaion_center_movement = False  # MSD etc.
flag_correct_center_outlier = True
mode_correct_center_outlier = 2
"""
0: use TIFF time info
1: use SD threshold (+ use TIFF time info)
2: use Outlier treatment algorithm (+ use TIFF time info)
"""
num_std_center = 5

## About Angular Velocity
flag_evaluate_angular_velocity_abs = False  # Evaluate absolute values of angular velocity

flag_correct_av_outlier = False  # Correct drop data
mode_correct_av_outlier = 0
"""
0: use TIFF time info
1: use SD threshold
"""
num_std_av = 8

flag_av_completion = True  # Complement data
mode_av_completion = 0
"""
0: Nan
1: Mean Value
2: Normal Random Number Completion
"""

flag_kmean = False

min_ref_av_num = 10  # Average

## Switching
flag_eval_switching_with_averaged_av = False

# Angular Velocity Switching Frequency Analysis (post-rise)
av_switching_window_width_sec = 2.0  # Window width in seconds
av_switching_window_shift_sec = 0.5   # Window shift step in seconds

# Background ROI Configuration (repellent response)
bg_roi_offset_um_downward = 10.0  # Downward offset from centroid in micrometers

# fluctuation analysis
# SD_window_width_list = [0.1, 0.5, 1.0]
SD_window_width_list = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

# SD time-series fluctuation mode
mode_evaluate_SD_fluctuation = 1
"""
0: SD
1: SD / Mean
"""


def get_SD_mode_label():
    if mode_evaluate_SD_fluctuation == 0:
        return "SD"
    elif mode_evaluate_SD_fluctuation == 1:
        return "SD_per_mean"


# Obtain SD with gaussian window
flag_apply_gaussian_window = True
edge_peak_divisor = 10.0

# evaluate SD FFT low Amp and decline
flag_evaluate_SD_FFT = True
flag_use_std_df_to_FFT = True

# compare fluctuation property
flag_share_y_axis_across_width_time = False
