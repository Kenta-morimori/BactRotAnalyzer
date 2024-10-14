import configparser
import os

import pandas as pd

# directory information
curr_dir = os.getcwd()
input_dir_bef = f"{curr_dir}/data"
save_dir_bef = f"{curr_dir}/outputs"


def get_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    flag_use_tiff_log = config.getboolean("Settings", "flag_use_tiff_log")
    sample_num = config.getint("Settings", "sample_num")
    # Frame Rate, Total Time
    FrameRate_list = []
    total_time_list = []
    if flag_use_tiff_log:
        csv_save_dir = f"{save_dir_bef}/{day}/time_list.csv"
        df = pd.read_csv(csv_save_dir)
        time_list = df.values.T.tolist()
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
flag_get_angle_with_cell_direcetion = True

## About Angular Velocity
flag_angular_velocity_correction = True  # Trimming with thresholds
num_std_dev = 5

flag_evaluate_angular_velocity_abs = False  # Evaluate absolute values of angular velocity.

flag_evaluating_switching = True  # evaluate switching of rotation

# fluctuation analysis
SD_window_width_list = [0.1, 0.5, 1.0]
# SD_window_width_list = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

# evaluate SD FFT low Amp and decline
flag_evaluate_SD_FFT = True
