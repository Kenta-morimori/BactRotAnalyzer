import configparser
import os

# directory information
curr_dir = os.getcwd()
input_dir_bef = f"{curr_dir}/../data"
save_dir_bef = f"{curr_dir}/../outputs"


def get_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    # データ数
    sample_num = config.getint("Settings", "sample_num")
    # 撮影条件
    FrameRate = config.getint("Settings", "FrameRate")
    total_time = config.getint("Settings", "total_time")

    return sample_num, FrameRate, total_time

# rotational analysis
flag_angular_velocity_correction = True
flag_evaluating_switching = True  # evaluate switching of rotation

# fluctuation analysis
SD_window_width_list = [0.1, 0.5, 1.0, 2.0]
