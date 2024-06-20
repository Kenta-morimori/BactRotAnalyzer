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

    sample_num = config.getint("Settings", "sample_num")
    try:
        FrameRate = config.getfloat("Settings", "FrameRate")
    except (ValueError, TypeError):
        FrameRate = config.getint("Settings", "FrameRate")
    try:
        total_time = config.getfloat("Settings", "total_time")
    except (ValueError, TypeError):
        total_time = config.getint("Settings", "total_time")

    return sample_num, FrameRate, total_time


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


# rotational analysis
## Determine the angle by the direction of the cell.
flag_get_angle_with_cell_direcetion = False
## About Angular Velocity
flag_angular_velocity_correction = True
flag_evaluate_angular_velocity_abs = True  # Evaluate absolute values of angular velocity.

flag_evaluating_switching = True  # evaluate switching of rotation

# fluctuation analysis
SD_window_width_list = [0.1, 0.5, 1.0]
# SD_window_width_list = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
