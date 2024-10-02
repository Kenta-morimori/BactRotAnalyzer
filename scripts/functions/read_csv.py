import os

import numpy as np
import pandas as pd

from . import param


def read_center_coordinates(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/center_coordinate.csv"
    sample_num, _, _ = param.get_config(day)

    df = pd.read_csv(csv_save_dir)
    center_x_list = df.filter(regex="No.\d+_x").iloc[0, :sample_num].tolist()
    center_y_list = df.filter(regex="No.\d+_y").iloc[0, :sample_num].tolist()

    return center_x_list, center_y_list


def read_angle(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/angle.csv"

    df = pd.read_csv(csv_save_dir)
    angle_list = df.values.T.tolist()

    return angle_list


def read_time_list(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/time_list.csv"

    df = pd.read_csv(csv_save_dir)
    time_list = df.values.T.tolist()

    return time_list


def get_timelist(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/time_list.csv"
    if os.path.isfile(csv_save_dir):
        time_list = read_time_list(day)
    else:
        sample_num, FrameRate_list, total_time_list = param.get_config(day)
        time_list = [
            np.linspace(0, total_time_list[i], int(total_time_list[i] * FrameRate_list[i])).tolist() for i in range(sample_num)
        ]

    return time_list
