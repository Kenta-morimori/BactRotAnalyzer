import os
import subprocess

import pandas as pd

from utils import param

from utils.features import ROTATION_FEATURES
from utils.functions import rot_df_manage, read_csv


def input_centroid_coordinate(day):
    save_dir = f"{param.save_dir_bef}/{day}"
    csv_dir = f"{save_dir}/centroid_coordinate.csv"

    if not os.path.isfile(csv_dir):
        subprocess.run(["Python3", "utils/functions/get_centroid_coordinate.py", day])
    else:
        long_axis_list, short_axis_list = read_csv.get_rot_axes(day)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_long_axis, long_axis_list, day)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_short_axis, short_axis_list, day)

    x_list, y_list = [], []
    df = pd.read_csv(csv_dir)
    column_list = df.columns.tolist()

    for i, column_name in enumerate(column_list):
        if i % 2 == 0:
            x_list.append(df[column_name])
        else:
            y_list.append(df[column_name])

    return x_list, y_list
