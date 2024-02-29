import os
import subprocess

import pandas as pd

from . import param


def input_centroid_coordinate(day):
    save_dir = f"{param.save_dir_bef}/{day}/"
    csv_dir = f"{save_dir}/saved_centroid_coordinate.csv"

    # 中心座標の保存
    if not os.path.isfile(csv_dir):
        subprocess.run(["Python3", "functions/get_centroid_coordinate.py", day])

    # 中心座標の取得
    x_list, y_list = [], []
    df = pd.read_csv(csv_dir)
    column_list = df.columns.tolist()

    for i, column_name in enumerate(column_list):
        if i % 2 == 0:
            x_list.append(df[column_name])
        else:
            y_list.append(df[column_name])

    return x_list, y_list
