import os

import pandas as pd

from utils import param
from utils.features import ROTATION_FEATURES, SD_WIDTH_DEPEND_COLS


def get_cols():
    col_list = []
    for feature in ROTATION_FEATURES.__annotations__.keys():
        if feature in SD_WIDTH_DEPEND_COLS:
            col_list.extend([f"{feature}_{sd_time}s" for sd_time in param.SD_window_width_list])
        else:
            col_list.append(feature)
    return col_list


def create_rot_df(day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/"
    csv_save_name = "rot_df.csv"
    os.makedirs(csv_save_dir, exist_ok=True)

    col_list = get_cols()
    rot_df = pd.DataFrame(columns=col_list)
    rot_df.to_csv(f"{csv_save_dir}/{csv_save_name}", index=False)


def update_rot_df(col, data, day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/rot_df.csv"

    rot_df = pd.read_csv(csv_save_dir, index_col=None)
    rot_df[col] = data
    rot_df.to_csv(csv_save_dir, index=False)
