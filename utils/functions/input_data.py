import os
import glob
import re
import subprocess

import pandas as pd

from utils import param

# from utils.features import ROTATION_FEATURES
# from utils.functions import read_csv, rot_df_manage


def get_ordered_avi_paths(day):
    """Resolve AVI inputs in the same order as the dataset's TIFF config."""
    input_dir = f"{param.input_dir_bef}/{day}"
    avi_paths = glob.glob(f"{input_dir}/*.avi")
    if not avi_paths:
        raise FileNotFoundError(f"No AVI files found in {input_dir}.")

    if not param.get_flag_use_tiff_log(day):
        return sorted(avi_paths, key=lambda path: int(re.findall(r"\d+", os.path.basename(path))[-1]))

    tiff_names = param.get_tiffinfo_config(day)
    if len(tiff_names) != len(set(tiff_names)):
        raise ValueError(f"Tiff_info.tiff_data contains duplicate sample names for {day}.")
    expected_names = [f"{name}.avi" for name in tiff_names]
    avi_by_name = {os.path.basename(path): path for path in avi_paths}
    if len(avi_by_name) != len(avi_paths):
        raise ValueError(f"Duplicate AVI basenames found for {day}.")

    missing = sorted(set(expected_names) - set(avi_by_name))
    unexpected = sorted(set(avi_by_name) - set(expected_names))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ValueError(f"TIFF/AVI mapping mismatch for {day}: " + ", ".join(details))
    return [avi_by_name[name] for name in expected_names]


def get_tiff_avi_sample_map(day):
    """Return a traceable config-order mapping for TIFF-log datasets."""
    paths = get_ordered_avi_paths(day)
    if not param.get_flag_use_tiff_log(day):
        return [
            {"sample_no": index + 1, "tiff_data": "", "avi_filename": os.path.basename(path), "mapping_status": "sorted"}
            for index, path in enumerate(paths)
        ]
    names = param.get_tiffinfo_config(day)
    return [
        {
            "sample_no": index + 1,
            "tiff_data": name,
            "avi_filename": os.path.basename(path),
            "mapping_status": "matched",
        }
        for index, (name, path) in enumerate(zip(names, paths))
    ]


def input_centroid_coordinate(day):
    save_dir = f"{param.save_dir_bef}/{day}"
    csv_dir = f"{save_dir}/centroid_coordinate.csv"

    if not os.path.isfile(csv_dir):
        subprocess.run(["Python3", "utils/functions/get_centroid_coordinate.py", day])
    """
    else:
        long_axis_list, short_axis_list, aspect_ratio_list = read_csv.get_rot_axes(day)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_long_axis, long_axis_list, day)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_short_axis, short_axis_list, day)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_aspect_ratio, aspect_ratio_list, day)
    """

    x_list, y_list = [], []
    df = pd.read_csv(csv_dir)
    column_list = df.columns.tolist()

    for i, column_name in enumerate(column_list):
        # Preserve missing frames so each coordinate remains aligned with its
        # original TIFF/AVI frame and the matching time-series entry.
        col_data = pd.to_numeric(df[column_name], errors="coerce")
        if i % 2 == 0:
            x_list.append(col_data)
        else:
            y_list.append(col_data)

    return x_list, y_list
