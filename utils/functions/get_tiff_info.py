import os
import re
from datetime import datetime

import numpy as np
from PIL import Image

from utils import param
from utils.functions import save2csv


def extract_number(filename):
    match = re.search(r"_(\d+)\.tif$", filename)
    return int(match.group(1)) if match else float("inf")


def get_timelist(day):
    input_dir = f"{param.input_dir_bef}/{day}/tiff_data"
    data_name_list = param.get_tiffinfo_config(day)

    time_list_all = []
    for data_name in data_name_list:
        data_dir = f"{input_dir}/{data_name}"
        file_list = [f for f in os.listdir(data_dir) if f.endswith(".tif")]
        file_list_sorted = sorted(file_list, key=extract_number)

        base_time = None
        time_list = [0]
        for file_name in file_list_sorted:
            file_path = os.path.join(data_dir, file_name)
            if file_name.endswith(".tiff") or file_name.endswith(".tif"):
                with Image.open(file_path) as img:
                    # メタデータの取得
                    metadata = img.tag_v2
                    # 特定の時間関連情報を表示
                    datetime_tag = 306  # DateTime tag ID
                    time = datetime.strptime(metadata[datetime_tag], "%m/%d/%Y %H:%M:%S.%f")
                    if base_time is None:
                        base_time = time
                    else:
                        time_diff = time - base_time
                        time_list.append(time_diff.total_seconds())
        time_list_all.append(time_list)
    save2csv.save_time_list(time_list_all, day)


def detect_time_jumps_with_sd(time_list, day):
    sample_num, _, _ = param.get_config(day)

    jump_time_index_list: list[list[float]] = [[] for _ in range(sample_num)]

    for i in range(sample_num):
        time_diff = [time_list[i][j + 1] - time_list[i][j] for j in range(len(time_list[i]) - 1)]
        mean_diff = np.mean(time_diff)
        sd_diff = np.std(time_diff)
        threshold = mean_diff + 2 * sd_diff

        jump_time_index = [j for j, diff in enumerate(time_diff) if diff > threshold]
        jump_time_index_list[i].extend(jump_time_index)

    return jump_time_index_list
