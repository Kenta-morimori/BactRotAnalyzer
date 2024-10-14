import os
import re
from datetime import datetime

from PIL import Image

from utils import param
from utils.functions import save2csv


def extract_number(filename):
    match = re.search(r"_(\d+)\.tif$", filename)
    return int(match.group(1)) if match else float("inf")  # 数字がなければ大きな数を返す


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
                    datetime_tag = 306  # 306はDateTimeのタグID
                    time = datetime.strptime(metadata[datetime_tag], "%m/%d/%Y %H:%M:%S.%f")
                    if base_time is None:
                        # 最初のファイルの時間を基準時間として保存
                        base_time = time
                    else:
                        # 基準時間との差分を計算
                        time_diff = time - base_time
                        time_list.append(time_diff.total_seconds())
        time_list_all.append(time_list)
    save2csv.save_time_list(time_list_all, day)
