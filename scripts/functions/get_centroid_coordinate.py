import csv
import glob
import os
import re
import sys

import cv2
import numpy as np

# 現在のスクリプトのディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir の親ディレクトリパスを取得（functionsディレクトリの親、つまりscriptsディレクトリを指す）
parent_dir = os.path.dirname(current_dir)
# 親ディレクトリをシステムパスに追加
sys.path.append(parent_dir)

from functions import param


# 重心座標の取得
def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0.0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        # 輪郭ポイントの平均値を重心とする
        mean_x = np.mean(max_contour[:, 0, 0].astype(float))
        mean_y = np.mean(max_contour[:, 0, 1].astype(float))
        return mean_x, mean_y
    else:
        # 輪郭が見つからない場合はNoneを返す
        return None, None


# 重心座標の保存(subprocessの関係で save2csv.py内には書き込み不可)
def save_centorid_cordinate(save_dir, save_name, data_num, x_list, y_list):
    data_len = len(x_list[0])
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/{save_name}", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーを書き込む
        headers = [f"{xy}_{i+1}" for i in range(data_num) for xy in ("x", "y")]
        csvwriter.writerow(headers)
        # データを書き込む
        for i in range(data_len):
            # xとyのデータを交互に配置
            row = [item for pair in zip(x_list, y_list) for item in [pair[0][i], pair[1][i]]]
            csvwriter.writerow(row)


def extract_centroid(day):
    input_dir = f"{param.input_dir_bef}/{day}"
    save_dir = f"{param.save_dir_bef}/{day}"
    csv_save_name = "saved_centroid_coordinate.csv"
    # input dataのファイル名を取得
    file_name_list_bef = glob.glob(f"{input_dir}/*.avi")
    file_name_list_aft = []
    for x in file_name_list_bef:
        match = re.search(r"(\d+)\.avi$", x)
        if match:  # ここで match が None でないことを確認
            file_name_list_aft.append(x)

    x_list, y_list = [], []
    for file_name in file_name_list_aft:
        movie = cv2.VideoCapture(file_name)
        # 重心座標の取得
        add_x_list, add_y_list = [], []
        while True:
            ret, frame = movie.read()
            if not ret:
                break
            x, y = contours(frame)
            add_x_list.append(x)
            add_y_list.append(y)
        x_list.append(add_x_list)
        y_list.append(add_y_list)

    # csv形式に保存
    save_centorid_cordinate(save_dir, csv_save_name, len(x_list), x_list, y_list)


if __name__ == "__main__":
    day = sys.argv[1]
    extract_centroid(day)
