import csv
import glob
import os
import re
import sys
from typing import List

import cv2
import numpy as np

# 現在のスクリプトのディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir の親ディレクトリパスを取得（functionsディレクトリの親、つまりscriptsディレクトリを指す）
parent_dir = os.path.dirname(current_dir)
# 親ディレクトリをシステムパスに追加
sys.path.append(parent_dir)

from functions import param


def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 時系列輪郭plotはテストしておいた方が良い

    max_area = 0.0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        mean_x = np.mean(max_contour[:, 0, 0].astype(float))
        mean_y = np.mean(max_contour[:, 0, 1].astype(float))
        if param.flag_get_angle_with_cell_direcetion:
            ellipse = cv2.fitEllipse(max_contour)
            angle = ellipse[2]
            return mean_x, mean_y, angle
        else:
            return mean_x, mean_y
    else:
        # If no contour is found, return None.
        return None, None


# Save centroid coordinates (cannot be written in save2csv.py due to subprocess)
def save_centorid_cordinate(save_dir, x_list, y_list):
    data_num = len(x_list)
    data_len = len(x_list[0])
    os.makedirs(save_dir, exist_ok=True)
    save_name = "centroid_coordinate.csv"

    with open(f"{save_dir}/{save_name}", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = [f"{xy}_{i+1}" for i in range(data_num) for xy in ("x", "y")]
        csvwriter.writerow(headers)
        for i in range(data_len):
            row = [item for pair in zip(x_list, y_list) for item in [pair[0][i], pair[1][i]]]
            csvwriter.writerow(row)


def save_angle(save_dir, angle_list):
    data_num = len(angle_list)
    data_len = len(angle_list[0])
    os.makedirs(save_dir, exist_ok=True)
    save_name = "angle.csv"

    with open(f"{save_dir}/{save_name}", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = [f"No.{i+1}" for i in range(data_num)]
        csvwriter.writerow(headers)
        for i in range(data_len):
            row = [angle_list[j][i] for j in range(data_num)]
            csvwriter.writerow(row)


def extract_centroid(day):
    input_dir = f"{param.input_dir_bef}/{day}"
    save_dir = f"{param.save_dir_bef}/{day}"

    file_name_list_bef = glob.glob(f"{input_dir}/*.avi")
    sort_num: List[tuple[str, int]] = []
    for x in file_name_list_bef:
        match = re.search(r"_([0-9]+)\.avi$", x)
        if match:
            sort_num.append((x, int(match.group(1))))
    sort_num.sort(key=lambda x: x[1])
    file_name_list_aft: List[str] = [x[0] for x in sort_num]

    x_list, y_list, angle_list = [], [], []
    for file_name in file_name_list_aft:
        movie = cv2.VideoCapture(file_name)
        add_x_list, add_y_list, add_angle_list = [], [], []
        while True:
            ret, frame = movie.read()
            if not ret:
                break
            if param.flag_get_angle_with_cell_direcetion:
                x, y, angle = contours(frame)
                add_angle_list.append(angle)
            else:
                x, y = contours(frame)
            add_x_list.append(x)
            add_y_list.append(y)
        x_list.append(add_x_list)
        y_list.append(add_y_list)
        if param.flag_get_angle_with_cell_direcetion:
            angle_list.append(add_angle_list)

    # save
    save_centorid_cordinate(save_dir, x_list, y_list)
    if param.flag_get_angle_with_cell_direcetion:
        save_angle(save_dir, angle_list)


if __name__ == "__main__":
    day = sys.argv[1]
    extract_centroid(day)
