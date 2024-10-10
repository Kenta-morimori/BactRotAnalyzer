import csv
import glob
import math
import os
import re
import sys
from typing import List

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils import param


def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Centroid coordinates were taken as the mean of the contours.
    if max_contour is not None:
        mean_x = np.mean(max_contour[:, 0, 0].astype(float))
        mean_y = np.mean(max_contour[:, 0, 1].astype(float))
        ellipse = cv2.fitEllipse(max_contour)
        return mean_x, mean_y, ellipse
    else:
        return None, None, None


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


def adjust_angle(angle_list_bef):
    angle_list_aft = [angle_list_bef[0] + 90]
    for j in range(len(angle_list_bef) - 1):
        d_deg = angle_list_bef[j + 1] - angle_list_bef[j]
        if d_deg < -100:
            angle_list_aft.append(angle_list_aft[j] + 180 - angle_list_bef[j] + angle_list_bef[j + 1])
        elif d_deg > 100:
            angle_list_aft.append(angle_list_aft[j] - (180 - angle_list_bef[j + 1] + angle_list_bef[j]))
        else:
            angle_list_aft.append(angle_list_aft[j] + d_deg)
    # degree --> radian
    angle_list_aft_rad = [math.radians(angle) for angle in angle_list_aft]

    return angle_list_aft_rad


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


def get_center(x):
    A, B, C, D, E = x[0], x[1], x[2], x[3], x[4]
    mat = np.array([[A, B / 2], [B / 2, C]])
    eig_val, eig_vec = np.linalg.eig(mat)

    center_x_bef = -1 * (D * eig_vec[0][0] + E * eig_vec[1][0]) / (2 * eig_val[0])
    center_y_bef = -1 * (D * eig_vec[0][1] + E * eig_vec[1][1]) / (2 * eig_val[1])
    center_x = eig_vec[0][0] * center_x_bef + eig_vec[0][1] * center_y_bef
    center_y = eig_vec[1][0] * center_x_bef + eig_vec[1][1] * center_y_bef

    return center_x, center_y


def get_center_coordinate(X, Y):
    size = len(X)
    X = X.reshape([size, 1])
    Y = Y.reshape([size, 1])

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    center_x, center_y = get_center(x.tolist())

    return center_x, center_y


def scale_center_zero(coordinates_bef, center):
    coordinates_aft = []
    for i in range(len(coordinates_bef)):
        coordinates_aft.append(coordinates_bef[i] - center[i])

    return coordinates_aft


def save_center_of_rotation(save_dir, center_x_list, center_y_list):
    save_name = "center_coordinate.csv"

    sample_num, _, _ = param.get_config(day)
    with open(f"{save_dir}/{save_name}", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = []
        center_list = []
        for i in range(sample_num):
            headers.extend([f"No.{i+1}_x", f"No.{i+1}_y"])
            center_list.extend([center_x_list[i], center_y_list[i]])
        csvwriter.writerow(headers)
        csvwriter.writerow(center_list)


# Trimming with thresholds
def correct_angular_velocity(data):
    num_std_dev = 3
    data_aft = []

    mean = np.mean(data)
    std_dev = np.std(data)
    lower_th = mean - num_std_dev * std_dev
    upper_th = mean + num_std_dev * std_dev

    for x in data:
        if x < lower_th or upper_th < x:
            data_aft.append(mean)
        else:
            data_aft.append(x)

    return data_aft


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
        add_x_list, add_y_list, add_angle_list_bef = [], [], []
        while True:
            ret, frame = movie.read()
            if not ret:
                break
            if param.flag_get_angle_with_cell_direcetion:
                x, y, ellipse = contours(frame)
                add_angle_list_bef.append(ellipse[2])
            else:
                x, y, _ = contours(frame)
            add_x_list.append(x)
            add_y_list.append(y)

        x_list.append(add_x_list)
        y_list.append(add_y_list)
        if param.flag_get_angle_with_cell_direcetion:
            # adjust angle (-π/2 ~ π/2)
            add_angle_list_aft = adjust_angle(add_angle_list_bef)
            angle_list.append(add_angle_list_aft)

    # exact center of rotation
    center_x_list, center_y_list = [], []
    for i in range(len(x_list)):
        x_arr, y_arr = np.array(x_list[i]), np.array(y_list[i])
        center_x, center_y = get_center_coordinate(x_arr, y_arr)
        center_x_list.append(center_x)
        center_y_list.append(center_y)

    # Fix x_list, y_list as center is zero
    x_list_aft = scale_center_zero(x_list, center_x_list)
    y_list_aft = scale_center_zero(y_list, center_y_list)

    # save
    save_centorid_cordinate(save_dir, x_list_aft, y_list_aft)
    save_center_of_rotation(save_dir, center_x_list, center_y_list)
    if param.flag_get_angle_with_cell_direcetion:
        save_angle(save_dir, angle_list)


if __name__ == "__main__":
    day = sys.argv[1]
    extract_centroid(day)
