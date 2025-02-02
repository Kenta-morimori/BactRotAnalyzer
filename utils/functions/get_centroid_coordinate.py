import csv
import glob
import math
import os
import re

# import statistics
import sys
from typing import List

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import frequency_analysis, make_graph, read_csv, rot_df_manage


def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def save_center_of_rotation(save_dir, center_x_list, center_y_list):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate/"
    csv_save_dir = f"{save_dir}/center_coordinate.csv"
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_x"] = center_x_list[i]
        data[f"No.{i+1}_y"] = center_y_list[i]
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


def save_rot_axes(long_axis_list, short_axis_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate/"
    csv_save_dir = f"{save_dir}/rotation_axes.csv"
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_long_axis"] = long_axis_list[i]
        data[f"No.{i+1}_short_axis"] = short_axis_list[i]
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


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


def calculate_ellipse_properties(X, Y, index):
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x_arr = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    x = x_arr.tolist()
    flag_Warning = False

    # Get information on ellipses using quadratic form
    A, B, C, D, E = x[0], x[1], x[2], x[3], x[4]
    mat = np.array([[A, B / 2], [B / 2, C]])
    eig_val_bef, eig_vec_bef = np.linalg.eig(mat)
    # Sort eigenvalues in descending order
    indices = np.argsort(eig_val_bef)[::-1]
    eig_val = eig_val_bef[indices]
    eig_vec = eig_vec_bef[:, indices]

    center_x_bef = -1 * (D * eig_vec[0][0] + E * eig_vec[1][0]) / (2 * eig_val[0])
    center_y_bef = -1 * (D * eig_vec[0][1] + E * eig_vec[1][1]) / (2 * eig_val[1])
    center_x = eig_vec[0][0] * center_x_bef + eig_vec[0][1] * center_y_bef
    center_y = eig_vec[1][0] * center_x_bef + eig_vec[1][1] * center_y_bef

    alfa = (
        1
        + ((D * eig_vec[0][0] + E * eig_vec[1][0]) ** 2) / (4 * eig_val[0])
        + ((D * eig_vec[0][1] + E * eig_vec[1][1]) ** 2) / (4 * eig_val[1])
    )
    if alfa / eig_val[0] < 0:
        # print(f"[Warning] No.{index + 1}   alfa / eig_val[0] is negative value")
        flag_Warning = True
        long_axis = math.sqrt(abs(alfa / eig_val[0]))
    else:
        long_axis = math.sqrt(alfa / eig_val[0])
    if alfa / eig_val[1] < 0:
        # print(f"[Warning] No.{index + 1}   alfa / eig_val[1] is negative value")
        flag_Warning = True
        short_axis = math.sqrt(abs(alfa / eig_val[1]))
    else:
        short_axis = math.sqrt(alfa / eig_val[1])

    return center_x, center_y, long_axis, short_axis, flag_Warning


def get_ellipse_info(X, Y, index, day):
    _, FrameRate, total_time = param.get_config(day)
    center_x_list, center_y_list = [], []
    long_axis_list, short_axis_list = [], []

    if param.flag_get_angle_with_cell_direcetion:
        size = len(X)
        X = X.reshape([size, 1])
        Y = Y.reshape([size, 1])
        center_x, center_y, long_axis, short_axis, _ = calculate_ellipse_properties(X, Y, index)
        center_x_list = [center_x] * size
        center_y_list = [center_y] * size
        long_axis_list = [long_axis] * size
        short_axis_list = [short_axis] * size
    else:
        time_list = read_csv.get_timelist(day)
        time_arr = np.array(time_list[index])

        # Determine the time window for obtaining the contour based on the FFT of the x_list
        x_freq_list, x_Amp_list = frequency_analysis.fft(X, 1 / FrameRate[index])
        y_freq_list, y_Amp_list = frequency_analysis.fft(Y, 1 / FrameRate[index])
        freq_th = 5  # Hz
        x_mask, y_mask = x_freq_list > freq_th, y_freq_list > freq_th
        x_freq_list, x_Amp_list = x_freq_list[x_mask], x_Amp_list[x_mask]
        y_freq_list, y_Amp_list = y_freq_list[y_mask], y_Amp_list[y_mask]

        width_time = param.n_rotations / max(x_freq_list[np.argmax(x_Amp_list)], y_freq_list[np.argmax(y_Amp_list)])
        # width_time = 5
        # width_time = time_arr[-1]

        start_time = 0.0
        flag_warning = False
        while 1:
            condition = (time_arr >= start_time) & (time_arr < start_time + width_time)
            X_aft = X[condition].reshape([np.sum(condition), 1])
            Y_aft = Y[condition].reshape([np.sum(condition), 1])
            center_x, center_y, long_axis, short_axis, add_flag_warning = calculate_ellipse_properties(
                X_aft, Y_aft, index
            )
            center_x_list.append(center_x)
            center_y_list.append(center_y)
            long_axis_list.append(long_axis)
            short_axis_list.append(short_axis)
            if add_flag_warning:
                flag_warning = True

            start_time += 1 / FrameRate[index]
            # width_timeの幅でSDが算出できない場合break
            if start_time + width_time >= total_time[index]:
                rest_data_num = len(time_arr) - len(center_x_list)
                center_x_list.extend([center_x] * rest_data_num)
                center_y_list.extend([center_y] * rest_data_num)
                long_axis_list.extend([long_axis] * rest_data_num)
                short_axis_list.extend([short_axis] * rest_data_num)
                break
        if flag_warning:
            print(f"[Warning] No.{index + 1}   alfa / eig_val[0] is negative value")
    return center_x_list, center_y_list, np.array(long_axis_list), np.array(short_axis_list)


def calculate_msd_sd(x_list, y_list, day):
    sample_num, FrameRate_list, _ = param.get_config(day)

    msd_list = []
    D_list = []  # diffusion coefficient
    for i in range(sample_num):
        x_arr = np.asarray(x_list[i])
        y_arr = np.asarray(y_list[i])
        dt = 1.0 / FrameRate_list[i]
        n_frames = len(x_arr)

        msds = [0.0]
        for tau in range(1, n_frames):
            dx = x_arr[tau:] - x_arr[:-tau]
            dy = y_arr[tau:] - y_arr[:-tau]
            msds.append(np.mean(dx**2 + dy**2))
        msd_list.append(np.array(msds))

        time_lags = np.arange(1, n_frames) * dt
        slope, _ = np.polyfit(time_lags, msds[1:], 1)
        D_list.append(slope / 4.0)

    return np.array(msd_list), D_list


def get_max_dist(x_list, y_list, day):
    sample_num, _, _ = param.get_config(day)

    max_dist_list = []
    for i in range(sample_num):
        N = len(x_list[i])
        max_dist_sq = 0
        for j in range(N):
            for k in range(j + 1, N):
                dist_sq = (x_list[i][j] - x_list[i][k]) ** 2 + (y_list[i][j] - y_list[i][k]) ** 2
                max_dist_sq = max(max_dist_sq, dist_sq)
        max_dist_list.append(np.sqrt(max_dist_sq))
    return max_dist_list


def extract_centroid(day):
    sample_num, _, _ = param.get_config(day)
    input_dir = f"{param.input_dir_bef}/{day}"
    save_dir = f"{param.save_dir_bef}/{day}"
    px2um_x, px2um_y = param.get_px2um_config(day)

    file_name_list_bef = glob.glob(f"{input_dir}/*.avi")
    sort_num: List[tuple[str, int]] = []
    for x in file_name_list_bef:
        match = re.search(r"_([0-9]+)\.avi$", x)
        if match:
            sort_num.append((x, int(match.group(1))))
    sort_num.sort(key=lambda x: x[1])
    file_name_list_aft: List[str] = [x[0] for x in sort_num]

    x_list_bef, y_list_bef, angle_list = [], [], []
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
            add_x_list.append(x * px2um_x)
            add_y_list.append(y * px2um_y)

        x_list_bef.append(add_x_list)
        y_list_bef.append(add_y_list)
        if param.flag_get_angle_with_cell_direcetion:
            # adjust angle (-π/2 ~ π/2)
            add_angle_list_aft = adjust_angle(add_angle_list_bef)
            angle_list.append(add_angle_list_aft)
    x_arr_bef = np.array(x_list_bef)
    y_arr_bef = np.array(y_list_bef)
    # dev
    make_graph.dev_plot_fft_coordinates(x_arr_bef, y_arr_bef, day)

    # exact center of rotation
    center_x_list, center_y_list = [], []
    long_axis_list, short_axis_list = [], []
    aspect_ratio_list = []
    for i in range(sample_num):
        center_x, center_y, long_axis, short_axis = get_ellipse_info(x_arr_bef[i], y_arr_bef[i], i, day)
        center_x_list.append(center_x)
        center_y_list.append(center_y)
        long_axis_list.append(long_axis)
        short_axis_list.append(short_axis)
        aspect_ratio_list.append(short_axis / long_axis)
    center_x_arr = np.array(center_x_list)
    center_y_arr = np.array(center_y_list)
    long_axis_arr = np.array(long_axis_list)
    short_axis_arr = np.array(short_axis_list)
    aspect_ratio_arr = np.array(aspect_ratio_list)

    # Fix x_list, y_list as center is zero
    x_list_aft = x_arr_bef - center_x_arr
    y_list_aft = y_arr_bef - center_y_arr

    # save
    save_centorid_cordinate(save_dir, x_list_aft, y_list_aft)
    make_graph.plot_coordinate(x_list_aft, y_list_aft, day, "centroid")
    save_center_of_rotation(save_dir, center_x_arr, center_y_arr)
    make_graph.plot_coordinate(center_x_arr, center_y_arr, day, "center")
    make_graph.plot_coordinate_with_center(x_arr_bef, y_arr_bef, center_x_arr, center_y_arr, day)

    # save long_axis, short_axis
    save_rot_axes(long_axis_arr, short_axis_arr, day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_long_axis, np.mean(long_axis_arr, axis=1), day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_short_axis, np.mean(short_axis_arr, axis=1), day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_aspect_ratio, np.mean(aspect_ratio_arr, axis=1), day)

    # rotaion center analysis
    if param.flag_evaluate_rotaion_center:
        max_dist_list = get_max_dist(center_x_arr, center_y_arr, day)
        # MSD
        msd_2d, D_list = calculate_msd_sd(center_x_arr, center_y_arr, day)
        make_graph.plot_msd(msd_2d, D_list, max_dist_list, day)

    if param.flag_get_angle_with_cell_direcetion:
        save_angle(save_dir, angle_list)


if __name__ == "__main__":
    day = sys.argv[1]
    extract_centroid(day)
