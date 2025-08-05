import csv
import glob
import math
import os
import re

# import statistics
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import (
    clean_data,
    frequency_analysis,
    make_graph,
    read_csv,
    rot_df_manage,
)


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
    max_len = max(len(x_list[i]) for i in range(data_num))
    os.makedirs(save_dir, exist_ok=True)
    save_name = "centroid_coordinate.csv"

    with open(os.path.join(save_dir, save_name), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = [f"{xy}_{i+1}" for i in range(data_num) for xy in ("x", "y")]
        csvwriter.writerow(headers)
        for i in range(max_len):
            row = []
            for j in range(data_num):
                x_val = x_list[j][i] if i < len(x_list[j]) else ""
                y_val = y_list[j][i] if i < len(y_list[j]) else ""
                row.extend([x_val, y_val])
            csvwriter.writerow(row)


def save_center_of_rotation(save_dir, center_x_list, center_y_list):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate/"
    csv_save_dir = f"{save_dir}/center_coordinate.csv"
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_x"] = pd.Series(center_x_list[i])
        data[f"No.{i+1}_y"] = pd.Series(center_y_list[i])
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


def save_rot_axes(long_axis_list, short_axis_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/other_rot_features/"
    csv_save_dir = f"{save_dir}/rotation_axes.csv"
    os.makedirs(save_dir, exist_ok=True)

    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_long_axis"] = long_axis_list[i]
        data[f"No.{i+1}_short_axis"] = short_axis_list[i]
    # Pad lists to the same length if necessary
    max_len = max(len(data[f"No.{i+1}_long_axis"]) for i in range(sample_num))
    for i in range(sample_num):
        long_axis = data[f"No.{i+1}_long_axis"]
        short_axis = data[f"No.{i+1}_short_axis"]
        if len(long_axis) < max_len:
            long_axis = list(long_axis) + [None] * (max_len - len(long_axis))
            data[f"No.{i+1}_long_axis"] = long_axis
        if len(short_axis) < max_len:
            short_axis = list(short_axis) + [None] * (max_len - len(short_axis))
            data[f"No.{i+1}_short_axis"] = short_axis
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


def calculate_ellipse_properties(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

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

    if isinstance(X, list):
        X = np.array(X)
    if isinstance(Y, list):
        Y = np.array(Y)

    if param.flag_get_angle_with_cell_direcetion:
        size = len(X)
        X = X.reshape([size, 1])
        Y = Y.reshape([size, 1])
        center_x, center_y, long_axis, short_axis, _ = calculate_ellipse_properties(X, Y)
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
        print(f"No.{index + 1}   width_time: {width_time:.2f} s   min_ref_centroid_num: {param.min_ref_centroid_num}")

        start_time = 0.0
        flag_warning = False
        while 1:
            condition = (time_arr >= start_time) & (time_arr < start_time + width_time)
            condition = np.array(condition, dtype=bool)
            X_aft = X[condition].reshape(-1, 1)
            Y_aft = Y[condition].reshape(-1, 1)

            # Set a threshold for the number of centroid coordinate data points.
            if (len(X_aft) < param.min_ref_centroid_num) or (len(X_aft) < param.min_ref_centroid_num):
                center_x = np.nan
                center_y = np.nan
                long_axis = np.nan
                short_axis = np.nan
                add_flag_warning = False
            else:
                center_x, center_y, long_axis, short_axis, add_flag_warning = calculate_ellipse_properties(X_aft, Y_aft)

            center_x_list.append(center_x)
            center_y_list.append(center_y)
            long_axis_list.append(long_axis)
            short_axis_list.append(short_axis)
            if add_flag_warning:
                flag_warning = True

            start_time += 1 / FrameRate[index]
            if start_time + width_time >= total_time[index]:
                rest_data_num = len(time_arr) - len(center_x_list)

                # Imputation of missing values
                if np.isnan(center_x_list).any():
                    valid_center_x = [v for v in center_x_list if not np.isnan(v)]
                    mean_center_x = np.mean(valid_center_x) if valid_center_x else 0
                    center_x_list = [mean_center_x if np.isnan(v) else v for v in center_x_list]
                if np.isnan(center_y_list).any():
                    valid_center_y = [v for v in center_y_list if not np.isnan(v)]
                    mean_center_y = np.mean(valid_center_y) if valid_center_y else 0
                    center_y_list = [mean_center_y if np.isnan(v) else v for v in center_y_list]

                # center_x_list.extend([np.mean(center_x_list)] * rest_data_num)
                # center_y_list.extend([np.mean(center_y_list)] * rest_data_num)
                center_x_list.extend([center_x_list[-1]] * rest_data_num)
                center_y_list.extend([center_y_list[-1]] * rest_data_num)
                break
        if flag_warning:
            print(f"[Warning] No.{index + 1}   alfa / eig_val[0] is negative value")
    return center_x_list, center_y_list, np.array(long_axis_list), np.array(short_axis_list), rest_data_num


def calculate_msd(x_arr, y_arr, rest_data_num_list, day):
    sample_num, FrameRate_list, _ = param.get_config(day)

    msd_list = []
    D_list = []  # diffusion coefficient
    intercept_list = []
    for i in range(sample_num):
        if not isinstance(x_arr[i], np.ndarray):
            x_arr_i = np.asarray(x_arr[i])
        else:
            x_arr_i = x_arr[i]
        if not isinstance(y_arr[i], np.ndarray):
            y_arr_i = np.asarray(y_arr[i])
        else:
            y_arr_i = y_arr[i]

        x_arr_i = x_arr_i[: -rest_data_num_list[i]]
        y_arr_i = y_arr_i[: -rest_data_num_list[i]]

        dt = 1.0 / FrameRate_list[i]
        n_frames = len(x_arr_i)

        msds = [0.0]
        for tau in range(1, n_frames):
            dx = x_arr_i[tau:] - x_arr_i[:-tau]
            dy = y_arr_i[tau:] - y_arr_i[:-tau]
            msds.append(np.mean(dx**2 + dy**2))
        msd_list.append(np.array(msds))

        time_lags = np.arange(1, n_frames) * dt
        slope, intercept = np.polyfit(time_lags, msds[1:], 1)
        D_list.append(slope / 4.0)
        intercept_list.append(intercept)

    return np.array(msd_list, dtype=object), D_list, intercept_list


def get_max_dist(x_arr, y_arr, rest_data_num_list, day):
    sample_num, _, _ = param.get_config(day)

    max_dist_list = []
    print("*** Calculating maximum distance ***")
    for i in range(sample_num):
        if not isinstance(x_arr[i], np.ndarray):
            x_arr_i = np.asarray(x_arr[i])
        else:
            x_arr_i = x_arr[i]
        if not isinstance(y_arr[i], np.ndarray):
            y_arr_i = np.asarray(y_arr[i])
        else:
            y_arr_i = y_arr[i]

        x_arr_i = x_arr_i[: -rest_data_num_list[i]]
        y_arr_i = y_arr_i[: -rest_data_num_list[i]]

        # Check if lengths of x_list and y_list match
        try:
            if len(x_arr_i) != len(y_arr_i):
                raise ValueError(
                    f"Length mismatch in sample {i + 1}: x_list length is {len(x_arr_i)}, y_list length is {len(y_arr_i)}"
                )
        except ValueError as e:
            print(e)
            continue

        N = len(x_arr_i)
        max_dist_sq = 0
        print(f"No.{i + 1}")
        for j in range(N):
            if (j + 1) % 1000 == 0:
                print(f"{j + 1} / {N}")
            for k in range(j + 1, N):
                dist_sq = (x_arr_i[j] - x_arr_i[k]) ** 2 + (y_arr_i[j] - y_arr_i[k]) ** 2
                max_dist_sq = max(max_dist_sq, dist_sq)
        max_dist_list.append(np.sqrt(max_dist_sq))
    return max_dist_list


def fill_trailing_nan(row):
    row = np.array(row, dtype=float)
    valid_idx = np.where(~np.isnan(row))[0]
    if valid_idx.size == 0:
        return row
    last_valid_idx = valid_idx[-1]
    row[last_valid_idx + 1 :] = row[last_valid_idx]
    return row


def save_msd(msd_2d, D_list, max_dist_list, day):
    sample_num, _, FrameRate_list = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate/"
    os.makedirs(save_dir, exist_ok=True)

    # MSD
    csv_save_dir = f"{save_dir}/center_msd.csv"
    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_t"] = np.arange(0, len(msd_2d[i])) * (1.0 / FrameRate_list[i])
        data[f"No.{i+1}_msd"] = msd_2d[i]
    max_len = max(len(v) for v in data.values())
    df = pd.DataFrame({k: list(v) + [None] * (max_len - len(v)) for k, v in data.items()})
    df.to_csv(csv_save_dir, index=False)

    # D_list, max_dist
    csv_save_dir = f"{save_dir}/Diffusion.csv"
    data = {}
    for i in range(sample_num):
        data[f"No.{i+1}_D"] = [D_list[i]]
        data[f"No.{i+1}_max_dist"] = [max_dist_list[i]]
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


def dev_get_max_dists(x_list, y_list, day, split_time=0.5):
    sample_num, _, _ = param.get_config(day)
    time_list = read_csv.get_timelist(day)

    x_list = [np.array(row)[~np.isnan(row)] for row in x_list]
    y_list = [np.array(row)[~np.isnan(row)] for row in y_list]

    max_dist_list = []
    for i in range(sample_num):
        x_arr_org = np.array(x_list[i])
        y_arr_org = np.array(y_list[i])
        time_arr = np.array(time_list[i])
        time_arr = time_arr[: min(len(time_arr), len(x_arr_org), len(y_arr_org))]

        max_dists = []
        time_th = 0
        while time_th + split_time <= time_arr[-1]:
            x_arr = x_arr_org[np.where((time_arr >= time_th) & (time_arr <= time_th + split_time))[0]]
            y_arr = y_arr_org[np.where((time_arr >= time_th) & (time_arr <= time_th + split_time))[0]]

            N = len(x_arr)
            max_dist_sq = 0
            for j in range(N):
                for k in range(j + 1, N):
                    dist_sq = (x_arr[j] - x_arr[k]) ** 2 + (y_arr[j] - y_arr[k]) ** 2
                    max_dist_sq = max(max_dist_sq, dist_sq)
            max_dists.append(np.sqrt(max_dist_sq))
            time_th += split_time
        max_dist_list.append(max_dists)

    return np.array(max_dist_list, dtype=object)


def extract_centroid(day):
    sample_num, _, _ = param.get_config(day)
    input_dir = f"{param.input_dir_bef}/{day}"
    save_dir = f"{param.save_dir_bef}/{day}"
    px2um_x, px2um_y = param.get_px2um_config(day)

    file_name_list_bef = glob.glob(f"{input_dir}/*.avi")
    file_name_list_aft = sorted(file_name_list_bef, key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))

    if len(file_name_list_aft) == 0:
        print("Error: No .avi files found in the input directory. Please check the path and file existence.")
        sys.exit(1)

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
    x_arr_bef = np.array(x_list_bef, dtype=object)
    y_arr_bef = np.array(y_list_bef, dtype=object)
    # dev
    make_graph.dev_plot_fft_coordinates(x_arr_bef, y_arr_bef, day)

    # exact center of rotation
    center_x_list, center_y_list = [], []
    long_axis_list, short_axis_list = [], []
    aspect_ratio_list = []
    rest_data_num_list = []
    if param.flag_correct_center_outlier:
        center_x_list_bef, center_y_list_bef = [], []
    for i in range(sample_num):
        center_x_bef, center_y_bef, long_axis, short_axis, rest_data_num = get_ellipse_info(
            x_arr_bef[i], y_arr_bef[i], i, day
        )

        # Correct rotation center
        if param.flag_correct_center_outlier:
            center_x_list_bef.append(center_x_bef)
            center_y_list_bef.append(center_y_bef)
            center_x_aft = clean_data.correct_rotation_center(center_x_bef, i, day)
            center_y_aft = clean_data.correct_rotation_center(center_y_bef, i, day)
        else:
            center_x_aft = center_x_bef
            center_y_aft = center_y_bef

        center_x_list.append(center_x_aft)
        center_y_list.append(center_y_aft)
        long_axis_list.append(long_axis)
        short_axis_list.append(short_axis)
        aspect_ratio_list.append(short_axis / long_axis)
        rest_data_num_list.append(rest_data_num)
    center_x_arr = np.array(center_x_list, dtype=object)
    center_y_arr = np.array(center_y_list, dtype=object)
    long_axis_arr = np.array(long_axis_list, dtype=object)
    short_axis_arr = np.array(short_axis_list, dtype=object)
    aspect_ratio_arr = np.array(aspect_ratio_list, dtype=object)
    make_graph.plot_rot_axes(long_axis_arr, short_axis_arr, aspect_ratio_arr, day)

    if param.flag_correct_center_outlier:
        make_graph.plot_center_colleration(
            np.array(center_x_list_bef, dtype=object), np.array(center_y_list_bef, dtype=object), day
        )
        make_graph.dev_plot_centroid_and_center(x_arr_bef, y_arr_bef, center_x_list_bef, center_y_list_bef, day)

    # rotaion center analysis
    if param.flag_evaluate_rotaion_center_movement:
        max_dist_list = get_max_dist(center_x_arr, center_y_arr, rest_data_num_list, day)
        # MSD
        msd_2d, D_list, intercept_list = calculate_msd(center_x_arr, center_y_arr, rest_data_num_list, day)
        make_graph.plot_msd(msd_2d, D_list, intercept_list, max_dist_list, day)
        save_msd(msd_2d, D_list, max_dist_list, day)
        # dev
        max_dist_list_st = dev_get_max_dists(center_x_arr, center_y_arr, day)
        make_graph.dev_plot_max_dist_stat(max_dist_list_st, max_dist_list, day)

    # Completes missing values with the last value
    # center_x_arr = np.apply_along_axis(fill_trailing_nan, 1, center_x_arr)
    # center_y_arr = np.apply_along_axis(fill_trailing_nan, 1, center_y_arr)
    center_x_arr = np.array([fill_trailing_nan(row) for row in center_x_arr], dtype=object)
    center_y_arr = np.array([fill_trailing_nan(row) for row in center_y_arr], dtype=object)

    # Fix x_list, y_list as center is zero
    x_list_aft = x_arr_bef - center_x_arr
    y_list_aft = y_arr_bef - center_y_arr

    # save
    save_center_of_rotation(save_dir, center_x_arr, center_y_arr)
    make_graph.plot_coordinate(center_x_arr, center_y_arr, day, "center")
    make_graph.plot_coordinate_with_center(x_arr_bef, y_arr_bef, center_x_arr, center_y_arr, day)
    save_centorid_cordinate(save_dir, x_list_aft, y_list_aft)
    make_graph.plot_coordinate(x_list_aft, y_list_aft, day, "centroid")

    # save long_axis, short_axis
    save_rot_axes(long_axis_arr, short_axis_arr, day)
    rot_df_manage.update_rot_df(
        ROTATION_FEATURES.rot_long_axis, np.array([np.mean(x) if np.size(x) else np.nan for x in long_axis_arr]), day
    )
    rot_df_manage.update_rot_df(
        ROTATION_FEATURES.rot_short_axis, np.array([np.mean(x) if np.size(x) else np.nan for x in short_axis_arr]), day
    )
    rot_df_manage.update_rot_df(
        ROTATION_FEATURES.rot_aspect_ratio,
        np.array([np.mean(x) if np.size(x) else np.nan for x in aspect_ratio_arr]),
        day,
    )

    if param.flag_get_angle_with_cell_direcetion:
        save_angle(save_dir, angle_list)

    # obtaion r_list
    r_list = []
    for i in range(sample_num):
        if not isinstance(x_list_bef[i], np.ndarray):
            x_arr_i = np.asarray(x_list_bef[i])
        else:
            x_arr_i = x_list_bef[i]
        if not isinstance(y_list_bef[i], np.ndarray):
            y_arr_i = np.asarray(y_list_bef[i])
        else:
            y_arr_i = y_list_aft[i]
        center_x_arr_i = center_x_arr[i]
        center_y_arr_i = center_y_arr[i]
        r_list.append(np.sqrt((x_arr_i - center_x_arr_i) ** 2 + (y_arr_i - center_y_arr_i) ** 2))
    r_arr = np.array(r_list, dtype=object)
    make_graph.plot_r(r_arr, day)


if __name__ == "__main__":
    day = sys.argv[1]
    extract_centroid(day)
