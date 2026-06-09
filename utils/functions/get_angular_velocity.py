import copy
import csv

import numpy as np

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import (
    clean_data,
    frequency_analysis,
    make_evaluate_switching,
    make_graph,
    read_csv,
    rot_df_manage,
    save2csv,
)


# Calculate centre coordinates using quadratic form.
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


# normalized angle to -π~π for FFT
def normalized_angle(angle_bef):
    angle_aft = [angle % (2 * np.pi) - np.pi for angle in angle_bef]

    return angle_aft


def get_angular_velocity(x_list, y_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    angle_list, angular_velocity_list = [], []
    angular_velocity_list_bef_corr = []

    if param.flag_get_angle_with_cell_direcetion:
        angle_list = read_csv.read_angle(day)

    for i in range(sample_num):
        # obtain angle
        if param.flag_get_angle_with_cell_direcetion:
            angle = np.array(angle_list[i])
        else:
            # center is zero
            x_arr, y_arr = np.array(x_list[i]), np.array(y_list[i])
            angle = np.arctan2(y_arr, x_arr)
            angle_list.append(angle)

        # obtain angular velocitiy
        add_angular_velocity = np.array([])
        for j in range(1, len(angle)):
            angle_diff = angle[j] - angle[j - 1]
            # 角度変化から回転方向を設定
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            # CCWを正にするために-1をかける
            add_angular_velocity = np.append(add_angular_velocity, -1 * angle_diff * FrameRate[i])

        # correct angular velocity
        add_angular_velocity_bef_corr = np.array(copy.deepcopy(add_angular_velocity))
        if param.flag_correct_av_outlier:
            add_angular_velocity = np.array(clean_data.correct_angular_velocity_outlier(add_angular_velocity, i, day))
        if param.flag_evaluate_angular_velocity_abs:
            angular_velocity_list.append(np.abs(add_angular_velocity))
            angular_velocity_list_bef_corr.append(np.abs(add_angular_velocity_bef_corr))
        else:
            angular_velocity_list.append(add_angular_velocity)
            angular_velocity_list_bef_corr.append(add_angular_velocity_bef_corr)

    # save
    # plot angle, angular velocity
    make_graph.plot_angular_velocity(angle_list, angular_velocity_list, day)

    if param.flag_correct_av_outlier:
        # check colleration
        make_graph.plot_av_colleration(angular_velocity_list_bef_corr, day)
    save2csv.save_angle_angular_velocity(angle_list, angular_velocity_list, day)
    rot_df_manage.update_rot_df(
        ROTATION_FEATURES.angular_velosity_mean,
        [np.nanmean(sublist) for sublist in angular_velocity_list],
        day,
    )
    rot_df_manage.update_rot_df(
        ROTATION_FEATURES.angular_velosity_sd,
        [np.nanstd(sublist) for sublist in angular_velocity_list],
        day,
    )

    # FFT
    frequency_analysis.fft_angle(angle_list, day)
    frequency_analysis.fft_angular_velocity(angular_velocity_list, day)

    # evaluate switching
    if param.flag_eval_switching_with_averaged_av:
        # obtain Angular Velocity mean
        time_list = read_csv.get_timelist(day)

        flag_use_conts_width_time = True
        conts_width_time = 0.1
        if not flag_use_conts_width_time:
            freq_list, Amp_list = read_csv.get_angle_FFT(day)

        angular_velocity_mean_list = []
        for i in range(sample_num):
            time_arr = np.array(time_list[i][: len(angular_velocity_list[i])])
            total_time_i = time_arr[-1]
            if flag_use_conts_width_time:
                width_time = conts_width_time
            else:
                if len(freq_list[i]) == 0 or len(Amp_list[i]) == 0 or not np.isfinite(Amp_list[i]).any():
                    width_time = conts_width_time
                else:
                    width_time = param.n_rotations / freq_list[i][int(np.nanargmax(Amp_list[i]))]

            add_angular_velocity_mean = []
            start_time = 0.0
            while 1:
                condition = (time_arr >= start_time) & (time_arr < start_time + width_time)
                condition = np.array(condition, dtype=bool)
                # av_i = angular_velocity_list[i][condition].reshape(-1, 1)
                av_i = angular_velocity_list[i][condition]

                if len(av_i) < param.min_ref_av_num:
                    add_angular_velocity_mean.append(np.nan)
                else:
                    add_angular_velocity_mean.append(np.nanmean(av_i))
                start_time += 1 / FrameRate[i]
                if start_time + width_time >= total_time_i:
                    break
            angular_velocity_mean_list.append(add_angular_velocity_mean)

        csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angular-velocity_time-series_mean.csv"
        header = [f"No.{i + 1}" for i in range(sample_num)]
        with open(csv_save_dir, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            for row in zip(*angle_list):
                csvwriter.writerow(row)

        # plot Averaged Angular Velocity
        make_graph.plot_averaged_angular_velocity(angular_velocity_list, angular_velocity_mean_list, day)

        # evaluate switching
        cw_ratio_list, switching_count_list = make_evaluate_switching.evaluate_switching(
            angular_velocity_mean_list, day
        )
    else:
        cw_ratio_list, switching_count_list = make_evaluate_switching.evaluate_switching(angular_velocity_list, day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.cw_ratio, cw_ratio_list, day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.switching_count, switching_count_list, day)

    return angle_list, angular_velocity_list
