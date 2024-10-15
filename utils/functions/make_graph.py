import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from utils import param
from utils.features import ROTATION_FEATURES, SD_WIDTH_DEPEND_COLS
from utils.functions import read_csv, rot_df_manage, save2csv

font_size = 20
fig_size_x = 20
fig_size_y = 23


def plot_centroid_coordinate(x_list, y_list, day):
    sample_num, _, _ = param.get_config(day)

    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    # Pixel to µm conversion.
    # x_list, y_list = (np.array(x_list) * px2um_x).tolist(), (np.array(y_list) * px2um_y).tolist()

    # plot centroid coordinate
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, sample_num // 2, figure=fig, wspace=0.38, hspace=0.2)
    for i in range(sample_num):
        # detect x_lim, y_lim
        x_range = max(x_list[i]) - min(x_list[i])
        y_range = max(y_list[i]) - min(y_list[i])
        max_range = 1.1 * max(x_range, y_range) / 2

        row = i // (sample_num // 2)
        col = i % (sample_num // 2)

        ax = fig.add_subplot(gs[row, col])
        ax.plot(x_list[i], y_list[i])
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.scatter(0, 0, c="red")  # center is zero
        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.set_title(f"Trajectory No.{i+1}", fontsize=16)
        ax.set_xlabel(r"x [$\mu$m]", fontsize=16)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
    plt.savefig(f"{save_dir}/trajectory.png")
    plt.close(fig)

    # plot x, y
    time_list = read_csv.get_timelist(day)
    xy_plot_label = [r"x [$\mu$m]", r"y [$\mu$m]"]
    xy_save_label = ["x_coordinate.png", "y_coordinate.png"]

    for label_i, xy_list in enumerate([x_list, y_list]):
        fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
        for i in range(sample_num):
            row = i // 2
            col = i % 2
            axs[row, col].plot(time_list[i], xy_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"Trajectory No.{i+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=18)
            axs[row, col].set_ylabel(xy_plot_label[label_i], fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{xy_save_label[label_i]}")
        plt.close(fig)


def plot_angular_velocity(angle_list, angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[i], angle_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Angle Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angle [rad]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angle_time-series.png")
    plt.close(fig)

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular-velocity_time-series.png")
    plt.close(fig)


def plot_av_colleration(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    num_std_dev = param.num_std_dev
    time_list = read_csv.get_timelist(day)

    fig, axs = plt.subplots(
        5,
        2 * sample_num // 5,
        figsize=(fig_size_x, fig_size_y),
        gridspec_kw={"width_ratios": [5, 1] * (sample_num // 5)},
    )
    axs = axs.flatten()
    for i in range(sample_num):
        mean = np.mean(angular_velocity_list[i])
        std_dev = np.std(angular_velocity_list[i])
        lower_th = mean - num_std_dev * std_dev
        upper_th = mean + num_std_dev * std_dev
        # Time series plot (left plot)
        ax_ts = axs[2 * i]
        ax_ts.plot(time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i])
        ax_ts.axhline(lower_th, color="red", linestyle="--")
        ax_ts.axhline(upper_th, color="red", linestyle="--")
        ax_ts.set_title(f"Angular Velocity No.{i + 1}")
        # Distribution plot (right plot)
        ax_dist = axs[2 * i + 1]
        ax_dist.hist(angular_velocity_list[i], bins=30, orientation="horizontal", alpha=0.7)
        ax_dist.axhline(lower_th, color="red", linestyle="--")
        ax_dist.axhline(upper_th, color="red", linestyle="--")
        ax_dist.set_title(f"Distribution No.{i + 1}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/av_colleration.png")
    plt.close(fig)


def plot_averaged_angular_velocity(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Anglular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular-velocity_time-series_averaged.png")
    plt.close(fig)


def plot_fft(freq_list, Amp_list, save_dir, save_name, day, flag_add_peak=False):
    sample_num, _, _ = param.get_config(day)
    os.makedirs(save_dir, exist_ok=True)
    peak_list = []

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(freq_list[i], Amp_list[i])
        if flag_add_peak:
            max_amp_index = np.argmax(Amp_list[i])
            freq_at_max_amp = freq_list[i][max_amp_index]
            axs[row, col].axvline(x=freq_at_max_amp, color="r", alpha=0.6)
            peak_list.append(freq_at_max_amp)
        axs[row, col].grid(True)
        axs[row, col].set_title(f"No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Freqency [Hz]", fontsize=font_size)
        axs[row, col].set_ylabel("Amp", fontsize=font_size)
        # axs[row, col].set_xscale("log")
        axs[row, col].set_yscale("log")
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")
    plt.close(fig)
    if flag_add_peak:
        save2csv.save_fft_peak(save_dir, save_name, peak_list)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.angle_FFT_peak, peak_list, day)


def plot_SD_list(SD_list, day, flag_std):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        time_list = read_csv.get_timelist(day)

        fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(time_list[j][: len(SD_list[j][i])], SD_list[j][i])
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("Standardized SD", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("SD", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        plt.tight_layout()
        if flag_std:
            plt.savefig(f"{save_dir}/SD-time-series_{width_time}s_standardized.png")
        else:
            plt.savefig(f"{save_dir}/SD-time-series_{width_time}s.png")
        plt.close(fig)

    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = []
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i, width_time in enumerate(width_time_list):
        time_list = read_csv.get_timelist(day)

        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(
                time_list[j][: len(SD_list[j][i])], SD_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7
            )
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("Standardized SD", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("SD", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    axs[-1][-1].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if flag_std:
        plt.savefig(f"{save_dir}/SD-time-series_all_standardized.png")
    else:
        plt.savefig(f"{save_dir}/SD-time-series_all.png")
    plt.close(fig)


def plot_SD_list_fft(freq_list, Amp_list, day, flag_std):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i])
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD time-series No.{j+1}", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Freqency [Hz]", fontsize=font_size)
            axs[row, col].set_ylabel("Amp", fontsize=font_size)
            # axs[row, col].set_xscale("log")
            axs[row, col].set_yscale("log")
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        plt.tight_layout()
        if flag_std:
            plt.savefig(f"{save_dir}/SD-time-series_FFT_{width_time}s_standardized.png")
        else:
            plt.savefig(f"{save_dir}/SD-time-series_FFT_{width_time}s.png")
        plt.close(fig)

    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = []
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i, width_time in enumerate(width_time_list):
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Freqency [Hz]", fontsize=font_size)
            axs[row, col].set_ylabel("Amp", fontsize=font_size)
            # axs[row, col].set_xscale("log")
            axs[row, col].set_yscale("log")
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        axs[-1][-1].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if flag_std:
        plt.savefig(f"{save_dir}/SD-time-series_FFT_all_standardized.png")
    else:
        plt.savefig(f"{save_dir}/SD-time-series_FFT_all.png")
    plt.close(fig)


def dev_plot_time_list(day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)

    time_list = read_csv.get_timelist(day)
    time_diff_list = [
        [time_list[i][j] - time_list[i][j - 1] for j in range(1, len(time_list[i]))] for i in range(len(time_list))
    ]

    fig, axs = plt.subplots(5, 2 * sample_num // 5, figsize=(2 * fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        # time list
        axs[row, 2 * col].plot(time_list[i])
        axs[row, 2 * col].grid(True)
        axs[row, 2 * col].set_title(f"time list No.{i+1}", fontsize=font_size)
        axs[row, 2 * col].set_xlabel("Data Num", fontsize=font_size)
        axs[row, 2 * col].set_ylabel("Time [s]", fontsize=font_size)
        axs[row, 2 * col].tick_params(axis="both", which="major", labelsize=font_size)
        # time diff
        axs[row, 2 * col + 1].plot(time_list[i][1:], time_diff_list[i])
        axs[row, 2 * col + 1].grid(True)
        axs[row, 2 * col + 1].set_title(f"time diff list No.{i+1}", fontsize=font_size)
        axs[row, 2 * col + 1].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, 2 * col + 1].set_ylabel("Time diff [s]", fontsize=font_size)
        axs[row, 2 * col + 1].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_list.png")
    plt.close(fig)


def dev_plot_sd_data_num(data_num_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)

    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = []
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i, width_time in enumerate(width_time_list):
        time_list = read_csv.get_timelist(day)

        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(
                time_list[j][: len(data_num_list[j][i])],
                data_num_list[j][i],
                label=f"SD {width_time}s",
                c=color_list[i],
                alpha=0.7,
            )
            axs[row, col].grid(True)
            axs[row, col].set_title(f"No.{j+1}", fontsize=font_size)
            axs[row, col].set_ylabel("data num", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    axs[-1][-1].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SD_data_num.png")
    plt.close(fig)


def plot_SD_FFT_decline(decrease_list, ref_point_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, sample_num + 1, figsize=(50, 5))
    for i in range(sample_num):
        # Amp Decreace
        axes[i].plot(width_time_list, decrease_list[i], "-o")
        axes[i].set_title(f"No.{i+1}", fontsize=font_size)
        axes[i].set_xlabel("Window Width [s]", fontsize=font_size)
        axes[i].set_ylabel("Amp Decrease Ratio", fontsize=font_size)
        axes[i].tick_params(axis="both", which="major", labelsize=font_size)
        # Low Amp Reference Points
    axes[-1].plot(range(len(ref_point_list)), ref_point_list, "o")
    axes[-1].set_title("Low Amp Reference Points", fontsize=font_size)
    axes[-1].set_xlabel("Data Number", fontsize=font_size)
    axes[-1].set_ylabel("Low Amp Reference Points", fontsize=font_size)
    axes[-1].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    fig.suptitle("SD FFT features", size=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.savefig(f"{save_dir}/SD_FFT_Amp_decrease.png")
    plt.close(fig)


def plot_compare_SD_FFT_decline(decrease_list1, decrease_list2, day1, day2):
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/compare_SD_FFT_decline/{day1}-{day2}/"
    os.makedirs(save_dir, exist_ok=True)
    label_list = [day1, day2]

    fig, axes = plt.subplots(1, 10, figsize=(50, 5))
    for i in range(2):
        if i == 0:
            decrease_list = decrease_list1
            c = "#1f77b4"
        else:
            decrease_list = decrease_list2
            c = "#ff7f0e"
        # 色分けしてplot
        for j in range(len(decrease_list)):
            axes[0].plot(width_time_list, decrease_list[j], "-o", color=c, alpha=0.7)

        # 平均値・標準偏差plot
        mean_arr = np.mean(decrease_list, axis=0)
        std_arr = np.std(decrease_list, axis=0)
        # axes[1].errorbar(width_time_list, mean_arr, std_arr)
        axes[1].errorbar(width_time_list, mean_arr, std_arr, fmt="o", label=label_list[i], alpha=0.7)

    for i in range(10):
        axes[i].set_xlabel("Window Width [s]", fontsize=font_size)
        axes[i].set_ylabel("Amp Decrease Ratio", fontsize=font_size)
        axes[i].tick_params(axis="both", which="major", labelsize=font_size)
    axes[1].legend(label_list, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    fig.suptitle("Amp decrease", size=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.savefig(f"{save_dir}/SD_FFT_Amp_decrease.png")
    plt.close(fig)


def plot_Amp_dec_rot_param(day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    rot_df = rot_df_manage.get_rot_df(day)
    # col_list = rot_df_manage.get_cols()
    col_list_org = ROTATION_FEATURES.__annotations__.keys()
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    fig_mag = len(col_list_org) / 2.5
    fig, axs = plt.subplots(
        len(col_list_org),
        len(col_list_org),
        figsize=(fig_size_x * fig_mag, fig_size_x * fig_mag)
    )
    label_list = [f"SD {width}s" for width in width_time_list]
    for i, i_col in enumerate(col_list_org):
        for j, j_col in enumerate(col_list_org):
            if i == j:
                axs[i][j].set_title(f"{j_col} column", fontsize=font_size)
                axs[i][j].set_aspect("equal", "box")
                continue
            # obtain rotation param
            data_i_bef, data_j_bef = [], []
            flag_i_width_depend, flag_j_width_depend = False, False
            if i_col in SD_WIDTH_DEPEND_COLS:
                flag_i_width_depend = True
                for width in width_time_list:
                    data_i_bef.append(rot_df[f"{i_col}_{width}s"].tolist())
            else:
                data_i_bef.append(rot_df[i_col].tolist())
            if j_col in SD_WIDTH_DEPEND_COLS:
                flag_j_width_depend = True
                for width in width_time_list:
                    data_j_bef.append(rot_df[f"{j_col}_{width}s"].tolist())
            else:
                data_j_bef.append(rot_df[j_col].tolist())

            # prepair plot
            if flag_i_width_depend:
                data_i_aft = data_i_bef.copy()
            else:
                data_i_aft = [data_i_bef[0] for _ in range(sample_num)]
            if flag_j_width_depend:
                data_j_aft = data_j_bef.copy()
            else:
                data_j_aft = [data_j_bef[0] for _ in range(sample_num)]

            # plot
            if flag_i_width_depend or flag_j_width_depend:
                for k, width in enumerate(width_time_list):
                    axs[i][j].plot(data_i_aft[k], data_j_aft[k], "o", label=label_list[k], ms=5 * fig_mag)
                axs[i][j].legend(label_list, loc="upper left", bbox_to_anchor=(1, 1))
            else:
                axs[i][j].plot(data_i_aft[0], data_j_aft[0], "o", ms=5 * fig_mag)
            # axs[i][j].set_aspect("equal")
            axs[i][j].grid(True)
            axs[i][j].set_title(f"{i_col}\nvs\n{j_col}", fontsize=font_size)
            axs[i][j].set_xlabel(i_col, fontsize=font_size)
            axs[i][j].set_ylabel(j_col, fontsize=font_size)
            axs[i][j].tick_params(axis="both", which="major", labelsize=font_size)
    # plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rot_param_relation.png")
    plt.close(fig)
