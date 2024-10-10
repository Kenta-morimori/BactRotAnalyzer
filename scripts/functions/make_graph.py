import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from . import param, read_csv, save2csv

font_size = 20
fig_size_x = 20
fig_size_y = 23


def plot_centroid_coordinate(x_list, y_list, day):
    sample_num, _, _ = param.get_config(day)
    px2um_x, px2um_y = param.get_px2um_config(day)

    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    # Pixel to µm conversion.
    x_list, y_list = (np.array(x_list) * px2um_x).tolist(), (np.array(y_list) * px2um_y).tolist()

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


def plot_SD_list(SD_list, day, flag_std):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        data_len = len(SD_list[0][i])
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
        data_len = len(SD_list[0][i])
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
