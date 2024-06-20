import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from . import param, read_csv, save2csv

font_size = 20
fig_size_x = 20
fig_size_y = 23


def plot_centroid_coordinate(x_list, y_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
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
    time_list = np.linspace(0, total_time, int(total_time * FrameRate))
    xy_plot_label = [r"x [$\mu$m]", r"y [$\mu$m]"]
    xy_save_label = ["x_coordinate.png", "y_coordinate.png"]

    for label_i, xy_list in enumerate([x_list, y_list]):
        fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
        for i in range(sample_num):
            row = i // 2
            col = i % 2
            axs[row, col].plot(time_list, xy_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"Trajectory No.{i+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=18)
            axs[row, col].set_ylabel(xy_plot_label[label_i], fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{xy_save_label[label_i]}")
        plt.close(fig)


def plot_angular_velocity(angle_list, angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, int(total_time * FrameRate))

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list, angle_list[i])
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
        axs[row, col].plot(time_list[:-1], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular-velocity_time-series.png")
    plt.close(fig)


def plot_averaged_angular_velocity(angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, int(total_time * FrameRate))

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[: len(angular_velocity_list[i])], angular_velocity_list[i])
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
    sample_num, FrameRate, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        data_len = len(SD_list[0][i])
        time_list = np.linspace(0, data_len / FrameRate, data_len)

        fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(time_list, SD_list[j][i])
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
        time_list = np.linspace(0, data_len / FrameRate, data_len)

        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(time_list, SD_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
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


# dev validation1: 角度と角速度
def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    return standardized_data


def plot_validation1(angle_list, angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, int(total_time * FrameRate))

    fig, axs = plt.subplots(5, sample_num // 5, figsize=(fig_size_x, fig_size_y))
    for i in range(sample_num):
        angle_stand, angular_velocity_stand = standardize(angle_list[i]), standardize(angular_velocity_list[i])
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list, angle_stand, alpha=0.6)
        axs[row, col].plot(time_list[: len(angular_velocity_list[i])], angular_velocity_stand, alpha=0.6)
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angle / Anglular velocity", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angle__anglular_velocity.png")
    plt.close(fig)


def normalized(data):
    mean = np.mean(data)

    return data / mean


# dev validation2: 回転中心と重心座標の距離の平均 vs (正規化)角速度のSD
def plot_validation2(x_list, y_list, angular_velocity_list, day):
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)
    sample_num, _, _ = param.get_config(day)

    # get center coordinate
    center_x_list, center_y_list = read_csv.read_center_coordinates(day)

    dist_mean_list, av_sd_list = [], []
    for i in range(sample_num):
        x_coords = np.array(x_list[i])
        y_coords = np.array(y_list[i])
        center_x = center_x_list[i]
        center_y = center_y_list[i]
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        dist_mean_list.append(np.mean(dist))

        angular_velocity = np.array(angular_velocity_list[i])
        angular_velocity_aft = normalized(angular_velocity)

        # av_sd_list.append(np.std(angular_velocity))
        av_sd_list.append(np.std(angular_velocity_aft))

    # dist_mean_list, av_sd_list をplot
    plt.figure(figsize=(10, 6))
    plt.scatter(dist_mean_list, av_sd_list, c="blue", label="Data points")
    for i, (dist_mean, av_sd) in enumerate(zip(dist_mean_list, av_sd_list)):
        plt.text(dist_mean, av_sd, str(i + 1), fontsize=12, ha="center", va="bottom")
    plt.xlabel("Average Distance from Center", fontsize=font_size)
    plt.ylabel("Standard Deviation of Normalized Angular Velocity", fontsize=font_size)
    plt.title("Average Distance from Center vs. Angular Velocity SD", fontsize=font_size)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/r__av_sd.png")


# dev validation3: 角速度
def detect_outlier(data):
    num_std_dev = 3

    mean = np.mean(data)
    std_dev = np.std(data)
    lower_th = mean - num_std_dev * std_dev
    upper_th = mean + num_std_dev * std_dev

    return lower_th, upper_th


def plot_validation3(angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, int(total_time * FrameRate))

    fig, axs = plt.subplots(
        5, 3 * sample_num // 5, figsize=(20, 25), gridspec_kw={"width_ratios": [1, 5, 0.5, 1, 5, 0.5]}
    )
    for i in range(sample_num):
        row = i // 2
        col = (i % 2) * 3 + 1

        lower_th, upper_th = detect_outlier(angular_velocity_list[i])

        axs[row, col].plot(time_list[: len(angular_velocity_list[i])], angular_velocity_list[i])
        axs[row, col].set_title(f"Time Series {i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].grid(True)
        axs[row, col].hlines(
            y=[lower_th, upper_th], xmin=time_list[0], xmax=time_list[len(angular_velocity_list[i])], colors="r"
        )

        scatter_x = np.random.uniform(-0.1, 0.1, len(angular_velocity_list[i]))
        axs[row, col + 1].scatter(scatter_x, angular_velocity_list[i], s=10)
        axs[row, col + 1].grid(True)
        axs[row, col + 1].tick_params(left=False, right=False, labelleft=False, labelbottom=False)
        axs[row, col + 1].set_xlim(-0.2, 0.2)
        axs[row, col + 1].hlines(y=[lower_th, upper_th], xmin=-0.2, xmax=0.2, colors="r")

        axs[row, col - 1].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.95, wspace=0.05, hspace=0.4)
    plt.savefig(f"{save_dir}/angular_velocity_distribution.png")
    plt.close(fig)
