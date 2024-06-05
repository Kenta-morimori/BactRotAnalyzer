import os

import matplotlib.pyplot as plt
import numpy as np

from . import param


def plot_centroid_coordinate(x_list, y_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)

    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    # 重心座標をplot
    fig, axs = plt.subplots(2, sample_num // 2, figsize=(20, 8))
    for i in range(sample_num):
        row = i // 5
        col = i % 5
        axs[row, col].plot(x_list[i], y_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Trajectory No.{i+1}")
        axs[row, col].set_xlabel("x [pixel]")
        axs[row, col].set_ylabel("y [pixel]")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectory.png")

    # x座標, y座標をplot
    time_list = np.linspace(0, total_time, total_time * FrameRate)
    xy_plot_label = ["x [pixel]", "y [pixel]"]
    xy_save_label = ["x_coordinate.png", "y_coordinate.png"]

    for label_i, xy_list in enumerate([x_list, y_list]):
        fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
        for i in range(sample_num):
            row = i // 2
            col = i % 2
            axs[row, col].plot(time_list, xy_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"Trajectory No.{i+1}")
            axs[row, col].set_xlabel("time [s]")
            axs[row, col].set_ylabel(xy_plot_label[label_i])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{xy_save_label[label_i]}")


def plot_angular_velocity(angle_list, angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, total_time * FrameRate)

    # 時系列角度をplot
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list, angle_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Angle time-series No.{i+1}")
        axs[row, col].set_xlabel("time [s]")
        axs[row, col].set_ylabel("angle [rad]")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angle_time-series.png")

    # 時系列角速度をplot
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[:-1], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity time-series No.{i+1}")
        axs[row, col].set_xlabel("time [s]")
        axs[row, col].set_ylabel("angle velocity [rad/s]")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular-velocity_time-series.png")


def plot_averaged_angular_velocity(angular_velocity_list, day):
    sample_num, FrameRate, total_time = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = np.linspace(0, total_time, total_time * FrameRate)

    # 時系列角速度をplot
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(time_list[: len(angular_velocity_list[i])], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity time-series No.{i+1}")
        axs[row, col].set_xlabel("time [s]")
        axs[row, col].set_ylabel("angle velocity [rad/s]")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular-velocity_time-series_averaged.png")


def plot_fft(freq_list, Amp_list, save_dir, save_name, day):
    sample_num, _, _ = param.get_config(day)
    os.makedirs(save_dir, exist_ok=True)

    # 時系列角速度をplot
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        axs[row, col].plot(freq_list[i], Amp_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_xlabel("freqency [Hz]")
        axs[row, col].set_ylabel("Amp")
        axs[row, col].set_xscale("log")
        axs[row, col].set_yscale("log")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")


def plot_SD_list(SD_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        data_len = len(SD_list[0][i])
        time_list = np.linspace(0, data_len / FrameRate, data_len)

        fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(time_list, SD_list[j][i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"SD time-series No.{j+1}")
            axs[row, col].set_xlabel("time [s]")
            axs[row, col].set_ylabel("SD")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SD-time-series_{width_time}s.png")
    
    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = []
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")
    
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i, width_time in enumerate(width_time_list):
        data_len = len(SD_list[0][i])
        time_list = np.linspace(0, data_len / FrameRate, data_len)

        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(time_list, SD_list[j][i], label=f"SD {width_time}s", c=color_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"SD time-series No.{j+1}")
            axs[row, col].set_xlabel("time [s]")
            axs[row, col].set_ylabel("SD")
    axs[-1][-1].legend(plot_label_list, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SD-time-series_all.png")
    

def plot_SD_list_fft(freq_list, Amp_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"

    # sepalate save
    for i, width_time in enumerate(width_time_list):
        fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"SD time-series No.{j+1}")
            axs[row, col].set_xlabel("Amp")
            axs[row, col].set_ylabel("freqency [Hz]")
            axs[row, col].set_xscale("log")
            axs[row, col].set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SD-time-series_FFT_{width_time}s.png")

    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = []
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")
    
    fig, axs = plt.subplots(5, sample_num // 5, figsize=(20, 20))
    for i, width_time in enumerate(width_time_list):
        for j in range(sample_num):
            row = j // 2
            col = j % 2
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i], label=f"SD {width_time}s", c=color_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"SD time-series No.{j+1}")
            axs[row, col].set_xlabel("Amp")
            axs[row, col].set_ylabel("freqency [Hz]")
            axs[row, col].set_xscale("log")
            axs[row, col].set_yscale("log")
        axs[-1][-1].legend(plot_label_list, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SD-time-series_FFT_all.png")
