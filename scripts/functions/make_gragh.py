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
        axs[row, col].set_aspect("equal")
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
