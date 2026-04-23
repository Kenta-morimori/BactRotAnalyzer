import math
import os
from typing import Optional, Sequence

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import param
from utils.features import IGNORE_PLOT_COLS, ROTATION_FEATURES, SD_WIDTH_DEPEND_COLS
from utils.functions import (
    frequency_analysis,
    get_tiff_info,
    read_csv,
    rot_df_manage,
    save2csv,
)

font_size = 20
fig_size_x = 20
fig_size_y = 23


def plot_coordinate(x_list, y_list, day, mode):
    sample_num, _, _ = param.get_config(day)

    save_dir = f"{param.save_dir_bef}/{day}/{mode}_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    # Pixel to µm conversion.
    # x_list, y_list = (np.array(x_list) * px2um_x).tolist(), (np.array(y_list) * px2um_y).tolist()

    cols = 5
    rows = max(1, math.ceil(sample_num / cols))
    # plot centroid coordinate
    fig = plt.figure(figsize=(20, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.38, hspace=0.2)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.plot(x_list[i], y_list[i])

        if mode == "centroid":
            # detect x_lim, y_lim
            x_range = max(x_list[i]) - min(x_list[i])
            y_range = max(y_list[i]) - min(y_list[i])
            max_range = 1.1 * max(x_range, y_range) / 2
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            # ax.scatter(0, 0, c="red")  # center is zero
        ax.set_aspect("equal", "box")
        ax.grid(True)
        ax.set_title(f"Trajectory No.{i+1}", fontsize=16)
        ax.set_xlabel(r"x [$\mu$m]", fontsize=16)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.38, hspace=0.2)
    plt.savefig(f"{save_dir}/trajectory.png")
    plt.close(fig)

    # plot x, y
    time_list = read_csv.get_timelist(day)
    xy_plot_label = [r"x [$\mu$m]", r"y [$\mu$m]"]
    xy_save_label = ["x_coordinate.png", "y_coordinate.png"]

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    for label_i, xy_list in enumerate([x_list, y_list]):
        fig, axs = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axs = np.atleast_2d(axs)
        for i in range(sample_num):
            row = i // cols
            col = i % cols
            axs[row, col].plot(time_list[i], xy_list[i])
            axs[row, col].grid(True)
            axs[row, col].set_title(f"Trajectory No.{i+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=18)
            axs[row, col].set_ylabel(xy_plot_label[label_i], fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{xy_save_label[label_i]}")
        plt.close(fig)


def plot_coordinate_with_center(x_list, y_list, center_x_list, center_y_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    cols = 5
    rows = max(1, math.ceil(sample_num / cols))

    # plot centroid coordinate
    fig = plt.figure(figsize=(20, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.38, hspace=0.2)
    label = ["centroid", "center"]

    coef = 1.1
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])

        ax.plot(x_list[i], y_list[i], label="centroid")
        ax.plot(center_x_list[i], center_y_list[i], label="center", alpha=0.5)

        # detect x_lim, y_lim
        ax.set_xlim(
            ((1 + coef) * min(x_list[i]) + (1 - coef) * max(x_list[i])) / 2,
            ((1 - coef) * min(x_list[i]) + (1 + coef) * max(x_list[i])) / 2,
        )
        ax.set_ylim(
            ((1 + coef) * min(y_list[i]) + (1 - coef) * max(y_list[i])) / 2,
            ((1 - coef) * min(y_list[i]) + (1 + coef) * max(y_list[i])) / 2,
        )
        ax.set_aspect("equal", "box")

        ax.grid(True)
        ax.set_title(f"Trajectory No.{i+1}", fontsize=16)
        ax.set_xlabel(r"x [$\mu$m]", fontsize=16)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=16)
    ax.legend(label, loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"{save_dir}/trajectory_with_center.png")
    plt.close(fig)

    time_list = read_csv.get_timelist(day)
    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    for axis_label in ["x", "y"]:
        fig, axs = plt.subplots(rows, cols, figsize=(20, 4 * rows))
        axs = np.atleast_2d(axs)
        plot_label = ["centroid", "center"]
        if axis_label == "x":
            xy_list = x_list
            xy_center_list = center_x_list
            xy_plot_label = r"x [$\mu$m]"
            xy_save_label = "x_centroid_center.png"
        elif axis_label == "y":
            xy_list = y_list
            xy_center_list = center_y_list
            xy_plot_label = r"y [$\mu$m]"
            xy_save_label = "y_centroid_center.png"

        for i in range(sample_num):
            row = i // cols
            col = i % cols
            axs[row, col].plot(time_list[i], xy_list[i], label="centroid", alpha=0.7)
            axs[row, col].plot(time_list[i][: len(xy_center_list[i])], xy_center_list[i], label="center", alpha=0.7)
            axs[row, col].grid(True)
            axs[row, col].set_title(f"Trajectory No.{i+1}", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=18)
            axs[row, col].set_ylabel(xy_plot_label, fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        axs[row, col].legend(plot_label, loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{xy_save_label}")
        plt.close(fig)


def plot_msd(msd, D_list, intercept_list, max_dist_list, day):
    sample_num, FrameRate_list, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    cols = 5
    rows = max(1, math.ceil(sample_num / cols))
    fig = plt.figure(figsize=(20, 4 * rows), layout="constrained")
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.5, hspace=0.2)
    for i in range(sample_num):
        t_list = np.arange(0, len(msd[i])) * (1.0 / FrameRate_list[i])
        fit_func = t_list * (D_list[i] * 4) + intercept_list[i]

        row = i // cols
        col = i % cols
        axs = fig.add_subplot(gs[row, col])

        axs.plot(t_list, msd[i] * 10**4, linewidth=4)
        axs.plot(t_list, fit_func * 10**4, "--", linewidth=4)
        axs.grid(True)
        axs.set_title(
            # f"MSD No.{i+1} | D={D_list[i]:.2e} | max_dist:{1000 * round(max_dist_list[i], 5)}" + r"[$\mu$m]",
            f"MSD No.{i+1} | {round(len(msd[i]) * (1.0 / FrameRate_list[i]), 3)} s | D={D_list[i]:.2e} | max_dist:{round(1000 * max_dist_list[i], 5)} nm",
            fontsize=font_size,
        )
        axs.set_xlabel("Δt(s)", fontsize=font_size)
        axs.set_ylabel("MSD (×10$^{-10}$cm$^2$)", fontsize=font_size)
        axs.tick_params(axis="both", which="major", labelsize=font_size)
        axs.set_box_aspect(1)
    # plt.tight_layout()
    plt.savefig(f"{save_dir}/MSD_2d.png")
    plt.close(fig)


def plot_rot_axes(long_axis_arr, short_axis_arr, aspect_ratio_arr, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/other_rot_features"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    time_list = read_csv.get_timelist(day)
    plot_label = ["long_axis", "short_axis"]

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, 2 * cols, figsize=(2 * fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        # long axis, short axis
        axs[row, 2 * col].plot(time_list[i][: len(long_axis_arr[i])], long_axis_arr[i], label="long_axis")
        axs[row, 2 * col].plot(time_list[i][: len(short_axis_arr[i])], short_axis_arr[i], label="short_axis")
        axs[row, 2 * col].grid(True)
        axs[row, 2 * col].set_title(f"Rotation Axes No.{i+1}", fontsize=font_size)
        axs[row, 2 * col].set_ylabel("Axes length [μm]", fontsize=font_size)
        axs[row, 2 * col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, 2 * col].tick_params(axis="both", which="major", labelsize=font_size)
        # aspect ratio
        axs[row, 2 * col + 1].plot(time_list[i][: len(aspect_ratio_arr[i])], aspect_ratio_arr[i])
        axs[row, 2 * col + 1].grid(True)
        axs[row, 2 * col + 1].set_title(f"Rotation Axes Ratio No.{i+1}", fontsize=font_size)
        axs[row, 2 * col + 1].set_ylabel("Axes Ratio", fontsize=font_size)
        axs[row, 2 * col + 1].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, 2 * col + 1].tick_params(axis="both", which="major", labelsize=font_size)
    axs[row, col].legend(plot_label, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rotation_axes.png")
    plt.close(fig)


def plot_r(r_arr, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/other_rot_features"
    os.makedirs(save_dir, exist_ok=True)

    # sepalate save
    time_list = read_csv.get_timelist(day)
    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, 23 * rows / 5))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(time_list[i], r_arr[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Rotation Axes No.{i+1}", fontsize=font_size)
        axs[row, col].set_ylabel("Axes length [μm]", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/r_time-series.png")
    plt.close(fig)


def plot_angular_velocity(angle_list, angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(time_list[i], angle_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Angle Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlim(0, time_list[i][-1])
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angle [rad]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angle_time-series.png")
    plt.close(fig)

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlim(0, time_list[i][-1])
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_time-series.png")
    plt.close(fig)

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(time_list[i][: len(angular_velocity_list[i])], [abs(x) for x in angular_velocity_list[i]])
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity (abs) [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        axs[row, col].set_xlim(0, time_list[i][len(angular_velocity_list[i])])
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_time-series_abs.png")
    plt.close(fig)


def plot_av_colleration(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    mode_correct_av_outlier = param.mode_correct_av_outlier
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)
    jump_time_index_list = get_tiff_info.detect_time_jumps(time_list, day)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(fig_size_x, fig_size_y * rows / 5),
        gridspec_kw={"width_ratios": [5, 1] * (sample_num // 5)},
    )
    axs = axs.flatten()
    for i in range(sample_num):
        # Time series plot (left plot)
        ax_ts = axs[2 * i]
        ax_ts.plot(time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i])
        ax_ts.set_xlim(0, time_list[i][len(angular_velocity_list[i])])

        if mode_correct_av_outlier == 0:  # use SD threshold
            num_std_av = param.num_std_av
            mean = np.mean(angular_velocity_list[i])
            std_dev = np.std(angular_velocity_list[i])
            lower_th = mean - num_std_av * std_dev
            upper_th = mean + num_std_av * std_dev
            ax_ts.axhline(lower_th, color="red", linestyle="--")
            ax_ts.axhline(upper_th, color="red", linestyle="--")
        elif mode_correct_av_outlier == 1:  # use TIFF time info
            for jump_time_index in jump_time_index_list[i]:
                ax_ts.axvline(time_list[i][jump_time_index], color="red", linestyle="--")
        ax_ts.set_title(f"Angular Velocity No.{i + 1}")

        # Distribution plot (right plot)
        ax_dist = axs[2 * i + 1]
        ax_dist.hist(angular_velocity_list[i], bins=30, orientation="horizontal", alpha=0.7)

        if mode_correct_av_outlier == 0:  # use SD threshold
            ax_dist.axhline(lower_th, color="red", linestyle="--")
            ax_dist.axhline(upper_th, color="red", linestyle="--")
        elif mode_correct_av_outlier == 1:  # use TIFF time info
            for jump_time_index in jump_time_index_list[i]:
                ax_dist.axhline(angular_velocity_list[i][jump_time_index], color="red", linestyle="--")
        ax_dist.set_title(f"Distribution No.{i + 1}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_outlier.png")
    plt.close(fig)


def plot_center_colleration(center_x_list, center_y_list, complement_index_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate/bef_correction"
    os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    time_list = read_csv.get_timelist(day)
    for center_i, center_list in enumerate([center_x_list, center_y_list]):
        fig, axs = plt.subplots(
            rows,
            2 * cols,
            figsize=(fig_size_x, fig_size_y * rows / 5),
            gridspec_kw={"width_ratios": [5, 1] * cols},
        )
        axs = axs.flatten()
        for i in range(sample_num):
            # Time series plot (left plot)
            ax_ts = axs[2 * i]
            ax_ts.plot(time_list[i][: len(center_list[i])], center_list[i])
            if len(complement_index_list[i][0]) > 0:
                for complement_index in complement_index_list[i][0]:
                    ax_ts.axvline(time_list[i][complement_index], color="red", linestyle="--")
            if len(complement_index_list[i][1]) > 0:
                for complement_index in complement_index_list[i][1]:
                    ax_ts.axvline(time_list[i][complement_index], color="red", linestyle="--")
            ax_ts.set_title(f"Center Coodinate No.{i + 1}")

            # Distribution plot (right plot)
            ax_dist = axs[2 * i + 1]
            ax_dist.hist(center_list[i], bins=30, orientation="horizontal", alpha=0.7)

            if len(complement_index_list[i][0]) > 0:
                for complement_index in complement_index_list[i][0]:
                    ax_dist.axhline(center_list[i][complement_index], color="red", linestyle="--")
            if len(complement_index_list[i][1]) > 0:
                for complement_index in complement_index_list[i][1]:
                    ax_dist.axhline(center_list[i][complement_index], color="red", linestyle="--")
            if param.mode_correct_center_outlier == 1:  # use SD threshold
                num_std_dev = param.num_std_center
                mean = np.nanmean(center_list[i])
                std_dev = np.nanstd(center_list[i])
                lower_th = mean - num_std_dev * std_dev
                upper_th = mean + num_std_dev * std_dev
                ax_dist.axhline(lower_th, color="green", linestyle="--")
                ax_dist.axhline(upper_th, color="green", linestyle="--")
            ax_dist.set_title(f"Distribution No.{i + 1}")
        plt.tight_layout()
        if center_i == 0:
            plt.savefig(f"{save_dir}/center_outlier_x.png")
        else:
            plt.savefig(f"{save_dir}/center_outlier_y.png")
        plt.close(fig)


def plot_angular_velocity_rot_part(angular_velocity_list, th_list, th_list_means, th_list_median, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)
    labels = ["original AV", "original k-means", "mean k-means", "median k-means"]

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(
            time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i], alpha=0.7, label="original AV"
        )
        # axs[row, col].axhline(th_list[i], color="red", linestyle="--")
        axs[row, col].axhline(th_list[i], linestyle="--", alpha=0.9, c="#ff7f0e", label="original k-means")
        axs[row, col].axhline(th_list_means[i], linestyle="--", alpha=0.9, c="#2ca02c", label="mean k-means")
        axs[row, col].axhline(th_list_median[i], linestyle="--", alpha=0.9, c="#d62728", label="median k-means")
        axs[row, col].grid(True)
        axs[row, col].set_xlim(0, time_list[i][len(angular_velocity_list[i])])
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    axs[-1][-1].legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_rot_part.png")
    plt.close(fig)


def plot_averaged_angular_velocity(angular_velocity_list, angular_velocity_mean_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    plot_label = ["original", "mean"]
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(
            time_list[i][: len(angular_velocity_list[i])], angular_velocity_list[i], label="original", alpha=0.6
        )
        axs[row, col].plot(
            time_list[i][: len(angular_velocity_mean_list[i])], angular_velocity_mean_list[i], label="mean", alpha=0.6
        )
        axs[row, col].grid(True)
        axs[row, col].set_xlim(0, time_list[i][len(angular_velocity_list[i])])
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Anglular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        axs[row, col].set_xlim(0, time_list[i][len(angular_velocity_list[i])])
    axs[row, col].legend(plot_label, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_time-series_averaged.png")
    plt.close(fig)


def plot_fft(freq_list, Amp_list, save_dir, save_name, day, flag_add_peak=False):
    sample_num, _, _ = param.get_config(day)
    os.makedirs(save_dir, exist_ok=True)
    peak_list = []

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        freq_arr = np.asarray(freq_list[i], dtype=float)
        amp_arr = np.asarray(Amp_list[i], dtype=float)
        if freq_arr.size == 0 or amp_arr.size == 0:
            axs[row, col].axis("off")
            if flag_add_peak:
                peak_list.append(np.nan)
            continue

        axs[row, col].plot(freq_arr, amp_arr)
        if flag_add_peak:
            finite_amp_mask = np.isfinite(amp_arr)
            if np.any(finite_amp_mask):
                max_amp_index = int(np.nanargmax(amp_arr))
                freq_at_max_amp = float(freq_arr[max_amp_index])
                axs[row, col].axvline(x=freq_at_max_amp, color="r", alpha=0.6)
                peak_list.append(freq_at_max_amp)
            else:
                peak_list.append(np.nan)
        axs[row, col].grid(True)
        freq_max = float(np.nanmax(freq_arr))
        if np.isfinite(freq_max) and freq_max > 0:
            axs[row, col].set_xlim(0, freq_max)
        axs[row, col].set_title(f"No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Frequency [Hz]", fontsize=font_size)
        axs[row, col].set_ylabel("Amp", fontsize=font_size)
        # axs[row, col].set_xscale("log")
        if np.any(np.isfinite(amp_arr) & (amp_arr > 0)):
            axs[row, col].set_yscale("log")
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")
    plt.close(fig)
    if flag_add_peak:
        save2csv.save_fft_peak(save_dir, save_name, peak_list)
        rot_df_manage.update_rot_df(ROTATION_FEATURES.angle_FFT_peak, peak_list, day)


def plot_SD_list(df, day, flag_std=False):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}"
    os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    # sepalate save
    for i, width_time in enumerate(width_time_list):
        fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
        axs = np.atleast_2d(axs)
        for j in range(sample_num):
            row = j // cols
            col = j % cols

            time = df[f"No.{j + 1}_time"].dropna().tolist()
            if flag_std:
                sd_list = df[f"No.{j + 1}_{width_time}s_sd_std"].tolist()
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("Standardized SD", fontsize=font_size)
            else:
                sd_list = df[f"No.{j + 1}_{width_time}s_sd"].tolist()
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("SD", fontsize=font_size)

            axs[row, col].plot(time, sd_list[: len(time)])
            axs[row, col].grid(True)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
            axs[row, col].set_xlim(0, time[-1])
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

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i, width_time in enumerate(width_time_list):
        for j in range(sample_num):
            row = j // cols
            col = j % cols

            time = df[f"No.{j + 1}_time"].dropna().tolist()
            if flag_std:
                sd_list = df[f"No.{j + 1}_{width_time}s_sd_std"].tolist()
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("Standardized SD", fontsize=font_size)
            else:
                sd_list = df[f"No.{j + 1}_{width_time}s_sd"].tolist()
                axs[row, col].set_title(f"SD time-series No.{j+1}", fontsize=font_size)
                axs[row, col].set_ylabel("SD", fontsize=font_size)

            axs[row, col].plot(time, sd_list[: len(time)], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
            axs[row, col].grid(True)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
            axs[row, col].set_xlim(0, time[-1])
    axs[row, col].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if flag_std:
        plt.savefig(f"{save_dir}/SD-time-series_all_standardized.png")
    else:
        plt.savefig(f"{save_dir}/SD-time-series_all.png")
    plt.close(fig)


def plot_sd_mean(df, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}"
    os.makedirs(save_dir, exist_ok=True)

    cols = 5
    rows = max(1, math.ceil(sample_num / cols))
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.38, hspace=0.2)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs = fig.add_subplot(gs[row, col])

        sd_mean_list = []
        for width_time in width_time_list:
            sd_arr_i = np.asarray(df[f"No.{i + 1}_{width_time}s_sd"].dropna().tolist())
            sd_mean_list.append(np.mean(sd_arr_i))

        axs.plot(width_time_list, sd_mean_list, linewidth=4)
        axs.grid(True)
        axs.set_title(f"No.{i+1}", fontsize=font_size)
        axs.set_xlabel("Width time [s]", fontsize=font_size)
        axs.set_ylabel("Average of SD", fontsize=font_size)
        axs.tick_params(axis="both", which="major", labelsize=font_size)
    fig.suptitle("SD means", size=12)
    # plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"{save_dir}/SD_mean.png")
    plt.close(fig)


def plot_SD_list_fft(freq_list, Amp_list, day, flag_std):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}_FFT"
    os.makedirs(save_dir, exist_ok=True)

    max_x_lim_list = []
    for i in range(sample_num):
        max_x_lim_list.append(max([freq_list[i][j][-1] for j in range(len(width_time_list))]))

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    # sepalate save
    for i, width_time in enumerate(width_time_list):
        fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
        axs = np.atleast_2d(axs)
        for j in range(sample_num):
            row = j // cols
            col = j % cols
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i])
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD time-series No.{j+1}", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
            axs[row, col].set_xlim(0, max_x_lim_list[j])
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

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y * rows / 5))
    axs = np.atleast_2d(axs)
    for i, width_time in enumerate(width_time_list):
        for j in range(sample_num):
            row = j // cols
            col = j % cols
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
            axs[row, col].grid(True)
            if flag_std:
                axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
            else:
                axs[row, col].set_title(f"SD Time-series No.{j+1}", fontsize=font_size)
            axs[row, col].set_xlim(0, freq_list[j][i][-1])
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


def plot_SD_FFT_feats(ratio_list, ratio_reciprocal_list, decrease_list, ref_point_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}_FFT"
    os.makedirs(save_dir, exist_ok=True)

    max_cols = 5
    cols = max_cols
    rows = math.ceil(sample_num / cols)

    for mode in ["ratio", "ratio_reciprocal", "decrease", "ref_points"]:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 12), squeeze=False)
        for i in range(sample_num):
            row = i // cols
            col = i % cols

            if mode == "ratio":
                # Amp Ratio
                axes[row, col].plot(width_time_list, ratio_list[i], "-o")
                axes[row, col].set_title(f"No.{i+1} Amp Ratio", fontsize=font_size)
            elif mode == "ratio_reciprocal":
                # Amp Ratio (reciprocal)
                axes[row, col].plot(width_time_list, ratio_reciprocal_list[i], "-o")
                axes[row, col].set_title(f"No.{i+1} Amp Ratio", fontsize=font_size)
            elif mode == "decrease":
                # Amp Decrease
                axes[row, col].plot(width_time_list, decrease_list[i], "-o")
                axes[row, col].set_title(f"No.{i+1} Amp Decrease", fontsize=font_size)
                axes[row, col].set_ylim(0)
            else:
                # Reference Points
                axes[row, col].plot(width_time_list, ref_point_list[i], "-o")
                axes[row, col].set_title(f"No.{i+1} Ref. Points", fontsize=font_size)
                axes[row, col].set_ylim(0)

            axes[row, col].set_xlabel("Window Width [s]", fontsize=font_size)
            axes[row, col].tick_params(axis="both", which="major", labelsize=font_size)
        fig.suptitle("SD FFT features", size=12)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(f"{save_dir}/SD_FFT_Amp_{mode}.png")
        plt.close(fig)


def plot_rot_param(day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    rot_df = rot_df_manage.get_rot_df(day)
    # col_list = rot_df_manage.get_cols()
    col_list_org = [col for col in ROTATION_FEATURES.__annotations__.keys() if col not in IGNORE_PLOT_COLS]
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"
    os.makedirs(save_dir, exist_ok=True)

    plot_cols_num = len(col_list_org)
    fig_mag = plot_cols_num / 5.0
    fig, axs = plt.subplots(
        plot_cols_num,
        plot_cols_num,
        figsize=(fig_size_x * fig_mag, fig_size_x * fig_mag),
        constrained_layout=True,
    )
    label_list = [f"SD {width}s" for width in width_time_list]
    for i, i_col in enumerate(tqdm(col_list_org)):
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
                data_i_aft = [data_i_bef[0] for _ in range(len(width_time_list))]
            if flag_j_width_depend:
                data_j_aft = data_j_bef.copy()
            else:
                data_j_aft = [data_j_bef[0] for _ in range(len(width_time_list))]

            # plot
            if flag_i_width_depend or flag_j_width_depend:
                for k, width in enumerate(width_time_list):
                    axs[i][j].plot(data_i_aft[k], data_j_aft[k], "o", label=label_list[k], ms=3 * fig_mag)
                axs[i][j].legend(label_list, loc="upper left", bbox_to_anchor=(1, 1))
            else:
                axs[i][j].plot(data_i_aft[0], data_j_aft[0], "o", ms=5 * fig_mag)
            axs[i][j].set_box_aspect(1)
            axs[i][j].grid(True)
            axs[i][j].set_title(f"{i_col}\nvs\n{j_col}", fontsize=font_size / 2)
            axs[i][j].set_xlabel(i_col, fontsize=font_size / 2)
            axs[i][j].set_ylabel(j_col, fontsize=font_size / 2)
            axs[i][j].tick_params(axis="both", which="major", labelsize=font_size / 2)
    plt.savefig(f"{save_dir}/rot_param_relation.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def dev_plot_centroid_and_center(x_list, y_list, center_x_list, center_y_list, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/centroid_coordinate/bef_correction"
    os.makedirs(save_dir, exist_ok=True)

    flag_normalize = False

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    # plot x, y
    time_list = read_csv.get_timelist(day)

    for flag_normalize in [False, True]:
        for label in ["x", "y"]:
            fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
            axs = np.atleast_2d(axs)
            plot_label = ["centroid", "center"]
            if label == "x":
                xy_list = x_list
                xy_center_list = center_x_list
                xy_plot_label = r"x [$\mu$m]"
                if flag_normalize:
                    xy_save_label = "x_centroid_center_bef_corr_normalized.png"
                else:
                    xy_save_label = "x_centroid_center_bef_corr.png"
            elif label == "y":
                xy_list = y_list
                xy_center_list = center_y_list
                xy_plot_label = r"y [$\mu$m]"
                if flag_normalize:
                    xy_save_label = "y_centroid_center_bef_corr_normalized.png"
                else:
                    xy_save_label = "y_centroid_center_bef_corr.png"

            for i in range(sample_num):
                row = i // cols
                col = i % cols

                if flag_normalize:
                    xy_arr_norm = np.array(xy_list[i])
                    xy_arr_norm = (xy_arr_norm - np.nanmean(xy_arr_norm)) / np.nanstd(xy_arr_norm)
                    xy_center_arr_norm = np.array(xy_center_list[i])
                    xy_center_arr_norm = (xy_center_arr_norm - np.nanmean(xy_center_arr_norm)) / np.nanstd(
                        xy_center_arr_norm
                    )
                    axs[row, col].plot(time_list[i], xy_arr_norm, label="centroid", alpha=0.7)
                    axs[row, col].plot(
                        time_list[i][: len(xy_center_arr_norm)], xy_center_arr_norm, label="center", alpha=0.7
                    )
                else:
                    axs[row, col].plot(time_list[i], xy_list[i], label="centroid", alpha=0.7)
                    axs[row, col].plot(
                        time_list[i][: len(xy_center_list[i])], xy_center_list[i], label="center", alpha=0.7
                    )
                axs[row, col].grid(True)
                axs[row, col].set_title(f"Trajectory No.{i+1}", fontsize=font_size)
                axs[row, col].set_xlabel("Time [s]", fontsize=18)
                axs[row, col].set_ylabel(xy_plot_label, fontsize=font_size)
                axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
            axs[row, col].legend(plot_label, loc="upper left", bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{xy_save_label}")
            plt.close(fig)


def dev_plot_time_list(day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)

    time_list = read_csv.get_timelist(day)
    time_diff_list = [
        [time_list[i][j] - time_list[i][j - 1] for j in range(1, len(time_list[i]))] for i in range(len(time_list))
    ]

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, 2 * cols, figsize=(2 * fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
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

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i, width_time in enumerate(width_time_list):
        time_list = read_csv.get_timelist(day)

        for j in range(sample_num):
            row = j // cols
            col = j % cols
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
    plt.savefig(f"{save_dir}/{param.get_SD_mode_label()}_data_num.png")
    plt.close(fig)


def dev_plot_av_with_stats(av_list, av_means, av_medians, day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    time_list = read_csv.get_timelist(day)
    labels = ["original", "mean", "median"]

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        row = i // cols
        col = i % cols
        axs[row, col].plot(time_list[i][: len(av_list[i])], av_list[i], alpha=0.7)
        axs[row, col].plot(time_list[i][: len(av_means[i])], av_means[i], alpha=0.9)
        axs[row, col].plot(time_list[i][: len(av_medians[i])], av_medians[i], alpha=0.9)
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Anglar Velocity Time-series No.{i+1}", fontsize=font_size)
        axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, col].set_ylabel("Angular Velocity [rad/s]", fontsize=font_size)
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    axs[-1][-1].legend(labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/angular_velocity_with_stats.png")
    plt.close(fig)


def dev_plot_av_with_mean_sd(av_list, df_sd, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}/av_with_sd"
    os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    # sepalate save
    for width_time in width_time_list:
        time_list = read_csv.get_timelist(day)

        fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
        axs = np.atleast_2d(axs)
        for i in range(sample_num):
            row = i // cols
            col = i % cols

            # plot angular velocity
            time_arr = np.array(time_list[i])
            av_arr = np.abs(np.array(av_list[i]))
            axs[row, col].plot(time_arr[: len(av_arr)], av_arr, c="black", alpha=0.6)

            # plot mean and sd
            time_arr2 = df_sd[f"No.{i + 1}_time"]
            sd_arr = df_sd[f"No.{i + 1}_{width_time}s_sd"]
            mean_arr = df_sd[f"No.{i + 1}_{width_time}s_mean"]
            axs[row, col].fill_between(
                time_arr2,
                mean_arr - sd_arr,
                mean_arr + sd_arr,
                color="red",
                alpha=0.4,
            )
            axs[row, col].plot(time_arr2, mean_arr, alpha=0.6)

            axs[row, col].grid(True)
            axs[row, col].set_title(f"SD Time-series No.{i+1}", fontsize=font_size)
            axs[row, col].set_ylabel("SD", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
            axs[row, col].set_xlim(0, time_arr[len(av_arr) - 1])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SD_with_mean_sd_{width_time}s.png")
        plt.close(fig)


def dev_plot_sd_FFT_with_rotation(freq_list, Amp_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/{param.get_SD_mode_label()}"
    os.makedirs(save_dir, exist_ok=True)

    av_freq_list, av_Amp_list = read_csv.get_angle_FFT(day)

    # Stacking save
    color_list = ["m", "g", "b", "y", "c", "r"]
    plot_label_list = ["Rotation Data"]
    for width_time in width_time_list:
        plot_label_list.append(f"SD {width_time}s")

    max_x_lim_list = []
    for i in range(sample_num):
        max_x_lim_list.append(max([freq_list[i][j][-1] for j in range(len(width_time_list))]))

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y + 2))
    axs = np.atleast_2d(axs)
    # fig, axs = plt.subplots(5, sample_num // 5, figsize=(15, 23))
    axs = np.atleast_2d(axs)
    # Angular Velocisy
    for j in range(sample_num):
        row = j // cols
        col = j % cols
        axs[row, col].plot(av_freq_list[j], av_Amp_list[j], label="Rotation Data", c="black", alpha=0.8)
        axs[row, col].grid(True)
        axs[row, col].set_title(f"Standardized SD Time-series No.{j+1}", fontsize=font_size)
        axs[row, col].set_xlim(0, max_x_lim_list[j])
        axs[row, col].set_xlabel("Freqency [Hz]", fontsize=font_size)
        axs[row, col].set_ylabel("Amp", fontsize=font_size)
        # axs[row, col].set_xscale("log")
        axs[row, col].set_yscale("log")
        axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)

    # SD
    for i, width_time in enumerate(width_time_list):
        for j in range(sample_num):
            row = j // cols
            col = j % cols
            axs[row, col].plot(freq_list[j][i], Amp_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
        axs[-1][-1].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SD-time-series_FFT_all_standardized_with_av.png")
    plt.close(fig)


def dev_plot_fft_coordinates(X, Y, day):
    sample_num, FrameRate, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate"
    save_name = "centroid_coodinate_fft.png"
    os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    fig, axs = plt.subplots(rows, 2 * cols, figsize=(2 * fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(sample_num):
        x_freq_list, x_Amp_list = frequency_analysis.fft(X[i], 1 / FrameRate[i])
        y_freq_list, y_Amp_list = frequency_analysis.fft(Y[i], 1 / FrameRate[i])
        peak = max(x_freq_list[np.argmax(x_Amp_list)], y_freq_list[np.argmax(y_Amp_list)])

        row = i // cols
        col = i % cols
        axs[row, 2 * col].plot(x_freq_list, x_Amp_list)
        axs[row, 2 * col + 1].plot(y_freq_list, y_Amp_list)
        axs[row, 2 * col].set_xlim(0, x_freq_list[-1])
        axs[row, 2 * col + 1].set_xlim(0, y_freq_list[-1])
        xy = ["x", "y"]
        for j in [0, 1]:
            axs[row, 2 * col + j].grid(True)
            axs[row, 2 * col + j].axvline(x=peak, color="r", alpha=0.6)
            axs[row, 2 * col + j].set_title(
                f"No.{i+1}_{xy[j]}  peak:{round(peak, 3)}  width_time:{round(param.n_rotations / peak, 3)}s",
                fontsize=font_size,
            )
            axs[row, 2 * col + j].set_xlabel("Freqency [Hz]", fontsize=font_size)
            axs[row, 2 * col + j].set_ylabel("Amp", fontsize=font_size)
            axs[row, 2 * col + j].set_yscale("log")
            axs[row, 2 * col + j].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")
    plt.close(fig)


def dev_plot_max_dist_stat(max_dists, max_dists_all, day):
    sample_num, _, _ = param.get_config(day)

    save_dir = f"{param.save_dir_bef}/{day}/center_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    mag = 1

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(50 * mag / 4, 20 * mag / 4))
    axs = np.atleast_2d(axs)
    axs = axs.flatten()
    for i, data in enumerate(max_dists):
        axs[i].boxplot(data * 1000, positions=[1])
        axs[i].plot(2, max_dists_all[i] * 1000, "ro", markersize=8)
        axs[i].set_title(f"No.{i+1} | max_dist={round(max_dists_all[i], 3)} | med:{round(np.median(data), 6)}")
        axs[i].set_ylabel("Maximum distance moved [nm]", fontsize=font_size * mag / 2.1)
        axs[i].set_xlabel("Time [s]", fontsize=font_size * mag / 2)
        axs[i].set_xticks([1, 2])
        axs[i].set_xticklabels(["0.5", "30"])
        axs[i].set_xlim(0.5, 2.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/max_dist_validation.png")
    plt.close(fig)


def plot_repellent_background_intensity(time_list, background_list, rise_indices, day):
    sample_num = min(len(time_list), len(background_list), len(rise_indices))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/01_brightness_change"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 6
    label_fs = font_size + 4
    tick_fs = font_size + 2

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue

        time_arr = np.asarray(time_list[i], dtype=float)
        bg_arr = np.asarray(background_list[i], dtype=float)
        n = min(time_arr.size, bg_arr.size)
        time_arr = time_arr[:n]
        bg_arr = bg_arr[:n]
        ax.plot(time_arr, bg_arr, linewidth=2)

        rise_idx = rise_indices[i]
        if np.isfinite(rise_idx):
            rise_idx = int(rise_idx)
            if 0 <= rise_idx < n:
                ax.axvline(time_arr[rise_idx], color="red", linestyle="--", alpha=0.8)

        ax.grid(True)
        ax.set_title(f"Background Intensity No.{i+1}", fontsize=title_fs)
        ax.set_xlabel("Time [s]", fontsize=label_fs)
        ax.set_ylabel("Intensity", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/background_intensity_time_series.png")
    plt.close(fig)


def plot_repellent_background_and_av_stacked(
    time_list,
    background_list,
    angular_velocity_list,
    day,
    sample_indices=None,
    rise_time_list=None,
):
    sample_num = min(len(time_list), len(background_list), len(angular_velocity_list))
    if sample_num <= 0:
        return

    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/01_brightness_change"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 4
    label_fs = font_size + 2
    tick_fs = font_size

    nrows = sample_num * 3 - 1
    height_ratios = []
    for i in range(sample_num):
        height_ratios.extend([1.0, 1.0])
        if i < sample_num - 1:
            # Spacer row between pairs to avoid title overlap.
            height_ratios.append(0.22)
    fig = plt.figure(figsize=(24, max(10.0, sample_num * 7.2)))
    gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios, hspace=0.1)

    for i in range(sample_num):
        row_base = i * 3
        ax_bg = fig.add_subplot(gs[row_base, 0])
        ax_av = fig.add_subplot(gs[row_base + 1, 0])

        t_arr = np.asarray(time_list[i], dtype=float)
        bg_arr = np.asarray(background_list[i], dtype=float)
        av_arr = np.asarray(angular_velocity_list[i], dtype=float)

        n_bg = min(t_arr.size, bg_arr.size)
        n_av = min(t_arr.size, av_arr.size)
        t_bg = t_arr[:n_bg]
        y_bg = bg_arr[:n_bg]
        t_av = t_arr[:n_av]
        y_av = av_arr[:n_av]

        sample_no = i + 1
        if sample_indices is not None and i < len(sample_indices):
            sample_no = int(sample_indices[i])

        ax_bg.plot(t_bg, y_bg, linewidth=1.8)
        ax_av.plot(t_av, y_av, linewidth=1.8)

        rise_time = np.nan
        if rise_time_list is not None and i < len(rise_time_list) and np.isfinite(rise_time_list[i]):
            rise_time = float(rise_time_list[i])
        if np.isfinite(rise_time):
            ax_bg.axvline(rise_time, color="red", linestyle="--", linewidth=1.6, alpha=0.85)
            ax_av.axvline(rise_time, color="red", linestyle="--", linewidth=1.6, alpha=0.85)

        if np.isfinite(t_arr).any():
            t_min = float(np.nanmin(t_arr))
            t_max = float(np.nanmax(t_arr))
            if np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min:
                ax_bg.set_xlim(t_min, t_max)
                ax_av.set_xlim(t_min, t_max)

        ax_bg.grid(True)
        ax_av.grid(True)
        ax_bg.set_title(f"No.{sample_no}", fontsize=title_fs, pad=4)

        ax_bg.set_ylabel("Intensity", fontsize=label_fs)
        ax_av.set_ylabel("AV [rad/s]", fontsize=label_fs)
        ax_av.set_xlabel("Time [s]", fontsize=label_fs)

        # Top panel is for paired viewing only: hide x ticks/labels.
        ax_bg.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_bg.tick_params(axis="y", which="major", labelsize=tick_fs)
        ax_av.tick_params(axis="both", which="major", labelsize=tick_fs)

    fig.subplots_adjust(top=0.985, bottom=0.045, left=0.08, right=0.98)
    plt.savefig(f"{save_dir}/background_intensity_and_angular_velocity_time_series.png")
    plt.close(fig)


def plot_repellent_post_rise_centroid(time_list, x_list, y_list, day):
    sample_num = min(len(time_list), len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)

    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue

        time_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        n = min(time_arr.size, x_arr.size, y_arr.size)
        time_arr = time_arr[:n]
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]

        ax.plot(time_arr, x_arr, linewidth=2, label="x")
        ax.plot(time_arr, y_arr, linewidth=2, label="y")
        ax.grid(True)
        ax.set_title(f"Post-rise Centroid No.{i+1}", fontsize=font_size)
        ax.set_xlabel("Time [s]", fontsize=font_size)
        ax.set_ylabel(r"Coordinate [$\mu$m]", fontsize=font_size)
        ax.tick_params(axis="both", which="major", labelsize=font_size)
        ax.legend(loc="best", fontsize=font_size - 6)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/post_rise_centroid_time_series.png")
    plt.close(fig)


def plot_repellent_center_coordinate(time_list, x_list, y_list, day):
    sample_num = min(len(time_list), len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 6
    label_fs = font_size + 4
    tick_fs = font_size + 2

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))

    # xy trajectory
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        n = min(x_arr.size, y_arr.size)
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
        ax.plot(x_arr, y_arr)
        ax.grid(True)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Center Trajectory No.{i+1}", fontsize=title_fs)
        ax.set_xlabel(r"x [$\mu$m]", fontsize=label_fs)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectory.png")
    plt.close(fig)

    # x time-series
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue
        t_arr = np.asarray(time_list[i], dtype=float)
        x_arr = np.asarray(x_list[i], dtype=float)
        n = min(t_arr.size, x_arr.size)
        t_arr = t_arr[:n]
        x_arr = x_arr[:n]
        ax.plot(t_arr, x_arr)
        ax.grid(True)
        ax.set_title(f"Center X No.{i+1}", fontsize=title_fs)
        ax.set_xlabel("Time [s]", fontsize=label_fs)
        ax.set_ylabel(r"x [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/x_coordinate.png")
    plt.close(fig)

    # y time-series
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue
        t_arr = np.asarray(time_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        n = min(t_arr.size, y_arr.size)
        t_arr = t_arr[:n]
        y_arr = y_arr[:n]
        ax.plot(t_arr, y_arr)
        ax.grid(True)
        ax.set_title(f"Center Y No.{i+1}", fontsize=title_fs)
        ax.set_xlabel("Time [s]", fontsize=label_fs)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/y_coordinate.png")
    plt.close(fig)


def plot_repellent_center_x_components(time_list, x_raw_list, x_center_list, x_corrected_list, day):
    sample_num = min(len(time_list), len(x_raw_list), len(x_center_list), len(x_corrected_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 6
    label_fs = font_size + 4
    tick_fs = font_size + 2

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    plot_defs = [
        ("x_centroid_before.png", x_raw_list, "Centroid Before Correction X"),
        ("x_rotation_center.png", x_center_list, "Rotation Center X"),
        ("x_centroid_corrected.png", x_corrected_list, "Centroid Corrected X"),
    ]

    for save_name, x_series_list, title_prefix in plot_defs:
        fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
        axs = np.atleast_2d(axs)
        for i in range(rows * cols):
            row = i // cols
            col = i % cols
            ax = axs[row, col]
            if i >= sample_num:
                ax.axis("off")
                continue

            t_arr = np.asarray(time_list[i], dtype=float)
            x_arr = np.asarray(x_series_list[i], dtype=float)
            n = min(t_arr.size, x_arr.size)
            t_arr = t_arr[:n]
            x_arr = x_arr[:n]

            ax.plot(t_arr, x_arr, alpha=0.85)
            ax.grid(True)
            ax.set_title(f"{title_prefix} No.{i+1}", fontsize=title_fs)
            ax.set_xlabel("Time [s]", fontsize=label_fs)
            ax.set_ylabel(r"x [$\mu$m]", fontsize=label_fs)
            ax.tick_params(axis="both", which="major", labelsize=tick_fs)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{save_name}")
        plt.close(fig)


def plot_repellent_component_panels(
    time_list,
    x_list,
    y_list,
    day,
    mode_label,
    save_name,
    save_subdir="03_post_rise_analysis/centroid_coordinate",
    rise_time_list=None,
    overlay_x_list=None,
    overlay_y_list=None,
):
    sample_num = min(len(time_list), len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/{save_subdir}"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 6
    label_fs = font_size + 4
    tick_fs = font_size + 2

    rows = max(1, sample_num)
    cols = 3  # x-y, x-t, y-t
    fig, axs = plt.subplots(rows, cols, figsize=(3 * fig_size_x / 2, max(6, rows * 4)))
    axs = np.atleast_2d(axs)
    if rows == 1:
        axs = np.array([axs])

    for i in range(rows):
        x_arr = np.asarray(x_list[i], dtype=float)
        y_arr = np.asarray(y_list[i], dtype=float)
        t_arr = np.asarray(time_list[i], dtype=float)
        x_overlay_arr = np.array([], dtype=float)
        y_overlay_arr = np.array([], dtype=float)
        flag_overlay = False
        if (
            overlay_x_list is not None
            and overlay_y_list is not None
            and i < len(overlay_x_list)
            and i < len(overlay_y_list)
        ):
            x_overlay_arr = np.asarray(overlay_x_list[i], dtype=float)
            y_overlay_arr = np.asarray(overlay_y_list[i], dtype=float)
            flag_overlay = True
        rise_time = np.nan
        if rise_time_list is not None and i < len(rise_time_list):
            rise_time = float(rise_time_list[i]) if np.isfinite(rise_time_list[i]) else np.nan
        n = min(x_arr.size, y_arr.size, t_arr.size)
        if flag_overlay:
            n = min(n, x_overlay_arr.size, y_overlay_arr.size)
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
        t_arr = t_arr[:n]
        if flag_overlay:
            x_overlay_arr = x_overlay_arr[:n]
            y_overlay_arr = y_overlay_arr[:n]

        # x-y
        ax = axs[i, 0]
        ax.plot(x_arr, y_arr, linewidth=1.8)
        if flag_overlay:
            ax.plot(x_overlay_arr, y_overlay_arr, color="orange", linewidth=1.8, alpha=0.9)
        ax.grid(True)
        ax.set_aspect("equal", "box")
        if n > 0 and np.isfinite(x_arr).any() and np.isfinite(y_arr).any():
            x_min = float(np.nanmin(x_arr))
            x_max = float(np.nanmax(x_arr))
            y_min = float(np.nanmin(y_arr))
            y_max = float(np.nanmax(y_arr))
            x_mid = 0.5 * (x_min + x_max)
            y_mid = 0.5 * (y_min + y_max)
            half = 0.55 * max(x_max - x_min, y_max - y_min, 1e-6)
            ax.set_xlim(x_mid - half, x_mid + half)
            ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_title(f"{mode_label} No.{i+1} | x-y", fontsize=title_fs)
        ax.set_xlabel(r"x [$\mu$m]", fontsize=label_fs)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

        # x-t
        ax = axs[i, 1]
        ax.plot(t_arr, x_arr, linewidth=1.8)
        if flag_overlay:
            ax.plot(t_arr, x_overlay_arr, color="orange", linewidth=1.8, alpha=0.9)
        if np.isfinite(rise_time):
            ax.axvline(rise_time, color="red", linestyle="--", linewidth=1.6, alpha=0.85)
        ax.grid(True)
        ax.set_title(f"{mode_label} No.{i+1} | x-t", fontsize=title_fs)
        ax.set_xlabel("Time [s]", fontsize=label_fs)
        ax.set_ylabel(r"x [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

        # y-t
        ax = axs[i, 2]
        ax.plot(t_arr, y_arr, linewidth=1.8)
        if flag_overlay:
            ax.plot(t_arr, y_overlay_arr, color="orange", linewidth=1.8, alpha=0.9)
        if np.isfinite(rise_time):
            ax.axvline(rise_time, color="red", linestyle="--", linewidth=1.6, alpha=0.85)
        ax.grid(True)
        ax.set_title(f"{mode_label} No.{i+1} | y-t", fontsize=title_fs)
        ax.set_xlabel("Time [s]", fontsize=label_fs)
        ax.set_ylabel(r"y [$\mu$m]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{save_name}")
    plt.close(fig)


def plot_repellent_time_list(time_list, day):
    sample_num = len(time_list)
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/00_time_list"
    os.makedirs(save_dir, exist_ok=True)
    title_fs = font_size + 6
    label_fs = font_size + 4
    tick_fs = font_size + 2

    cols = 2
    rows = max(1, math.ceil(sample_num / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))
    axs = np.atleast_2d(axs)
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        if i >= sample_num:
            ax.axis("off")
            continue
        t_arr = np.asarray(time_list[i], dtype=float)
        if t_arr.size == 0:
            ax.axis("off")
            continue
        ax.plot(np.arange(t_arr.size), t_arr, linewidth=2)
        ax.grid(True)
        ax.set_title(f"Time List No.{i+1}", fontsize=title_fs)
        ax.set_xlabel("Frame", fontsize=label_fs)
        ax.set_ylabel("Time [s]", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_list.png")
    plt.close(fig)


def plot_repellent_angular_velocity_onecol(time_list, angle_list, angular_velocity_list, save_dir, rise_time_list=None):
    sample_num = min(len(time_list), len(angle_list), len(angular_velocity_list))
    os.makedirs(save_dir, exist_ok=True)

    title_fs = font_size + 4
    label_fs = font_size + 2
    tick_fs = font_size

    def _plot_panel(y_lists, save_name, y_label, title_prefix, use_abs=False):
        rows = max(1, sample_num)
        fig, axs = plt.subplots(rows, 1, figsize=(24, max(5.0, rows * 4.2)))
        axs = np.atleast_1d(axs)
        for i in range(rows):
            ax = axs[i]
            if i >= sample_num:
                ax.axis("off")
                continue

            t_arr = np.asarray(time_list[i], dtype=float)
            y_arr = np.asarray(y_lists[i], dtype=float)
            if use_abs:
                y_arr = np.abs(y_arr)
            n = min(len(t_arr), len(y_arr))
            t_arr = t_arr[:n]
            y_arr = y_arr[:n]
            if n <= 0:
                ax.axis("off")
                continue

            ax.plot(t_arr, y_arr, linewidth=1.8)
            rise_time = np.nan
            if rise_time_list is not None and i < len(rise_time_list) and np.isfinite(rise_time_list[i]):
                rise_time = float(rise_time_list[i])
            if np.isfinite(rise_time):
                ax.axvline(rise_time, color="red", linestyle="--", linewidth=1.6, alpha=0.85)
            ax.grid(True)
            ax.set_title(f"{title_prefix} No.{i + 1}", fontsize=title_fs)
            ax.set_xlabel("Time [s]", fontsize=label_fs)
            ax.set_ylabel(y_label, fontsize=label_fs)
            if np.isfinite(t_arr).any():
                t_min = float(np.nanmin(t_arr))
                t_max = float(np.nanmax(t_arr))
                if np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min:
                    ax.set_xlim(t_min, t_max)
            ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{save_name}")
        plt.close(fig)

    _plot_panel(
        angle_list,
        "angle_time-series.png",
        "Angle [rad]",
        "Angle Time-series",
        use_abs=False,
    )
    _plot_panel(
        angular_velocity_list,
        "angular_velocity_time-series.png",
        "AV [rad/s]",
        "Angular Velocity Time-series",
        use_abs=False,
    )
    _plot_panel(
        angular_velocity_list,
        "angular_velocity_time-series_abs.png",
        "AV [rad/s]",
        "Angular Velocity Abs Time-series",
        use_abs=True,
    )


def plot_angular_velocity_switching_frequency(
    time_list: Sequence[Sequence[float]],
    frequency_list: Sequence[Sequence[float]],
    day: str,
    sample_indices: Optional[Sequence[int]] = None,
) -> None:
    """Plot switching frequency time-series."""
    if not time_list or not frequency_list:
        return

    if sample_indices is None:
        sample_indices = list(range(1, len(time_list) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, (t_series, f_series) in enumerate(zip(time_list, frequency_list)):
        if len(t_series) > 0 and len(f_series) > 0:
            sample_no = sample_indices[idx] if idx < len(sample_indices) else idx + 1
            ax.plot(t_series, f_series, marker="o", markersize=4, label=f"No.{sample_no}", linewidth=2)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Sign Reversal Frequency (1/s)", fontsize=12)
    ax.set_title("Angular Velocity Sign-Reversal Frequency (Post-rise)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/switching_frequency.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
