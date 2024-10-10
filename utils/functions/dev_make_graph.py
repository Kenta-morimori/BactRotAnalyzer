import os

import matplotlib.pyplot as plt

from utils import param
from utils.functions import read_csv

font_size = 20
fig_size_x = 20
fig_size_y = 23


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
            axs[row, col].plot(time_list[j][:len(data_num_list[j][i])], data_num_list[j][i], label=f"SD {width_time}s", c=color_list[i], alpha=0.7)
            axs[row, col].grid(True)
            axs[row, col].set_title(f"No.{j+1}", fontsize=font_size)
            axs[row, col].set_ylabel("data num", fontsize=font_size)
            axs[row, col].set_xlabel("Time [s]", fontsize=font_size)
            axs[row, col].tick_params(axis="both", which="major", labelsize=font_size)
    axs[-1][-1].legend(plot_label_list, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/SD_data_num.png")
    plt.close(fig)


def dev_plot_time_list(day):
    sample_num, _, _ = param.get_config(day)
    save_dir = f"{param.save_dir_bef}/{day}/dev"
    os.makedirs(save_dir, exist_ok=True)

    time_list = read_csv.get_timelist(day)
    time_diff_list = [[time_list[i][j] - time_list[i][j-1] for j in range(1, len(time_list[i]))] for i in range(len(time_list))]

    fig, axs = plt.subplots(5, 2 * sample_num // 5, figsize=(2 * fig_size_x, fig_size_y))
    for i in range(sample_num):
        row = i // 2
        col = i % 2
        # time list
        axs[row, 2*col].plot(time_list[i])
        axs[row, 2*col].grid(True)
        axs[row, 2*col].set_title(f"time list No.{i+1}", fontsize=font_size)
        axs[row, 2*col].set_xlabel("Data Num", fontsize=font_size)
        axs[row, 2*col].set_ylabel("Time [s]", fontsize=font_size)
        axs[row, 2*col].tick_params(axis="both", which="major", labelsize=font_size)
 
        # time diff
        axs[row, 2*col+1].plot(time_list[i][1:], time_diff_list[i])
        axs[row, 2*col+1].grid(True)
        axs[row, 2*col+1].set_title(f"time diff list No.{i+1}", fontsize=font_size)
        axs[row, 2*col+1].set_xlabel("Time [s]", fontsize=font_size)
        axs[row, 2*col+1].set_ylabel("Time diff [s]", fontsize=font_size)
        axs[row, 2*col+1].tick_params(axis="both", which="major", labelsize=font_size)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time_list.png")
    plt.close(fig)
