import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import param

DATA_KEYS = [
    "SJW46_10",
    "SJW46_23",
    "SJW46_30",
]
AV_DIR_DICT = {  # Angular velocity
    "SJW46_10": "outputs/SJW46_temp=10/angular_velocity/angular-velocity_time-series.csv",
    "SJW46_23": "outputs/SJW46_temp=23/angular_velocity/angular-velocity_time-series.csv",  # SJW46 23°C
    "SJW46_30": "outputs/SJW46_temp=30/angular_velocity/angular-velocity_time-series.csv",  # SJW46 30°C
}
TIME_DIR_DICT = {
    "SJW46_10": "outputs/SJW46_temp=10/time_list.csv",
    "SJW46_23": "outputs/SJW46_temp=23/time_list.csv",
    "SJW46_30": "outputs/SJW46_temp=30/time_list.csv",
}


def plot_distibustion(df: pd.DataFrame, out_dir: Path) -> None:
    """時系列角速度データの分布チェック

    Args:
        - df (pd.DataFrame): 時系列角速度データセット
        - out_dir (str): 保存先のディレクトリ

    Returns:

    """
    out_dir = out_dir / "av_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    for save_i, data_key in enumerate(DATA_KEYS):
        df_selected = df[df["label"] == data_key]
        No_list = df_selected["No"].unique().tolist()
        No_num = len(No_list)

        if No_num == 0:
            continue

        # plot
        cols = 2
        rows = max(1, math.ceil(No_num / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axs = np.array(axs).ravel()

        for i, No_i in enumerate(No_list):
            axes = axs[i]

            data = df_selected[df_selected["No"] == No_i]["av"]
            axes.hist(data, bins=50)
            axes.grid(True)
            axes.set_title(f"No.{i+1}")
            axes.set_xlabel("Angular Velocity")
            axes.set_ylabel("Counts")
        for j in range(No_num, rows * cols):
            fig.delaxes(axs[j])
        fig.suptitle(f"{data_key} Angular Velocity Distribution")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{save_i + 1}_{data_key}_av_distribution.png")
        plt.close(fig)


def main():
    ############
    # Load Data
    ############
    df_dict = defaultdict(list)
    for data_key in DATA_KEYS:
        df_av = pd.read_csv(AV_DIR_DICT[data_key])
        df_time = pd.read_csv(TIME_DIR_DICT[data_key])

        # obtain data number
        cols = df_av.columns.tolist()

        # make dataset
        for col in cols:
            data_len = len(df_av[col])
            # data param
            df_dict["label"].extend([data_key] * data_len)
            df_dict["No"].extend([col] * data_len)

            # angular velocity and time
            df_dict["av"].extend(df_av[col])
            df_dict["time"].extend(df_time[col][:data_len])
    df = pd.DataFrame(df_dict)

    ############
    # Data analysis
    ############
    out_dir = Path(param.save_dir_bef) / "03_av_analysis"
    plot_distibustion(df, out_dir)


if __name__ == "__main__":
    # TODO: select directory (and data number) to analyze
    main()
