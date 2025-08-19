import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.functions import make_graph, rot_df_manage  # noqa

"""
def compare_SD_FFT(day1, day2, ignore_data_day1, ignore_data_day2):
    decrease_list1 = read_csv.get_SD_FFT_decline(day1, ignore_data_day1)
    decrease_list2 = read_csv.get_SD_FFT_decline(day2, ignore_data_day2)
    ref_point_list1 = read_csv.get_SD_FFT_refpoints(day1, ignore_data_day1)
    ref_point_list2 = read_csv.get_SD_FFT_refpoints(day2, ignore_data_day2)
    # plot
    make_graph.plot_compare_SD_FFT_decline(decrease_list1, decrease_list2, ref_point_list1, ref_point_list2, day1, day2)
"""


def main(day_list, plot_labels=None):
    save_label = "-".join(day_list)

    rot_df_list = []
    for day in day_list:
        add_rot_df = rot_df_manage.get_rot_df(day)
        add_rot_df["day"] = day
        rot_df_list.append(add_rot_df)
    rot_df_all = pd.concat(rot_df_list, ignore_index=True)

    make_graph.plot_rot_param_compairison(day_list, rot_df_all, save_label, plot_labels=plot_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", type=str, help="data labels")
    parser.add_argument("--plot-labels", nargs="+", type=str, help="data labels")

    args = parser.parse_args()
    day_list = args.days
    plot_labels = args.plot_labels

    main(day_list, plot_labels)
