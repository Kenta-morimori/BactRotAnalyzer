import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.functions import make_graph, read_csv  # noqa


def compare_SD_FFT(day1, day2, ignore_data_day1, ignore_data_day2):
    decrease_list1 = read_csv.get_SD_FFT_decline(day1, ignore_data_day1)
    decrease_list2 = read_csv.get_SD_FFT_decline(day2, ignore_data_day2)
    # plot
    make_graph.plot_compare_SD_FFT_decline(decrease_list1, decrease_list2, day1, day2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--day1", type=str, default="20240912_temp=10")
    parser.add_argument("--day2", type=str, default="20240912_temp=23")
    parser.add_argument("--ignore-data-day1", nargs="+", type=int, default=[], help="ignore data numbers")
    parser.add_argument("--ignore-data-day2", nargs="+", type=int, default=[], help="ignore data numbers")

    args = parser.parse_args()
    compare_SD_FFT(
        day1=args.day1, day2=args.day2, ignore_data_day1=args.ignore_data_day1, ignore_data_day2=args.ignore_data_day2
    )
