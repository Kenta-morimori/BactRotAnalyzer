import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils.functions import (  # noqa
    fluctuation_analysis,
    get_angular_velocity,
    get_tiff_info,
    input_data,
    make_graph,
    rot_df_manage,
)


def main(
    day: str,
    flag_flctuation_analysis: bool,
):
    rot_df_manage.create_rot_df(day)
    # make time list
    get_tiff_info.get_timelist(day)

    # obtain centroid coordinates
    x_list, y_list = input_data.input_centroid_coordinate(day)
    # obtain angle, angular velocity (+ evaluate switching)
    _, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)

    if flag_flctuation_analysis:
        # evaluate fluctuation
        fluctuation_analysis.main(angular_velocity_list, day)

        # Comparison of SD_FFT_Amp_refpoints with other rot_df parameters
        make_graph.plot_rot_param(day)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--day", type=str, default="test_data")
    parser.add_argument("--fluc", action="store_true", help="fluctuation analysis")

    args = parser.parse_args()
    day = args.day
    flag_fluctuation_analysis = args.fluc

    main(
        day=day,
        flag_flctuation_analysis=flag_fluctuation_analysis,
    )
