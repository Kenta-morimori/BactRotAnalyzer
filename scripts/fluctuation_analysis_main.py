import os
import sys

from functions import (
    dev_make_graph,
    fluctuation_analysis,
    frequency_analysis,
    get_angular_velocity,
    get_tiff_info,
    input_data,
    make_graph,
)


def main(day):
    # make time list
    get_tiff_info.get_timelist(day)
    # obtain centroid coordinates
    x_list, y_list = input_data.input_centroid_coordinate(day)
    # obtain angle, angular velocity
    angle_list, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)

    # dev
    dev_make_graph.dev_plot_time_list(day)

    # plot centroid coordinates
    make_graph.plot_centroid_coordinate(x_list, y_list, day)
    # plot angle, angular velocity
    make_graph.plot_angular_velocity(angle_list, angular_velocity_list, day)

    # FFT
    frequency_analysis.fft_angle(angle_list, day)
    frequency_analysis.fft_angular_velocity(angular_velocity_list, day)

    # evaluate fluctuation
    fluctuation_analysis.main(angular_velocity_list, day)


if __name__ == "__main__":
    day = sys.argv[1]
    main(day)
