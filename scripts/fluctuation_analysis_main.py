import sys

from functions import (
    fluctuation_analysis,
    frequency_analysis,
    get_angular_velocity,
    input_data,
    make_gragh,
)


def main(day):
    x_list, y_list = input_data.input_centroid_coordinate(day)
    make_gragh.plot_centroid_coordinate(x_list, y_list, day)

    angle_list, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)
    make_gragh.plot_angular_velocity(angle_list, angular_velocity_list, day)

    # FFT
    frequency_analysis.fft_angle(angle_list, day)
    frequency_analysis.fft_angular_velocity(angular_velocity_list, day)

    # evaluate fluctuation
    fluctuation_analysis.main(angular_velocity_list, day)


if __name__ == "__main__":
    day = sys.argv[1]
    main(day)
