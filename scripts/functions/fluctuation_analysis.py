import statistics

import numpy as np

from . import frequency_analysis, make_gragh, param


def get_sd_time_series(angular_velocity, day):
    _, FrameRate, total_time = param.get_config(day)
    width_time_list = [float(width_time) for width_time in param.SD_window_width_list]

    sd_list = []
    for width_time in width_time_list:
        add_sd = []
        width_frame_num = int(width_time * FrameRate)

        for index_1, time_num in enumerate(np.arange(1/FrameRate, float(total_time + 1/FrameRate), 1/FrameRate)):
            time_num = round(time_num, 4)
            if ((index_1 + width_frame_num) < FrameRate * total_time):
                add_sd.append(statistics.stdev(angular_velocity[index_1:(index_1 + width_frame_num)]))
        sd_list.append(add_sd)
    
    return sd_list


def main(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    sd_list = []

    # get SD time-series
    for i in range(sample_num):
        add_sd_list = get_sd_time_series(angular_velocity_list[i], day)
        sd_list.append(add_sd_list)

    # plot
    make_gragh.plot_SD_list(sd_list, day)
    
    # FFT
    frequency_analysis.fft_sd_list(sd_list, day)
