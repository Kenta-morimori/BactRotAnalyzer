import statistics

import numpy as np

from . import dev_make_graph, frequency_analysis, make_graph, param, read_csv


def get_sd_time_series(i, angular_velocity, day):
    _, FrameRate_list, total_time_list = param.get_config(day)
    width_time_list = param.SD_window_width_list

    sd_list = []
    data_num_list = []  # develop
    # 時間ベースの取得
    time_list = read_csv.get_timelist(day)
    for width_time in width_time_list:
        add_sd = []
        add_data_num = []  # develop
        start_time = 0.0
        while 1:
            data = []
            # start_time ~ start_time + width_time のデータ
            for j in range(len(time_list[i])):
                if (time_list[i][j] >= start_time) and (time_list[i][j] < start_time + width_time):
                    data.append(angular_velocity[j])
            if len(data) > 2:  # SDの算出は最低3データ必要
                add_sd.append(statistics.stdev(data))
                add_data_num.append(len(data))
            start_time += 1 / FrameRate_list[i]
            # width_timeの幅でSDが算出できない場合break
            if start_time + width_time >= total_time_list[i]:
                break
        sd_list.append(add_sd)
        data_num_list.append(add_data_num)  # develop
    """
    # indexベースの取得
    for width_time in width_time_list:
        add_sd = []
        width_frame_num = int(width_time * FrameRate_list[i])

        for index_1, time_num in enumerate(np.arange(1 / FrameRate_list[i], float(total_time_list[i] + 1 / FrameRate_list[i]), 1 / FrameRate_list[i])):
            time_num = round(time_num, 4)
            if (index_1 + width_frame_num) < FrameRate_list[i] * total_time_list[i]:
                print(index_1, index_1 + width_frame_num)
                add_sd.append(statistics.stdev(angular_velocity[index_1 : (index_1 + width_frame_num)]))
        sd_list.append(add_sd)
    """

    return sd_list, data_num_list


def standardize_sd_time_series(sd_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list

    std_sd_list = []
    for i in range(sample_num):
        add_std_sd_list = []
        for j in range(len(width_time_list)):
            sd = np.array(sd_list[i][j])
            mean = np.mean(sd)
            std = np.std(sd, axis=0)
            add_std_sd_list.append((sd - mean) / std)
        std_sd_list.append(add_std_sd_list)

    return std_sd_list


def main(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    sd_list = []
    data_num_list = []  # develop

    # get SD time-series
    for i in range(sample_num):
        add_sd_list, add_data_num_list = get_sd_time_series(i, angular_velocity_list[i], day)
        sd_list.append(add_sd_list)
        data_num_list.append(add_data_num_list)

    dev_make_graph.dev_plot_sd_data_num(data_num_list, day)  # develop

    # plotv
    flag_std = False
    make_graph.plot_SD_list(sd_list, day, flag_std)
    # FFT
    # frequency_analysis.fft_sd_list(sd_list, day, flag_std)

    # get standardized sd time-series
    flag_std = True
    std_sd_list = standardize_sd_time_series(sd_list, day)
    make_graph.plot_SD_list(std_sd_list, day, flag_std)
    frequency_analysis.fft_sd_list(std_sd_list, day, flag_std)
