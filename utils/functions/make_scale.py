import numpy as np

from utils import param
from utils.functions import read_csv


def scale_mean_one(data):
    mean = np.mean(data)

    return data / mean


def scale_mean_zero(data):
    mean = np.mean(data)

    return data - mean


def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    return standardized_data


def get_data_stat(data_bef_list, day, stat_type="mean"):
    """
    stat_type: mean or median
    """
    sample_num, FrameRate_list, total_time_list = param.get_config(day)
    time_list = read_csv.get_timelist(day)
    width_coef = 0.001

    data_aft = []
    for i in range(sample_num):
        width_time = FrameRate_list[i] * width_coef
        add_data = []
        start_time = 0.0
        while 1:
            data = []
            # start_time ~ start_time + width_time
            for j in range(len(time_list[i])):
                if (time_list[i][j] >= start_time) and (time_list[i][j] < start_time + width_time):
                    data.append(data_bef_list[i][j])
            if len(data) > 2:
                if stat_type == "mean":
                    add_data.append(np.mean(data))
                elif stat_type == "median":
                    add_data.append(np.median(data))
            start_time += 1 / FrameRate_list[i]
            # width_timeの幅でSDが算出できない場合break
            if start_time + width_time >= total_time_list[i]:
                break
        data_aft.append(add_data)

    return data_aft
