import numpy as np

from utils import param
from utils.functions import get_tiff_info, read_csv


# Trimming with thresholds
def correct_angular_velocity_outlier(data, idx, day):
    num_std_dev = param.num_std_dev
    mode_correct_av_outlier = param.mode_correct_av_outlier
    data_aft = []

    if mode_correct_av_outlier == 0:  # use SD threshold
        num_std_dev = param.num_std_dev
        mean = np.mean(data)
        std_dev = np.std(data)
        lower_th = mean - num_std_dev * std_dev
        upper_th = mean + num_std_dev * std_dev

        # for x in data:
        for i, x in enumerate(data):
            if x < lower_th or upper_th < x:
                # data_aft.append(mean)  # Average
                data_aft.append(np.nan)
            else:
                data_aft.append(x)
    elif mode_correct_av_outlier == 1:  # use TIFF time info
        time_list = read_csv.get_timelist(day)
        jump_time_index_list = get_tiff_info.detect_time_jumps_with_sd(time_list, day)

        for j in range(len(data)):
            if j in jump_time_index_list[idx]:
                data_aft.append(np.nan)
            else:
                data_aft.append(data[j])

    return data_aft


def angular_velocity_completion(data):
    sample_num = len(data)

    for i in range(sample_num):
        print("hogehoge")
