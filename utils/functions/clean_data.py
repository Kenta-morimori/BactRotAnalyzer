import numpy as np

from utils import param


# Trimming with thresholds
def correct_angular_velocity_outlier(data):
    num_std_dev = param.num_std_dev
    data_aft = []

    mean = np.mean(data)
    std_dev = np.std(data)
    lower_th = mean - num_std_dev * std_dev
    upper_th = mean + num_std_dev * std_dev

    for x in data:
        if x < lower_th or upper_th < x:
            # data_aft.append(mean)  # Average
            data_aft.append(np.nan)
        else:
            data_aft.append(x)

    return data_aft


def angular_velocity_completion(data):
    sample_num = len(data)

    for i in range(sample_num):
        print("hogehoge")
