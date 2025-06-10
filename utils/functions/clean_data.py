import numpy as np

from utils import param
from utils.functions import get_tiff_info, read_csv


# Trimming with thresholds
def correct_angular_velocity_outlier(data, idx, day):
    mode_correct_av_outlier = param.mode_correct_av_outlier
    data_aft = []
    if mode_correct_av_outlier == 0:  # use SD threshold
        num_std_av = param.num_std_av
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)
        lower_th = mean - num_std_av * std_dev
        upper_th = mean + num_std_av * std_dev

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


def correct_rotation_center(data, idx, day):
    """
    The code replaces abnormal data points with the average of their neighboring values.
    input:
        data: list of rotation center coordinates
        idx: index of rotation data
        day: data name
        mode: 0 for using TIFF time info, 1 for using SD threshold
    output:
        data_aft: list of corrected rotation center coordinates
    """
    n_neighbors = 2  # number of neighbors to consider for averaging

    if param.mode_correct_av_outlier == 0:  # use TIFF time info
        time_list = read_csv.get_timelist(day)
        detect_time_jumps_with_sd = get_tiff_info.detect_time_jumps_with_sd(time_list, day)
        complement_index_list = detect_time_jumps_with_sd[idx]
    elif param.mode_correct_av_outlier == 1:  # use SD threshold
        num_std_dev = param.num_std_center
        mean = np.nanmean(data)
        std_dev = np.nanstd(data)
        lower_th = mean - num_std_dev * std_dev
        upper_th = mean + num_std_dev * std_dev

        complement_index_list = [i for i, x in enumerate(data) if x < lower_th or upper_th < x]

    data_aft = []
    for j in range(len(data)):
        if j in complement_index_list:
            neighbors = []
            for k in range(1, n_neighbors + 1):
                if j - k >= 0 and (j - k) not in complement_index_list:
                    neighbors.append(data[j - k])
                if j + k < len(data) and (j + k) not in complement_index_list:
                    neighbors.append(data[j + k])
            if neighbors:
                data_aft.append(np.mean(neighbors))
            else:
                # Warning if all neighbors are in complement_index_list
                print(f"Warning: All neighbors of index {j} are in complement_index_list.")
                valid_data = [x for index, x in enumerate(data) if index not in complement_index_list]
                if valid_data:
                    data_aft.append(np.mean(valid_data))
                else:
                    data_aft.append(np.nan)  # If no valid data is found
        else:
            data_aft.append(data[j])

    return data_aft
