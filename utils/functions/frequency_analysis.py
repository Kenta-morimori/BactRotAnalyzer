import numpy as np

from utils import param
from utils.functions import get_angular_velocity, make_graph, save2csv


def fft(data_bef, dt):
    data_arr = np.asarray(data_bef, dtype=float)
    if data_arr.size == 0 or not np.isfinite(data_arr).any():
        empty = np.asarray([], dtype=float)
        return empty, empty

    data_aft = data_arr - np.nanmean(data_arr)
    N = len(data_aft)
    # FFT
    F = np.fft.fft(data_aft)
    freq = np.fft.fftfreq(N, d=dt)
    F = F / (N / 2)
    Amp = np.abs(F)

    return freq[1 : N // 2], Amp[1 : N // 2]


def fft_angle(angle_list, day):
    sample_num, FrameRate_list, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        if param.flag_get_angle_with_cell_direcetion:
            # normalize angle to -π~π
            angle = get_angular_velocity.normalized_angle(angle_list[i])
        else:
            angle = angle_list[i]
        freq, Amp = fft(angle, 1 / FrameRate_list[i])
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    save_name = "angle_FFT"
    make_graph.plot_fft(freq_list, Amp_list, save_dir, save_name, day, flag_add_peak=True)
    save2csv.save_fft(save_dir, save_name, freq_list, Amp_list)


def fft_angular_velocity(angular_velocity_list, day):
    sample_num, FrameRate_list, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        # Nan --> Mean value
        if np.any(np.isnan(angular_velocity_list[i])):
            mean = np.nanmean(angular_velocity_list[i])
            angular_velocity_list[i] = np.where(np.isnan(angular_velocity_list[i]), mean, angular_velocity_list[i])
        freq, Amp = fft(angular_velocity_list[i], 1 / FrameRate_list[i])
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity"
    save_name = "angular_velocity_FFT"
    make_graph.plot_fft(freq_list, Amp_list, save_dir, save_name, day)
    save2csv.save_fft(save_dir, save_name, freq_list, Amp_list)


def fft_sd_list(df, day, flag_std=False):
    sample_num, FrameRate_list, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        add_freq_list, add_Amp_list = [], []
        for j in range(len(width_time_list)):
            if flag_std:
                sd_list = df[f"No.{i + 1}_{width_time_list[j]}s_sd_std"].dropna().tolist()
            else:
                sd_list = df[f"No.{i + 1}_{width_time_list[j]}s_sd"].dropna().tolist()
            freq, Amp = fft(sd_list, 1 / FrameRate_list[i])
            add_freq_list.append(freq.tolist())
            add_Amp_list.append(Amp.tolist())
        freq_list.append(add_freq_list)
        Amp_list.append(add_Amp_list)
    freq_arr = np.array(freq_list, dtype=object)
    Amp_arr = np.array(Amp_list, dtype=object)
    # plot
    make_graph.plot_SD_list_fft(freq_arr, Amp_arr, day, flag_std)
    make_graph.dev_plot_sd_FFT_with_rotation(freq_arr, Amp_arr, day)

    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    save2csv.save_sd_fft(save_dir, freq_arr, Amp_arr)

    return freq_arr, Amp_arr
