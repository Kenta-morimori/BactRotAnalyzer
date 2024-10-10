import numpy as np

from . import get_angular_velocity, make_graph, param, save2csv


def fft(data_bef, dt):
    data_aft = np.array(data_bef) - np.average(np.array(data_bef))
    N = len(data_aft)
    # FFT
    F = np.fft.fft(data_aft)
    freq = np.fft.fftfreq(N, d=dt)
    F = F / (N / 2)
    Amp = np.abs(F)

    return freq[1 : N // 2].tolist(), Amp[1 : N // 2].tolist()


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


def fft_sd_list(sd_list, day, flag_std):
    sample_num, FrameRate_list, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        add_freq_list, add_Amp_list = [], []
        for j in range(len(width_time_list)):
            freq, Amp = fft(sd_list[i][j], 1 / FrameRate_list[i])
            add_freq_list.append(freq)
            add_Amp_list.append(Amp)
        freq_list.append(add_freq_list)
        Amp_list.append(add_Amp_list)
    # plot
    make_graph.plot_SD_list_fft(freq_list, Amp_list, day, flag_std)
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series"
    save2csv.save_sd_fft(save_dir, freq_list, Amp_list)
