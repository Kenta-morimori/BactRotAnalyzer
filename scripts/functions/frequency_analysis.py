import numpy as np

from . import make_gragh, param


def fft(data_bef, dt):
    data_aft = np.array(data_bef) - np.average(np.array(data_bef))
    N = len(data_aft)
    # FFT
    F = np.fft.fft(data_aft)
    freq = np.fft.fftfreq(N, d=dt)
    F = F / (N / 2)
    Amp = np.abs(F)

    return freq[1:N//2].tolist(), Amp[1:N//2].tolist()


def fft_angle(angle_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        freq, Amp = fft(angle_list[i], 1/FrameRate)
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"
    save_name = "angle_FFT"
    make_gragh.plot_fft(freq_list, Amp_list, save_dir, save_name, day)


def fft_angular_velocity(angular_velocity_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        freq, Amp = fft(angular_velocity_list[i], 1/FrameRate)
        freq_list.append(freq)
        Amp_list.append(Amp)
    # plot
    save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis"
    save_name = "angular_velocity_FFT"
    make_gragh.plot_fft(freq_list, Amp_list, save_dir, save_name, day)


def fft_sd_list(sd_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    freq_list, Amp_list = [], []
    # FFT
    for i in range(sample_num):
        add_freq_list, add_Amp_list = [], []
        for j in range(len(width_time_list)):
            freq, Amp = fft(sd_list[i][j], 1/FrameRate)
            add_freq_list.append(freq)
            add_Amp_list.append(Amp)
        freq_list.append(add_freq_list)
        Amp_list.append(add_Amp_list)
    # plot
    make_gragh.plot_SD_list_fft(freq_list, Amp_list, day)