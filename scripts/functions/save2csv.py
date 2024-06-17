import os

import csv
import pandas as pd

from . import param


def save_angle_angular_velocity(angle_list, angular_velocity_list, day):
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/"
    os.makedirs(save_dir, exist_ok=True)
    header = [f"No.{i+1}" for i in range(len(angle_list))]

    csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angle_time-series.csv"
    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row in zip(*angle_list):
            csvwriter.writerow(row)

    csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angular-velocity_time-series.csv"
    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row in zip(*angular_velocity_list):
            csvwriter.writerow(row)


def save_switching_value(switching_value_list, day, flag_averaged=False):
    if flag_averaged:
        csv_save_dir = f"{param.save_dir_bef}/{day}/switching_value_averagedAV.csv"
    else:
        csv_save_dir = f"{param.save_dir_bef}/{day}/switching_value.csv"
    sample_num, _, _ = param.get_config(day)

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = [f"No.{i+1}" for i in range(sample_num)]
        csvwriter.writerow(headers)
        csvwriter.writerow(switching_value_list)


def save_fft(save_dir, save_name, freq_list, Amp_list):
    csv_save_dir = f"{save_dir}/{save_name}.csv"
    headers, row = [], []
    for i in range(len(freq_list)):
        headers.extend([f"No.{i+1}_freq", f"No.{i+1}_Amp"])

    index = 0
    while(1):
        count_no_data = 0
        add_row = []
        for i in range(len(freq_list)):
            if len(freq_list[i]) >= index + 1:
                add_row.extend([freq_list[i][index], Amp_list[i][index]])
            else:
                count_no_data += 1
        if count_no_data == len(freq_list):
            break
        row.append(add_row)
        index += 1

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(row)


def save_fft_peak(save_dir, save_name, peak_list):
    csv_save_dir = f"{save_dir}/{save_name}_peak.csv"
    sample_num = len(peak_list)

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = [f"No.{i+1}" for i in range(sample_num)]
        csvwriter.writerow(headers)
        csvwriter.writerow(peak_list)


def save_sd_fft(save_dir, freq_list, Amp_list):
    csv_save_dir = f"{save_dir}/SD-time-series_fft.csv"
    width_list = param.SD_window_width_list
    data = {}
    
    for i in range(len(freq_list)):
        for j, width in enumerate(width_list):
            freq_key = f"No.{i+1}_{width}s_freq"
            amp_key = f"No.{i+1}_{width}s_Amp"
            
            freq_values = freq_list[i][j]
            amp_values = Amp_list[i][j]

            data[freq_key] = freq_values
            data[amp_key] = amp_values

    max_len = max(len(v) for v in data.values())
    df = pd.DataFrame({k: v + [None]*(max_len - len(v)) for k, v in data.items()})
    df.to_csv(csv_save_dir, index=False)
