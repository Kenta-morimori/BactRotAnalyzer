import csv
import os
from itertools import zip_longest
from typing import Sequence

import pandas as pd

from utils import param


def save_time_list(time_list, day):
    save_dir = f"{param.save_dir_bef}/{day}/"
    os.makedirs(save_dir, exist_ok=True)

    header = [f"No.{i+1}" for i in range(len(time_list))]
    csv_save_path = os.path.join(save_dir, "time_list.csv")

    with open(csv_save_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row in zip_longest(*time_list, fillvalue=None):
            csvwriter.writerow(row)


def save_angle_angular_velocity(angle_list, angular_velocity_list, day):
    save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/"
    os.makedirs(save_dir, exist_ok=True)
    header = [f"No.{i+1}" for i in range(len(angle_list))]

    csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angle_time-series.csv"
    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row in zip_longest(*angle_list, fillvalue=None):
            csvwriter.writerow(row)

    csv_save_dir = f"{param.save_dir_bef}/{day}/angular_velocity/angular-velocity_time-series.csv"
    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for row in zip_longest(*angular_velocity_list, fillvalue=None):
            csvwriter.writerow(row)


def save_fft(save_dir, save_name, freq_list, Amp_list):
    csv_save_dir = f"{save_dir}/{save_name}.csv"

    headers = []
    for i in range(len(freq_list)):
        headers.append(f"No.{i + 1}_freq")
        headers.append(f"No.{i + 1}_Amp")
    rows = []
    for index in range(max(map(len, freq_list))):
        row = []
        for freq, amp in zip(freq_list, Amp_list):
            if index < len(freq):
                row.append(freq[index])
                row.append(amp[index])
            else:
                row.extend([None, None])
        rows.append(row)

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)


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
    df = pd.DataFrame()

    cols = []
    for i in range(len(freq_list)):
        for j, width in enumerate(width_list):
            f = pd.Series(freq_list[i][j], dtype="float64", name=f"No.{i+1}_{width}s_freq")
            a = pd.Series(Amp_list[i][j], dtype="float64", name=f"No.{i+1}_{width}s_Amp")
            cols.extend([f, a])
    df = pd.concat(cols, axis=1, copy=False)
    df.to_csv(csv_save_dir, index=False)


def save_SD_FFT_decline(decrease_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/SD_FFT_Amp_decrease.csv"

    data = {
        "No": [i + 1 for i in range(sample_num) for _ in width_time_list],
        "width": width_time_list * sample_num,
        "decrease": [decrease_list[i][j] for i in range(sample_num) for j in range(len(width_time_list))],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


def save_SD_FFT_refpoints(ref_point_list, day):
    sample_num, _, _ = param.get_config(day)
    width_time_list = param.SD_window_width_list
    csv_save_dir = f"{param.save_dir_bef}/{day}/fluctuation_analysis/SD-time-series/SD_FFT_Amp_refpoints.csv"

    data = {
        "No": [i + 1 for i in range(sample_num) for _ in width_time_list],
        "width": width_time_list * sample_num,
        "decrease": [ref_point_list[i][j] for i in range(sample_num) for j in range(len(width_time_list))],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_save_dir, index=False)


def save_repellent_background_intensity(time_list, background_list, day):
    sample_num = min(len(time_list), len(background_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/01_brightness_change"
    os.makedirs(save_dir, exist_ok=True)
    csv_save_dir = f"{save_dir}/background_intensity_time_series.csv"

    data = {}
    for i in range(sample_num):
        time_arr = pd.Series(time_list[i], dtype="float64")
        bg_arr = pd.Series(background_list[i], dtype="float64")
        n = min(len(time_arr), len(bg_arr))
        data[f"No.{i+1}_time"] = time_arr.iloc[:n].reset_index(drop=True)
        data[f"No.{i+1}_background"] = bg_arr.iloc[:n].reset_index(drop=True)
    pd.DataFrame(data).to_csv(csv_save_dir, index=False)


def save_repellent_rise_summary(results, day):
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/01_brightness_change"
    os.makedirs(save_dir, exist_ok=True)
    csv_save_dir = f"{save_dir}/rise_summary.csv"
    pd.DataFrame(results).to_csv(csv_save_dir, index=False)


def save_repellent_post_rise_centroid(time_list, x_list, y_list, day):
    sample_num = min(len(time_list), len(x_list), len(y_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/centroid_coordinate"
    os.makedirs(save_dir, exist_ok=True)
    csv_save_dir = f"{save_dir}/post_rise_centroid_time_series.csv"

    data = {}
    for i in range(sample_num):
        time_arr = pd.Series(time_list[i], dtype="float64")
        x_arr = pd.Series(x_list[i], dtype="float64")
        y_arr = pd.Series(y_list[i], dtype="float64")
        n = min(len(time_arr), len(x_arr), len(y_arr))
        data[f"No.{i+1}_time"] = time_arr.iloc[:n].reset_index(drop=True)
        data[f"No.{i+1}_x"] = x_arr.iloc[:n].reset_index(drop=True)
        data[f"No.{i+1}_y"] = y_arr.iloc[:n].reset_index(drop=True)
    pd.DataFrame(data).to_csv(csv_save_dir, index=False)


def save_repellent_time_list(time_list, day):
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/00_time_list"
    os.makedirs(save_dir, exist_ok=True)
    csv_save_dir = f"{save_dir}/time_list.csv"

    data = {}
    for i in range(len(time_list)):
        data[f"No.{i+1}"] = pd.Series(time_list[i], dtype="float64")
    pd.DataFrame(data).to_csv(csv_save_dir, index=False)


def save_angular_velocity_switching_frequency(
    time_list: Sequence[Sequence[float]],
    frequency_list: Sequence[Sequence[float]],
    day: str,
) -> None:
    """Save angular velocity switching frequency to CSV."""
    sample_num = min(len(time_list), len(frequency_list))
    save_dir = f"{param.save_dir_bef}/{day}/repellent_response/03_post_rise_analysis/angular_velocity"
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = f"{save_dir}/switching_frequency.csv"
    
    data = {}
    for i in range(sample_num):
        t_arr = pd.Series(time_list[i], dtype="float64")
        f_arr = pd.Series(frequency_list[i], dtype="float64")
        n = min(len(t_arr), len(f_arr))
        data[f"No.{i+1}_time"] = t_arr.iloc[:n].reset_index(drop=True)
        data[f"No.{i+1}_frequency"] = f_arr.iloc[:n].reset_index(drop=True)
    
    pd.DataFrame(data).to_csv(csv_path, index=False)
