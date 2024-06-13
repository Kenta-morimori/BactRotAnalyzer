import csv

from . import param


def save_switching_value(switching_value_list, day, flag_averaged=False):
    if flag_averaged:
        csv_save_dir = f"{param.save_dir_bef}/{day}/switching_value_averagedAV.csv"
    else:
        csv_save_dir = f"{param.save_dir_bef}/{day}/switching_value.csv"
    sample_num, _, _ = param.get_config(day)

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーを書き込む
        headers = [f"No.{i+1}" for i in range(sample_num)]
        csvwriter.writerow(headers)
        # データを書き込む
        csvwriter.writerow(switching_value_list)


def save_center_of_rotation(center_x_list, center_y_list, day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/center_coordinate.csv"

    sample_num, _, _ = param.get_config(day)
    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        headers = []
        center_list = []
        for i in range(sample_num):
            headers.extend([f"No.{i+1}_x", f"No.{i+1}_y"])
            center_list.extend([center_x_list[i], center_y_list[i]])
        # ヘッダーを書き込む
        csvwriter.writerow(headers)
        # データを書き込む
        csvwriter.writerow(center_list)
