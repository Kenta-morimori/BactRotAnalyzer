import csv

from . import param


def save_switching_value(switching_value_list, day):
    csv_save_dir = f"{param.save_dir_bef}/{day}/switching_value.csv"
    sample_num, _, _ = param.get_config(day)

    with open(csv_save_dir, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # ヘッダーを書き込む
        headers = [f"No.{i+1}" for i in range(sample_num)]
        csvwriter.writerow(headers)
        # データを書き込む
        csvwriter.writerow(switching_value_list)
