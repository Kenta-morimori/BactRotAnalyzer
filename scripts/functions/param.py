import configparser
import os

# ディレクトリ
curr_dir = os.getcwd()
input_dir_bef = f"{curr_dir}/../data"
save_dir_bef = f"{curr_dir}/../outputs"


def get_config(day):
    config_dir = f"{input_dir_bef}/{day}/config.ini"
    config = configparser.ConfigParser()
    config.read(config_dir)

    # データ数
    sample_num = config.getint("Settings", "sample_num")
    # 撮影条件
    FrameRate = config.getint("Settings", "FrameRate")
    total_time = config.getint("Settings", "total_time")

    return sample_num, FrameRate, total_time


# fluctuation analysis
SD_window_width_list = ["0.5", "1.0"]
