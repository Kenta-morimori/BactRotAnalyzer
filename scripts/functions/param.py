import os

# データ数
sample_num = 10
# 撮影条件
FrameRate = 200  # [Hz]
total_time = 6  # [s]

# ディレクトリ
curr_dir = os.getcwd()
input_dir_bef = f"{curr_dir}/../data"
save_dir_bef = f"{curr_dir}/../outputs"


# window_width_list [s]
window_width_list = ["0.5", "0.1"]
