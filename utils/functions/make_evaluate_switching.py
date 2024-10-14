import numpy as np

from utils import param
from utils.functions import make_graph, save2csv


def evaluate_switching(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
    # スイッチング頻度をCCW/CWと定義
    switching_value_list = []
    for i in range(sample_num):
        cw_count, ccw_count = 0, 0
        for j in range(len(angular_velocity_list[i])):
            if angular_velocity_list[i][j] >= 0:
                cw_count += 1
            else:
                ccw_count += 1
        switching_value_list.append(round(ccw_count / cw_count, 3))
    # 保存
    save2csv.save_switching_value(switching_value_list, day)


def evaluate_switching_averaged(angular_velocity_list_bef, day):
    sample_num, _, total_time = param.get_config(day)

    # angular_velocity_list を窓関数を用いて平均化
    # 窓関数の幅
    window_width_sec = 0.1  # [s]
    angular_velocity_list_aft = []
    for i in range(sample_num):
        add_av_data = []
        total_frame = len(angular_velocity_list_bef[i])
        # 0.1 sの窓関数を使用
        window_frame = int(window_width_sec * total_frame / total_time)

        for frame_i in range(int(total_frame - window_frame)):
            add_av_data.append(np.mean(angular_velocity_list_bef[i][frame_i : frame_i + window_frame]).astype(float))
        angular_velocity_list_aft.append(add_av_data)

    # 平均化角速度をplot
    make_graph.plot_averaged_angular_velocity(angular_velocity_list_aft, day)

    # スイッチング頻度をCCW/CWと定義
    switching_value_list = []
    for i in range(sample_num):
        cw_count, ccw_count = 0, 0
        for j in range(len(angular_velocity_list_aft[i])):
            if angular_velocity_list_aft[i][j] >= 0:
                cw_count += 1
            else:
                ccw_count += 1
        switching_value_list.append(round(ccw_count / cw_count, 3))
    # 保存
    save2csv.save_switching_value(switching_value_list, day, flag_averaged=True)
