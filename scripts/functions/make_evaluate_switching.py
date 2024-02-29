from . import param, save2csv


def evaluate_switching(angular_velocity_list, day):
    sample_num = param.sample_num
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
