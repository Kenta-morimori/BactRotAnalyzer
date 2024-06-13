import sys

from functions import (
    get_angular_velocity,
    input_data,
    make_evaluate_switching,
    make_graph
)


# 回転解析
def main(day):
    # 重心座標の取得
    x_list, y_list = input_data.input_centroid_coordinate(day)
    # 重心座標, x座標, y座標のplot
    make_graph.plot_centroid_coordinate(x_list, y_list, day)

    # 角度・角速度の取得
    angle_list, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)
    # 角度, 角速度のplot
    make_graph.plot_angular_velocity(angle_list, angular_velocity_list, day)

    # option: 角速度のスイッチングを評価
    make_evaluate_switching.evaluate_switching(angular_velocity_list, day)
    make_evaluate_switching.evaluate_switching_averaged(angular_velocity_list, day)


if __name__ == "__main__":
    day = sys.argv[1]
    main(day)
