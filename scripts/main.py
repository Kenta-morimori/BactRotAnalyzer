import sys

from functions import (
    input_data,
    make_gragh,
    get_angular_velocity,
    make_evaluate_switching,
)


# 回転解析
def analyze_rotation(day):
    # 重心座標の取得
    x_list, y_list = input_data.input_centroid_coordinate(day)
    # 重心座標, x座標, y座標のplot
    make_gragh.plot_centroid_coordinate(x_list, y_list, day)

    # 角度・角速度の取得
    angle_list, angular_velocity_list = get_angular_velocity.get_angular_velocity(
        x_list, y_list
    )
    # 角度, 角速度のplot
    make_gragh.plot_angular_velocity(angle_list, angular_velocity_list, day)

    # option: 角速度のスイッチングを評価
    make_evaluate_switching.evaluate_switching(angular_velocity_list, day)


# 回転揺らぎの解析
def analyze_fluctuation(day):
    print("hogehoge")


if __name__ == "__main__":
    day = sys.argv[2]
    if sys.argv[1] == "rotation_analysis":
        analyze_rotation(day)
