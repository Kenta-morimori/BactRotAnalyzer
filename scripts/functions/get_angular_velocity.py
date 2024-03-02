import numpy as np

from . import param


# 二次形式を用いて中心座標を算出
def get_center(x):
    A, B, C, D, E = x[0], x[1], x[2], x[3], x[4]
    mat = np.array([[A, B / 2], [B / 2, C]])
    eig_val, eig_vec = np.linalg.eig(mat)

    center_x_bef = -1 * (D * eig_vec[0][0] + E * eig_vec[1][0]) / (2 * eig_val[0])
    center_y_bef = -1 * (D * eig_vec[0][1] + E * eig_vec[1][1]) / (2 * eig_val[1])
    center_x = eig_vec[0][0] * center_x_bef + eig_vec[0][1] * center_y_bef
    center_y = eig_vec[1][0] * center_x_bef + eig_vec[1][1] * center_y_bef

    return center_x, center_y


def get_center_coordinate(X, Y):
    size = len(X)
    X = X.reshape([size, 1])
    Y = Y.reshape([size, 1])

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    center_x, center_y = get_center(x.tolist())

    return center_x, center_y


def get_angular_velocity(x_list, y_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    angle_list, angular_velocity_list = [], []

    for i in range(sample_num):
        x_arr, y_arr = np.array(x_list[i]), np.array(y_list[i])
        center_x, center_y = get_center_coordinate(x_arr, y_arr)

        # 角度の取得
        angle = np.arctan2(y_arr - center_y, x_arr - center_x)
        angle_list.append(angle)

        # 角速度の取得
        add_angular_velocity = []
        for j in range(1, len(angle)):
            angle_diff = angle[j] - angle[j - 1]
            # 角度変化から回転方向を設定
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            # CCWを正にするために-1をかける
            add_angular_velocity.append(-1*angle_diff * FrameRate)
        angular_velocity_list.append(add_angular_velocity)

    return angle_list, angular_velocity_list
