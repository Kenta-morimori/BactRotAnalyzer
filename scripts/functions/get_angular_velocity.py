import numpy as np

from . import param, read_csv, save2csv


# Calculate centre coordinates using quadratic form.
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


# Trimming with thresholds
def correct_angular_velocity(data):
    num_std_dev = 3
    data_aft = []

    mean = np.mean(data)
    std_dev = np.std(data)
    lower_th = mean - num_std_dev * std_dev
    upper_th = mean + num_std_dev * std_dev

    for x in data:
        if x < lower_th or upper_th < x:
            data_aft.append(mean)
        else:
            data_aft.append(x)

    return data_aft


def get_angular_velocity(x_list, y_list, day):
    sample_num, FrameRate, _ = param.get_config(day)
    angle_list, angular_velocity_list = [], []

    if param.flag_get_angle_with_cell_direcetion:
        angle_list = read_csv.read_angle(day)

    for i in range(sample_num):
        x_arr, y_arr = np.array(x_list[i]), np.array(y_list[i])

        # obtain angle
        if param.flag_get_angle_with_cell_direcetion:
            angle = np.array(angle_list[i])
        else:
            # center is zero
            angle = np.arctan2(y_arr, x_arr)
            angle_list.append(angle)

        # obtain angular velocitiy
        add_angular_velocity = []
        for j in range(1, len(angle)):
            angle_diff = angle[j] - angle[j - 1]
            # 角度変化から回転方向を設定
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            # CCWを正にするために-1をかける
            add_angular_velocity.append(-1 * angle_diff * FrameRate)

        # correct angular velocity
        if param.flag_angular_velocity_correction:
            add_angular_velocity_aft = correct_angular_velocity(add_angular_velocity)
            if param.flag_evaluate_angular_velocity_abs:
                angular_velocity_list.append(np.abs(add_angular_velocity_aft))
            else:
                angular_velocity_list.append(add_angular_velocity_aft)
        else:
            if param.flag_evaluate_angular_velocity_abs:
                angular_velocity_list.append(np.abs(add_angular_velocity))
            else:
                angular_velocity_list.append(add_angular_velocity)
    # save
    save2csv.save_angle_angular_velocity(angle_list, angular_velocity_list, day)

    return angle_list, angular_velocity_list
