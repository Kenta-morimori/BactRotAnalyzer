import numpy as np
from sklearn.cluster import KMeans

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import make_graph, make_scale, rot_df_manage, save2csv


def evaluate_switching(angular_velocity_list, day):
    sample_num, _, _ = param.get_config(day)
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

    # get ngular_velocity_list average using window function
    window_width_sec = 0.1  # [s]: window width
    angular_velocity_list_aft = []
    for i in range(sample_num):
        add_av_data = []
        total_frame = len(angular_velocity_list_bef[i])
        window_frame = int(window_width_sec * total_frame / total_time)

        for frame_i in range(int(total_frame - window_frame)):
            add_av_data.append(np.mean(angular_velocity_list_bef[i][frame_i : frame_i + window_frame]).astype(float))
        angular_velocity_list_aft.append(add_av_data)

    # plot Averaged Angular Velocity
    make_graph.plot_averaged_angular_velocity(angular_velocity_list_aft, day)

    # CCW/CW
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


def k_means_av(av_list, day):
    sample_num, _, _ = param.get_config(day)

    th_list, mean_list = [], []
    for i in range(sample_num):
        data = av_list[i]
        if isinstance(data, list):
            data = np.array(data)

        data = data[~np.isnan(data)]
        threshold_distance = 5

        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(data.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        center_distance = np.abs(centers[1] - centers[0])

        if center_distance < threshold_distance:
            threshold = None
        threshold = (centers[0] + centers[1]) / 2
        th_list.append(threshold)

        if threshold is not None:
            mean_value = np.mean(data[data > threshold])
            mean_list.append(mean_value)
        else:
            mean_list.append(None)
    return th_list, mean_list


def get_angular_velocity_rot_part(angular_velocity_list, day):
    th_list, mean_list = k_means_av(angular_velocity_list, day)
    # save
    rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_angular_velosity_th, th_list, day)
    rot_df_manage.update_rot_df(ROTATION_FEATURES.angular_velosity_mean_rot_part, mean_list, day)

    # dev
    angular_velocity_means = make_scale.get_data_stat(angular_velocity_list, day, stat_type="mean")
    angular_velocity_medians = make_scale.get_data_stat(angular_velocity_list, day, stat_type="median")
    make_graph.dev_plot_av_with_stats(angular_velocity_list, angular_velocity_means, angular_velocity_medians, day)
    # angular_velocity_means, angular_velocity_medians で再度K-means (関数化して利用)
    th_list_means, _ = k_means_av(angular_velocity_means, day)
    th_list_median, _ = k_means_av(angular_velocity_medians, day)

    # plot
    make_graph.plot_angular_velocity_rot_part(angular_velocity_list, th_list, th_list_means, th_list_median, day)
