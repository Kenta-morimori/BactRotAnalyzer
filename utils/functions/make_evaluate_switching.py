import numpy as np
from sklearn.cluster import KMeans

from utils import param
from utils.features import ROTATION_FEATURES
from utils.functions import make_graph, make_scale, rot_df_manage


def evaluate_switching_averaged(angular_velocity_mean_list, day):
    sample_num, _, _ = param.get_config(day)

    cw_ratio_list = []
    # get ngular_velocity_list average using window function
    for i in range(sample_num):
        cw_count, ccw_count = 0, 0
        for j in range(len(angular_velocity_mean_list[i])):
            if angular_velocity_mean_list[i][j] == np.nan:
                continue
            elif angular_velocity_mean_list[i][j] >= 0:
                cw_count += 1
            else:
                ccw_count += 1
        if ccw_count == 0:
            cw_ratio_list.append(-1)
        else:
            cw_ratio_list.append(round(cw_count / ccw_count, 3))

    return cw_ratio_list


def k_means_av(av_list, day):
    sample_num, _, _ = param.get_config(day)

    th_list = []
    # mean_list, sd_list = [], []
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

        """
        if threshold is not None:
            mean_list.append(np.nanmean(data[data > threshold]))
            sd_list.append(np.nanstd(data[data > threshold]))
        else:
            mean_list.append(None)
        """
    return th_list


def get_angular_velocity_rot_part(angular_velocity_list, day):
    flag_kmean = param.flag_kmean

    if flag_kmean:
        th_list = k_means_av(angular_velocity_list, day)
        # save
        rot_df_manage.update_rot_df(ROTATION_FEATURES.rot_angular_velosity_th, th_list, day)
        # rot_df_manage.update_rot_df(ROTATION_FEATURES.angular_velosity_mean_rot_part, mean_list, day)
        # rot_df_manage.update_rot_df(ROTATION_FEATURES.angular_velosity_sd_rot_part, sd_list, day)

    # dev
    angular_velocity_means = make_scale.get_data_stat(angular_velocity_list, day, stat_type="mean")
    angular_velocity_medians = make_scale.get_data_stat(angular_velocity_list, day, stat_type="median")
    make_graph.dev_plot_av_with_stats(angular_velocity_list, angular_velocity_means, angular_velocity_medians, day)

    if flag_kmean:
        # angular_velocity_means, angular_velocity_medians で再度K-means (関数化して利用)
        th_list_means, _, _ = k_means_av(angular_velocity_means, day)
        th_list_median, _, _ = k_means_av(angular_velocity_medians, day)
        # plot
        make_graph.plot_angular_velocity_rot_part(angular_velocity_list, th_list, th_list_means, th_list_median, day)
