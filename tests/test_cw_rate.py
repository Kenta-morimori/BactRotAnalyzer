import numpy as np
import pandas as pd

from utils import param
from utils.functions import repellent_response, save2csv


def test_cw_rate_counts_only_negative_angular_velocity_samples():
    time, rate = repellent_response.calculate_angular_velocity_cw_rate(
        [[0.0, 0.5, 1.0, 1.5]], [[-2.0, 0.0, 3.0, np.nan]], window_width_sec=1.0
    )

    assert time == [[0.5, 1.0]]
    assert np.allclose(rate[0], [1.0 / 3.0, 0.0])


def test_cw_rate_excludes_nan_and_matches_switching_window_times():
    time_list = [[0.0, 0.5, 1.0, 1.5]]
    av_list = [[-1.0, np.nan, 0.0, 1.0]]

    cw_time, cw_rate = repellent_response.calculate_angular_velocity_cw_rate(time_list, av_list)
    switching_time, _ = repellent_response.calculate_angular_velocity_switching_count(time_list, av_list)

    assert cw_time == switching_time
    assert np.allclose(cw_rate[0], [0.5, 0.0])


def test_cw_rate_csv_has_time_and_rate_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))

    save2csv.save_angular_velocity_cw_rate([[0.5, 1.0]], [[0.25, 0.5]], "day")

    output_path = tmp_path / "day" / "repellent_response" / "00_all_rotational_analysis" / "angular_velocity" / "cw_rate.csv"
    output = pd.read_csv(output_path)
    assert output.columns.tolist() == ["No.1_time", "No.1_cw_rate"]
    assert output["No.1_cw_rate"].tolist() == [0.25, 0.5]
