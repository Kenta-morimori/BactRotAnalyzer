import configparser

import numpy as np
import pandas as pd

from utils import param
from utils.functions import repellent_response


def test_tiff_log_uses_tiff_data_length_not_settings_sample_num(tmp_path, monkeypatch):
    input_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    day_dir = input_root / "day"
    day_dir.mkdir(parents=True)
    output_day = output_root / "day"
    output_day.mkdir(parents=True)

    config = configparser.ConfigParser()
    config["Settings"] = {
        "sample_num": "99",
        "flag_use_tiff_log": "True",
        "px2um_x": "1",
        "px2um_y": "1",
    }
    config["Tiff_info"] = {"tiff_data": "first, second"}
    with (day_dir / "config.ini").open("w", encoding="utf-8") as config_file:
        config.write(config_file)

    pd.DataFrame({"No.1": [0.0, 0.5, 1.0], "No.2": [0.0, 0.25, 0.5]}).to_csv(
        output_day / "time_list.csv", index=False
    )
    monkeypatch.setattr(param, "input_dir_bef", str(input_root))
    monkeypatch.setattr(param, "save_dir_bef", str(output_root))

    sample_num, frame_rates, total_times = param.get_config("day")

    assert sample_num == 2
    assert frame_rates == [3.0, 6.0]
    assert total_times == [1.0, 0.5]


def test_manual_rise_time_tokens_and_per_sample_fallback(tmp_path, monkeypatch):
    input_root = tmp_path / "data"
    day_dir = input_root / "day"
    day_dir.mkdir(parents=True)
    config = configparser.ConfigParser()
    config["Settings"] = {"flag_use_tiff_log": "True"}
    config["Tiff_info"] = {"tiff_data": "first, second, third, fourth, fifth"}
    config["RepellentResponse"] = {"manual_rise_time_sec": "None, NaN, auto, , 40.0"}
    with (day_dir / "config.ini").open("w", encoding="utf-8") as config_file:
        config.write(config_file)
    monkeypatch.setattr(param, "input_dir_bef", str(input_root))

    assert param.get_manual_rise_time_sec_config("day") == [None, None, None, None, 40.0]

    results = [
        {"rise_index": 1.0, "baseline_mean": 3.0, "baseline_std": 1.0, "threshold": 6.0},
        {"rise_index": 1.0, "baseline_mean": 4.0, "baseline_std": 2.0, "threshold": 10.0},
    ]
    monkeypatch.setattr(param, "get_flag_use_manual_rise_time", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(param, "get_manual_rise_time_sec_config", lambda _day: [None, 40.0])

    repellent_response.add_rise_time_to_results("day", results, [[0.0, 10.0, 20.0], [0.0, 30.0, 40.0, 50.0]])

    assert results[0]["rise_index"] == 1.0
    assert results[0]["rise_time"] == 10.0
    assert results[0]["baseline_mean"] == 3.0
    assert results[1]["rise_index"] == 2.0
    assert results[1]["rise_time"] == 40.0
    assert np.isnan(results[1]["baseline_mean"])
