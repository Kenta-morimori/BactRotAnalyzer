import configparser

import pandas as pd

from utils import param


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
