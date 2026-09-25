import configparser

import numpy as np
import pytest

from utils import param
from utils.functions import frequency_analysis, get_angular_velocity, repellent_response


def _write_config(path, ranges):
    config = configparser.ConfigParser()
    config["Settings"] = {"flag_use_tiff_log": "True"}
    config["Tiff_info"] = {"tiff_data": "first, second"}
    config["RepellentResponse"] = {"analysis_frame_ranges": ranges}
    with path.open("w", encoding="utf-8") as fp:
        config.write(fp)


def test_analysis_frame_range_tokens_and_validation(tmp_path, monkeypatch):
    day_dir = tmp_path / "data" / "day"
    day_dir.mkdir(parents=True)
    monkeypatch.setattr(param, "input_dir_bef", str(tmp_path / "data"))

    _write_config(day_dir / "config.ini", "auto, 2049-16384")
    assert param.get_analysis_frame_ranges_config("day") == [None, (2049, 16384)]

    _write_config(day_dir / "config.ini", "None, NaN")
    assert param.get_analysis_frame_ranges_config("day") == [None, None]

    _write_config(day_dir / "config.ini", "1-2")
    with pytest.raises(ValueError, match="one value per sample"):
        param.get_analysis_frame_ranges_config("day")

    _write_config(day_dir / "config.ini", "0-2, auto")
    with pytest.raises(ValueError, match="positive"):
        param.get_analysis_frame_ranges_config("day")


def test_analysis_frame_ranges_keep_source_time_and_indices():
    time = [list(np.arange(20, dtype=float)), list(np.arange(30, dtype=float))]
    x = [list(np.arange(20, dtype=float)), list(np.arange(30, dtype=float))]
    y = [list(np.arange(20, dtype=float) + 100), list(np.arange(30, dtype=float) + 100)]
    background = [list(np.arange(20, dtype=float) + 200), list(np.arange(30, dtype=float) + 200)]

    selected_time, selected, rows = repellent_response.apply_analysis_frame_ranges(
        time, [x, y, background], [None, (5, 18)]
    )

    assert selected_time[0] == time[0]
    assert selected_time[1] == list(np.arange(4, 18, dtype=float))
    assert len(selected_time[1]) == 14
    assert selected[0][1][0] == 4.0
    assert selected[1][1][-1] == 117.0
    assert selected[2][1][0] == 204.0
    assert rows[1]["source_start_index_0based"] == 4
    assert rows[1]["source_end_index_0based"] == 17

    with pytest.raises(ValueError, match="exceeds available"):
        repellent_response.apply_analysis_frame_ranges(time, [x], [None, (1, 31)])


def test_longest_continuous_segment_excludes_timestamp_gap():
    time = np.r_[np.arange(0.0, 1.0, 0.1), np.arange(5.0, 5.6, 0.1)]
    values = np.sin(time)
    (segment_time, segment_values), metadata = frequency_analysis.longest_continuous_segment(time, values)

    assert len(segment_time) == 10
    assert np.allclose(segment_time, np.arange(0.0, 1.0, 0.1))
    assert np.allclose(segment_values, np.sin(np.arange(0.0, 1.0, 0.1)))
    assert metadata["start_index"] == 0
    assert metadata["end_index"] == 9
    assert metadata["effective_frame_rate_hz"] == pytest.approx(10.0)


def test_cleanup_removes_stale_response_outputs(tmp_path, monkeypatch):
    root = tmp_path / "outputs" / "day" / "repellent_response"
    stale = root / "01_brightness_change" / "background_intensity_time_series.png"
    stale.parent.mkdir(parents=True)
    stale.touch()
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path / "outputs"))

    repellent_response.cleanup_legacy_repellent_outputs("day")

    assert not root.exists()


def test_angular_velocity_is_missing_across_timestamp_jump(monkeypatch):
    monkeypatch.setattr(param, "get_config", lambda _day: (1, [100.0], [1.03]))
    monkeypatch.setattr(param, "flag_get_angle_with_cell_direcetion", False)
    monkeypatch.setattr(param, "flag_correct_av_outlier", False)
    monkeypatch.setattr(param, "flag_evaluate_angular_velocity_abs", False)
    monkeypatch.setattr(param, "flag_eval_switching_with_averaged_av", False)
    monkeypatch.setattr(get_angular_velocity.make_graph, "plot_angular_velocity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(get_angular_velocity.save2csv, "save_angle_angular_velocity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(get_angular_velocity.rot_df_manage, "update_rot_df", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(get_angular_velocity.frequency_analysis, "fft_angle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(get_angular_velocity.frequency_analysis, "fft_angular_velocity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(get_angular_velocity.make_evaluate_switching, "evaluate_switching", lambda *_args, **_kwargs: ([], []))

    angle = np.arange(5, dtype=float) * 0.1
    _, av = get_angular_velocity.get_angular_velocity(
        [np.cos(angle)], [np.sin(angle)], "day", time_list=[[0.0, 0.01, 0.02, 1.02, 1.03]]
    )

    assert np.isfinite(av[0][0])
    assert np.isfinite(av[0][1])
    assert np.isnan(av[0][2])
    assert np.isfinite(av[0][3])


def test_all_time_angular_velocity_can_skip_duplicate_time_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path / "outputs"))
    monkeypatch.setattr(repellent_response.make_graph, "plot_repellent_angular_velocity_onecol", lambda *_args, **_kwargs: None)

    repellent_response.save_segment_angular_velocity_outputs(
        day="day",
        segment_subdir="00_all_rotational_analysis",
        time_list=[[0.0, 0.1, 0.2]],
        angle_list=[[0.0, 0.1, 0.2]],
        angular_velocity_list=[[1.0, 1.0]],
        write_time_list=False,
    )

    root = tmp_path / "outputs" / "day" / "repellent_response" / "00_all_rotational_analysis"
    assert (root / "angular_velocity" / "angle_time-series.csv").is_file()
    assert not (root / "time_list.csv").exists()
