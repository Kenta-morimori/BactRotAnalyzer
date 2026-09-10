import configparser

import numpy as np
import pytest

from utils import param
from utils.functions import frequency_analysis, repellent_response


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
