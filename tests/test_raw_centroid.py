import configparser

import cv2
import numpy as np
import pytest

from utils import param
from utils.functions import raw_centroid, repellent_response


def test_shared_raw_centroid_preserves_frames_and_cache(tmp_path, monkeypatch):
    input_root = tmp_path / "data"
    output_root = tmp_path / "outputs"
    day_dir = input_root / "day"
    day_dir.mkdir(parents=True)
    config = configparser.ConfigParser()
    config["Settings"] = {"flag_use_tiff_log": "True", "px2um_x": "0.02", "px2um_y": "0.02"}
    config["Tiff_info"] = {"tiff_data": "sample"}
    with (day_dir / "config.ini").open("w") as stream:
        config.write(stream)
    avi = day_dir / "sample.avi"
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), 200, (64, 64))
    if not writer.isOpened():
        pytest.skip("MJPG AVI writer unavailable")
    for center in ((20, 22), None, (32, 28), (34, 30)):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        if center is not None:
            cv2.ellipse(image, center, (8, 6), 0, 0, 360, (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    monkeypatch.setattr(param, "input_dir_bef", str(input_root))
    monkeypatch.setattr(param, "save_dir_bef", str(output_root))
    times = [[0.0, 0.005, 0.010, 0.015]]

    standard = raw_centroid.load_all("day", times)[0]
    assert standard.source_frame_1based.tolist() == [1, 2, 3, 4]
    assert standard.detected.tolist() == [True, False, True, True]
    assert np.isnan(standard.x_px.iloc[1])
    assert np.allclose(standard.x_um, standard.x_px * 0.02, equal_nan=True)

    def must_not_reopen(_path):
        raise AssertionError("Raw AVI was decoded twice")

    monkeypatch.setattr(raw_centroid.cv2, "VideoCapture", must_not_reopen)
    repellent = raw_centroid.load_all("day", times)[0]
    selected_time, selected, _ = repellent_response.apply_analysis_frame_ranges(
        times, [[repellent.x_um.tolist()], [repellent.y_um.tolist()]], [(2, 4)]
    )
    assert selected_time[0] == times[0][1:4]
    np.testing.assert_allclose(selected[0][0], standard.x_um.iloc[1:4], equal_nan=True)
    np.testing.assert_allclose(selected[1][0], standard.y_um.iloc[1:4], equal_nan=True)
    assert raw_centroid.load("day", 1, times[0]).equals(standard)
    with pytest.raises(ValueError, match="Timestamps"):
        raw_centroid.load("day", 1, [0.0, 0.006, 0.010, 0.015])

    assert not raw_centroid.standard_cache_matches("day", times)
    (output_root / "day" / "centroid_coordinate.csv").write_text("x_1,y_1\n")
    raw_centroid.save_standard_manifest("day", times)
    assert raw_centroid.standard_cache_matches("day", times)
    assert not raw_centroid.standard_cache_matches("day", [[0.0, 0.006, 0.010, 0.015]])
