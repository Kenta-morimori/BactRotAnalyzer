import configparser
import json

import cv2
import numpy as np
import pytest

from utils import param
from utils.functions import raw_centroid, repellent_response


@pytest.mark.parametrize(
    "time_values",
    [[0.0, 0.005, 0.010, 0.015], [0.0, np.nan, 0.010, 0.015], [0.0, 0.005, np.nan, np.nan], [np.nan] * 4],
)
def test_shared_raw_centroid_preserves_frames_and_cache(tmp_path, monkeypatch, time_values):
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
    writer = cv2.VideoWriter(str(avi), getattr(cv2, "VideoWriter_fourcc")(*"MJPG"), 200, (64, 64))
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
    times = [time_values]

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
    np.testing.assert_allclose(selected_time[0], times[0][1:4], equal_nan=True)
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


def test_contour_mean_cache_is_rejected_and_reextracted_with_frame_identity(tmp_path, monkeypatch):
    source = tmp_path / "sample.avi"
    source.touch()
    monkeypatch.setattr(raw_centroid, "source_path", lambda *_args: source)
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path / "outputs"))
    monkeypatch.setattr(param, "get_px2um_config", lambda _day: (0.02, 0.03))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:45, 12:25] = 255
    mask[35:45, 25:52] = 255
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frames = [image, np.zeros_like(image), image]
    times = [0.0, 0.005, 0.025]
    opened = []

    class Capture:
        def __init__(self, _path):
            opened.append(_path)
            self.frames = iter(frames)

        def isOpened(self):
            return True

        def read(self):
            frame = next(self.frames, None)
            return (frame is not None), frame

        def release(self):
            pass

    monkeypatch.setattr(raw_centroid.cv2, "VideoCapture", Capture)
    old_method = "standard-largest-contour-mean-threshold-120-v2"
    assert raw_centroid.METHOD != old_method
    # A historical cache may have valid frame/time metadata but wrong centroids.
    with monkeypatch.context() as historical:
        historical.setattr(raw_centroid, "METHOD", old_method)
        cached = raw_centroid.extract_or_load("day", 1, times)
        cached.loc[cached.detected, ["x_px", "x_um"]] = -99.0
        cached.to_csv(raw_centroid.coordinate_path("day", 1), index=False)
        (tmp_path / "outputs" / "day" / "centroid_coordinate.csv").write_text("x_1,y_1\n")
        raw_centroid.save_standard_manifest("day", [times])

    assert not raw_centroid.standard_cache_matches("day", [times])
    with pytest.raises(ValueError, match="Stale or malformed"):
        raw_centroid.load("day", 1, times)

    result = raw_centroid.extract_or_load("day", 1, times)
    expected_y, expected_x = np.nonzero(mask)
    np.testing.assert_allclose(result.x_px, [expected_x.mean(), np.nan, expected_x.mean()], equal_nan=True)
    np.testing.assert_allclose(result.y_px, [expected_y.mean(), np.nan, expected_y.mean()], equal_nan=True)
    np.testing.assert_allclose(result.x_um, result.x_px * 0.02, equal_nan=True)
    np.testing.assert_allclose(result.y_um, result.y_px * 0.03, equal_nan=True)
    assert result.source_frame_1based.tolist() == [1, 2, 3]
    assert result.time_sec.tolist() == times
    assert result.detected.tolist() == [True, False, True]
    assert len(opened) == 2
    with raw_centroid.coordinate_path("day", 1).with_suffix(".json").open() as stream:
        assert json.load(stream)["method"] == raw_centroid.METHOD

    assert raw_centroid.extract_or_load("day", 1, times).equals(result)
    assert len(opened) == 2
    raw_centroid.save_standard_manifest("day", [times])
    assert raw_centroid.standard_cache_matches("day", [times])
