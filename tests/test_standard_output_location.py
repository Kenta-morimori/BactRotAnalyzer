"""Synthetic checks for the standard-result root and shared raw observations."""

from pathlib import Path
from runpy import run_path

import pytest

from utils import param
from utils.functions import raw_centroid
from utils.run_inputs import dataset_inputs


def test_standard_results_and_shared_raw_centroids_use_distinct_roots(tmp_path, monkeypatch):
    results = tmp_path / "outputs" / "rotation_analysis"
    shared = tmp_path / "outputs"
    source = tmp_path / "data" / "day" / "sample.avi"
    monkeypatch.setattr(param, "save_dir_bef", str(results))
    monkeypatch.setattr(param, "raw_centroid_root", str(shared))
    monkeypatch.setattr(raw_centroid, "source_path", lambda *_args: source)

    assert (
        raw_centroid.coordinate_path("day", 1) == shared / "day" / "raw_centroid" / raw_centroid.METHOD / "sample.csv"
    )
    assert list(dataset_inputs("day", tmp_path, results))[0] == tmp_path / "data" / "day"
    assert results / "day" / "centroid_coordinate.csv" in list(dataset_inputs("day", tmp_path, results))


def test_grouped_av_reader_prefers_new_results_and_falls_back_to_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/03_av_analysis/make_dataset.py"
    input_paths = run_path(str(script))["input_paths"]
    root = tmp_path / "outputs" / "rotation_analysis"
    current = root / "SJW46_temp=10" / "angular_velocity"
    current.mkdir(parents=True)
    (current / "angular-velocity_time-series.csv").write_text("No.1\n1\n")
    (current.parent / "time_list.csv").write_text("No.1\n0\n")

    angular_velocity, time = input_paths(root)
    assert angular_velocity["SJW46_10"] == current / "angular-velocity_time-series.csv"
    assert time["SJW46_10"] == root / "SJW46_temp=10" / "time_list.csv"
    assert angular_velocity["SJW46_23"] == Path(
        "outputs/SJW46_temp=23/angular_velocity/angular-velocity_time-series.csv"
    )

    (current / "angular-velocity_time-series.csv").unlink()
    with pytest.raises(ValueError, match="Incomplete standard rotation output"):
        input_paths(root)
