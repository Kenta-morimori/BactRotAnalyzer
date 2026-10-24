"""Run isolation, failed-run records, and plotting against the stored contract."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.synthetic.stops import Case, generate
from utils import param
from utils.functions import repellent_plotting as plotting
from utils.functions.repellent_output import (
    current,
    response_root,
    response_run,
    set_ranges,
)
from utils.functions.repellent_sign_reference import analyze_sample


def test_runs_keep_existing_outputs_and_separate_modes_suffix_and_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    old = tmp_path / "day/repellent_response/00_time_list/time_list.csv"
    old.parent.mkdir(parents=True)
    old.write_text("existing timeline")
    paths = []
    for suffix in (None, "comparison"):
        for all_frames in (False, True):
            for _ in range(2):
                with response_run("day", all_frames=all_frames, output_suffix=suffix) as run:
                    paths.append(run.root)
                    assert response_root("day") == run.root
                    assert response_root("other") == tmp_path / "other/repellent_response"
                    assert json.loads((run.root / "run.json").read_text())["status"] == "running"
                    set_ranges("day", [dict(sample_no=1, source_start_frame_1based=201, source_end_frame_1based=400)])
                record = json.loads((run.root / "run.json").read_text())
                assert record["status"] == "success"
                assert record["source_frame_ranges"][0]["source_start_frame_1based"] == 201
                assert run.root.parent.name == ("all_frames" if all_frames else "configured")
                assert "JST" in run.run_id
                if suffix:
                    assert run.root.is_relative_to(tmp_path / suffix)
    assert len(set(paths)) == 8
    with pytest.raises(RuntimeError, match="failure"):
        with response_run("day") as failed:
            raise RuntimeError("failure")
    assert json.loads((failed.root / "run.json").read_text())["status"] == "failed"
    assert current() is None
    assert param.save_dir_bef == str(tmp_path)
    assert old.read_text() == "existing timeline"


def test_entry_failure_restores_configuration_and_records_resolved_options(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("entry_failure", path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    original = param.post_rise_center_mode

    def fail(*args):
        param.post_rise_center_mode = 99
        raise ValueError("synthetic failure")

    monkeypatch.setattr(entry, "_main", fail)
    with pytest.raises(ValueError, match="synthetic failure"):
        entry.main("day", 0.5, 3, 3, 1, 0.2, 1, "comparison", all_frames=True)
    records = list(tmp_path.glob("comparison/day/repellent_response/runs/all_frames/*/run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "failed" and record["center_estimator"] == "v2"
    assert param.post_rise_center_mode == original and current() is None


def test_plot_lines_preserve_gaps_missing_values_and_singletons():
    clock = np.array([0, 0.025, 0.05, 0.6, 0.625, 0.65, 0.675])
    values = np.array([1, 2, np.nan, 3, np.nan, 4, 5])
    assert list(plotting.continuous_parts(clock, values)) == [(0, 2), (3, 4), (5, 7)]
    fig, axes = plt.subplots(1, 2)
    plotting.plot_series(axes[0], clock, values)
    plotting.plot_series(axes[1], clock, np.column_stack((values, values)), xy=True)
    assert [len(line.get_xdata()) for line in axes[0].lines] == [2, 1, 2]
    assert [len(line.get_xdata()) for line in axes[1].lines] == [2, 1, 2]
    assert axes[0].lines[1].get_marker() == "."
    # Identical clock spacing can still belong to separate original recording runs.
    fig2, axis = plt.subplots()
    plotting.plot_series(axis, [0, 0.025, 0.05], [1, 2, 3], segments=[0, 1, 1])
    assert [len(line.get_xdata()) for line in axis.lines] == [1, 2]
    plt.close(fig)
    plt.close(fig2)


def test_reference_and_velocity_artists_match_quality_values():
    data = generate(Case("step", shape="step"), 10400)
    _, frame, episodes = analyze_sample(data["time"], data["observed"])
    fig, axes = plt.subplots(1, 3)
    for axis, components in zip(axes[:2], (False, True)):
        plotting.reference_series(axis, frame, component=0, components=components)
        actual = sorted((float(x), float(y)) for line in axis.lines for x, y in zip(line.get_xdata(), line.get_ydata()))
        valid = frame.final_reference_x_um.notna()
        expected = sorted(zip(frame.loc[valid, "time_sec"], frame.loc[valid, "final_reference_x_um"]))
        assert actual == expected
    plotting.velocities(axes[2], frame)
    crosses = [line for line in axes[2].lines if line.get_marker() == "x"]
    assert crosses and all(line.get_linestyle() == "None" for line in crosses)
    actual = sorted((float(x), float(y)) for line in crosses for x, y in zip(line.get_xdata(), line.get_ydata()))
    low = frame.stop_velocity_quality.eq("low_confidence")
    assert actual == sorted(zip(frame.loc[low, "time_sec"], frame.loc[low, "stop_angular_velocity_raw_rad_s"]))
    plotting.candidate_bands([axes[2]], episodes)
    assert len(axes[2].patches) == len(episodes)
    plt.close(fig)


def test_phase_tables_and_plots_share_original_values(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    data = generate(Case("quiet"), 10400)
    _, frame, episodes = analyze_sample(data["time"], data["observed"], source_start=201)
    saved = []

    def capture(fig, path, **kwargs):
        saved.append(Path(path))
        plt.close(fig)

    monkeypatch.setattr(plotting, "save_figure", capture)
    samples = [(data["time"], data["observed"], frame, episodes)]
    with response_run("day") as run:
        plotting.phase_figures("day", samples, [3.6], source_starts=[201], rise_indices=[720])
    assert len(saved) == 15
    for name in (
        "centroid_before.png",
        "rotation_center.png",
        "centroid_before_components.png",
        "rotation_center_components.png",
        "angular_velocity_with_stop.png",
    ):
        assert sum(p.name == name for p in saved) == 3
    tables = [
        pd.read_csv(run.root / segment / "rotation_center/no01/timeseries.csv")
        for segment in ("00_all_rotational_analysis", "02_pre_rise_fluctuation", "03_post_rise_analysis")
    ]
    pd.testing.assert_frame_equal(tables[0], pd.concat(tables[1:], ignore_index=True))
    assert tables[1].time_sec.lt(3.6).all() and tables[2].time_sec.ge(3.6).all()
    assert np.array_equal(tables[0].source_frame_1based, tables[0].frame_0based + 201)
    np.testing.assert_allclose(tables[0].final_reference_x_um, frame.final_reference_x_um, equal_nan=True)


def test_cached_timeline_keeps_internal_missing_timestamp(tmp_path, monkeypatch):
    from utils.functions.repellent_response import ensure_time_list

    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    folder = tmp_path / "day"
    folder.mkdir()
    pd.DataFrame({"No.1": [0, np.nan, 0.01, 0.015], "No.2": [0, 0.005, np.nan, np.nan]}).to_csv(
        folder / "time_list.csv", index=False
    )
    before = (folder / "time_list.csv").read_bytes()
    result = ensure_time_list("day")
    assert len(result[0]) == 4 and np.isnan(result[0][1])
    assert result[1] == [0, 0.005]
    assert (folder / "time_list.csv").read_bytes() == before


def test_production_settings_merge_run_metadata(tmp_path, monkeypatch):
    from utils.functions import repellent_sign_reference, rotation_center_output

    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    monkeypatch.setattr(rotation_center_output, "plot_sample", lambda *args: None)
    with response_run("short", options=dict(center_estimator="v2", stop_measurement_enabled=False)) as run:
        repellent_sign_reference.analyze(
            "short", [np.arange(3) / 200], [[0, 0, 0]], [[0, 0, 0]], stop_measurement_enabled=False
        )
    settings = json.loads((run.root / "rotation_center/settings.json").read_text())
    assert settings["output_schema_version"] == 2
    assert settings["center_estimator"] == "v2" and not settings["stop_measurement_enabled"]
    assert settings["run_id"] == run.run_id and settings["mode"] == "configured"
    summary = pd.read_csv(run.root / "rotation_center/summary.csv")
    assert summary.final_reference_denominator.iloc[0] == 0
    assert summary.stop_quality_denominator.iloc[0] == 0


def test_cached_timeline_preserves_missing_final_frames_from_source_count(tmp_path, monkeypatch):
    from utils.functions import input_data, repellent_response

    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    monkeypatch.setattr(input_data, "get_ordered_avi_paths", lambda day: ["source.avi"])
    folder = tmp_path / "day"
    (folder / "raw_centroid").mkdir(parents=True)
    pd.DataFrame({"No.1": [0, 0.005, np.nan]}).to_csv(folder / "time_list.csv", index=False)
    (folder / "raw_centroid/source.json").write_text(json.dumps({"frame_count": 3}))
    times = repellent_response.ensure_time_list("day")[0]
    assert len(times) == 3 and np.isnan(times[-1])


@pytest.mark.parametrize("times", [[0.0, 0.005, np.nan], [np.nan, np.nan, np.nan]])
def test_cached_timeline_without_centroid_cache_keeps_source_frame_count(tmp_path, monkeypatch, times):
    from utils.functions import input_data, repellent_response

    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    monkeypatch.setattr(input_data, "get_ordered_avi_paths", lambda day: ["source.avi"])
    folder = tmp_path / "day"
    folder.mkdir()
    path = folder / "time_list.csv"
    pd.DataFrame({"No.1": times}).to_csv(path, index=False)
    original = path.read_bytes()
    released = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, _property):
            return 3

        def release(self):
            released.append(True)

    monkeypatch.setattr(repellent_response.cv2, "VideoCapture", lambda _path: Capture())
    result = repellent_response.ensure_time_list("day")
    np.testing.assert_allclose(result[0], times, equal_nan=True)
    assert path.read_bytes() == original and released == [True]


def test_generated_timeline_is_read_without_compacting_missing_times(tmp_path, monkeypatch):
    from utils.functions import get_tiff_info, repellent_response

    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))

    def generate_timeline(day):
        folder = tmp_path / day
        folder.mkdir()
        pd.DataFrame({"No.1": [0.0, np.nan, 0.010]}).to_csv(folder / "time_list.csv", index=False)

    monkeypatch.setattr(get_tiff_info, "get_timelist", generate_timeline)
    result = repellent_response.ensure_time_list("day")
    np.testing.assert_allclose(result[0], [0.0, np.nan, 0.010], equal_nan=True)


def test_phase_export_preserves_sparse_original_sample_number(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    _, frame, episodes = analyze_sample(np.arange(3) / 200, np.zeros((3, 2)))
    labels: list[str] = []

    def capture(fig, path, **kwargs):
        labels.extend(axis.get_title() for axis in fig.axes)
        assert kwargs["samples"] == [14]
        plt.close(fig)

    monkeypatch.setattr(plotting, "save_figure", capture)
    plotting.phase_figures(
        "short", [(np.arange(3) / 200, np.zeros((3, 2)), frame, episodes)], [np.nan], sample_numbers=[14]
    )
    root = tmp_path / "short/repellent_response"
    assert len(list(root.glob("*/rotation_center/no14/timeseries.csv"))) == 3
    assert not list(root.glob("*/rotation_center/no01"))
    assert any("No.14" in label for label in labels)


def test_one_missing_coordinate_breaks_both_time_traces():
    clock = np.arange(3) / 200
    points = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    fig, axes = plt.subplots(1, 2)
    for component, axis in enumerate(axes):
        values = plotting.coordinate_component(points, component)
        plotting.plot_series(axis, clock, values)
        assert len(axis.lines) == 2
        assert all(len(line.get_xdata()) == 1 for line in axis.lines)
    plt.close(fig)
