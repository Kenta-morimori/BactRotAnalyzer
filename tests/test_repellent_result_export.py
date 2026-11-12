"""Aggregated output, source identity and display-only FFT contracts."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from utils import param
from utils.functions import repellent_result_export as output
from utils.functions.repellent_result_adapter import collect_results


def observations(no=10):
    return pd.DataFrame(
        dict(
            sample_no=no,
            source_frame_1based=np.arange(1, 8),
            time_sec=[0, 0.005, 0.010, 0.5, 0.505, 0.510, 0.515],
            raw_x_um=[1.0, 1.2, np.nan, 1.1, 1.2, 1.0, 1.1],
            raw_y_um=[2.0, 2.1, np.nan, 2.2, 2.1, 2.0, 2.1],
            center_x_um=[0.8] * 7,
            center_y_um=[1.8] * 7,
            angular_velocity_rad_s=[np.nan, 2, np.nan, np.nan, -2, -3, 1],
            primary_angular_velocity_rad_s=[np.nan, 2, np.nan, np.nan, np.nan, np.nan, 1],
            center_status=[
                "one_sided_mean_hold",
                "direct",
                "unavailable",
                "direct",
                "two_sided_mean_linear",
                "two_sided_mean_linear",
                "direct",
            ],
            centroid_missing_reason=["", "", "contact_unresolved", "", "", "", ""],
            source_id=f"synthetic_{no}",
        )
    )


def settings(**extra):
    return dict(
        sign_convention="CCW_positive_image_y_down",
        series_role="provisional_diagnostic",
        angular_sampling="adjacent_source_frames",
        rise_source_frames={"10": 4, "14": 4},
        **extra,
    )


@pytest.fixture
def captured_figures(monkeypatch):
    captured = {}

    def save(fig, filename):
        captured[Path(filename).name] = fig
        Path(filename).write_bytes(b"synthetic plot placeholder")
        plt.close(fig)

    monkeypatch.setattr(output, "_figure_end", save)
    return captured


def test_log_spectrum_masks_display_only_and_keeps_axes_log():
    f = np.array([0, 0.1, 1, 2, 3, 4.0])
    a = np.array([4, 3, 0, -1, np.nan, 2.0])
    before = (f.copy(), a.copy())
    fig, ax = plt.subplots()
    valid = output.plot_log_spectrum(ax, f, a)
    assert ax.get_xscale() == ax.get_yscale() == "log"
    np.testing.assert_array_equal(valid, [False, True, False, False, False, True])
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.1, 4])
    np.testing.assert_array_equal(f, before[0])
    np.testing.assert_array_equal(a, before[1])
    plt.close(fig)


def test_aggregate_tables_pages_labels_and_missing_values(tmp_path, captured_figures):
    data = pd.concat([observations(), observations(14)], ignore_index=True)
    before = data.copy(deep=True)
    output.export_results(tmp_path, data, settings=settings(samples_per_page=1))
    pd.testing.assert_frame_equal(before, data)
    table = pd.read_csv(tmp_path / output.ALL / "angular_velocity/angular_velocity_time_series.csv")
    np.testing.assert_allclose(table.angular_velocity_rad_s, data.angular_velocity_rad_s, equal_nan=True)
    assert table.sample_no.tolist() == [10] * 7 + [14] * 7
    assert table.source_frame_1based.tolist() == list(range(1, 8)) * 2
    inventory = pd.read_csv(tmp_path / "output_manifest.csv")
    assert all((tmp_path / p).is_file() for p in inventory.path)
    assert not list(tmp_path.rglob("no10")) and not list(tmp_path.rglob("runs"))
    assert "centroid_center_time_series_labeled_part02.png" in captured_figures
    assert "SD_fft_loglog.png" not in captured_figures
    fig = captured_figures["angular_velocity_time_series_labeled_part01.png"]
    assert any(np.asarray(line.get_ydata()).tolist() == [0, 0] for line in fig.axes[0].lines)
    assert fig.axes[0].patches  # completion and recording gap bands
    statuses = json.loads((tmp_path / "settings.json").read_text())
    assert statuses["analysis_status"]["physical_reversal"] == "not_assessed"
    frequency = pd.read_csv(tmp_path / output.POST / "switching_analysis/reversal_frequency.csv")
    assert frequency.frequency_hz.isna().all()


def test_line_breaks_at_missing_pairs_and_source_gaps():
    data = observations()
    data.loc[2, "raw_x_um"] = 10
    assert [p.tolist() for p in output._parts(data, ["raw_x_um"])] == [[0, 1], [3, 4, 5, 6]]
    fig, ax = plt.subplots()
    output._line(ax, data, ["raw_x_um"])
    assert len(ax.lines) == 2
    plt.close(fig)


def test_wrong_sign_duplicate_frames_and_cross_gap_velocity_rejected(tmp_path):
    data = observations()
    with pytest.raises(ValueError, match="CCW"):
        output.export_results(tmp_path, data, settings=dict(sign_convention="CW", series_role="primary"))
    data.loc[1, "source_frame_1based"] = 1
    with pytest.raises(ValueError, match="duplicate"):
        output.export_results(tmp_path, data, settings=settings())
    data = observations()
    data.loc[3, "angular_velocity_rad_s"] = 4
    with pytest.raises(ValueError, match="unavailable observation"):
        output.export_results(tmp_path, data, settings=settings())


def test_spectrum_values_and_before_centers_survive_export(tmp_path, captured_figures):
    data = observations()
    data["center_before_x_um"] = np.nan
    data["center_before_y_um"] = np.nan
    data.loc[1, ["center_before_x_um", "center_before_y_um"]] = [0.7, 1.7]
    spectrum = pd.DataFrame(
        dict(sample_no=[10] * 4, sd_window_sec=[0.1] * 4, frequency_hz=[0, 1, 2, 3], amplitude=[5, 2, 0, np.nan])
    )
    output.export_results(tmp_path, data, settings=settings(details=True), spectrum=spectrum)
    saved = pd.read_csv(tmp_path / output.PRE / "fluctuation_analysis/SD_fft.csv")
    np.testing.assert_allclose(saved.amplitude, spectrum.amplitude, equal_nan=True)
    axes = captured_figures["SD_fft_loglog.png"].axes
    assert axes[0].get_xscale() == axes[0].get_yscale() == "log"
    assert (tmp_path / "details/center_before_correction.csv").exists()
    assert not any("comparison" in x for x in pd.read_csv(tmp_path / "output_manifest.csv").path)


def test_reexport_removes_only_previous_manifest_owned_stale_files(tmp_path, captured_figures):
    data = observations()
    data["center_before_x_um"] = 0.7
    data["center_before_y_um"] = 1.7
    output.export_results(tmp_path, data, settings=settings(details=True))
    keep = tmp_path / "details/user_experiment.csv"
    keep.write_text("preserve")
    output.export_results(tmp_path, data, settings=settings(details=False))
    assert keep.read_text() == "preserve"
    assert not (tmp_path / "details/center_before_correction.csv").exists()


def test_existing_backend_adapter_uses_explicit_source_frames_and_flips_once(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    raw = pd.DataFrame(dict(x_um=np.arange(17), y_um=np.arange(17), reason="accepted"))
    rotation = dict(valid_indices=[0], angular_velocity_list=[[2, -3]], source_frame_list=[[12, 16]])
    bundle = collect_results(
        "day",
        [dict(source_start_frame_1based=11)],
        [raw],
        [[0, 0.005, 0.01, 0.5, 0.505, 0.51, 0.515]],
        [[1] * 7],
        [[2] * 7],
        [[0.8] * 7],
        [[1.8] * 7],
        rotation,
        [[100] * 7],
        [dict(rise_index=3)],
        "v2",
    )
    frame = bundle["observations"]
    finite = frame.loc[frame.angular_velocity_rad_s.notna()]
    assert finite.source_frame_1based.tolist() == [12, 16]
    assert finite.angular_velocity_rad_s.tolist() == [-2, 3]
    assert frame.centroid_missing_reason.eq("").all()
    assert len(frame) == 7 and bundle["settings"]["rise_source_frames"] == {"1": 14}


def test_main_publishes_fixed_phase_layout_and_discards_intermediates(tmp_path, monkeypatch, captured_figures):
    path = Path(__file__).parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("grouped_entry", path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    from utils.functions.repellent_output import current

    stages = []

    def computed(*args, **kwargs):
        stage = current().root
        stages.append(stage)
        (stage / "unwanted_debug.txt").write_text("intermediate")
        return dict(observations=observations(), settings=settings())

    monkeypatch.setattr(entry, "_main", computed)
    entry.main("day", 0.5, 3, 3, 1, 0.2, 1, None, center_estimator="v2")
    root = tmp_path / "day/repellent_response"
    assert (root / "output_manifest.csv").exists() and not (root / "runs").exists()
    assert not list(root.rglob("unwanted_debug.txt"))
    assert all(not stage.exists() for stage in stages)


def test_existing_switching_statistics_are_preserved_separately_from_physical_reversals(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    folder = tmp_path / "day/repellent_response/00_all_rotational_analysis/angular_velocity"
    folder.mkdir(parents=True)
    pd.DataFrame({"No.1_time": [0.6, 1.6], "No.1_count": [2.0, 4.0]}).to_csv(
        folder / "switching_count.csv", index=False
    )
    pd.DataFrame({"No.1_time": [0.6, 1.6], "No.1_cw_rate": [0.2, 0.7]}).to_csv(folder / "cw_rate.csv", index=False)
    clock = np.arange(500) * 0.005
    raw = pd.DataFrame(dict(x_um=clock, y_um=clock))
    rotation = dict(valid_indices=[0], angular_velocity_list=[np.ones(500)], source_frame_list=[np.arange(1, 501)])
    bundle = collect_results(
        "day",
        [dict(source_start_frame_1based=1)],
        [raw],
        [clock],
        [np.ones(500)],
        [np.ones(500)],
        [np.zeros(500)],
        [np.zeros(500)],
        rotation,
        [np.ones(500)],
        [dict(rise_index=200)],
        "v2",
    )
    metrics = bundle["frequency"]
    assert metrics.time_sec.tolist() == [1.6]  # entire saved 1-second window is after rise
    assert metrics.diagnostic_switch_count.tolist() == [4.0]
    assert metrics.cw_fraction.tolist() == [0.7]
    assert metrics.frequency_hz.isna().all()
    fig, ax = plt.subplots()
    output._frequency_line(ax, metrics, "diagnostic_zero_crossing_rate_hz")
    assert len(ax.lines) == 1 and ax.lines[0].get_ydata().tolist() == [4.0]
    plt.close(fig)


def test_empty_spectrum_is_labeled_without_inventing_positive_data():
    fig, axis = plt.subplots()
    valid = output.plot_log_spectrum(axis, [], [])
    assert not valid.any() and len(axis.lines[0].get_xdata()) == 0
    assert any("No positive finite spectrum" in t.get_text() for t in axis.texts)
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize("estimator", ["validated", "legacy"])
def test_compatibility_absolute_display_uses_signed_measurement_for_ccw_export(tmp_path, monkeypatch, estimator):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    monkeypatch.setattr(param, "flag_evaluate_angular_velocity_abs", True)
    raw = pd.DataFrame(dict(x_um=np.arange(17), y_um=np.arange(17)))
    rotation = dict(
        valid_indices=[0],
        angular_velocity_list=[[2, 3]],
        signed_angular_velocity_list=[[2, -3]],
        source_frame_list=[[12, 16]],
    )
    result = collect_results(
        "day",
        [dict(source_start_frame_1based=11)],
        [raw],
        [[0, 0.005, 0.01, 0.5, 0.505, 0.51, 0.515]],
        [[1] * 7],
        [[2] * 7],
        [[0.8] * 7],
        [[1.8] * 7],
        rotation,
        [[100] * 7],
        [dict(rise_index=3)],
        estimator,
    )
    finite = result["observations"].dropna(subset=["angular_velocity_rad_s"])
    assert finite.source_frame_1based.tolist() == [12, 16]
    assert finite.angular_velocity_rad_s.tolist() == [-2, 3]
