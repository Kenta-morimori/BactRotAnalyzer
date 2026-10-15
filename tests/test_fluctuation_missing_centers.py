import numpy as np
import pandas as pd

from utils import param
from utils.functions import (
    fluctuation_analysis,
    frequency_analysis,
    make_graph,
    repellent_response,
)


def test_missing_angular_velocity_stays_missing_in_fluctuation_windows(monkeypatch):
    time = (np.arange(21) / 20).tolist()
    velocity = np.linspace(1.0, 2.0, 20)
    velocity[10] = np.nan
    monkeypatch.setattr(param, "get_config", lambda day: (1, [20.0], [1.0]))
    monkeypatch.setattr(param, "SD_window_width_list", [0.2])
    monkeypatch.setattr(param, "mode_evaluate_SD_fluctuation", 1)
    monkeypatch.setattr(param, "flag_apply_gaussian_window", True)
    monkeypatch.setattr(fluctuation_analysis.read_csv, "get_timelist", lambda day: [time])

    result = fluctuation_analysis.get_sd_time_series(0, velocity, "sample")
    sd = result["No.1_0.2s_sd"]
    mean = result["No.1_0.2s_mean"]
    assert np.isfinite(sd.iloc[5])
    assert np.isnan(sd.iloc[10]) and np.isnan(mean.iloc[10])
    assert result["No.1_0.2s_data_num"].iloc[10] < 4
    assert np.isfinite(sd.iloc[15])


def test_pre_rise_fluctuation_excludes_samples_without_valid_velocity(monkeypatch):
    monkeypatch.setattr(param, "min_ref_av_num", 10)
    time = (np.arange(21) / 20).tolist()
    selected = repellent_response._get_valid_pre_rise_indices(
        [time, time, time],
        [[np.nan] * 20, [1.0] * 9 + [np.nan] * 11, [1.0] * 10 + [np.nan] * 10],
    )
    assert selected == [2]


def test_sd_fft_keeps_missing_gap_and_empty_width(monkeypatch):
    monkeypatch.setattr(param, "get_config", lambda day: (1, [20.0], [1.0]))
    monkeypatch.setattr(param, "SD_window_width_list", [0.2, 2.0])
    monkeypatch.setattr(make_graph, "plot_SD_list_fft", lambda *args: None)
    monkeypatch.setattr(make_graph, "dev_plot_sd_FFT_with_rotation", lambda *args: None)
    monkeypatch.setattr(frequency_analysis.save2csv, "save_sd_fft", lambda *args: None)
    df = pd.DataFrame(
        {
            "No.1_time": np.arange(20) / 20,
            "No.1_0.2s_sd_std": [1.0] * 8 + [np.nan] * 4 + [2.0] * 8,
            "No.1_2.0s_sd_std": [np.nan] * 20,
        }
    )
    frequencies, amplitudes = frequency_analysis.fft_sd_list(df, "sample", flag_std=True)
    assert len(frequencies[0][0]) == 3  # longest contiguous 8 points, not 16 joined points
    assert len(amplitudes[0][1]) == 0
    assert make_graph._max_available_frequency(frequencies[0]) > 0
    assert make_graph._max_available_frequency([[], []]) == 1.0


def test_continuous_segment_does_not_join_regularly_spaced_missing_points():
    time = np.arange(10) / 20
    values = np.array([1.0, np.nan] * 5)
    (_, segment), metadata = frequency_analysis.longest_continuous_segment(time, values)
    assert len(segment) == 1
    assert metadata["sample_count"] == 1
