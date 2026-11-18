"""Synthetic checks for rotation-only poster comparisons."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd

from utils.functions.poster_rotation_only import (
    WINDOWS_S,
    band_metrics,
    detect_stops,
    evaluate_interval,
    longest_rotation_interval,
    summarize_groups,
)


def _frame(duration_s=6.0, rate_hz=200.0):
    t = np.arange(round(duration_s * rate_hz)) / rate_hz
    return pd.DataFrame(
        {
            "source_frame_1based": np.arange(1, len(t) + 1),
            "time_sec": t,
            "x_um": 0.1 * np.cos(2 * np.pi * 8 * t),
            "y_um": 0.1 * np.sin(2 * np.pi * 8 * t),
            "long_axis_omega_rad_s": 50
            + 2 * np.sin(2 * np.pi * 0.7 * t)
            + np.random.default_rng(4).normal(0, 1, len(t)),
        }
    )


def test_longest_interval_keeps_source_frames_and_stop_margin():
    frame = _frame()
    frame.loc[10, "long_axis_omega_rad_s"] = np.nan
    stops = pd.DataFrame([{"start_s": 1.0, "end_s": 1.5}])
    selected = longest_rotation_interval(frame, stops)
    assert selected.source_frame_1based.iloc[0] > 1.6 * 200
    assert selected.source_frame_1based.iloc[-1] == len(frame)
    assert np.all(np.diff(selected.source_frame_1based) == 1)
    assert np.allclose(selected.time_sec, (selected.source_frame_1based - 1) / 200)


def test_missing_frame_splits_instead_of_compacting():
    frame = _frame()
    frame.loc[600, ["x_um", "y_um"]] = np.nan
    selected = longest_rotation_interval(frame, pd.DataFrame(columns=["start_s", "end_s"]))
    assert selected.source_frame_1based.iloc[0] == 1
    assert selected.source_frame_1based.iloc[-1] == 600


def test_low_band_bin_gate_rejects_short_rotation():
    interval = _frame(duration_s=3.4)
    _, _, bands, reason = evaluate_interval(interval)
    assert reason == "fewer_than_3_low_bins"
    assert set(bands.window_s) == set(WINDOWS_S)
    assert bands.loc[bands.window_s.eq(1.0), "low_bin_count"].iloc[0] < 3


def test_correlated_angle_and_centroid_pause_is_detected():
    rate = 250
    time = np.arange(6 * rate) / rate
    progress = np.where(time < 5.45, time, np.where(time < 5.8, 5.45, time - 0.35))
    phase = 2 * np.pi * 8 * progress
    frame = pd.DataFrame(
        {"time_sec": time, "x_um": 0.2 * np.cos(phase), "y_um": 0.2 * np.sin(phase), "long_axis_unwrapped_rad": phase}
    )
    _, stops = detect_stops(frame)
    assert ((stops.start_s < 5.55) & (stops.end_s > 5.75)).any()


def test_band_definitions_and_group_equal_cell_weight():
    spectrum = pd.DataFrame(
        {
            "window_s": [0.1] * 5 + [0.5] * 5 + [1.0] * 5,
            "frequency_hz": [0.25, 0.5, 0.75, 80, 90] * 3,
            "amplitude": [2, 2, 2, 1, 1] * 3,
        }
    )
    bands = band_metrics(spectrum, (80, 100))
    assert (bands.low_bin_count == 3).all()
    assert (bands.high_bin_count == 2).all()
    assert (bands.low_to_high_amplitude_ratio == 2).all()
    cells = pd.DataFrame(
        [
            {
                "day": "a",
                "sample_no": cell,
                "window_s": width,
                "selection_reason": "included",
                "low_to_high_amplitude_ratio": ratio,
                "low_mean_amplitude": ratio,
                "high_mean_amplitude": 1.0,
            }
            for cell, ratio in [(1, 1.0), (2, 9.0)]
            for width in WINDOWS_S
        ]
    )
    summary = summarize_groups(cells)
    assert (summary.cell_count == 2).all()
    assert np.allclose(summary.ratio_geometric_mean, 3.0)
    assert np.allclose(summary.normalized_ratio_mean, 1.0)


def test_sjw46_reference_uses_only_seven_23c_cells_and_normalizes_each(tmp_path):
    reference_rows = []
    for number in range(1, 8):
        relative = f"23C/No.{number:02d}.csv"
        target = tmp_path / relative
        target.parent.mkdir(exist_ok=True)
        pd.DataFrame(
            {"window_s": WINDOWS_S, "low_to_high_amplitude_ratio": [2 * number, 4 * number, 8 * number]}
        ).to_csv(target, index=False)
        reference_rows.append({"temperature_c": 23, "sample_no": number, "ratio_file": relative})
    reference_rows.append({"temperature_c": 10, "sample_no": 1, "ratio_file": "missing.csv"})
    pd.DataFrame(reference_rows).to_csv(tmp_path / "cell_summary.csv", index=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "10_poster_rotation_only_stats.py"
    load_reference = run_path(str(script))["load_sjw46_reference"]
    result = load_reference(tmp_path)
    assert len(result) == 21
    assert set(result.sample_no) == set(range(1, 8))
    assert np.allclose(result.groupby("sample_no").normalized_ratio.max(), 1)
    assert np.allclose(result.groupby("window_s").normalized_ratio.mean(), [0.25, 0.5, 1])
