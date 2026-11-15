import importlib.util
from pathlib import Path

import numpy as np


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts/11_poster_repellent_screening.py"
    spec = importlib.util.spec_from_file_location("poster_repellent_screening", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screening_keeps_missing_center_gap_and_counts_unproven_speed():
    script = _script()
    time = np.arange(10, dtype=float)
    centers = np.array([0, 0, 0, 0, np.nan, 5, 5, 5, 5, 5], dtype=float)
    omega = np.array([10, 10, 10, 10, 10, 0, 0, 10, 10, np.nan], dtype=float)
    background = np.array([10] * 5 + [20] * 5, dtype=float)
    metrics, step_time, step_ratio, active = script.summarize_cell(
        time,
        np.ones(10),
        np.zeros(10),
        centers,
        np.zeros(10),
        time,
        omega,
        np.append(time[1:], np.nan),
        "point",
        time,
        background,
        5.0,
    )
    assert metrics["background_median_post_over_pre"] == 2.0
    assert metrics["center_step_over_radius_max"] == 0.0
    assert 5.0 not in step_time  # Do not measure a center jump across the missing observation.
    assert np.all(step_ratio == 0)
    assert metrics["velocity_valid_fraction"] == 0.9
    assert metrics["speed_above_0.2x_pre_median_fraction_of_record"] == 0.7
    assert metrics["speed_above_0.2x_pre_median_time_fraction_of_record"] == 6 / 9
    assert np.isnan(active[-1])
    assert np.array_equal(script.velocity_time_from_export(time, time, omega[:-1])[0], time[:-1])
    assert script.velocity_time_from_export(time, time, omega[:-1])[1] == "interval_start"
    assert script.velocity_time_from_export(time, time, omega)[1] == "point"


def test_time_fraction_does_not_credit_a_timestamp_gap():
    script = _script()
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
    omega = np.full(4, 10.0)
    metrics, _, _, _ = script.summarize_cell(
        time,
        np.ones(5),
        np.zeros(5),
        np.zeros(5),
        np.zeros(5),
        time[:-1],
        omega,
        time[1:],
        "interval_start",
        time,
        np.ones(5),
        2.0,
    )
    assert metrics["speed_above_0.2x_pre_median_time_fraction_of_record"] == 3 / 11


def test_inventory_separates_alternate_exports_from_independent_cells(tmp_path):
    script = _script()
    outputs, raw = tmp_path / "outputs", tmp_path / "data"
    for name in ("20260312", "20260312_mode=1", "raw_only"):
        (outputs / name / "repellent_response").mkdir(parents=True)
    (raw / "20260312").mkdir(parents=True)
    (raw / "raw_only").mkdir()
    for name in script.REQUIRED:
        (outputs / "20260312" / "repellent_response" / name).parent.mkdir(parents=True, exist_ok=True)
        (outputs / "20260312" / "repellent_response" / name).touch()
    inventory = script.source_inventory(outputs, raw).set_index("dataset")
    assert inventory.loc["20260312", "status"] == "screened"
    assert inventory.loc["20260312_mode=1", "status"] == "alternate_export_not_independent"
    assert inventory.loc["raw_only", "status"] == "incomplete_export"
