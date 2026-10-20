"""Behavioral requirements for the drift-and-stop reference candidate."""

import numpy as np

from tests.synthetic.audit import AuditCase, generate
from tests.synthetic.sign import StudyCase
from utils.functions.sign_reference_center_v2 import estimate_sign_reference_v2


def test_fast_drift_can_report_both_image_space_directions():
    for cartesian_clockwise in (False, True):
        data = generate(AuditCase("fast", StudyCase("fast", 0.35, cartesian_clockwise=cartesian_clockwise), 44), 10000)
        frame = estimate_sign_reference_v2(data.time, data.observed)
        accepted = frame.rotation_sign.dropna()
        assert len(accepted) > 150
        assert accepted.eq(-1 if cartesian_clockwise else 1).all()
        assert frame.loc[frame.rotation_sign.notna(), "sign_area"].eq(accepted).all()


def test_drifting_stop_has_no_reference_updates_and_recovers():
    data = generate(AuditCase("pause", StudyCase("pause", 0.1, perturbation="dwell_drift_resume")), 10000)
    frame = estimate_sign_reference_v2(data.time, data.observed)
    stopped = frame.loc[frame.time_sec.between(2.55, 3.45)]
    assert stopped.rotation_sign.isna().all()
    assert stopped.reference_x_um.isna().all()
    assert frame.loc[frame.time_sec.gt(4.0), "rotation_sign"].notna().any()


def test_gap_equals_processing_each_continuous_run_separately():
    data = generate(AuditCase("gap", StudyCase("gap", 0.1), 44, gap=True), 10000)
    frame = estimate_sign_reference_v2(data.time, data.observed)
    for start, end in ((0, 600), (600, 1200)):
        isolated = estimate_sign_reference_v2(data.time[start:end], data.observed[start:end])
        selected = frame.loc[frame.frame_0based.between(start, end - 1)]
        assert np.allclose(selected.time_sec, isolated.time_sec)
        assert np.allclose(selected.rotation_sign, isolated.rotation_sign, equal_nan=True)
        assert np.allclose(selected.reference_x_um, isolated.reference_x_um, equal_nan=True)


def test_partial_arc_retains_some_sign_information_and_line_abstains():
    data = generate(AuditCase("arc", StudyCase("arc", 0.1, orbit="arc_240")), 10000)
    frame = estimate_sign_reference_v2(data.time, data.observed)
    assert frame.rotation_sign.notna().sum() > 40
    data = generate(AuditCase("line", StudyCase("line", 0.1, perturbation="line")), 10000)
    frame = estimate_sign_reference_v2(data.time, data.observed)
    assert frame.rotation_sign.isna().all()
