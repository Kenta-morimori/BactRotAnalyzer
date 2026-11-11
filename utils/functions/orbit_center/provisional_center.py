"""Versioned research reference: local conic centers and explicit short fills.

Passing the model diagnostics is conditional on the local conic/time model.
It is not a calibrated5% statement about an experimental physical pivot.
The default20-frame evaluation cadence is explicit; full-row audit is optional.
"""

import copy
from typing import Any

import numpy as np

from utils.functions.orbit_center.bounded_center_interpolation import (
    interpolate_center_anchors,
)
from utils.functions.orbit_center.local_center_selection import estimate_local_center
from utils.functions.orbit_center.raw_edges import input_hash

METHOD = "local-conic-2d-provisional-v1"
_CONFIG: dict[str, Any] = dict(
    windows_s=[0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
    center_degrees=[1, 2],
    relative_error_limit=0.05,
    relative_residual_limit=0.3,
    gaussian_se_multiplier=2.45,
    validation_gap_rows=4,
    anchor_step_frames=20,
    default_evaluation_step_frames=20,
    model_switch_review_ratio=0.05,
)


def configuration():
    return dict(
        method=METHOD,
        parameters=copy.deepcopy(_CONFIG),
        target="2D image-plane model center; no3D motor-base correction",
        accuracy="5% target retained on declared known-truth controls; model diagnostics are not experimental accuracy certification",
        known_unverified=[
            "simultaneous correlated radius and center changes",
            "phase-locked/nonconic radial motion",
            "rolling-shutter/phase-contrast image formation",
        ],
        sampling="All source rows/times retained; center fit cadence explicit. Default20-frame anchors with short interpolation; evaluation_step=1 performs full-row fit audit.",
        interpolation="Only bracketed original-valid observations; requested failures and model-switch review points are barriers; no end holds or extrapolation; known failure rows remain unavailable.",
        primary_output="research reference only; V2/stop outputs and original coordinates unchanged",
    )


def estimate_provisional_center(
    frame,
    time,
    xy,
    *,
    valid,
    segment_ids,
    frame_period_s,
    timestamp_quantum_s=1e-6,
    evaluation_step=20,
    chunk_size=256,
    progress=None,
    parameters=None,
):
    """Bounded-memory query chunks always see the complete original source.

    Switch review flags describe model dependence, not a physical movement
    limit. Neither flags nor interpolation change directly fitted centers.
    """
    config = copy.deepcopy(_CONFIG)
    if parameters is not None:
        unknown = set(parameters) - set(config)
        if unknown:
            raise ValueError(f"Unknown center parameters: {sorted(unknown)}")
        config.update(parameters)
    for key in ("anchor_step_frames", "default_evaluation_step_frames"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"positive integer {key} required")
    if not np.isfinite(config["model_switch_review_ratio"]) or config["model_switch_review_ratio"] <= 0:
        raise ValueError("positive model switch review ratio required")
    f, t, p = np.asarray(frame), np.asarray(time, float), np.asarray(xy, float)
    valid = np.asarray(valid, bool)
    segments = np.asarray(segment_ids)
    n = len(t)
    if f.shape != (n,) or p.shape != (n, 2) or valid.shape != (n,) or segments.shape != (n,):
        raise ValueError("aligned original observations/validity/identities required")
    if not np.isfinite([frame_period_s, timestamp_quantum_s]).all() or min(frame_period_s, timestamp_quantum_s) <= 0:
        raise ValueError("explicit positive acquisition period and saved-time quantum required")
    if type(evaluation_step) is not int or evaluation_step < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("positive integer evaluation step and chunk size required")
    before = input_hash(f, t, p)
    gap = frame_period_s + timestamp_quantum_s
    span = config["anchor_step_frames"] * frame_period_s + 2 * timestamp_quantum_s
    layout = estimate_local_center(
        f,
        t,
        p,
        max_gap_s=gap,
        valid=valid,
        segment_ids=segments,
        query_mask=np.zeros(n, bool),
        **{
            k: v
            for k, v in config.items()
            if k
            in [
                "windows_s",
                "center_degrees",
                "relative_error_limit",
                "relative_residual_limit",
                "gaussian_se_multiplier",
                "validation_gap_rows",
            ]
        },
    )
    ids = layout["run_id"]
    eligible = ids >= 0
    query = np.zeros(n, bool)
    anchors = query.copy()
    query[::evaluation_step] = True
    anchors[:: config["anchor_step_frames"]] = True
    for rid in np.unique(ids[eligible]):
        rows = np.flatnonzero(ids == rid)
        query[rows[[0, -1]]] = True
        anchors[rows[[0, -1]]] = True
    # Reporting anchors are always directly evaluated; no invented seed values.
    query |= anchors
    query &= eligible
    anchors &= eligible
    computed = np.full((n, 2), np.nan)
    conditional = computed.copy()
    records = [
        dict(
            source_row=i,
            status="unavailable_original_observation" if not eligible[i] else "not_evaluated",
            reason="source_barrier" if not eligible[i] else "not_requested",
        )
        for i in range(n)
    ]
    requested = np.flatnonzero(query)
    previous = None
    for offset in range(0, len(requested), chunk_size):
        rows = requested[offset : offset + chunk_size]
        mask = np.zeros(n, bool)
        mask[rows] = True
        result = estimate_local_center(
            f,
            t,
            p,
            max_gap_s=gap,
            valid=valid,
            segment_ids=segments,
            query_mask=mask,
            **{
                k: v
                for k, v in config.items()
                if k
                in [
                    "windows_s",
                    "center_degrees",
                    "relative_error_limit",
                    "relative_residual_limit",
                    "gaussian_se_multiplier",
                    "validation_gap_rows",
                ]
            },
        )
        np.testing.assert_array_equal(result["run_id"], ids)
        computed[rows] = result["computed_center"][rows]
        conditional[rows] = result["center"][rows]
        for i in rows:
            r = result["records"][i]
            record = {k: v for k, v in r.items() if k != "candidates"}
            record.update(
                model_selection_changed=False,
                model_switch_distance_native_px=None,
                model_switch_distance_over_fitted_radius=None,
                review_model_switch=False,
            )
            if previous is not None and ids[previous] == ids[i] and r.get("selected_window_s") is not None:
                prior = records[previous]
                family = (r["selected_window_s"], r["selected_center_degree"])
                oldfamily = (prior.get("selected_window_s"), prior.get("selected_center_degree"))
                record["previous_evaluated_source_frame"] = float(f[previous])
                record["evaluation_spacing_s"] = float(t[i] - t[previous])
                if family != oldfamily and oldfamily[0] is not None:
                    record["model_selection_changed"] = True
                    old = next(
                        (
                            c
                            for c in r["candidates"]
                            if c["status"] == "computed" and (c["window_s"], c["center_degree"]) == oldfamily
                        ),
                        None,
                    )
                    chosen = next(
                        c
                        for c in r["candidates"]
                        if c["status"] == "computed" and (c["window_s"], c["center_degree"]) == family
                    )
                    if old is not None:
                        distance = float(np.linalg.norm(np.asarray(old["center"]) - np.asarray(chosen["center"])))
                        ratio = distance / min(old["projected_major_semiaxis"], chosen["projected_major_semiaxis"])
                        record.update(
                            model_switch_distance_native_px=distance,
                            model_switch_distance_over_fitted_radius=float(ratio),
                            review_model_switch=bool(ratio > config["model_switch_review_ratio"]),
                        )
                    else:
                        record.update(
                            review_model_switch=True,
                            model_switch_review_reason="previous_family_not_identified_at_current_time",
                        )
            if record["status"] == "candidate_supported":
                record["status"] = (
                    "review_required_model_switch" if record["review_model_switch"] else "model_diagnostics_pass"
                )
                if record["review_model_switch"]:
                    record["reason"] = "same_time_window_family_center_disagreement"
            record["measurement_role"] = "conditional2D model reference; no calibrated experimental accuracy"
            records[i] = record
            previous = int(i)
        if progress is not None:
            progress(min(offset + len(rows), len(requested)), len(requested))
    review = np.array([bool(r.get("review_model_switch")) for r in records])
    seeds = conditional.copy()
    seeds[~anchors | review] = np.nan
    audit_ids = ids.copy()
    audit_ids[query & (~np.isfinite(conditional).all(axis=1) | review)] = -1
    completion = interpolate_center_anchors(
        f, t, seeds, run_id=audit_ids, requested=anchors, max_gap_s=gap, max_span_s=span
    )
    assert before == input_hash(f, t, p)
    return dict(
        method=METHOD,
        configuration=dict(configuration(), parameters=copy.deepcopy(config)),
        run_id=ids,
        eligible=eligible,
        evaluated=query,
        reporting_anchor=anchors,
        computed_center=computed,
        conditional_center=conditional,
        completed_center=completion["center"],
        interpolated=completion["interpolated"],
        completion_reason=completion["reason"],
        left_support_row=completion["left_support_row"],
        right_support_row=completion["right_support_row"],
        records=records,
        review_model_switch=review,
        max_gap_s=gap,
        max_interpolation_span_s=span,
        evaluation_step=evaluation_step,
        unqueried_center_fit_validity="not assessed" if evaluation_step > 1 else "all eligible rows evaluated",
    )
