"""Select a 2D moving orbit using the SAME local held-out observations.

Candidate support windows and center-polynomial degrees are explicit inputs.
Prediction tests use identical source rows around the requested observation,
so distant data cannot conceal local temporal bias by lowering an average over
a larger time interval. Numerical candidates and conditionally assessed centers
are separate. Neither is a measurement of the motor's three-dimensional base.
"""

from typing import Any

import numpy as np

from utils.functions.orbit_center.center_covariance import linearized_center_covariance
from utils.functions.orbit_center.dependent_center_scale import irls_hac_coefficients
from utils.functions.orbit_center.moving_center import (
    OrbitModel,
    center_at,
    fit_orbit,
    observation_runs,
    orbit_prediction_residual,
    reanchor_fit,
)
from utils.functions.orbit_center.orbit_validation import validation_blocks


def estimate_local_center(
    frame,
    time,
    xy,
    *,
    max_gap_s,
    windows_s,
    valid,
    segment_ids,
    relative_error_limit,
    relative_residual_limit,
    gaussian_se_multiplier,
    center_degrees=(1, 2),
    query_mask=None,
    validation_gap_rows=0
):
    frame, time, xy = np.asarray(frame), np.asarray(time, float), np.asarray(xy, float)
    if not windows_s or any(not np.isfinite(w) or w <= 0 for w in windows_s):
        raise ValueError("explicit positive support windows required")
    if not center_degrees or any(d not in (1, 2) for d in center_degrees):
        raise ValueError("explicit linear/quadratic center candidates required")
    if type(validation_gap_rows) is not int or validation_gap_rows < 0:
        raise ValueError("nonnegative integer validation guard required")
    limits = [relative_error_limit, relative_residual_limit, gaussian_se_multiplier]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("explicit positive diagnostic limits required")
    n = len(time)
    query = np.ones(n, bool) if query_mask is None else np.asarray(query_mask, bool)
    if query.shape != (n,):
        raise ValueError("unaligned query mask")
    widths = sorted(set(windows_s))
    degrees = sorted(set(center_degrees))
    ids = observation_runs(frame, time, xy, max_gap_s=max_gap_s, valid=valid, segment_ids=segment_ids)
    assessed = np.full((n, 2), np.nan)
    computed = np.full((n, 2), np.nan)
    records: list[dict[str, Any]] = [
        dict(
            source_row=i,
            status="undefined",
            reason="unavailable_original_observation" if query[i] else "not_requested",
            candidates=[],
        )
        for i in range(n)
    ]
    starts = np.flatnonzero((ids >= 0) & np.r_[True, ids[1:] != ids[:-1]])
    ends = np.flatnonzero((ids >= 0) & np.r_[ids[:-1] != ids[1:], True]) + 1
    runs = {int(ids[a]): (int(a), int(b)) for a, b in zip(starts, ends)}
    full_cache: dict[Any, Any] = {}
    validation_cache: dict[Any, Any] = {}

    def support(a, b, i, width):
        ts = time[a:b]
        left = max(ts[0], min(time[i] - width / 2, ts[-1] - width))
        first = a + int(np.searchsorted(ts, left, "left"))
        last = max(i + 1, a + int(np.searchsorted(ts, left + width, "right")))
        return first, min(b, last)

    def full_fit(a, b, degree):
        key = a, b, degree
        if key not in full_cache:
            model = OrbitModel(degree, 2 * degree, True)
            fit = fit_orbit(time[a:b], xy[a:b], float(np.mean(time[a:b])), model)
            fit["selected_model"] = dict(center_degree=degree, radius_squared_degree=2 * degree, ellipse=True)
            covariance = None
            if fit["status"] == "computed":
                covariance = irls_hac_coefficients(
                    fit, time[a:b], xy[a:b], min(b - a - 1, int(np.ceil((b - a) ** (1 / 3))))
                )
            full_cache[key] = fit, covariance
        return full_cache[key]

    for i in np.flatnonzero(query & (ids >= 0)):
        run_a, run_b = runs[int(ids[i])]
        core_a, core_b = support(run_a, run_b, i, widths[0])
        local = np.arange(core_a, core_b)
        blocks = validation_blocks(len(local), max(2 + 2 * (d + 1) + (2 * d + 1) for d in degrees))
        candidates = []
        seen = set()
        for width in widths:
            a, b = support(run_a, run_b, i, width)
            for degree in degrees:
                if (a, b, degree) in seen:
                    continue
                seen.add((a, b, degree))
                full, hac = full_fit(a, b, degree)
                base = dict(
                    window_s=float(width),
                    center_degree=degree,
                    n_support=b - a,
                    support_first_source_frame=float(frame[a]),
                    support_last_source_frame=float(frame[b - 1]),
                )
                if full["status"] != "computed":
                    candidates.append(dict(base, status="undefined", reason=full["reason"]))
                    continue
                key = a, b, core_a, core_b, degree
                if key not in validation_cache:
                    trials = []
                    errors = []
                    failure = None
                    for block in blocks:
                        held = local[block]
                        train = np.ones(b - a, bool)
                        # This is a temporary validation exclusion, not a source
                        # gap, coordinate edit or calculated angular edge.
                        lo = max(0, int(held[0] - a) - validation_gap_rows)
                        hi = min(b - a, int(held[-1] - a) + validation_gap_rows + 1)
                        train[lo:hi] = False
                        if train.sum() <= full["parameter_count"]:
                            failure = "insufficient_training_degrees_of_freedom"
                            break
                        trial = fit_orbit(
                            time[a:b][train],
                            xy[a:b][train],
                            float(np.mean(time[a:b][train])),
                            OrbitModel(degree, 2 * degree, True),
                        )
                        if trial["status"] != "computed":
                            failure = trial["reason"]
                            break
                        residual = orbit_prediction_residual(trial, time[held], xy[held])
                        if not np.isfinite(residual).all():
                            failure = "nonfinite_local_prediction"
                            break
                        trials.append(trial)
                        errors.append(residual**2)
                    if failure is None:
                        sizes = np.array([len(e) for e in errors], float)
                        mse = np.array([e.mean() for e in errors])
                        mean = float(sum(e.sum() for e in errors) / sizes.sum())
                        # An empirical block standard error, not a calibrated CI.
                        se = float(np.std(mse, ddof=1) / np.sqrt(len(mse))) if len(mse) > 1 else np.inf
                        validation_cache[key] = dict(trials=trials, mse=mean, se=se, failure=None)
                    else:
                        validation_cache[key] = dict(failure=failure)
                cv = validation_cache[key]
                if cv["failure"] is not None:
                    candidates.append(dict(base, status="undefined", reason="local_validation:" + cv["failure"]))
                    continue
                at = reanchor_fit(full, time[i])
                point = at["center"]
                if at["status"] != "computed":
                    candidates.append(dict(base, status="undefined", reason=at["reason"]))
                    continue
                conditional = linearized_center_covariance(full, prediction_time=time[i])
                if hac is None or not np.isfinite(conditional).all():
                    candidates.append(dict(base, status="undefined", reason="center_covariance_unavailable"))
                    continue
                dependent = linearized_center_covariance(
                    dict(full, coefficient_covariance=hac), prediction_time=time[i]
                )
                if not np.isfinite(dependent).all():
                    candidates.append(dict(base, status="undefined", reason="dependent_covariance_unavailable"))
                    continue
                se = max(
                    np.sqrt(max(0.0, np.linalg.eigvalsh(conditional)[-1])),
                    np.sqrt(max(0.0, np.linalg.eigvalsh(dependent)[-1])),
                )
                spread = max(np.linalg.norm(center_at(t, time[i]) - point) for t in cv["trials"])
                major = at["radius_at_anchor"] / np.sqrt(np.linalg.eigvalsh(full["quadratic"])[0])
                relative = max(gaussian_se_multiplier * se, spread) / major
                candidates.append(
                    dict(
                        base,
                        status="computed",
                        reason=None,
                        center=point.copy(),
                        local_prediction_mse=cv["mse"],
                        local_prediction_se=cv["se"],
                        relative_error_scale=float(relative),
                        relative_residual=float(full["residual_rms"] / at["radius_at_anchor"]),
                        center_standard_error=float(se),
                        local_deletion_spread=float(spread),
                        projected_major_semiaxis=float(major),
                    )
                )
        finite = [c for c in candidates if c["status"] == "computed"]
        if not finite:
            records[i] = dict(
                source_row=int(i),
                source_frame=float(frame[i]),
                status="undefined",
                reason="no_locally_validated_orbit",
                continuous_run_id=int(ids[i]),
                candidates=candidates,
            )
            continue
        # A good coordinate predictor may have an unidentified center. Select
        # among identified candidates first instead of rejecting the whole row
        # after an uncertain predictor wins over an available reliable model.
        identified = [
            c
            for c in finite
            if c["relative_error_scale"] <= relative_error_limit and c["relative_residual"] <= relative_residual_limit
        ]
        pool = identified or finite
        best = min(pool, key=lambda c: c["local_prediction_mse"])
        bound = best["local_prediction_mse"] + best["local_prediction_se"]
        competitive = [c for c in pool if c["local_prediction_mse"] <= bound]
        chosen = min(competitive, key=lambda c: (c["relative_error_scale"], c["center_degree"], c["window_s"]))
        computed[i] = chosen["center"]
        reasons = []
        if chosen["relative_error_scale"] > relative_error_limit:
            reasons.append("center_error_scale")
        if chosen["relative_residual"] > relative_residual_limit:
            reasons.append("orbit_model_residual")
        if not reasons:
            assessed[i] = computed[i]
        records[i] = dict(
            source_row=int(i),
            source_frame=float(frame[i]),
            continuous_run_id=int(ids[i]),
            status="candidate_supported" if not reasons else "undefined",
            reason=";".join(reasons) or None,
            computed_center=computed[i].copy(),
            reported_center=assessed[i].copy(),
            selected_window_s=chosen["window_s"],
            selected_center_degree=chosen["center_degree"],
            support_first_source_frame=chosen["support_first_source_frame"],
            support_last_source_frame=chosen["support_last_source_frame"],
            local_validation_first_source_frame=float(frame[core_a]),
            local_validation_last_source_frame=float(frame[core_b - 1]),
            competitive_local_mse_limit=float(bound),
            relative_error_scale=chosen["relative_error_scale"],
            validation_gap_rows=validation_gap_rows,
            selection_pool_role="identified_center_candidates" if identified else "numerical_candidates_only",
            unrestricted_best_local_mse=float(min(c["local_prediction_mse"] for c in finite)),
            relative_residual=chosen["relative_residual"],
            candidates=candidates,
            measurement_role="2D image-plane orbit center under local conic/time model; not motor-base position",
        )
    return dict(center=assessed, computed_center=computed, records=records, run_id=ids)
