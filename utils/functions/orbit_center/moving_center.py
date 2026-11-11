"""Local conic regression and evaluation of its time-varying center. No coordinate smoothing."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OrbitModel:
    """Declared candidate model, independent of synthetic case truth."""

    center_degree: int = 1
    radius_squared_degree: int = 2
    ellipse: bool = True
    reweight_iterations: int = 4

    def __post_init__(self):
        if self.center_degree not in (0, 1, 2):
            raise ValueError("center degree must be zero, one or two")
        if self.radius_squared_degree < 2 * self.center_degree:
            raise ValueError("radius nuisance polynomial must include squared center terms")
        if self.radius_squared_degree > 4 or self.reweight_iterations < 0:
            raise ValueError("unsupported polynomial degree or iteration count")


def fit_orbit(time, points, anchor_time, model=OrbitModel()):
    """Fit (p-c(t))' Q (p-c(t)) = r_squared(t), trace(Q)=2.

    For a linear center, expanding this identity gives a linear regression in
    x,y,tx,ty and 1,t,t²; an ellipse adds x²-y² and 2xy. Time and coordinates
    are normalized locally for numerical stability. Gradient weighting reduces
    algebraic residual scale effects; robust reweighting uses residual MAD.
    It does not smooth the observations. Reported standard errors are local
    linearized diagnostics, NOT calibrated confidence intervals.
    """
    time, points = np.asarray(time, float), np.asarray(points, float)
    result: dict[str, Any] = dict(status="undefined", reason=None, center=np.full(2, np.nan))
    if time.ndim != 1 or points.shape != (len(time), 2):
        raise ValueError("unaligned time/points")
    k = (2 if model.ellipse else 0) + 2 * (model.center_degree + 1) + model.radius_squared_degree + 1
    result.update(n=len(time), parameter_count=k)
    if len(time) <= k:
        return dict(result, reason="insufficient_points")
    if not np.isfinite(points).all() or not np.isfinite(time).all() or not np.isfinite(anchor_time):
        return dict(result, reason="nonfinite_input")
    if np.any(np.diff(time) <= 0) or not time[0] <= anchor_time <= time[-1]:
        return dict(result, reason="invalid_time_or_extrapolation")
    origin = np.median(points, axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points - origin) ** 2, axis=1))))
    half_span = float(max(abs(time - anchor_time)))
    if scale == 0 or half_span == 0:
        return dict(result, reason="no_spatial_or_temporal_extent")
    p = (points - origin) / scale
    tau = (time - anchor_time) / half_span
    powers = np.column_stack([tau**j for j in range(model.radius_squared_degree + 1)])
    x, y = p.T
    shape_columns = [x * x - y * y, 2 * x * y] if model.ellipse else []
    columns = shape_columns + [2 * p[:, axis] * powers[:, j] for j in range(model.center_degree + 1) for axis in (0, 1)]
    design = np.column_stack(columns + [powers[:, j] for j in range(model.radius_squared_degree + 1)])
    target = np.sum(p * p, axis=1)
    weights = np.ones(len(time))
    shape_n = len(shape_columns)
    numerical_tolerance = np.finfo(float).eps * max(design.shape)
    for iteration in range(model.reweight_iterations + 1):
        weighted = design * np.sqrt(weights[:, None])
        coef, _, rank, singular = np.linalg.lstsq(weighted, target * np.sqrt(weights), rcond=None)
        if rank < k:
            return dict(result, reason="unidentifiable_design", rank=int(rank))
        q = np.eye(2)
        if model.ellipse:
            q -= np.array([[coef[0], coef[1]], [coef[1], -coef[0]]])
        eigenvalues = np.linalg.eigvalsh(q)
        if eigenvalues[0] <= numerical_tolerance * eigenvalues[-1]:
            return dict(result, reason="nonellipse_quadratic", quadratic_eigenvalues=eigenvalues.tolist())
        center_coef = np.linalg.solve(q, coef[shape_n : shape_n + 2 * (model.center_degree + 1)].reshape(-1, 2).T).T
        c = powers[:, : model.center_degree + 1] @ center_coef
        gradient_norm = 2 * np.linalg.norm((p - c) @ q, axis=1)
        # Sampson distance approximates orthogonal distance to each local orbit.
        residual = (target - design @ coef) / np.maximum(gradient_norm, numerical_tolerance)
        mad = float(1.4826 * np.median(abs(residual - np.median(residual))))
        if iteration == model.reweight_iterations:
            break
        # Huber's conventional 1.345 is an experimental robust-fit parameter,
        # not a physical acceptance threshold or an outlier deletion rule.
        robust_weight = (
            np.ones(len(time))
            if mad <= numerical_tolerance
            else np.minimum(1, 1.345 * mad / np.maximum(abs(residual), numerical_tolerance))
        )
        weights = robust_weight / np.maximum(gradient_norm, numerical_tolerance) ** 2
        weights /= np.max(weights)
    nuisance = coef[shape_n + 2 * (model.center_degree + 1) :]
    radius_sq = np.einsum("ij,ij->i", c @ q, c) + powers @ nuisance
    anchor_radius_sq = center_coef[0] @ q @ center_coef[0] + nuisance[0]
    if anchor_radius_sq <= 0 or not np.isfinite(radius_sq).all():
        return dict(result, reason="nonpositive_anchor_radius")
    if np.any(radius_sq < -numerical_tolerance):
        return dict(result, reason="negative_radius_squared_on_support")
    center = origin + scale * center_coef[0]
    # Conditional linearized algebraic covariance; includes drift/radius nuisance.
    dof = len(time) - k
    sigma_sq = float(np.sum(weights * (target - design @ coef) ** 2) / dof)
    # Forming A.T@A squares the condition number and can discard weakly
    # identified directions that the fit itself retained. Invert A directly.
    inverse_design = np.linalg.pinv(weighted, rcond=numerical_tolerance)
    covariance = inverse_design @ inverse_design.T * sigma_sq
    jac = np.zeros((2, k))
    inv_q = np.linalg.inv(q)
    jac[:, shape_n : shape_n + 2] = inv_q
    if model.ellipse:
        jac[:, 0] = inv_q @ np.diag([1.0, -1.0]) @ center_coef[0]
        jac[:, 1] = inv_q @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ center_coef[0]
    center_cov = scale**2 * jac @ covariance @ jac.T
    return dict(
        result,
        status="computed",
        reason=None,
        center=center,
        center_coefficients=center_coef * scale,
        time_scale=half_span,
        origin=origin,
        coordinate_scale=scale,
        quadratic=q,
        radius_squared_coefficients=nuisance,
        normalized_coefficients=coef,
        anchor_time=float(anchor_time),
        coefficient_covariance=covariance,
        radius_at_anchor=float(scale * np.sqrt(anchor_radius_sq)),
        residual_rms=float(scale * np.sqrt(np.mean(residual**2))),
        residual_mad=scale * mad,
        center_linearized_se=float(np.sqrt(max(0.0, np.trace(center_cov)))),
        condition=float(singular[0] / singular[-1]),
        rank=int(rank),
        support_time_first=float(time[0]),
        support_time_last=float(time[-1]),
        negative_radius_rows=int(np.count_nonzero(radius_sq <= 0)),
    )


def orbit_prediction_residual(fit, time, points):
    """Held-out geometric/Sampson residual in the original coordinate units."""
    tau = (time - fit["anchor_time"]) / fit["time_scale"]
    powers = np.column_stack([tau**j for j in range(len(fit["radius_squared_coefficients"]))])
    c = powers[:, : len(fit["center_coefficients"])] @ fit["center_coefficients"] / fit["coordinate_scale"]
    p = (points - fit["origin"]) / fit["coordinate_scale"]
    q = fit["quadratic"]
    rsq = np.einsum("ij,ij->i", c @ q, c) + powers @ fit["radius_squared_coefficients"]
    if np.any(rsq <= 0):
        return np.full(len(time), np.inf)
    r = p - c
    gradient = 2 * np.linalg.norm(r @ q, axis=1)
    return fit["coordinate_scale"] * (np.einsum("ij,ij->i", r @ q, r) - rsq) / np.maximum(gradient, np.finfo(float).eps)


def center_at(fit, time):
    """Evaluate the fitted center polynomial, also for held-out diagnostics."""
    tau = (time - fit["anchor_time"]) / fit["time_scale"]
    powers = np.array([tau**j for j in range(len(fit["center_coefficients"]))])
    return fit["origin"] + powers @ fit["center_coefficients"]


def reanchor_fit(fit, time):
    """Evaluate one fixed-support fit and its covariance at another source time."""
    if fit["status"] != "computed":
        return dict(fit)
    result = dict(fit, center=center_at(fit, time), prediction_time_sec=float(time))
    tau = (time - fit["anchor_time"]) / fit["time_scale"]
    coeff = fit["center_coefficients"]
    nuisance = fit["radius_squared_coefficients"]
    p = np.array([tau**j for j in range(len(nuisance))])
    c = p[: len(coeff)] @ coeff / fit["coordinate_scale"]
    q = fit["quadratic"]
    rsq = c @ q @ c + p @ nuisance
    if rsq <= 0:
        return dict(result, status="undefined", reason="nonpositive_anchor_radius", center=np.full(2, np.nan))
    result["radius_at_anchor"] = float(fit["coordinate_scale"] * np.sqrt(rsq))
    cov = fit["coefficient_covariance"]
    shape_n = len(cov) - 2 * len(coeff) - len(nuisance)
    jac = np.zeros((2, len(cov)))
    inv_q = np.linalg.inv(q)
    for j in range(len(coeff)):
        jac[:, shape_n + 2 * j : shape_n + 2 * j + 2] = tau**j * inv_q
    if shape_n:
        jac[:, 0] = inv_q @ np.diag([1.0, -1.0]) @ c
        jac[:, 1] = inv_q @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ c
    variance = fit["coordinate_scale"] ** 2 * jac @ cov @ jac.T
    result["center_linearized_se"] = float(np.sqrt(max(0.0, np.trace(variance))))
    return result


def observation_runs(frame, time, points, *, valid, segment_ids, max_gap_s):
    """Original adjacent rows only, split at missing observations or identities."""
    frame, time, points = np.asarray(frame, float), np.asarray(time, float), np.asarray(points, float)
    n = len(time)
    if frame.shape != (n,) or points.shape != (n, 2):
        raise ValueError("unaligned source arrays")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("explicit positive acquisition gap required")
    segments = np.asarray(segment_ids, object)
    good = np.array(valid, bool, copy=True)
    if segments.shape != (n,) or good.shape != (n,):
        raise ValueError("aligned source validity and segment identities required")
    good &= np.isfinite(time) & np.isfinite(frame) & np.isfinite(points).all(axis=1)
    good &= np.array([s is not None and str(s).lower() not in ("", "nan", "unknown", "none") for s in segments])
    connected = good[:-1] & good[1:] & (np.diff(frame) == 1) & (np.diff(time) > 0) & (np.diff(time) <= max_gap_s)
    connected &= segments[:-1] == segments[1:]
    result = np.full(n, -1, int)
    rid = -1
    for i in range(n):
        if good[i]:
            if i == 0 or not connected[i - 1]:
                rid += 1
            result[i] = rid
    return result
