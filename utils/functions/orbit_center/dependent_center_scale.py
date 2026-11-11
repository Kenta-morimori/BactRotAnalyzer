"""IRLS influence covariance with Bartlett observation-lag dependence; conditional model diagnostic."""

import numpy as np

from utils.functions.orbit_center.moving_center import OrbitModel, fit_orbit


def bartlett_score_covariance(scores, lags):
    """PSD Gram form of lag0 + sum(1-h/(L+1))*(lag_h+lag_h.T)."""
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("finite source-ordered score vectors required")
    if type(lags) is not int or not 0 <= lags < len(scores):
        raise ValueError("integer bandwidth below observation count required")
    padded = np.pad(scores, ((lags, lags), (0, 0)))
    cumulative = np.vstack([np.zeros(scores.shape[1]), np.cumsum(padded, axis=0)])
    rolling = cumulative[lags + 1 :] - cumulative[: -lags - 1]
    return rolling.T @ rolling / (lags + 1)


def irls_hac_coefficients(fit, time, points, lags):
    """Recover the EXACT last weighted regression without editing the fitter.

    Iteration4 weights come from iteration3, so a three-iteration fit recovers
    those frozen weights. The recovered final coefficients must reproduce the
    original ones; numerical disagreement cannot support a diagnostic.
    """
    descriptor = fit["selected_model"]
    model = OrbitModel(**descriptor)
    if not model.ellipse or model.center_degree not in (1, 2):
        raise ValueError("this diagnostic requires a linear or quadratic moving ellipse")
    previous = fit_orbit(time, points, fit["anchor_time"], OrbitModel(**descriptor, reweight_iterations=3))
    if previous["status"] != "computed":
        return None
    p = (points - fit["origin"]) / fit["coordinate_scale"]
    tau = (time - fit["anchor_time"]) / fit["time_scale"]
    powers = np.column_stack([tau**j for j in range(model.radius_squared_degree + 1)])
    x, y = p.T
    design = np.column_stack(
        [x * x - y * y, 2 * x * y]
        + [2 * p[:, axis] * powers[:, j] for j in range(model.center_degree + 1) for axis in (0, 1)]
        + [powers[:, j] for j in range(model.radius_squared_degree + 1)]
    )
    target = np.sum(p * p, axis=1)
    c = powers[:, : model.center_degree + 1] @ previous["center_coefficients"] / fit["coordinate_scale"]
    tol = np.finfo(float).eps * max(design.shape)
    gradient = 2 * np.linalg.norm((p - c) @ previous["quadratic"], axis=1)
    residual = (target - design @ previous["normalized_coefficients"]) / np.maximum(gradient, tol)
    mad = previous["residual_mad"] / fit["coordinate_scale"]
    robust = np.ones(len(time)) if mad <= tol else np.minimum(1, 1.345 * mad / np.maximum(abs(residual), tol))
    weights = robust / np.maximum(gradient, tol) ** 2
    weights /= weights.max()
    weighted = design * np.sqrt(weights[:, None])
    recovered = np.linalg.lstsq(weighted, target * np.sqrt(weights), rcond=None)[0]
    if not np.allclose(recovered, fit["normalized_coefficients"], rtol=1e-7, atol=1e-10):
        return None
    influence = (np.linalg.pinv(weighted, rcond=tol) * np.sqrt(weights)).T
    influence *= (target - design @ fit["normalized_coefficients"])[:, None]
    covariance = bartlett_score_covariance(influence, lags) * len(time) / (len(time) - len(recovered))
    return covariance
