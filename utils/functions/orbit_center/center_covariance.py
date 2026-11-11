"""Recover the full conditional 2D center covariance from an orbit fit.

This preserves orientation information lost by the trace-only SE. It does not
calibrate robust/nonlinear fitting uncertainty or establish physical truth.
"""

import numpy as np


def linearized_center_covariance(fit, *, prediction_time=None, points=None):
    if fit["status"] != "computed":
        return np.full((2, 2), np.nan)
    if fit.get("geometric_refinement"):
        if points is None:
            raise ValueError("original support points required for geometric-circle covariance")
        lever = fit["center"] - np.asarray(points, float)
        lengths = np.linalg.norm(lever, axis=1)
        if np.any(lengths == 0) or len(lengths) <= 3:
            return np.full((2, 2), np.nan)
        jacobian = np.column_stack([lever / lengths[:, None], -np.ones(len(lengths))])
        if np.linalg.matrix_rank(jacobian) < 3:
            return np.full((2, 2), np.nan)
        inverse = np.linalg.pinv(jacobian, rcond=np.finfo(float).eps * max(jacobian.shape))
        variance = np.sum((lengths - fit["radius_at_anchor"]) ** 2) / (len(lengths) - 3)
        return (variance * inverse @ inverse.T)[:2, :2]
    when = fit["anchor_time"] if prediction_time is None else prediction_time
    tau = (when - fit["anchor_time"]) / fit["time_scale"]
    coefficients = fit["center_coefficients"]
    nuisance = fit["radius_squared_coefficients"]
    covariance = fit["coefficient_covariance"]
    shape_n = len(covariance) - 2 * len(coefficients) - len(nuisance)
    c = np.array([tau**j for j in range(len(coefficients))]) @ coefficients / fit["coordinate_scale"]
    inv_q = np.linalg.inv(fit["quadratic"])
    jacobian = np.zeros((2, len(covariance)))
    for j in range(len(coefficients)):
        jacobian[:, shape_n + 2 * j : shape_n + 2 * j + 2] = tau**j * inv_q
    if shape_n:
        jacobian[:, 0] = inv_q @ np.diag([1.0, -1.0]) @ c
        jacobian[:, 1] = inv_q @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ c
    return fit["coordinate_scale"] ** 2 * jacobian @ covariance @ jacobian.T
