"""Recover the full conditional 2D center covariance from an orbit fit.

This preserves orientation information lost by the trace-only SE. It does not
calibrate robust/nonlinear fitting uncertainty or establish physical truth.
"""

import numpy as np


def linearized_center_covariance(fit, *, prediction_time=None):
    if fit["status"] != "computed":
        return np.full((2, 2), np.nan)
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
