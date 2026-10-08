"""Reusable center candidates for diagnostics; production analysis does not use these.

Coordinates keep the caller's units. Missing rows are excluded within a fit only;
callers retain original frame/time arrays. Computational status is not a statement
that a physical rotation center has been identified.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ELLIPSE_METHODS = ("legacy", "ellipse_constrained", "geometric_ellipse")
METHODS = ELLIPSE_METHODS + ("centered_algebraic", "robust_circle")


@dataclass
class Ellipse:
    center: np.ndarray
    axes: np.ndarray
    basis: np.ndarray
    condition: float = np.nan


@dataclass
class CenterEstimate:
    method: str
    status: str
    center: Optional[np.ndarray]
    ellipse: Optional[Ellipse]
    converged: Optional[bool]
    boundary_reached: bool
    n_input: int
    n_used: int
    reason: str = ""

    @property
    def axes(self) -> Optional[np.ndarray]:
        return None if self.ellipse is None else self.ellipse.axes

    @property
    def angle_rad(self) -> Optional[float]:
        if self.ellipse is None:
            return None
        return float(np.arctan2(self.ellipse.basis[1, 0], self.ellipse.basis[0, 0]))

    @property
    def computationally_ok(self) -> bool:
        return self.status == "computed"


def robust_radius(points: np.ndarray) -> float:
    return float(0.5 * np.max(np.ptp(np.quantile(points, [0.05, 0.95], axis=0), axis=0)))


def ellipse_distance(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """Signed nearest-point distance with a dense seed and Newton refinement."""
    local = (points - ellipse.center) @ ellipse.basis
    a, b = ellipse.axes
    theta_grid = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    curve = np.column_stack((a * np.cos(theta_grid), b * np.sin(theta_grid)))
    theta = theta_grid[cKDTree(curve).query(local)[1]]
    u, v = local.T
    for _ in range(8):
        s, c = np.sin(theta), np.cos(theta)
        grad = (b * b - a * a) * s * c + a * u * s - b * v * c
        hessian = (b * b - a * a) * (c * c - s * s) + a * u * c + b * v * s
        step = np.divide(grad, hessian, out=np.zeros_like(grad), where=np.abs(hessian) > 1e-12)
        theta -= np.clip(step, -0.1, 0.1)
    closest = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    sign = np.where((u / a) ** 2 + (v / b) ** 2 >= 1, 1.0, -1.0)
    return np.linalg.norm(local - closest, axis=1) * sign


def estimate_center(
    points: np.ndarray,
    method: str,
    *,
    initial_center: str = "ellipse",
    bounds_scale: float = 1.0,
    max_nfev: int = 80,
) -> CenterEstimate:
    """Fit one window without selecting a scientifically valid candidate.

    ``initial_center`` and ``bounds_scale`` affect geometric ellipse fitting.
    Scale 1 uses median ± R for center and 0.1R–3R for axes; scale 2
    doubles the center allowance and upper axes, retaining the positive lower
    axes. These search settings are not physical acceptance thresholds.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown center method: {method}")
    if initial_center not in ("ellipse", "median") or bounds_scale <= 0 or max_nfev < 1:
        raise ValueError("Invalid initialization, bounds scale or iteration limit")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    n_input = len(points)
    points = points[np.isfinite(points).all(axis=1)]
    n_used = len(points)

    def output(status, center=None, ellipse=None, converged=None, boundary=False, reason=""):
        return CenterEstimate(method, status, center, ellipse, converged, boundary, n_input, n_used, reason)

    if n_used < 30:
        return output("insufficient_points", reason="Fewer than 30 finite points (diagnostic fitting minimum)")
    radius = robust_radius(points)
    if radius <= 1e-9 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        return output("degenerate", reason="Point cloud has no two-dimensional extent")
    middle = np.median(points, axis=0)
    scaled = (points - middle) / radius
    try:
        if method in ("legacy", "centered_algebraic"):
            x, y = (points if method == "legacy" else scaled).T
            design = np.column_stack((x * x, x * y, y * y, x, y))
            coef, _, rank, _ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)
            if rank < 5:
                return output("degenerate", reason="Conic design rank below five")
            a, b, c, d, e = coef
            quadratic = np.array([[a, b / 2], [b / 2, c]])
            eigenvalues, basis = np.linalg.eigh(quadratic)
            if min(abs(eigenvalues)) < 1e-12:
                return output("degenerate", reason="Singular conic quadratic form")
            center = -0.5 * np.linalg.solve(quadratic, np.array([d, e]))
            axes_squared = (1 + center @ quadratic @ center) / eigenvalues
            actual_center = center if method == "legacy" else middle + radius * center
            if np.any(axes_squared <= 0) or not np.isfinite(axes_squared).all():
                return output("nonellipse", center=actual_center, reason="Conic has no real ellipse axes")
            axes = np.sqrt(axes_squared) * (1 if method == "legacy" else radius)
            fit = Ellipse(actual_center, axes, basis)
            return output("computed", fit.center, fit)
        if method == "robust_circle":
            initial_radius = np.median(np.linalg.norm(points - middle, axis=1))
            result = least_squares(
                lambda z: np.linalg.norm(points - z[:2], axis=1) - z[2],
                np.r_[middle, initial_radius],
                bounds=(np.r_[middle - radius, 0.1 * radius], np.r_[middle + radius, 2 * radius]),
                loss="soft_l1",
                f_scale=0.2 * radius,
                max_nfev=150,
            )
            center, r = result.x[:2], result.x[2]
            boundary = bool(
                np.any(result.active_mask)
                or np.max(np.abs(center - middle)) >= 0.999 * radius
                or r >= 1.999 * radius
                or r <= 0.1001 * radius
            )
            fit = Ellipse(center, np.array([r, r]), np.eye(2))
        else:
            raw = cv2.fitEllipseDirect(scaled.astype(np.float32).reshape(-1, 1, 2))
            center, diameters, angle = raw
            axes = np.asarray(diameters) / 2 * radius
            theta = np.deg2rad(angle)
            basis = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            direct = Ellipse(middle + radius * np.asarray(center), axes, basis)
            if not np.isfinite(np.r_[direct.center, direct.axes]).all() or min(axes) <= 0:
                return output("failed", reason="Direct fit returned invalid geometry")
            if method == "ellipse_constrained":
                return output("computed", direct.center, direct)
            lower = np.r_[middle - bounds_scale * radius, np.log([0.1 * radius] * 2), -4 * np.pi]
            upper = np.r_[middle + bounds_scale * radius, np.log([3 * bounds_scale * radius] * 2), 4 * np.pi]

            def geometry(parameters):
                c, s = np.cos(parameters[4]), np.sin(parameters[4])
                return Ellipse(parameters[:2], np.exp(parameters[2:4]), np.array([[c, -s], [s, c]]))

            initial = np.r_[direct.center if initial_center == "ellipse" else middle, np.log(direct.axes), theta]
            initial = np.clip(initial, lower + 1e-6, upper - 1e-6)
            result = least_squares(
                lambda z: ellipse_distance(points, geometry(z)),
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.2 * radius,
                max_nfev=max_nfev,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            fit = geometry(result.x)
            boundary = bool(np.any(result.active_mask))
        status = "not_converged" if not result.success else ("boundary" if boundary else "computed")
        return output(status, fit.center, fit, bool(result.success), boundary, str(result.message))
    except (cv2.error, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return output("failed", reason=str(error))
