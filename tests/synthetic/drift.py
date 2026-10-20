"""Elliptic trajectories with controlled drift, arcs, and missing coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FPS = 200
R = 0.30


@dataclass(frozen=True)
class Case:
    name: str
    drift_um_per_5s: float = 0.0
    drift_shape: str = "linear"
    direction: str = "x"
    orbit: str = "full"
    noise_sd: float = 0.02
    duration_s: float = 6.0
    perturbation: str = "none"


@dataclass
class Generated:
    time: np.ndarray
    observed: np.ndarray
    ideal: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    rotating: np.ndarray


def generate(case: Case, seed: int) -> Generated:
    rng = np.random.default_rng(seed)
    n = int(round(case.duration_s * FPS))
    time = np.arange(n) / FPS
    speed = case.drift_um_per_5s / 5
    if case.drift_shape == "linear":
        travel = speed * time
    elif case.drift_shape == "slope_change":
        middle = case.duration_s / 2
        travel = speed * np.minimum(time, middle) + 2 * speed * np.maximum(time - middle, 0)
    elif case.drift_shape == "step":
        travel = speed * time + case.drift_um_per_5s * (time >= case.duration_s / 2)
    else:
        raise ValueError(case.drift_shape)
    vector = np.array([1.0, 0.0]) if case.direction == "x" else np.ones(2) / np.sqrt(2)
    center = np.array([1.3, 1.45]) + travel[:, None] * vector
    if case.orbit == "full":
        phase = 2 * np.pi * time / 0.3
    else:
        span = 360 if case.orbit == "reversal" else int(case.orbit.split("_")[1])
        # One leg takes 0.3 s; span and reversal frequency are independent.
        phase = np.deg2rad(span) / 2 * (2 / np.pi) * np.arcsin(np.sin(np.pi * time / 0.3))
    rotating = np.ones(n, dtype=bool)
    radius = np.full(n, R)
    if case.perturbation == "stop":
        phase[time >= case.duration_s / 2] = phase[np.flatnonzero(time < case.duration_s / 2)[-1]]
        rotating[time >= case.duration_s / 2] = False
    elif case.perturbation == "radius_drop":
        radius[time >= case.duration_s / 2] *= 0.4
    orbit = np.column_stack((radius * np.cos(phase), 0.6 * radius * np.sin(phase)))
    ideal = center + orbit
    observed = ideal + rng.normal(0, case.noise_sd, ideal.shape)
    if case.perturbation == "missing":
        observed[(time >= case.duration_s / 2) & (time < case.duration_s / 2 + 0.2)] = np.nan
    return Generated(time, observed, ideal, center, radius, rotating)
