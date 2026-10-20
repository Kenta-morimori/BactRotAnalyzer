"""Drift, dwell, and rotation direction fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.synthetic.drift import Case, generate


@dataclass(frozen=True)
class StudyCase:
    name: str
    drift: float = 0.0
    orbit: str = "full"
    shape: str = "linear"
    perturbation: str = "none"
    direction: str = "x"
    noise: float = 0.02
    duration: float = 6.0
    clockwise: bool = False
    dwell_noise: float = 0.008


def make_data(case: StudyCase, seed: int):
    mapped = Case(
        case.name,
        case.drift,
        case.shape,
        case.direction,
        case.orbit,
        case.noise,
        case.duration,
        case.perturbation if case.perturbation in {"radius_drop", "missing"} else "none",
    )
    data = generate(mapped, seed)
    if case.perturbation in {"dwell_resume", "dwell_drift_resume"}:
        a, b = case.duration * 0.4, case.duration * 0.6
        phase = 2 * np.pi * np.where(data.time < a, data.time, np.where(data.time < b, a, data.time - (b - a))) / 0.3
        if case.perturbation == "dwell_drift_resume":
            data.center[:, 0] += 0.10 * np.clip((data.time - a) / (b - a), 0, 1)
        data.rotating[(data.time >= a) & (data.time < b)] = False
        data.ideal = data.center + np.column_stack((data.radius * np.cos(phase), 0.6 * data.radius * np.sin(phase)))
        rng = np.random.default_rng(seed + 33000)
        data.observed = data.ideal + rng.normal(0, case.noise, data.ideal.shape)
        dwell = ~data.rotating
        data.observed[dwell] = data.ideal[dwell] + rng.normal(0, case.dwell_noise, (dwell.sum(), 2))
    if case.perturbation == "outliers":
        rng = np.random.default_rng(seed + 44000)
        selected = rng.choice(np.arange(100, len(data.time) - 100), 12, replace=False)
        data.observed[selected] += rng.normal(0, 0.12, (len(selected), 2))
    if case.perturbation in {"line", "stationary"}:
        phase = 2 * np.pi * data.time / 0.3
        orbit = (
            np.column_stack((0.3 * np.cos(phase), 0.18 * np.cos(phase)))
            if case.perturbation == "line"
            else np.tile([0.3, 0.0], (len(phase), 1))
        )
        data.ideal = data.center + orbit
        data.observed = data.ideal + np.random.default_rng(seed + 55000).normal(0, case.noise, data.ideal.shape)
        data.rotating[:] = False
    if case.clockwise:
        data.ideal[:, 1] = 2 * data.center[:, 1] - data.ideal[:, 1]
        data.observed[:, 1] = 2 * data.center[:, 1] - data.observed[:, 1]
    return data
