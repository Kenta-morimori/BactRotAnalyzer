"""Frequency and recording-gap variations of the sign fixtures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.synthetic.sign import StudyCase, make_data


@dataclass(frozen=True)
class AuditCase:
    name: str
    source: StudyCase
    frequency: float | None = None
    gap: bool = False
    colored_dwell: bool = False
    phase_offset: float = 0.0
    audit_only: bool = False


def generate(case: AuditCase, seed: int):
    data = make_data(case.source, seed)
    if case.frequency is not None:
        t = data.time
        clock = t.copy()
        if "dwell" in case.source.perturbation:
            a, b = 0.4 * case.source.duration, 0.6 * case.source.duration
            clock = np.where(t < a, t, np.where(t < b, a, t - (b - a)))
        phase = (-1 if case.source.cartesian_clockwise else 1) * 2 * np.pi * case.frequency * clock
        data.ideal = data.center + np.column_stack((data.radius * np.cos(phase), 0.6 * data.radius * np.sin(phase)))
        rng = np.random.default_rng(seed + 777)
        data.observed = data.ideal + rng.normal(0, case.source.noise, data.ideal.shape)
        stopped = ~data.rotating
        data.observed[stopped] = data.ideal[stopped] + rng.normal(0, case.source.dwell_noise, (stopped.sum(), 2))
    if case.colored_dwell:
        rng = np.random.default_rng(seed + 888)
        indices = np.flatnonzero(~data.rotating)
        jitter = np.zeros((len(indices), 2))
        for index in range(1, len(indices)):
            jitter[index] = 0.85 * jitter[index - 1] + rng.normal(0, 0.02, 2)
        data.observed[indices] = data.ideal[indices] + jitter
    if case.phase_offset:
        noise = data.observed - data.ideal
        relative = data.ideal - data.center
        phase = np.arctan2(relative[:, 1] / (0.6 * data.radius), relative[:, 0] / data.radius) + case.phase_offset
        data.ideal = data.center + np.column_stack((data.radius * np.cos(phase), 0.6 * data.radius * np.sin(phase)))
        data.observed = data.ideal + noise
    if case.gap:
        data.time[data.time >= case.source.duration / 2] += 5.5
    return data
