"""Stopped-orbit trajectories with known centers and angular fluctuations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    drift: float = 0.1
    stop_s: float = 1.2
    frequency: float = 24
    noise: float = 0.008
    phase: float = 0
    kind: str = "static"
    shape: str = "linear"
    radius: float = 0.3


def generate(case, seed):
    time = np.arange(0, 6 + case.stop_s, 0.005)
    a, b = 3.0, 3.0 + case.stop_s
    stopped = (time >= a) & (time < b)
    if case.kind in {"rotation", "radius_shrink"}:
        stopped[:] = False
    if case.kind == "one_sided":
        stopped[time >= a] = True
    clock = time.copy()
    if stopped.any():
        clock = np.where(time < a, time, np.where(time < b, a, time - case.stop_s))
    if case.kind == "one_sided":
        clock[time >= a] = a
    center = np.column_stack((1.3 + case.drift / 5 * time, np.full(len(time), 1.45)))
    if case.shape == "diagonal":
        center[:, 0] = 1.3 + case.drift / 5 / np.sqrt(2) * time
        center[:, 1] += case.drift / 5 / np.sqrt(2) * time
    elif case.shape == "speed_change":
        center[:, 0] += case.drift / 5 * np.maximum(time - (a + b) / 2, 0)
    elif case.shape == "step":
        center[:, 0] += 0.10 * (time >= (a + b) / 2)
    phase = 2 * np.pi * case.frequency * clock + case.phase
    if case.kind == "angular_jitter":
        phase[stopped] += 0.06 * np.sin(2 * np.pi * 2 * (time[stopped] - a))
    radius = np.full(len(time), case.radius)
    if case.kind == "radius_shrink":
        radius[time >= a] = 0.03
    relative = np.column_stack((radius * np.cos(phase), 0.6 * radius * np.sin(phase)))
    if case.kind == "center":
        relative[stopped] = 0
    elif case.kind == "near_center":
        relative[stopped] = [0.015, 0.005]
    ideal = center + relative
    rng = np.random.default_rng(seed)
    observed = ideal + rng.normal(0, 0.02, ideal.shape)
    observed[stopped] = ideal[stopped] + rng.normal(0, case.noise, (stopped.sum(), 2))
    if case.kind == "colored_noise":
        z = np.zeros((stopped.sum(), 2))
        for i in range(1, len(z)):
            z[i] = 0.85 * z[i - 1] + rng.normal(0, 0.008, 2)
        observed[stopped] = ideal[stopped] + z
    if case.kind == "gap":
        time[time >= (a + b) / 2] += 5.5
    return dict(
        time=time,
        observed=observed,
        ideal=ideal,
        center=center,
        stopped=stopped,
        radius=radius,
        true_angle=np.unwrap(np.arctan2(relative[:, 1], relative[:, 0])),
    )
