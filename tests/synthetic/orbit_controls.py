"""Known-center controls with radius changes on the rotation timescale.

These are declared alternatives to a fixed conic, not measured noise models.
In phase-locked radial motion the physical generating center need not equal a
best-fit ellipse center; small conic residuals alone cannot certify that center.
"""

import numpy as np


def make_radial_control(kind, *, speed_rev_s, dt_s, duration_s, amplitude, axis_ratio, seed):
    if kind not in ("independent_radius", "colored_radius", "fast_radius", "phase_locked_radius"):
        raise ValueError("unknown radius variation")
    settings = np.asarray([speed_rev_s, dt_s, duration_s, amplitude, axis_ratio], float)
    if not np.isfinite(settings).all() or min(speed_rev_s, dt_s, duration_s) <= 0:
        raise ValueError("finite positive acquisition settings required")
    if not 0 < amplitude < 1 or not 0 < axis_ratio <= 1:
        raise ValueError("invalid radial amplitude or projection")
    t = np.arange(int(np.floor(duration_s / dt_s)) + 1) * dt_s
    rng = np.random.default_rng(seed)
    phase = 2 * np.pi * speed_rev_s * t + rng.uniform(-np.pi, np.pi)
    if kind in ("independent_radius", "colored_radius"):
        # Bounded stochastic variation, never zero/negative radius.
        noise = rng.normal(size=len(t))
        if kind == "colored_radius":
            for i in range(1, len(t)):
                noise[i] = 0.7 * noise[i - 1] + np.sqrt(1 - 0.7**2) * noise[i]
        radius = 1 + amplitude * np.tanh(noise)
    elif kind == "fast_radius":
        radius = 1 + amplitude * np.sin(2 * np.pi * speed_rev_s * 0.37 * t + 0.4)
    else:
        radius = 1 + amplitude * np.cos(phase)
    center = np.column_stack([0.03 * t, -0.02 * t])
    clean = center + radius[:, None] * np.column_stack([np.cos(phase), -axis_ratio * np.sin(phase)])
    observed = clean + rng.normal(0, 0.02, clean.shape)
    return dict(
        source_frame=np.arange(1, len(t) + 1),
        time=t,
        raw_xy=observed,
        clean_xy=clean,
        true_center=center,
        true_radius=radius,
        true_phase=phase,
        truth_rotating=np.ones(len(t), bool),
        source_segment_id=np.zeros(len(t), int),
    )
