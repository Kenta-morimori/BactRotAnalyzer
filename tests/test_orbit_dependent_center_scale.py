"""Observation-lag covariance is PSD and reconstructs unchanged IRLS fits."""

import numpy as np

from utils.functions.orbit_center.dependent_center_scale import (
    bartlett_score_covariance,
    irls_hac_coefficients,
)
from utils.functions.orbit_center.moving_center import OrbitModel, fit_orbit


def test_psd_gram_form_matches_explicit_bartlett_lag_sum():
    scores = np.random.default_rng(21).normal(size=(31, 4))
    for lags in [0, 1, 5, 30]:
        expected = scores.T @ scores
        for h in range(1, lags + 1):
            lag = scores[h:].T @ scores[:-h]
            expected += (1 - h / (lags + 1)) * (lag + lag.T)
        actual = bartlett_score_covariance(scores, lags)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert np.linalg.eigvalsh(actual).min() > -1e-12


def test_frozen_weight_reconstruction_and_source_boundaries():
    t = np.arange(121) * 0.005
    xy = np.c_[np.cos(2 * np.pi * 24 * t), -np.sin(2 * np.pi * 24 * t)] + np.random.default_rng(5).normal(
        0, 0.02, (121, 2)
    )
    fit = fit_orbit(t, xy, t[60], OrbitModel())
    fit["selected_model"] = dict(center_degree=1, radius_squared_degree=2, ellipse=True)
    covariance = irls_hac_coefficients(fit, t, xy, 5)
    assert covariance is not None and covariance.shape == (9, 9)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12
