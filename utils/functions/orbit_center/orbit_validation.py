"""Held-out validation blocks for local center prediction."""

import math

import numpy as np


def validation_blocks(n_observations, max_parameters, *, minimum_splits=4):
    """Keep contiguous held-out blocks and at least one training residual df.

    With n=12 and k=9, four folds leave nine training points and cannot
    estimate residual variance. Six folds instead leave ten training points.
    If even leave-one-out has no residual df, the caller retains an explicit
    unsupported model; this function does not pretend to create information.
    """
    if n_observations < 1 or max_parameters < 1 or minimum_splits < 2:
        raise ValueError("positive observation/parameter counts and at least two splits required")
    largest_holdout = max(1, n_observations - max_parameters - 1)
    count = min(n_observations, max(minimum_splits, math.ceil(n_observations / largest_holdout)))
    return np.array_split(np.arange(n_observations), count)
