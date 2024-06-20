import numpy as np


def scale_mean_one(data):
    mean = np.mean(data)

    return data / mean


def scale_mean_zero(data):
    mean = np.mean(data)

    return data - mean


def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    return standardized_data
