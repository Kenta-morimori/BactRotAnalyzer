"""Fit an FFT-amplitude envelope and synthesize a real random-phase series."""

from dataclasses import dataclass

import numpy as np

REFERENCE_HZ = 1.0
MAX_FIT_BINS = 9


@dataclass(frozen=True)
class AmplitudeFit:
    scale: float
    exponent: float
    offset_hz: float
    point_count: int
    bin_count: int
    frequency_min_hz: float
    frequency_max_hz: float
    rmse_amplitude: float
    r_squared_amplitude: float
    log_bin_rmse: float
    log_bin_r_squared: float


def amplitude_envelope(frequency_hz: np.ndarray, scale: float, *, exponent: float, offset_hz: float) -> np.ndarray:
    """FFT amplitude A * ((f + f0) / 1 Hz)**(-alpha), for f > 0."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(scale) or scale < 0 or not np.isfinite(exponent) or exponent < 0:
        raise ValueError("Finite nonnegative scale and exponent are required")
    if not np.isfinite(offset_hz) or offset_hz <= 0:
        raise ValueError("A finite positive frequency offset is required")
    if np.any(~np.isfinite(frequency_hz)) or np.any(frequency_hz <= 0):
        raise ValueError("Frequencies must be finite and positive")
    return scale * ((frequency_hz + offset_hz) / REFERENCE_HZ) ** (-exponent)


def log_frequency_bins(
    frequency_hz: np.ndarray, amplitude: np.ndarray, max_bins: int = MAX_FIT_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize positive FFT amplitudes by medians in equal log-frequency bands."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if frequency_hz.shape != amplitude.shape:
        raise ValueError("Frequency and amplitude shapes differ")
    if max_bins < 3:
        raise ValueError("At least three logarithmic frequency bands are required")
    valid = np.isfinite(frequency_hz) & np.isfinite(amplitude) & (frequency_hz > 0) & (amplitude > 0)
    frequency_hz = frequency_hz[valid]
    amplitude = amplitude[valid]
    if len(frequency_hz) < 3 or np.unique(frequency_hz).size < 3:
        raise ValueError("At least three distinct positive-frequency FFT amplitudes are required")
    edges = np.geomspace(np.min(frequency_hz), np.max(frequency_hz), min(max_bins, len(frequency_hz)) + 1)
    centers = []
    medians = []
    counts = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (frequency_hz >= left) & (
            (frequency_hz <= right) if index == len(edges) - 2 else (frequency_hz < right)
        )
        if np.any(selected):
            centers.append(float(np.exp(np.mean(np.log(frequency_hz[selected])))))
            medians.append(float(np.median(amplitude[selected])))
            counts.append(int(np.sum(selected)))
    if len(centers) < 3:
        raise ValueError("At least three nonempty logarithmic frequency bands are required")
    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def equal_frequency_bins(
    frequency_hz: np.ndarray, amplitude: np.ndarray, width_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize positive FFT amplitudes in constant-width Hz bands from zero."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if frequency_hz.shape != amplitude.shape:
        raise ValueError("Frequency and amplitude shapes differ")
    if not np.isfinite(width_hz) or width_hz <= 0:
        raise ValueError("A finite positive frequency-band width is required")
    valid = np.isfinite(frequency_hz) & np.isfinite(amplitude) & (frequency_hz > 0) & (amplitude > 0)
    frequency_hz = frequency_hz[valid]
    amplitude = amplitude[valid]
    if len(frequency_hz) < 3 or np.unique(frequency_hz).size < 3:
        raise ValueError("At least three distinct positive-frequency FFT amplitudes are required")
    band_index = np.floor(frequency_hz / width_hz).astype(int)
    centers = []
    medians = []
    counts = []
    for index in np.unique(band_index):
        selected = band_index == index
        centers.append(float(np.mean(frequency_hz[selected])))
        medians.append(float(np.median(amplitude[selected])))
        counts.append(int(np.sum(selected)))
    if len(centers) < 3:
        raise ValueError("At least three nonempty equal-width frequency bands are required")
    return np.asarray(centers), np.asarray(medians), np.asarray(counts)


def fit_amplitude(
    frequency_hz: np.ndarray,
    amplitude: np.ndarray,
    *,
    max_bins: int = MAX_FIT_BINS,
    fixed_exponent: float | None = None,
    equal_width_hz: float | None = None,
) -> AmplitudeFit:
    """Fit band-median amplitudes; equal-width Hz bands are for sensitivity plots."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if frequency_hz.shape != amplitude.shape:
        raise ValueError("Frequency and amplitude shapes differ")
    positive_frequency = frequency_hz[np.isfinite(frequency_hz) & (frequency_hz > 0)]
    if len(positive_frequency) == 0:
        raise ValueError("At least one finite positive FFT frequency is required")
    offset_hz = float(np.min(positive_frequency))
    valid = np.isfinite(frequency_hz) & np.isfinite(amplitude) & (frequency_hz > 0) & (amplitude > 0)
    frequency_hz = frequency_hz[valid]
    amplitude = amplitude[valid]
    if equal_width_hz is None:
        bin_frequency, bin_amplitude, _ = log_frequency_bins(frequency_hz, amplitude, max_bins=max_bins)
    else:
        bin_frequency, bin_amplitude, _ = equal_frequency_bins(frequency_hz, amplitude, equal_width_hz)
    x = np.log((bin_frequency + offset_hz) / REFERENCE_HZ)
    y = np.log(bin_amplitude)
    if fixed_exponent is None:
        slope = float(np.dot(x - np.mean(x), y - np.mean(y)) / np.dot(x - np.mean(x), x - np.mean(x)))
        exponent = max(0.0, -slope)
    else:
        if not np.isfinite(fixed_exponent) or fixed_exponent < 0:
            raise ValueError("The fixed exponent must be finite and nonnegative")
        exponent = float(fixed_exponent)
    scale = float(np.exp(np.mean(y + exponent * x)))
    predicted_bins = amplitude_envelope(bin_frequency, scale, exponent=exponent, offset_hz=offset_hz)
    log_residual = y - np.log(predicted_bins)
    log_total = float(np.sum((y - np.mean(y)) ** 2))
    predicted = amplitude_envelope(frequency_hz, scale, exponent=exponent, offset_hz=offset_hz)
    residual = amplitude - predicted
    total = float(np.sum((amplitude - np.mean(amplitude)) ** 2))
    return AmplitudeFit(
        scale=scale,
        exponent=exponent,
        offset_hz=offset_hz,
        point_count=len(frequency_hz),
        bin_count=len(bin_frequency),
        frequency_min_hz=float(np.min(frequency_hz)),
        frequency_max_hz=float(np.max(frequency_hz)),
        rmse_amplitude=float(np.sqrt(np.mean(residual**2))),
        r_squared_amplitude=1.0 - float(np.sum(residual**2)) / total if total > 0 else float("nan"),
        log_bin_rmse=float(np.sqrt(np.mean(log_residual**2))),
        log_bin_r_squared=1.0 - float(np.sum(log_residual**2)) / log_total if log_total > 0 else float("nan"),
    )


def random_phase_inverse_fft(
    sample_count: int, sample_interval_s: float, scale: float, exponent: float, offset_hz: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a zero-mean real series whose saved positive-bin amplitudes equal the fit.

    The poster FFT omits DC and the final positive bin even when N is odd.
    Those two coefficients are kept at zero. Its amplitude convention is
    2*abs(FFT)/N on the retained positive bins.
    """
    if sample_count < 6 or not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("At least six regularly timed samples are required")
    frequency_hz = np.fft.rfftfreq(sample_count, d=sample_interval_s)[1 : sample_count // 2]
    target_amplitude = amplitude_envelope(frequency_hz, scale, exponent=exponent, offset_hz=offset_hz)
    phase_rad = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=len(frequency_hz))
    coefficients = np.zeros(sample_count // 2 + 1, dtype=complex)
    coefficients[1 : sample_count // 2] = sample_count / 2 * target_amplitude * np.exp(1j * phase_rad)
    series = np.fft.irfft(coefficients, n=sample_count)
    return frequency_hz, target_amplitude, phase_rad, series
