"""Fit a 1/f-shaped FFT amplitude and synthesize a real random-phase series."""

from dataclasses import dataclass

import numpy as np

OFFSET_HZ = 1.0


@dataclass(frozen=True)
class AmplitudeFit:
    scale: float
    offset_hz: float
    point_count: int
    frequency_min_hz: float
    frequency_max_hz: float
    rmse_amplitude: float
    r_squared_amplitude: float


def amplitude_envelope(frequency_hz: np.ndarray, scale: float, offset_hz: float = OFFSET_HZ) -> np.ndarray:
    """FFT amplitude A/(f+f0), with f and f0 both measured in Hz."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(scale) or scale < 0 or not np.isfinite(offset_hz) or offset_hz <= 0:
        raise ValueError("A nonnegative finite scale and positive finite offset are required")
    if np.any(~np.isfinite(frequency_hz)) or np.any(frequency_hz <= 0):
        raise ValueError("Frequencies must be finite and positive")
    return scale / (frequency_hz + offset_hz)


def fit_amplitude(frequency_hz: np.ndarray, amplitude: np.ndarray) -> AmplitudeFit:
    """Unweighted least squares on the saved FFT *amplitude*, not power."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    if frequency_hz.shape != amplitude.shape:
        raise ValueError("Frequency and amplitude shapes differ")
    valid = np.isfinite(frequency_hz) & np.isfinite(amplitude) & (frequency_hz > 0) & (amplitude >= 0)
    frequency_hz = frequency_hz[valid]
    amplitude = amplitude[valid]
    if len(frequency_hz) < 3 or np.max(amplitude) <= 0:
        raise ValueError("At least three valid positive-frequency FFT amplitudes are required")
    basis = 1.0 / (frequency_hz + OFFSET_HZ)
    scale = float(np.dot(basis, amplitude) / np.dot(basis, basis))
    predicted = amplitude_envelope(frequency_hz, scale)
    residual = amplitude - predicted
    error_sum = float(np.dot(residual, residual))
    centered = amplitude - np.mean(amplitude)
    total_sum = float(np.dot(centered, centered))
    return AmplitudeFit(
        scale=scale,
        offset_hz=OFFSET_HZ,
        point_count=len(frequency_hz),
        frequency_min_hz=float(np.min(frequency_hz)),
        frequency_max_hz=float(np.max(frequency_hz)),
        rmse_amplitude=float(np.sqrt(error_sum / len(frequency_hz))),
        r_squared_amplitude=1.0 - error_sum / total_sum if total_sum > 0 else float("nan"),
    )


def random_phase_inverse_fft(
    sample_count: int, sample_interval_s: float, scale: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make a zero-mean real series whose saved positive-bin amplitudes equal the fit.

    The poster FFT omits DC and the final positive bin even when N is odd.
    Those two coefficients are kept at zero. Its amplitude convention is
    2*abs(FFT)/N on the retained positive bins.
    """
    if sample_count < 6 or not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("At least six regularly timed samples are required")
    frequency_hz = np.fft.rfftfreq(sample_count, d=sample_interval_s)[1 : sample_count // 2]
    target_amplitude = amplitude_envelope(frequency_hz, scale)
    phase_rad = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=len(frequency_hz))
    coefficients = np.zeros(sample_count // 2 + 1, dtype=complex)
    coefficients[1 : sample_count // 2] = sample_count / 2 * target_amplitude * np.exp(1j * phase_rad)
    series = np.fft.irfft(coefficients, n=sample_count)
    return frequency_hz, target_amplitude, phase_rad, series
