"""Module to handle the signal-domain representations, i.e., envelope
amplitude, phase, and frequency"""

import numpy as np
import scipy as sp


def envelope_frequency(
    signal: np.ndarray, signal_step: float = 1 / (2 * np.pi)
):
    """Calculate the envelope and instantaneous frequency of a waveform

    Keyword arguments:
    - signal -- 1d or 2d array, first dimension used for the waveform(s)
    - signal_step (Optional) -- If unspecified, returns normalized frequency

    Returns:
    - envelope -- Envelope of waveform
    - frequency -- Instantaneous frequency of waveform
    - phase -- Instantaneous phase of waveform
    - complex_envelope -- Hilbert transform of waveform
    """
    if signal.real.ndim == 1:
        complex_envelope = sp.signal.hilbert(signal.real)
        envelope = np.abs(complex_envelope)
        phase = np.unwrap(np.angle(complex_envelope))
        frequency = np.zeros_like(phase)
        frequency[:-1] = (1 / signal_step) * np.diff(phase) / (2 * np.pi)
        frequency[-1] = frequency[-2]
    elif signal.real.ndim == 2:
        complex_envelope = sp.signal.hilbert(signal.real, axis=0)
        envelope = np.abs(complex_envelope)
        phase = np.unwrap(np.angle(complex_envelope), axis=0)
        frequency = np.zeros_like(phase)
        frequency[:-1, :] = (
            (1 / signal_step) * np.diff(phase, axis=0) / (2 * np.pi)
        )
        frequency[-1, :] = frequency[-2, :]

    return envelope, frequency, phase, complex_envelope
