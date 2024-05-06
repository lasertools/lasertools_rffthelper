"""Module to use RFFT to transform a signal from signal basis to frequency 
basis and vice versa"""

import numpy as np
import scipy as sp
from lasertools_rffthelper.axes import Axes


def spectrum_from_signal(signal: np.ndarray, axes: Axes):
    """Given a waveform, calculate the frequency domain amplitude
    and phase.

    Keyword arguments:
    signal -- 1d or 2d array, first dimension used for the waveform(s)
    axes -- Object representing a signal and frequency axes linked by RFFT
    """

    # Calculate the complex spectrum
    spectrum_complex = complex_spectrum_from_signal(signal, axes)

    # Amplitude of spectrum
    spectrum_amplitude = np.abs(spectrum_complex)

    # Phase of spectrum
    spectrum_phase = -1 * np.angle(spectrum_complex)

    return spectrum_amplitude, spectrum_phase


def complex_spectrum_from_signal(signal: np.ndarray, axes: Axes):
    """Given a waveform, calculate the complex frequency spectrum.

    Keyword arguments:
    signal -- 1d or 2d array, with the first dimension used for the waveform(s)
    axes -- Object representing a signal and frequency axes linked by RFFT
    """

    # Fourier transform to frequency
    spectrum_complex = axes.axes_parameters.signal_step * sp.fft.rfft(
        sp.fft.fftshift(signal, axes=0), axis=0
    )

    return spectrum_complex


def signal_from_spectrum(
    spectrum_amplitude: np.ndarray, spectrum_phase: np.ndarray, axes: Axes
):
    """Calculate the waveform from a frequency-domain spectrum defined by
    amplitude and phase.

    Keyword arguments:
    - spectrum_amplitude -- The amplitude of the spectrum
    - spectrum_phase -- The phase of the spectrum
    - axes -- Object representing a signal and frequency axes linked by RFFT
    """
    spectrum_complex = spectrum_amplitude * np.exp(-1j * spectrum_phase)

    return signal_from_complex_spectrum(spectrum_complex, axes)


def signal_from_complex_spectrum(spectrum_complex: np.ndarray, axes: Axes):
    """Calculate the waveform from a frequency-domain spectrum defined by a
    complex array.

    Keyword arguments:
    - spectrum_complex -- The complex spectrum
    - axes -- Object representing a signal and frequency axes linked by RFFT
    """

    # Transform spectrum to signal
    signal = (
        axes.axes_parameters.signal_samples
        * axes.axes_parameters.frequency_step
        * sp.fft.ifftshift(
            sp.fft.irfft(
                spectrum_complex, n=axes.axes_parameters.signal_samples, axis=0
            ),
            axes=0,
        )
    )
    return signal
