"""Module to shift a waveform in the signal domain"""

import numpy as np
from lasertools_rffthelper.axes import Axes
from lasertools_rffthelper.fft import signal_from_spectrum, spectrum_from_signal


def shift_signal(
    signal: np.ndarray,
    shift_amount: np.ndarray,
    axes: Axes,
    wrap: bool = False,
):
    """Returns shifted signal

    Keyword arguments:
    - signal -- 1d or 2d array, first dimension used for the waveform(s)
    - shift_amount -- Number or vector of amount(s) to shift signal
    - axes -- Object representing a signal and frequency axes linked by RFFT
    - wrap (Optional) -- False = prevent signal from wrapping around the axis
    """

    # Find input spectrum and phase
    spectrum_amplitude, spectrum_phase = spectrum_from_signal(signal, axes)

    # Calculate mask to shift signal
    if len(shift_amount) == 1:
        spectrum_amplitude_initial = spectrum_amplitude
        spectrum_phase_initial = spectrum_phase
    elif len(shift_amount) > 1:
        spectrum_amplitude_initial = np.tile(
            spectrum_amplitude, (len(shift_amount), 1)
        ).T
        spectrum_phase_initial = np.tile(
            spectrum_phase, (len(shift_amount), 1)
        ).T

    # Apply mask
    spectrum_phase_final = (
        calculate_phase_shift(shift_amount, axes) + spectrum_phase_initial
    )

    # Find shifted signal (with wrapping)
    signal_wrapped = signal_from_spectrum(
        spectrum_amplitude_initial,
        spectrum_phase_final,
        axes,
    )

    if not wrap:
        signal = signal_wrapped * calculate_amplitude_mask(shift_amount, axes)
    else:
        signal = signal_wrapped

    return signal


def calculate_phase_shift(shift_amount: np.ndarray, axes: Axes):
    """Returns phase mask to shift signal

    Keyword arguments:
    - shift_amount -- number or vector of amount(s) to shift signal
    - axes -- Object representing a signal and frequency axes linked by RFFT
    """

    if len(shift_amount) == 1:
        phase_shift = 2 * np.pi * axes.frequency_axis * shift_amount
    elif len(shift_amount) > 1:
        phase_shift = 2 * np.pi * np.outer(axes.frequency_axis, shift_amount)
    return phase_shift


def calculate_amplitude_mask(shift_amount: np.ndarray, axes: Axes):
    """Returns amplitude mask to prevent signal wrapping

    Keyword arguments:
    - shift_amount -- number or vector of amount(s) to shift signal
    - axes -- Object representing a signal and frequency axes linked by RFFT
    """

    shift_amount_steps = shift_amount / axes.axes_parameters.signal_step
    if len(shift_amount) == 1:
        amplitude_mask = 2 * np.pi * np.zeros_like(axes.signal_axis)
        if shift_amount_steps >= 0:
            amplitude_mask[int(np.floor(shift_amount_steps)) :] = 1
        if shift_amount_steps <= 0:
            amplitude_mask[: int(np.ceil(shift_amount_steps))] = 1
    elif len(shift_amount) > 1:
        amplitude_mask = np.zeros_like(
            np.outer(axes.signal_axis, shift_amount)
        )
        for shift_index, shift_steps in enumerate(shift_amount_steps):
            if shift_steps >= 0:
                amplitude_mask[int(np.floor(shift_steps)) :, shift_index] = 1
            if shift_steps <= 0:
                amplitude_mask[: int(np.ceil(shift_steps)), shift_index] = 1
    return amplitude_mask
