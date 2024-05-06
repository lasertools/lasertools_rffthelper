"""A unit test module for shift.py"""

import dataclasses
import numpy as np
import lasertools_rffthelper.axes
import lasertools_rffthelper.shift


def test_shift(
    pulse_gaussian_parameters=None,
    signal_shift_amount=None,
    signal_shift_points=100,
    maxima_threshold=0.95,
    determination_threshold=0.99999,
):
    if not pulse_gaussian_parameters:
        pulse_gaussian_parameters = PulseGaussianParameters()

    if not signal_shift_amount:
        signal_shift_amount = 1.5 * pulse_gaussian_parameters.signal_length

    pulse_gaussian_test, pulse_gaussian_test_axes = pulse_gaussian(
        pulse_gaussian_parameters
    )

    shift_amounts_test = np.linspace(
        -1 * signal_shift_amount, signal_shift_amount, signal_shift_points
    )

    valid_points = (
        np.abs(shift_amounts_test)
        < pulse_gaussian_parameters.signal_length / 2
    )
    invalid_points = np.invert(valid_points)

    signal_shifted = lasertools_rffthelper.shift.shift_signal(
        signal=pulse_gaussian_test,
        shift_amount=shift_amounts_test,
        axes=pulse_gaussian_test_axes,
    )

    shifted_maxima = np.max(signal_shifted, axis=0)

    shifted_maxima_indices = np.argmax(signal_shifted, axis=0)

    invalid_points = valid_points is False

    print(valid_points)

    test1 = np.all(shifted_maxima[invalid_points] < maxima_threshold)

    signal_center = pulse_gaussian_test_axes.axes_parameters.signal_length / 2

    test2_determination = find_determination(
        pulse_gaussian_test_axes.signal_axis[
            shifted_maxima_indices[valid_points]
        ]
        - signal_center,
        shift_amounts_test[valid_points],
    )

    test2 = test2_determination > determination_threshold

    assert test1 & test2


@dataclasses.dataclass
class PulseGaussianParameters:
    """Class to store Gaussian pulse parameters"""

    center_frequency: float = 1e15
    tau: float = 10e-15
    beta: float = 3e28
    signal_length: float = 100e-15
    signal_samples: int = 10001


def pulse_gaussian(pulse_parameters: PulseGaussianParameters):
    """Function to create a Gaussian pulse based on parameters

    Keyword arguments:
    - pulse_parameters -- Object specifying pulse parameters

    Returns:
    - field -- Field of pulse
    - axes -- Axes object with signal and frequency axes parameters"""

    axes = lasertools_rffthelper.axes.Axes(
        lasertools_rffthelper.axes.AxesParameters(
            signal_length=pulse_parameters.signal_length,
            signal_samples=pulse_parameters.signal_samples,
        )
    )

    signal_center = axes.axes_parameters.signal_length / 2
    signal_axis_centered = axes.signal_axis - signal_center

    field = np.exp(
        -1 * signal_axis_centered**2 / pulse_parameters.tau**2
    ) * np.cos(
        2 * np.pi * pulse_parameters.center_frequency * signal_axis_centered
        + 2 * pulse_parameters.beta * signal_axis_centered**2
    )

    return field, axes


def find_determination(test_array, reference_array):
    """Function to calculate residual R^2

    Keyword arguments:
    - test_array -- Array to compare with
    - reference_array -- Array to compare to

    Returns:
    - determination -- Residual R^2"""

    sos_residual = np.sum(np.square(test_array - reference_array))
    sos_total = np.var(reference_array) * np.size(reference_array)
    determination = 1 - sos_residual / sos_total
    return determination
