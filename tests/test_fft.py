"""A unit test module for fft.py"""

import numpy as np
import lasertools_rffthelper.axes
import lasertools_rffthelper.fft


def test_fft(
    num_indices=100,
    signal_samples=1024,
    signal_length=100,
    valid_threshold=0.9999999999999,
    num_transformations=2000,
):
    """Function to test the FFT module methods"""

    print("FFT: Testing.")

    # Define axes
    reference_axes_parameters = lasertools_rffthelper.axes.AxesParameters(
        signal_length=signal_length, signal_samples=signal_samples
    )
    reference_axes = lasertools_rffthelper.axes.Axes(reference_axes_parameters)
    signal_axis = reference_axes.signal_axis
    test_frequency_step = reference_axes.axes_parameters.frequency_step

    # Create signal with known frequencies
    test_signal = np.zeros((signal_samples, num_indices))
    for k in range(num_indices):
        test_signal[:, k] = np.cos(
            2 * np.pi * test_frequency_step * (k + 1) * signal_axis
        )

    # Check known frequencies
    r_sq_1 = check_transformed_frequency(
        reference_axes,
        test_frequency_step,
        test_signal,
        num_indices,
    )
    if r_sq_1 >= valid_threshold:
        print(f"R^2 = {r_sq_1}; 1st test passed")
    else:
        print(f"R^2 = {r_sq_1}; 1st test failed")

    r_sq_2 = check_gained_error(reference_axes, num_transformations)
    if r_sq_2 >= valid_threshold:
        print(f"R^2 = {r_sq_2}; 2nd test passed")
    else:
        print(f"R^2 = {r_sq_2}; 2nd test failed")

    assert (r_sq_1 >= valid_threshold) and (r_sq_2 >= valid_threshold)


def check_transformed_frequency(
    reference_axes, frequency_step, test_signal, num_indices
):
    """Function to compare transformed signal with known reference frequency
    values

    Keyword arguments:
    - reference_axes -- Object defining signal and frequency axes
    - frequency_step -- Increment of frequency axis
    - test_signal --

    Returns:
    - determination -- Residual R^2 after transformations"""

    # Find frequency domain amplitude
    (transformed_spectrum, _) = lasertools_rffthelper.fft.spectrum_from_signal(
        test_signal, reference_axes
    )

    # Initialize arrays for tranformed and reference frequencies
    peaks = np.zeros(num_indices)
    reference = np.zeros(num_indices)

    # Find transformed and reference frequencies
    for k in range(num_indices):
        peaks[k] = reference_axes.frequency_axis[
            np.argmax(transformed_spectrum[:, k])
        ]
        reference[k] = frequency_step * (k + 1)

    # Calculate coefficient of determination
    r_sq = find_determination(peaks, reference)
    return r_sq


def check_gained_error(reference_axes, num_transformations):
    """Function to compare a reference signal with one transformed back and
    forth many times to check for gained numerical error

    Keyword arguments:
    - reference_axes -- Object defining signal and frequency axes
    - num_transformations -- Number of time to Fourier transform back and forth

    Returns:
    - determination -- Residual R^2 after transformations"""

    # Create signal to test
    reference_signal = np.exp(-1 * (reference_axes.signal_axis) ** 2)

    # Transform back and forth many times
    signal = reference_signal
    for _i in range(num_transformations):
        spectrum, phase = lasertools_rffthelper.fft.spectrum_from_signal(
            signal, reference_axes
        )
        signal = lasertools_rffthelper.fft.signal_from_spectrum(
            spectrum, phase, reference_axes
        )

    # Compare signals
    determination = find_determination(signal, reference_signal)
    return determination


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
