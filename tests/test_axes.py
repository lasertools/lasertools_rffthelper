"""A unit test module for axes.py"""

import numpy as np
import scipy as sp
import lasertools_rffthelper.axes


def test_axes(signal_length=None, signal_samples_test=None):
    """Function that tests the class Axes"""

    print("Axes: Testing.")

    if signal_samples_test is None:
        signal_samples_test = np.random.randint(800, 5000)
        print("Signal samples = " + str(signal_samples_test))

    if signal_length is None:
        signal_length = 200e-15 + np.random.rand() * 800e-18
        print("Signal length = " + str(signal_length))

    results_test = pass_signal_samples(
        signal_samples_test, signal_length, True
    )
    results_best = pass_signal_samples(
        signal_samples_test, signal_length, False
    )

    print(results_test)
    print(results_best)
    assert all([all(results_test), all(results_best)])


def pass_signal_samples(signal_samples, signal_length, best_signal_samples):
    """Function to test based on signal samples and length"""

    # Find known true values
    if best_signal_samples:
        signal_axis = np.linspace(
            0,
            signal_length,
            sp.fft.next_fast_len(signal_samples, real=True),
        )
    else:
        signal_axis = np.linspace(0, signal_length, signal_samples)
    signal_step = signal_axis[1] - signal_axis[0]
    frequency_axis = np.fft.rfftfreq(np.size(signal_axis), d=signal_step)
    frequency_samples = len(frequency_axis)
    frequency_step = frequency_axis[1] - frequency_axis[0]
    frequency_length = np.max(frequency_axis)

    if len(signal_axis) % 2 != 0:
        signal_samples_parity_even = False
    else:
        signal_samples_parity_even = True

    print("Signal step: " + str(signal_step) + " (reference)")
    print("Frequency step: " + str(frequency_step) + " (reference)")

    combinations_valid = [
        {
            "frequency_samples": frequency_samples,
            "signal_step": signal_step,
        },
        {
            "frequency_samples": frequency_samples,
            "frequency_step": frequency_step,
        },
        {
            "frequency_samples": frequency_samples,
            "signal_length": signal_length,
        },
        {
            "frequency_samples": frequency_samples,
            "frequency_length": frequency_length,
        },
        {"signal_samples": len(signal_axis), "signal_step": signal_step},
        {
            "signal_samples": len(signal_axis),
            "frequency_step": frequency_step,
        },
        {"signal_samples": len(signal_axis), "signal_length": signal_length},
        {
            "signal_samples": len(signal_axis),
            "frequency_length": frequency_length,
        },
        {"signal_step": signal_step, "frequency_step": frequency_step},
        {"signal_step": signal_step, "signal_length": signal_length},
        {
            "frequency_step": frequency_step,
            "frequency_length": frequency_length,
        },
        {
            "signal_length": signal_length,
            "frequency_length": frequency_length,
        },
    ]

    combinations_valid_result = []
    for combination in combinations_valid:
        print(combination)
        combinations_valid_result.append(
            check_combination_valid(
                signal_axis,
                frequency_axis,
                0.9999999,
                signal_samples_parity_even,
                best_signal_samples,
                **combination
            )
        )
        print(combinations_valid_result[-1])

    combinations_invalid = [
        {"signal_step": signal_step, "frequency_length": frequency_length},
        {"frequency_step": frequency_step, "signal_length": signal_length},
        {
            "frequency_samples": frequency_samples,
            "signal_samples": len(signal_axis),
        },
        {"frequency_step": frequency_step},
        {
            "frequency_step": frequency_step,
            "frequency_length": frequency_length,
            "signal_samples": len(signal_axis),
        },
    ]

    combinations_invalid_result = []
    for combination in combinations_invalid:
        combinations_invalid_result.append(
            check_combination_invalid(**combination)
        )
        print(combination)
        print(combinations_invalid_result[-1])
    return all(combinations_valid_result), all(combinations_invalid_result)


def check_combination_valid(
    signal_axis, frequency_axis, valid_threshold, parity, optimize, **kwargs
):
    """Function to test a valid combination of parameters"""

    if len(kwargs.items()) != 2:
        raise ValueError("Too many arguments.")
    combination_parameters = lasertools_rffthelper.axes.AxesParameters(**kwargs)
    combination = lasertools_rffthelper.axes.Axes(
        combination_parameters, parity, optimize
    )
    determination_signal = find_determination(
        combination.signal_axis, signal_axis
    )
    determination_frequency = find_determination(
        combination.frequency_axis, frequency_axis
    )
    determination_combination = (determination_signal > valid_threshold) & (
        determination_frequency > valid_threshold
    )

    print(combination.signal_axis[0:3])
    print(signal_axis[0:3])
    print(combination.frequency_axis[0:3])
    print(frequency_axis[0:3])
    print(determination_signal)
    print(determination_frequency)

    return determination_combination


def check_combination_invalid(**kwargs):
    """Function to test an invalid combination of parameters"""

    combination_result = False
    try:
        combination_parameters = lasertools_rffthelper.axes.AxesParameters(**kwargs)
        lasertools_rffthelper.axes.Axes(combination_parameters)
    except ValueError:
        combination_result = True
    return combination_result


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
