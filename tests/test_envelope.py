"""A unit test module for envelope.py"""

import numpy as np
import lasertools_rffthelper.envelope


def test_envelope():
    """Function to test the envelope module functions"""

    determination_threshold = 0.90

    (
        reference_signal,
        reference_signal_envelope,
        reference_signal_frequency,
        reference_signal_axis,
    ) = create_signal_2d()

    reference_signal_step = reference_signal_axis[1] - reference_signal_axis[0]

    (
        calculated_signal_envelope,
        calculated_signal_frequency,
        _,
        _,
    ) = lasertools_rffthelper.envelope.envelope_frequency(
        reference_signal, reference_signal_step
    )

    signal_samples_sectioned = int(len(reference_signal_frequency) / 3)
    assert (
        find_determination(
            calculated_signal_envelope, reference_signal_envelope
        )
        > determination_threshold
    ) and (
        find_determination(
            calculated_signal_frequency[
                signal_samples_sectioned:-signal_samples_sectioned, :
            ],
            reference_signal_frequency[
                signal_samples_sectioned:-signal_samples_sectioned, :
            ],
        )
        > determination_threshold
    )

    reference_signal_1d = reference_signal[:, 0]
    (
        calculated_signal_envelope,
        _,
        _,
        _,
    ) = lasertools_rffthelper.envelope.envelope_frequency(
        reference_signal_1d, reference_signal_step
    )

    assert (
        find_determination(
            calculated_signal_envelope, reference_signal_envelope[:, 0]
        )
        > determination_threshold
    )


def create_signal_2d(
    reference_signal_max=100,
    reference_signal_points=10001,
    reference_signal_2d_cols=10,
):
    """function to create a 2-dimensional reference signal"""

    phase_coefficients = [
        (10 / reference_signal_max) * np.random.uniform(low=4, high=6),
        (30 / (2 * reference_signal_max) ** 2)
        * np.random.uniform(low=-1, high=1),
        (3000 / reference_signal_max)
        * np.random.uniform(low=-1, high=1)
        / (2 * reference_signal_max) ** 3,
    ]

    reference_signal_axis = np.linspace(
        -reference_signal_max, reference_signal_max, reference_signal_points
    )

    reference_signal_envelope = np.zeros(
        (reference_signal_points, reference_signal_2d_cols)
    )
    reference_signal_phase = np.zeros(
        (reference_signal_points, reference_signal_2d_cols)
    )
    reference_signal_frequency = np.zeros(
        (reference_signal_points, reference_signal_2d_cols)
    )

    for k in range(reference_signal_2d_cols):
        reference_signal_envelope[:, k] = np.exp(
            -(
                (
                    reference_signal_axis
                    / (reference_signal_max * np.random.uniform(0.1, 0.2))
                )
                ** 2
            )
        ) + np.random.uniform(0.2, 0.3) * np.exp(
            -(
                (
                    (
                        np.random.uniform(-0.3, 0.3) * reference_signal_max
                        + reference_signal_axis
                    )
                    / (reference_signal_max * np.random.uniform(0.3, 0.5))
                )
                ** 2
            )
        )

        reference_signal_phase[:, k] = (
            2
            * np.pi
            * (
                phase_coefficients[0]
                * (reference_signal_axis + reference_signal_max)
                + phase_coefficients[1]
                * (reference_signal_axis + reference_signal_max) ** 2
                + phase_coefficients[2]
                * (reference_signal_axis + reference_signal_max) ** 3
            )
        )
        reference_signal_frequency[:, k] = (
            phase_coefficients[0]
            + 2
            * phase_coefficients[1]
            * (reference_signal_axis + reference_signal_max)
            + 3
            * phase_coefficients[2]
            * (reference_signal_axis + reference_signal_max) ** 2
        )

    reference_signal = reference_signal_envelope * np.cos(
        reference_signal_phase
    )

    return (
        reference_signal,
        reference_signal_envelope,
        reference_signal_frequency,
        reference_signal_axis,
    )


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
