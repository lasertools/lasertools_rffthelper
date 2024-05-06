"""Module to define signal and frequency axes linked by FFT"""

# Import dependencies
import dataclasses
import numpy as np
import scipy as sp


@dataclasses.dataclass
class AxesParameters:
    """Class to store signal and frequency axis parameters, two of which must
    be provided to fully define axes"""

    signal_step: float = None
    signal_length: float = None
    signal_samples: int = None
    frequency_step: float = None
    frequency_length: float = None
    frequency_samples: int = None


class Axes:
    """Class representing a signal and frequency axes linked by FFT

    Keyword arguments:
    - axes_parameters -- Object defining two parameters
    - signal_samples_parity_even (Optional) -- True = even sample number
    - optimize (Optional) -- True = optimal computational sample number
    """

    def __init__(
        self,
        axes_parameters: AxesParameters,
        signal_samples_parity_even: bool = True,
        optimize: bool = True,
    ):
        self.optimize = optimize
        self.signal_samples_parity_even = signal_samples_parity_even
        self.axes_parameters = axes_parameters
        self.define_parameters()

    def define_parameters(
        self,
        axes_parameters: AxesParameters = None,
        signal_samples_parity_even: bool = None,
        optimize: bool = None,
    ):
        """Method to define/redefine axes

        Keyword arguments:
        - axes_parameters (Optional) -- Object defining two parameters
        - signal_samples_parity_even (Optional) -- True = even sample number
        - optimize (Optional) -- True = optimal computational sample number
        """
        if axes_parameters is not None:
            self.axes_parameters = axes_parameters
        if signal_samples_parity_even is not None:
            self.axes_parameters = axes_parameters
        if optimize is not None:
            self.axes_parameters = axes_parameters

        # Fully define parameters
        if self._count_undefined_parameters() == 4:
            self._define_parameters()
        else:
            raise ValueError(
                "Incorrect number of signal and frequency axis \
                             parameters specified."
            )

        if self._count_undefined_parameters() == 4:
            self._define_parameters_with_parity()

        # Check that all parameters are defined
        if self._count_undefined_parameters() != 0:
            raise ValueError(
                "Specified signal and frequency axis parameters \
                             are not independent."
            )

        # Calculate signal and frequency axes
        self.signal_axis = np.linspace(
            0,
            self.axes_parameters.signal_length,
            self.axes_parameters.signal_samples,
        )
        self.frequency_axis = np.linspace(
            0,
            self.axes_parameters.frequency_length,
            self.axes_parameters.frequency_samples,
        )

    def optimal_samples(self):
        """Returns the optimal RFFT grid length for computation"""
        return sp.fft.next_fast_len(
            self.axes_parameters.signal_samples - 1, real=True
        )

    def signal_axis_centered(self):
        """Returns the signal axis with the center set to zero"""
        return (
            self.signal_axis - (self.signal_axis[-1] - self.signal_axis[0]) / 2
        )

    def _count_undefined_parameters(self):
        """Method to count how many axis parameters are undefined"""

        count = sum(
            1
            for attribute in self.axes_parameters.__dict__.values()
            if attribute is None
        )
        return count

    def _signal_samples_padded(self):
        """Sets the optimal RFFT grid length for computation if enabled"""
        if self.optimize:
            self.axes_parameters.signal_samples = self.optimal_samples()

    def _frequency_from_signal(self):
        """Method to define frequency parameters using signal parameters"""

        if self.axes_parameters.signal_samples % 2 != 0:
            self.axes_parameters.frequency_samples = int(
                np.round((self.axes_parameters.signal_samples - 1) / 2 + 1, 0)
            )
            self.signal_samples_parity_even = False
        else:
            self.axes_parameters.frequency_samples = int(
                np.round((self.axes_parameters.signal_samples / 2) + 1, 0)
            )
            self.signal_samples_parity_even = True
        self.axes_parameters.frequency_step = 1 / (
            self.axes_parameters.signal_samples
            * self.axes_parameters.signal_step
        )
        self.axes_parameters.frequency_length = (
            self.axes_parameters.frequency_step
            * (self.axes_parameters.frequency_samples - 1)
        )

    def _signal_samples_from_frequency_samples(self):
        """Method to define the signal samples from frequency samples"""

        if self.signal_samples_parity_even is True:
            self.axes_parameters.signal_samples = 2 * (
                self.axes_parameters.frequency_samples - 1
            )
        elif self.signal_samples_parity_even is False:
            self.axes_parameters.signal_samples = (
                2 * (self.axes_parameters.frequency_samples - 1) + 1
            )

    def _signal_from_frequency(self):
        """Method to define signal parameters from frequency parameters"""

        self._signal_samples_from_frequency_samples()
        if self.signal_samples_parity_even is True:
            self.axes_parameters.signal_step = 1 / (
                2 * self.axes_parameters.frequency_length
            )
        elif self.signal_samples_parity_even is False:
            self.axes_parameters.signal_step = (
                (self.axes_parameters.signal_samples - 1)
                / self.axes_parameters.signal_samples
            ) / (2 * self.axes_parameters.frequency_length)

        self._signal_samples_padded()
        self.axes_parameters.signal_length = (
            self.axes_parameters.signal_step
            * (self.axes_parameters.signal_samples - 1)
        )
        self._frequency_from_signal()

    def _define_parameters(self):
        """Method to complete a fully defined parameter set"""

        if (
            self.axes_parameters.signal_length
            and self.axes_parameters.signal_step
        ):
            self.axes_parameters.signal_samples = int(
                np.round(
                    self.axes_parameters.signal_length
                    / self.axes_parameters.signal_step
                    + 1,
                    0,
                )
            )
            self._signal_samples_padded()
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()
        elif (
            self.axes_parameters.signal_length
            and self.axes_parameters.signal_samples
        ):
            self._signal_samples_padded()
            self.axes_parameters.signal_step = (
                self.axes_parameters.signal_length
                / (self.axes_parameters.signal_samples - 1)
            )
            self._frequency_from_signal()
        elif (
            self.axes_parameters.signal_step
            and self.axes_parameters.signal_samples
        ):
            self._signal_samples_padded()
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()
        elif (
            self.axes_parameters.frequency_step
            and self.axes_parameters.signal_samples
        ):
            self._signal_samples_padded()
            self.axes_parameters.signal_step = 1 / (
                self.axes_parameters.signal_samples
                * self.axes_parameters.frequency_step
            )
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()
        elif (
            self.axes_parameters.frequency_step
            and self.axes_parameters.signal_step
        ):
            self.axes_parameters.signal_samples = int(
                np.round(
                    1
                    / (
                        self.axes_parameters.signal_step
                        * self.axes_parameters.frequency_step
                    ),
                    0,
                )
            )
            self._signal_samples_padded()
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()
        elif (
            self.axes_parameters.frequency_length
            and self.axes_parameters.signal_samples
        ):
            self._signal_samples_padded()
            if self.axes_parameters.signal_samples % 2 != 0:
                self.axes_parameters.signal_step = (
                    (self.axes_parameters.signal_samples - 1)
                    / self.axes_parameters.signal_samples
                ) * (1 / (2 * self.axes_parameters.frequency_length))
            else:
                self.axes_parameters.signal_step = 1 / (
                    2 * self.axes_parameters.frequency_length
                )
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()

    def _define_parameters_with_parity(self):
        """Method to complete a fully defined parameter set with or assuming
        the parity of the signal samples"""

        if (
            self.axes_parameters.frequency_step
            and self.axes_parameters.frequency_length
        ):
            self.axes_parameters.frequency_samples = int(
                np.round(
                    self.axes_parameters.frequency_length
                    / (self.axes_parameters.frequency_step - 1)
                    + 1,
                    0,
                )
            )
            self._signal_from_frequency()
        elif (
            self.axes_parameters.frequency_step
            and self.axes_parameters.frequency_samples
        ):
            self.axes_parameters.frequency_length = (
                self.axes_parameters.frequency_samples - 1
            ) * self.axes_parameters.frequency_step
            self._signal_from_frequency()
        elif (
            self.axes_parameters.frequency_length
            and self.axes_parameters.frequency_samples
        ):
            self.axes_parameters.frequency_step = (
                self.axes_parameters.frequency_length
                / (self.axes_parameters.frequency_samples - 1)
            )
            self._signal_from_frequency()
        elif (
            self.axes_parameters.signal_step
            and self.axes_parameters.frequency_samples
        ):
            self._signal_samples_from_frequency_samples()
            self.axes_parameters.signal_length = (
                self.axes_parameters.signal_samples - 1
            ) * self.axes_parameters.signal_step
            self._frequency_from_signal()
        elif (
            self.axes_parameters.signal_length
            and self.axes_parameters.frequency_samples
        ):
            self._signal_samples_from_frequency_samples()
            self.axes_parameters.signal_step = (
                self.axes_parameters.signal_length
                / (self.axes_parameters.signal_samples - 1)
            )
            self._frequency_from_signal()
        elif (
            self.axes_parameters.signal_length
            and self.axes_parameters.frequency_length
        ):
            if self.signal_samples_parity_even is True:
                self.axes_parameters.signal_samples = int(
                    np.round(
                        (
                            2
                            * self.axes_parameters.signal_length
                            * self.axes_parameters.frequency_length
                            + 1
                        )
                    )
                )
                self._signal_samples_padded()
                self.axes_parameters.signal_step = (
                    self.axes_parameters.signal_length
                    / (self.axes_parameters.signal_samples - 1)
                )
                self._frequency_from_signal()
            elif self.signal_samples_parity_even is False:
                factor = (
                    self.axes_parameters.signal_length
                    * self.axes_parameters.frequency_length
                    + 1
                )
                self.axes_parameters.signal_samples = int(
                    np.round(factor + np.sqrt(factor**2 - 1))
                )
                self._signal_samples_padded()
                self.axes_parameters.signal_step = (
                    self.axes_parameters.signal_length
                    / (self.axes_parameters.signal_samples - 1)
                )
                self._frequency_from_signal()
