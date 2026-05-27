""" This module descibes classes that can be used to generate biases for measurements."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class MeasurementBias(ABC):
    def __init__(self, measurement_loader, measurement_covariance_loader):
        self.measurement_loader = measurement_loader
        self.measurement_covariance_loader = measurement_covariance_loader

    @abstractmethod
    def set_bias(self):
        pass

    @abstractmethod
    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        """Generate a bias for the given measurements.

        Args:
            measurements (xr.DataArray): The measurements to generate the bias for.

        Returns:
            xr.DataArray: The bias for the given measurements.
        """
        pass


class ConstantBias(MeasurementBias):
    def __init__(self, measurement_loader, measurement_covariance_loader, bias: float):
        """Bias loader for measurements with constant bias.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in inversion.
            measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
                covariance loader used in inversion.
            bias (float): The constant bias to apply to the measurements.
        """
        super().__init__(measurement_loader, measurement_covariance_loader)
        self.bias = bias

    def set_bias(self):
        pass

    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        return xr.full_like(measurements, self.bias * 1e-6)


class ConstantBiasTotalOnly(MeasurementBias):
    def __init__(
        self,
        measurement_loader,
        measurement_covariance_loader,
        bias: float,
        co2_ff_mplace_name: str = "co2_ff",
    ):
        """Constant bias that only applies to total CO2 measurements.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in inversion.
            measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
                covariance loader used in inversion.
            bias (float): The constant bias to apply to total CO2 measurements.
            co2_ff_mplace_name (str, optional): MPlace name used for CO2_ff measurements.
                Defaults to "co2_ff".
        """
        super().__init__(measurement_loader, measurement_covariance_loader)
        self.bias = bias
        self.co2_ff_mplace_name = co2_ff_mplace_name

    def set_bias(self):
        pass

    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        if "MPlace" not in measurements.coords:
            return xr.full_like(measurements, self.bias * 1e-6)
        mplace_name = self.co2_ff_mplace_name
        if (
            isinstance(mplace_name, str)
            and measurements.MPlace.dtype.kind == "S"
        ):
            mplace_name = np.array(mplace_name, dtype="S")
        return xr.where(
            measurements.MPlace != mplace_name,
            self.bias * 1e-6,
            0.0,
        )


class RandomStaticBias(MeasurementBias):
    def __init__(
        self, measurement_loader, measurement_covariance_loader, standard_devition_ppm
    ):
        """Bias loader for measurements with random static bias.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in inversion.
            measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
                covariance loader used in inversion.
            standard_devition_ppm (float): The standard deviation of the bias in ppm.
        """
        super().__init__(measurement_loader, measurement_covariance_loader)
        self.standard_devition_ppm = standard_devition_ppm
        self._bias = None

    def set_bias(self):
        mplaces = self.measurement_loader.measurements.unstack().MPlace
        mtimes = self.measurement_loader.measurements.unstack().MTime
        self._bias = (
            xr.DataArray(
                np.random.normal(0, self.standard_devition_ppm * 1e-6, len(mplaces)),
                coords=[("MPlace", mplaces.values)],
            )
            .expand_dims(MTime=mtimes.values)
            .stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
        )

    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        bias_selection = self._bias.isel(
            measurement=(
                self._bias.MTime.isin(measurements.unstack().MTime)
                & self._bias.MPlace.isin(measurements.unstack().MPlace)
            )
        )
        return bias_selection


class ConstantPlusRandomBias(MeasurementBias):
    def __init__(
        self,
        measurement_loader,
        measurement_covariance_loader,
        constant_bias,
        standard_devition_ppm,
    ):
        """Bias loader for measurements with constant bias and random static bias.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in inversion.
            measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
                covariance loader used in inversion.
            constant_bias (float): The constant bias to apply to the measurements.
            standard_devition_ppm (float): The standard deviation of the bias in ppm.
        """
        super().__init__(measurement_loader, measurement_covariance_loader)
        self.constant_bias = constant_bias
        self.standard_devition_ppm = standard_devition_ppm
        self._bias = None

    def set_bias(self):
        mplaces = self.measurement_loader.measurements.unstack().MPlace
        mtimes = self.measurement_loader.measurements.unstack().MTime
        self._bias = (
            xr.DataArray(
                np.random.normal(0, self.standard_devition_ppm * 1e-6, len(mplaces)),
                coords=[("MPlace", mplaces.values)],
            )
            .expand_dims(MTime=mtimes.values)
            .stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
        ) + self.constant_bias * 1e-6

    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        bias_selection = self._bias.isel(
            measurement=(
                self._bias.MTime.isin(measurements.unstack().MTime)
                & self._bias.MPlace.isin(measurements.unstack().MPlace)
            )
        )
        return bias_selection


class RelativeBias(MeasurementBias):
    def __init__(
        self,
        measurement_loader,
        measurement_covariance_loader,
        relative_bias,
    ):
        """Bias loader for measurements with relative bias.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in inversion.
            measurement_covariance_loader (MeasurementCovarianceLoader): Measurement
                covariance loader used in inversion.
            relative_bias (float): The relative bias to apply to the measurements.
        """
        super().__init__(measurement_loader, measurement_covariance_loader)
        self.relative_bias = relative_bias

    def set_bias(self):
        pass

    def generate_bias(self, measurements: xr.DataArray) -> xr.DataArray:
        return measurements * self.relative_bias
