""" This module descibes classes that can be used to load/build measurement covariance
 data for an inversion."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.measurement import MeasurementLoader

FLOAT_PRECISION = np.float32


class MeasurementCovarianceLoader(ABC):
    @abstractmethod
    def __init__(self, measurement_loader: MeasurementLoader, *args, **kwargs):
        self.measurement_loader = measurement_loader

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load a timeframe of the covariance.

        Args:
            start_time (np.datetime64): Start time of the convariance timeframe
            end_time (np.datetime64): End time of the convariance timeframe
        Returns:
            xr.DataArray: The covariance as 2D array. Coordinates should be stacked
                beforehand.
        """
        pass

    @staticmethod
    def _to_two_dimensions(one_d_dataarray: xr.DataArray):
        """Convert a vector to two dimensions by renaming the dimensions.

        Args:
            one_d_dataarray (xr.DataArray): Vector to convert.

        Returns:
            xr.DataArray: Outer product of vector with itself.
        """
        vector0 = one_d_dataarray.rename(
            dict([(dim, f"{dim}0") for dim in one_d_dataarray.coords.keys()])
        )
        vector1 = one_d_dataarray.rename(
            dict([(dim, f"{dim}1") for dim in one_d_dataarray.coords.keys()])
        )
        return vector0 * vector1


class ConstantNoCorrelation(MeasurementCovarianceLoader):
    def __init__(self, measurement_loader: MeasurementLoader, ppm_error: float):
        """Covariance loader for measurements with constant standard deviation and no
        correlation.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in
                 inversion.
            ppm_error (float): Error to apply to each measurment in ppm.
        """
        super().__init__(measurement_loader)
        self._ppm_error = ppm_error

    def _get_measurement_subset(self, start_time, end_time):
        return (
            self.measurement_loader.measurements.unstack()
            .sel(MTime=slice(start_time, end_time))
            .stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
        )

    def _set_constant_std(self, measurement_subset):
        std = xr.DataArray(
            np.ones_like(measurement_subset, dtype=measurement_subset.dtype),
            coords=measurement_subset.coords,
        )
        std *= self._ppm_error * 1e-6
        return std

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        measurement_subset = self._get_measurement_subset(start_time, end_time)
        std = self._set_constant_std(measurement_subset)
        return (
            (xr.zeros_like(self._to_two_dimensions(std)) + np.diag(std.data**2))
            .astype(FLOAT_PRECISION)
            .compute()
        )


class ConstantPlusRelativeNoCorrelation(ConstantNoCorrelation):
    def __init__(
        self,
        measurement_loader: MeasurementLoader,
        ppm_error: float,
        relative_error: float,
    ):
        """Covariance loader for measurements with constant standard deviation and no
        correlation.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in
                 inversion.
            ppm_error (float): Error to apply to each measurment in ppm.
            relative_error (float): Relative error to apply to each measurment.
        """
        super().__init__(measurement_loader, ppm_error)
        self._relative_error = relative_error

    def load_timeframe(self, start_time, end_time):
        measurement_subset = self._get_measurement_subset(start_time, end_time)
        std_const = self._set_constant_std(measurement_subset)
        std_rel = self._relative_error * np.abs(measurement_subset)
        return (
            (
                xr.zeros_like(self._to_two_dimensions(std_const))
                + np.diag(std_const.data**2)
                + np.diag(std_rel.data**2)
            )
            .astype(FLOAT_PRECISION)
            .compute()
        )


class ConstantNoCorrelationCO(MeasurementCovarianceLoader):
    def __init__(
        self,
        measurement_loader: MeasurementLoader,
        ppm_error: float,
        ppb_error: float,
    ):
        """Covariance loader for measurements with constant standard deviation for CO2
             and CO seperately and no correlation.

        Args:
            measurement_loader (MeasurementFromFile): Measurement loader used in
                 inversion.
            ppm_error (float): Error to apply to each CO2 measurement in ppm.
            ppb_error (float): Error to apply to each CO measurement in ppb.
        """
        super().__init__(measurement_loader)
        self._ppm_error = ppm_error
        self._ppb_error = ppb_error

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        measurement_subset = (
            self.measurement_loader.measurements.unstack()
            .sel(MTime=slice(start_time, end_time))
            .stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
        )
        std = xr.DataArray(
            np.ones_like(measurement_subset, dtype=measurement_subset.dtype),
            coords=measurement_subset.coords,
        )
        std *= (
            std.where(measurement_subset.species == "CO2", 0) * 1e-6 * self._ppm_error
        ) + (std.where(measurement_subset.species == "CO", 0) * 1e-9 * self._ppb_error)
        return (
            (xr.zeros_like(self._to_two_dimensions(std)) + np.diag(std.data**2))
            .astype(FLOAT_PRECISION)
            .compute()
        )


class FromFileNoCorrelation(MeasurementCovarianceLoader):
    def __init__(
        self,
        measurement_loader: MeasurementLoader,
        std_file: str,
        ppm_error: float = 0,
        add_quadratic: bool = True,
    ):
        """Covariance loader for measurements with constant standard deviation and no
        correlation.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in
                 inversion.
            std_file (str): Path to file with standard deviations.
            ppm_error (float): Error to apply to each measurment in ppm.
            add_quadratic (bool): Add additional errors quardatically or not.
        """
        super().__init__(measurement_loader)
        self._std_file = std_file
        self._ppm_error = ppm_error
        self._add_quadratic = add_quadratic
        self._std = None

    @property
    def std(self):
        if self._std is None:
            self._std = xr.open_dataarray(self._std_file).stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
            if self._ppm_error != 0:
                if self._add_quadratic:
                    self._std = np.sqrt(self._std**2 + self._ppm_error**2 * 1e-12)
                else:
                    self._std += self._ppm_error * 1e-6
        return self._std

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        measurement_subset = self.measurement_loader.measurements.unstack().sel(
            MTime=slice(start_time, end_time)
        )
        selection = dict(
            MTime=measurement_subset.MTime.values,
            MPlace=measurement_subset.MPlace.values,
        )
        if "species" in measurement_subset.dims:
            selection["species"] = measurement_subset.species.values

        std_subset = (
            self.std.unstack()
            .sel(**selection)
            .stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
        )
        return (
            (
                xr.zeros_like(self._to_two_dimensions(std_subset))
                + np.diag(std_subset.data**2)
            )
            .astype(FLOAT_PRECISION)
            .compute()
        )


class FromFileNoCorrelationCO(FromFileNoCorrelation):
    def __init__(
        self,
        measurement_loader: MeasurementLoader,
        std_file: str,
        ppm_error: float = 0,
        ppb_error: float = 0,
        add_quadratic: bool = True,
    ):
        """Covariance loader for measurements with constant standard deviation and no
        correlation.

        Args:
            measurement_loader (MeasurementLoader): Measurement loader used in
                 inversion.
            std_file (str): Path to file with standard deviations.
            ppm_error (float): Error to apply to each measurment in ppm.
            ppb_error (float): Error to apply to each measurment in ppb.
            add_quadratic (bool): Add additional errors quardatically or not.
        """
        super().__init__(measurement_loader, std_file, ppm_error, add_quadratic)
        self._ppb_error = ppb_error

    @property
    def std(self):
        if self._std is None:
            self._std = xr.open_dataarray(self._std_file).stack(
                measurement=self.measurement_loader.footprint_loader.MEASUREMENT_DIMS
            )
            if self._ppm_error != 0:
                self._std = xr.where(
                    self._std.species == "CO2",
                    (self._std + self._ppm_error * 1e-6)
                    if not self._add_quadratic
                    else np.sqrt(self._std**2 + self._ppm_error**2 * 1e-12),
                    self._std,
                )
            if self._ppb_error != 0:
                self._std = xr.where(
                    self._std.species == "CO",
                    (self._std + self._ppb_error * 1e-9)
                    if not self._add_quadratic
                    else np.sqrt(self._std**2 + self._ppb_error**2 * 1e-18),
                    self._std,
                )
        return self._std
