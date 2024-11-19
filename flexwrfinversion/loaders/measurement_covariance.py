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
            np.ones_like(measurement_subset, dtype=measurement_subset.dtype)
            * self._ppm_error
            * 1e-6,
            coords=measurement_subset.coords,
        )
        return (
            (xr.zeros_like(self._to_two_dimensions(std)) + np.diag(std.data**2))
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
