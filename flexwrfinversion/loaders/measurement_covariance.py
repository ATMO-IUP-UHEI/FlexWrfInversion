""" This module descibes classes that can be used to load/build measurement covariance
 data for an inversion."""

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.measurement import MeasurementFromFile, MeasurementLoader


class MeasurementCovarianceLoader(ABC):
    @abstractmethod
    def __init__(self, measurement_loader: MeasurementLoader, *args, **kwargs):
        pass

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the measurement data
        Returns:
            xr.DataArray: The measurement covariance data as 2D array. Coordinates should
                 be stacked beforehand.
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
    def __init__(self, measurement_loader: MeasurementFromFile, ppm_error: float):
        self.measurement_loader = measurement_loader
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
            .astype(np.float32)
            .compute()
        )
