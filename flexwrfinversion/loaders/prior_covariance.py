""" This module descibes classes that can be used to load/build prior covariance data for
     an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.prior import PriorLoader, ShiftToBiospheric


class PriorCovarianceLoader(ABC):
    @abstractmethod
    def __init__(self, prior_loader: PriorLoader, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def prior_std(self) -> xr.DataArray:
        """Load the prior data
        Returns:
            xr.DataArray: The prior standard deviation data as 1D array. Unstacked
                coordinates are supposed to be passed.
        """
        pass

    @abstractmethod
    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        """Load the prior data
        Returns:
            xr.DataArray: The prior covariance data as 2D array. Coordinates should be
                stacked beforehand.
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


class RelativeErrorWithSpatialCorrelation(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: ShiftToBiospheric,
        spatial_correlation_path: str | Path = None,
        relative_error: float = 1,
    ):
        self.prior_loader = prior_loader
        self._spatial_correlation_path = spatial_correlation_path
        self._relative_error = relative_error
        self._prior_std = None
        self._spatial_correlation = None

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.prior) * self._relative_error
        return self._prior_std

    @property
    def spatial_correlation(self) -> xr.DataArray:
        if self._spatial_correlation is None:
            if self._spatial_correlation_path is None:
                spatial_coordinate_values = (
                    self.prior_loader.prior.unstack().subsector.values
                )
                self._spatial_correlation = xr.DataArray(
                    np.eye(
                        len(spatial_coordinate_values),
                        dtype=self.prior_loader.prior.dtype,
                    ),
                    coords=[
                        ("subsector0", spatial_coordinate_values),
                        ("subsector1", spatial_coordinate_values),
                    ],
                ).compute()
            else:
                self._spatial_correlation = xr.open_dataarray(
                    self._spatial_correlation_path
                ).compute()
        return self._spatial_correlation

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        unstacked_prior_std = self.prior_std.unstack()
        time_values = unstacked_prior_std.Time.sel(
            Time=slice(start_time, end_time)
        ).values
        temporal_correlation = xr.DataArray(
            np.eye(
                len(time_values),
                dtype=self.prior_loader.prior.dtype,
            ),
            coords=[("Time0", time_values), ("Time1", time_values)],
        )
        correlation = (self.spatial_correlation * temporal_correlation).stack(
            state0=[
                state_dim + "0"
                for state_dim in self.prior_loader.target_loader.STATE_DIMS
            ],
            state1=[
                state_dim + "1"
                for state_dim in self.prior_loader.target_loader.STATE_DIMS
            ],
        )
        return (
            (
                self._to_two_dimensions(
                    unstacked_prior_std.sel(Time=slice(start_time, end_time)).stack(
                        state=self.prior_loader.target_loader.STATE_DIMS
                    )
                )
                * correlation
            )
            .astype(np.float32)
            .compute()
        )
