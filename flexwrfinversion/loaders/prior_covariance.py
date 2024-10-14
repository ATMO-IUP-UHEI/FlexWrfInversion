""" This module descibes classes that can be used to load/build prior covariance data for
     an inversion."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

from flexwrfinversion.loaders.prior import (
    FlexiblePriorLoaderTotal_ShiftToBiospheric,
    PriorLoader,
)


class PriorCovarianceLoader(ABC):
    @abstractmethod
    def __init__(self, prior_loader: PriorLoader, *args, **kwargs):
        self.prior_loader = prior_loader

    @property
    @abstractmethod
    def prior_std(self) -> xr.DataArray:
        """Property of prior standard deviation.
        Returns:
            xr.DataArray: The prior standard deviation data as 1D array. Unstacked
                coordinates are supposed to be passed.
        """
        pass

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


class RelativeErrorWithSpatialCorrelation(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: FlexiblePriorLoaderTotal_ShiftToBiospheric,
        spatial_correlation_path: str | Path = None,
        relative_error: float = 1,
    ):
        """Prior covariance loader with standard deviation that is given as relative
        error to the prior. Spatial correlation can be used if given in proper format.

        Args:
            prior_loader (FlexiblePriorLoaderTotal_ShiftToBiospheric): Prior loader used
                 in the Inversion.
            spatial_correlation_path (str | Path, optional): Path to file with spatial
                 correlations. Defaults to None.
            relative_error (float, optional): Relative error to use. `1` corresponds.
                 to a 100% error. Defaults to 1.
        """
        super().__init__(prior_loader)
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
        """Full spatial correlation matrix (no temporal part included).

        Returns:
            xr.DataArray: Spatial correlation matrix.
        """
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


class TargetAsErrorNoCorrelation(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: PriorLoader,
        minimum_error: float = None,
    ):
        """Use target values as std for the covariance. Cannot use correlation.

        Args:
            prior_loader (PriorLoader): Prior loader of the inversion.
            minimum_error (float, optional): Minimal error to allow. Defaults to None.
        """
        super().__init__(prior_loader)
        self._prior_std = None
        self._minimum_error = minimum_error

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.target_loader.target)
            if self._minimum_error is not None:
                self._prior_std = xr.where(
                    self._prior_std < self._minimum_error,
                    self._minimum_error,
                    self._prior_std,
                )

        return self._prior_std

    def load_timeframe(
        self, start_time: np.datetime64, end_time: np.datetime64
    ) -> xr.DataArray:
        prior_selection = (
            self.prior_std.unstack()
            .sel(Time=slice(start_time, end_time))
            .stack(state=self.prior_loader.target_loader.STATE_DIMS)
        )
        return self._to_two_dimensions(prior_selection) * np.eye(
            prior_selection.shape[0]
        )


class TargetAsErrorWithCO_Correlation(PriorCovarianceLoader):
    def __init__(
        self,
        prior_loader: PriorLoader,
        anth_co_correlation: float = 0,
        spatial_correlation_path: str | Path = None,
    ):
        """Prior covariance loader for anthropogenic and biogenic CO2 together with CO.
        A correlation is built into the covariace between CO2_anth and CO.

        Args:
            prior_loader (PriorLoader): Prior loader used in the inversion
            anth_co_correlation (float, optional): Correlation between anthropogenic CO2
                 and CO emissions. Defaults to 0.
            spatial_correlation_path (str | Path, optional): Path to the spatial
                 correlation to be used. Defaults to None.
        """
        super().__init__(prior_loader)
        self._anth_co_correlation = anth_co_correlation
        self._spatial_correlation_path = spatial_correlation_path
        self._prior_std = None
        self._sector_correlation = None
        self._spatial_correlation = None

    @property
    def prior_std(self) -> xr.DataArray:
        if self._prior_std is None:
            self._prior_std = np.abs(self.prior_loader.target_loader.target)
        return self._prior_std

    @property
    def spatial_correlation(self) -> xr.DataArray:
        """Spatial subpart of the covariance (2D)."""
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

    @property
    def sector_correlation(self) -> xr.DataArray:
        """Sector part of the correlation."""
        if self._sector_correlation is None:
            correlation = self._to_two_dimensions(
                xr.zeros_like(
                    self.prior_loader.prior.unstack().isel(Time=0, subsector=0)
                )
            )
            correlation = correlation + np.eye(correlation.shape[0])
            anth_index = np.argwhere(
                self.prior_loader.prior.unstack().sector.values
                == self.prior_loader.target_loader.ANTH_SECTOR_KEY
            ).item()
            co_index = np.argwhere(
                self.prior_loader.prior.unstack().sector.values
                == self.prior_loader.target_loader.CO_SECTOR_KEY
            ).item()
            correlation[anth_index, co_index] = self._anth_co_correlation
            correlation[co_index, anth_index] = self._anth_co_correlation
            self._sector_correlation = correlation
        return self._sector_correlation

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
        correlation = (
            self.spatial_correlation * temporal_correlation * self.sector_correlation
        ).stack(
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
